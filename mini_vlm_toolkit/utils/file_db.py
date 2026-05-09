"""Atomic per-example persistence primitives.

Replaces the cumulative-pickle-rewrite scheme in ``mp_util.track_progress_rich``
for tool-heavy datasets where the per-iter ``dump(res, save)`` write becomes
quadratic in N. Workers commit one example at a time via temp-dir + atomic
rename + done flag; resume is a directory listing, not a pickle load.

Layout under ``<work_dir>``::

    results/<shard>/<idx>/record.json   # source-of-truth per example
    _claims/<idx>.lock                  # O_EXCL claim, mtime touched by heartbeat
    _done/<idx>.flag                    # empty file, atomic create at commit

All atomicity guarantees assume a local filesystem (ext4/xfs/zfs/btrfs).
Not safe over NFS without ``flock``.
"""

import hashlib
import json
import os
import os.path as osp
import shutil
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Set, Tuple

DEFAULT_STALE_SECONDS = 15 * 60
DEFAULT_HEARTBEAT_SECONDS = 30


def shard_for(idx: Any) -> str:
    """Deterministic 2-char hex shard key. Stable across processes (unlike
    ``hash()``, which honors PYTHONHASHSEED)."""
    return f"{hashlib.md5(str(idx).encode('utf-8')).digest()[0]:02x}"


def _claim_dir(work_dir: str) -> str:
    return osp.join(work_dir, "_claims")


def _done_dir(work_dir: str) -> str:
    return osp.join(work_dir, "_done")


def _results_root(work_dir: str) -> str:
    return osp.join(work_dir, "results")


def _claim_path(work_dir: str, idx: Any) -> str:
    return osp.join(_claim_dir(work_dir), f"{idx}.lock")


def _done_path(work_dir: str, idx: Any) -> str:
    return osp.join(_done_dir(work_dir), f"{idx}.flag")


def _record_dir(work_dir: str, idx: Any) -> str:
    return osp.join(_results_root(work_dir), shard_for(idx), str(idx))


def ensure_layout(work_dir: str) -> None:
    """Create top-level ``_claims/`` and ``_done/`` dirs if missing."""
    os.makedirs(_claim_dir(work_dir), exist_ok=True)
    os.makedirs(_done_dir(work_dir), exist_ok=True)


def claim(work_dir: str, idx: Any, stale_seconds: int = DEFAULT_STALE_SECONDS) -> bool:
    """Atomically claim ``idx``. Returns True on success.

    Uses ``O_CREAT|O_EXCL`` for the atomic primitive. If a claim file exists
    but its mtime is older than ``stale_seconds``, the original heartbeat is
    presumed dead and the claim is reclaimed by overwriting.
    """
    os.makedirs(_claim_dir(work_dir), exist_ok=True)
    path = _claim_path(work_dir, idx)
    payload = json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.time(),
    }).encode("utf-8")

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(path)
        except FileNotFoundError:
            return claim(work_dir, idx, stale_seconds=stale_seconds)
        if age < stale_seconds:
            return False
        # Stale — reclaim. The race window where two workers both see a
        # stale mtime is bounded: only one will land its overwrite "last,"
        # and that one's heartbeat keeps the claim fresh.
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": time.time(),
                "reclaimed": True,
            }))
        return True


def release(work_dir: str, idx: Any) -> None:
    """Drop our claim on ``idx``. Idempotent."""
    try:
        os.unlink(_claim_path(work_dir, idx))
    except FileNotFoundError:
        pass


class Heartbeat:
    """Touch a claim file's mtime periodically while a worker is alive.

    Daemon thread; exits with the process. Stop with ``.stop()`` in the
    worker's ``finally`` block.
    """

    def __init__(self, path: str, interval: float = DEFAULT_HEARTBEAT_SECONDS):
        self._path = path
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Heartbeat":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                os.utime(self._path, None)
            except FileNotFoundError:
                return
            except OSError:
                # Best-effort. Don't let a transient FS error kill the worker.
                pass

    def stop(self) -> None:
        self._stop.set()


def start_heartbeat(
    work_dir: str,
    idx: Any,
    interval: float = DEFAULT_HEARTBEAT_SECONDS,
) -> Heartbeat:
    return Heartbeat(_claim_path(work_dir, idx), interval=interval).start()


def commit_atomic(
    work_dir: str,
    idx: Any,
    record: Any,
    *,
    json_encoder: Optional[type] = None,
    extra_files: Optional[dict] = None,
) -> None:
    """Write ``record.json`` (+ any ``extra_files``) and the done flag atomically.

    Sequence:
      1. write all files into a sibling temp dir on the same filesystem
      2. ``os.rename(tmpdir, results/<shard>/<idx>)`` — atomic on local FS
      3. ``open(_done/<idx>.flag, O_CREAT)`` — atomic create

    A SIGKILL between (1) and (2) leaves an orphan tmpdir but no consumer
    sees it. Between (2) and (3), the record exists but isn't marked done;
    the boot scan re-dispatches the example and step (2) replaces.

    ``extra_files`` is an optional ``{filename: bytes-or-str}`` map for
    additional sidecars to land atomically alongside ``record.json`` (e.g.,
    a per-example HTML report).
    """
    shard_dir = osp.join(_results_root(work_dir), shard_for(idx))
    os.makedirs(shard_dir, exist_ok=True)
    final_dir = osp.join(shard_dir, str(idx))

    tmp = tempfile.mkdtemp(prefix=f".{idx}.", dir=shard_dir)
    try:
        with open(osp.join(tmp, "record.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, cls=json_encoder)
        for name, content in (extra_files or {}).items():
            mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
            with open(osp.join(tmp, name), mode) as f:
                f.write(content)
        if osp.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp, final_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    os.makedirs(_done_dir(work_dir), exist_ok=True)
    fd = os.open(_done_path(work_dir, idx), os.O_CREAT | os.O_WRONLY, 0o644)
    os.close(fd)


def load_done_set(work_dir: str) -> Set[str]:
    """Return committed idx (as strings) by scanning ``_done/``."""
    d = _done_dir(work_dir)
    if not osp.isdir(d):
        return set()
    return {
        osp.splitext(name)[0]
        for name in os.listdir(d)
        if name.endswith(".flag")
    }


def load_fresh_claims(
    work_dir: str,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
) -> Set[str]:
    """Return idx (as strings) currently claimed by a live worker."""
    d = _claim_dir(work_dir)
    if not osp.isdir(d):
        return set()
    now = time.time()
    fresh: Set[str] = set()
    for name in os.listdir(d):
        if not name.endswith(".lock"):
            continue
        try:
            age = now - os.path.getmtime(osp.join(d, name))
        except FileNotFoundError:
            continue
        if age < stale_seconds:
            fresh.add(osp.splitext(name)[0])
    return fresh


def load_record(work_dir: str, idx: Any) -> Any:
    """Load a previously-committed ``record.json`` by idx."""
    with open(osp.join(_record_dir(work_dir, idx), "record.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(work_dir: str) -> Iterator[Tuple[str, Any]]:
    """Yield ``(idx_str, record)`` for every committed record. Order undefined."""
    root = _results_root(work_dir)
    if not osp.isdir(root):
        return
    for shard in sorted(os.listdir(root)):
        shard_dir = osp.join(root, shard)
        if not osp.isdir(shard_dir):
            continue
        for idx_name in os.listdir(shard_dir):
            rec_path = osp.join(shard_dir, idx_name, "record.json")
            if not osp.isfile(rec_path):
                continue
            with open(rec_path, "r", encoding="utf-8") as f:
                yield idx_name, json.load(f)


def atomic_write(path: str, content: Any, *, json_encoder: Optional[type] = None) -> None:
    """Write ``content`` to ``path`` via tmp + rename. Atomic on local FS.

    ``content`` may be ``bytes`` (written raw), ``str`` (UTF-8), or any
    JSON-serializable value (dumped with ``json_encoder`` if provided).
    """
    parent = osp.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{osp.basename(path)}.", dir=parent)
    try:
        if isinstance(content, (bytes, bytearray)):
            os.write(fd, content)
        elif isinstance(content, str):
            os.write(fd, content.encode("utf-8"))
        else:
            os.write(
                fd,
                json.dumps(content, ensure_ascii=False, cls=json_encoder).encode("utf-8"),
            )
        os.close(fd)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def mark_done(work_dir: str, idx: Any) -> None:
    """Create the done flag without writing a record. Used by seed migration
    paths (e.g., ``run_experience_learning`` carry-forward) where the record
    has already been written via ``commit_atomic``."""
    os.makedirs(_done_dir(work_dir), exist_ok=True)
    fd = os.open(_done_path(work_dir, idx), os.O_CREAT | os.O_WRONLY, 0o644)
    os.close(fd)


@contextmanager
def claim_and_heartbeat(
    work_dir: str,
    idx: Any,
    *,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_SECONDS,
):
    """Context manager: claim → start heartbeat → run → release.

    Yields ``True`` if the claim was acquired, ``False`` if another worker
    owns it. Caller must check the yielded value::

        with claim_and_heartbeat(work_dir, idx) as got:
            if not got:
                return  # someone else has it
            ...
    """
    if not claim(work_dir, idx, stale_seconds=stale_seconds):
        yield False
        return
    hb = start_heartbeat(work_dir, idx, interval=heartbeat_interval)
    try:
        yield True
    finally:
        hb.stop()
        release(work_dir, idx)
