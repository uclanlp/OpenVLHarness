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


def _claim_is_stale(path: str, stale_seconds: int) -> bool:
    """A claim is stale if (a) its mtime is older than ``stale_seconds`` OR
    (b) the recorded PID is on the same host but no longer alive (catches
    SIGKILL'd workers immediately, without waiting for the mtime timeout).
    """
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return True
    if time.time() - mtime > stale_seconds:
        return True
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return False  # can't read payload; trust the fresh mtime
    pid = payload.get("pid")
    host = payload.get("host")
    if not pid or host != socket.gethostname():
        return False  # different host; we can't check liveness, trust mtime
    try:
        os.kill(int(pid), 0)
        return False  # alive
    except ProcessLookupError:
        return True   # dead — claim is orphaned regardless of mtime
    except PermissionError:
        return False  # alive but owned by another uid; treat as fresh


def claim(work_dir: str, idx: Any, stale_seconds: int = DEFAULT_STALE_SECONDS) -> bool:
    """Atomically claim ``idx``. Returns True on success.

    Initial-claim path uses ``O_CREAT|O_EXCL`` for atomicity. Stale
    reclaim is serialized through a sibling ``<path>.reclaiming`` mutex
    (also ``O_EXCL``) so that under heavy concurrency only one worker
    actually performs the unlink+recreate sequence; a re-check of
    staleness inside the mutex prevents late arrivals from clobbering
    the just-installed fresh claim.
    """
    os.makedirs(_claim_dir(work_dir), exist_ok=True)
    path = _claim_path(work_dir, idx)
    payload = json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.time(),
    }).encode("utf-8")
    reclaim_payload = json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.time(),
        "reclaimed": True,
    }).encode("utf-8")

    # Step 1 — try to create the canonical claim outright.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        pass

    # Step 2 — claim already exists. Is it stale?
    if not _claim_is_stale(path, stale_seconds):
        return False

    # Step 3 — stale. Acquire the per-claim reclaim mutex so at most one
    # worker performs the unlink+recreate sequence. Workers that lose
    # the mutex back off; the winner re-verifies staleness inside the
    # mutex (another worker may have reclaimed while we waited) before
    # touching the canonical path.
    marker = path + ".reclaiming"
    try:
        mfd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(mfd)
    except FileExistsError:
        return False  # another worker holds the reclaim mutex

    try:
        # Re-check staleness now that we hold the mutex. If another
        # worker reclaimed between our staleness check and our mutex
        # acquisition, the canonical claim is fresh and we must back off.
        if not _claim_is_stale(path, stale_seconds):
            return False
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, reclaim_payload)
        finally:
            os.close(fd)
        return True
    finally:
        try:
            os.unlink(marker)
        except FileNotFoundError:
            pass


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
    """Return idx (as strings) currently claimed by a live worker.

    Uses ``_claim_is_stale`` so that SIGKILL'd workers' orphaned claims
    are recognized as stale immediately on resume, rather than waiting
    out the ``stale_seconds`` mtime timeout.
    """
    d = _claim_dir(work_dir)
    if not osp.isdir(d):
        return set()
    fresh: Set[str] = set()
    for name in os.listdir(d):
        if not name.endswith(".lock"):
            continue
        path = osp.join(d, name)
        if not _claim_is_stale(path, stale_seconds):
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


def _dir_size_bytes(path: str) -> int:
    """Best-effort recursive byte count. Unreadable entries are skipped."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += osp.getsize(osp.join(root, name))
            except OSError:
                pass
    return total


def is_safe_to_cleanup(
    work_dir: str,
    *,
    result_file_exists: bool,
    expected_n: int,
    has_failures: bool,
    stale_seconds: int = DEFAULT_STALE_SECONDS,
) -> Tuple[bool, str]:
    """Decide whether the per-example file-db under ``work_dir`` can be deleted.

    Returns ``(ok, reason)``. The heavy inputs (whether the aggregated
    prediction file exists, whether any FAIL_MSG rows remain) are computed by
    the caller and passed in, so this stays dependency-light and unit-testable.
    All of the following must hold:

      * the aggregated prediction file exists (durable copy of every record),
      * no FAIL_MSG rows remain (preserve the db so failures can be retried
        in place via ``--ignore-failed`` without rebuilding from the xlsx),
      * no live worker still holds a claim on this work_dir,
      * every expected example has a committed done flag (full coverage).
    """
    if not result_file_exists:
        return False, "aggregated result file not found"
    if has_failures:
        return False, "FAIL_MSG rows present; preserving db for retry"
    fresh = load_fresh_claims(work_dir, stale_seconds=stale_seconds)
    if fresh:
        return False, f"{len(fresh)} live claim(s) present"
    done = len(load_done_set(work_dir))
    if done < expected_n:
        return False, f"incomplete coverage: {done}/{expected_n} done flags"
    return True, "ok"


def cleanup(work_dir: str) -> dict:
    """Idempotently remove the per-example file-db (``results/``, ``_done/``,
    ``_claims/``) under ``work_dir``.

    Safe to call when the layout is already gone. Never raises: filesystem
    errors are swallowed because the run's durable outputs (the aggregated
    xlsx, ``full_trajectories/``, etc.) live elsewhere and are untouched.
    Returns ``{"removed": [<abs paths removed>], "freed_bytes": <int>}``.
    """
    targets = [_results_root(work_dir), _done_dir(work_dir), _claim_dir(work_dir)]
    removed = []
    freed = 0
    for path in targets:
        if not osp.isdir(path):
            continue
        freed += _dir_size_bytes(path)
        shutil.rmtree(path, ignore_errors=True)
        if not osp.exists(path):
            removed.append(path)
    return {"removed": removed, "freed_bytes": freed}


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
