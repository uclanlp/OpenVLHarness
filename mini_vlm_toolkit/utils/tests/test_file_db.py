"""Unit tests for the file_db atomic per-example persistence primitives."""

import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

# Avoid importing mini_vlm_toolkit/__init__.py (heavy dataset deps).
mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

from mini_vlm_toolkit.utils import file_db  # noqa: E402


def test_shard_for_is_deterministic_and_2hex():
    s1 = file_db.shard_for("val_001")
    s2 = file_db.shard_for("val_001")
    assert s1 == s2
    assert len(s1) == 2
    int(s1, 16)  # parses as hex


def test_shard_for_spreads_integers():
    shards = {file_db.shard_for(i) for i in range(1000)}
    # Far better than the leading-digit-of-int trap (which gives ~10).
    assert len(shards) > 100


def test_claim_is_exclusive():
    with tempfile.TemporaryDirectory() as work_dir:
        assert file_db.claim(work_dir, "x") is True
        assert file_db.claim(work_dir, "x") is False  # already held
        file_db.release(work_dir, "x")
        assert file_db.claim(work_dir, "x") is True  # released, can reclaim


def test_claim_race_n_threads_one_wins():
    with tempfile.TemporaryDirectory() as work_dir:
        winners = []
        lock = threading.Lock()

        def worker():
            if file_db.claim(work_dir, "shared"):
                with lock:
                    winners.append(threading.current_thread().name)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(winners) == 1


def test_stale_claim_is_reclaimable():
    with tempfile.TemporaryDirectory() as work_dir:
        assert file_db.claim(work_dir, "y") is True
        # Backdate the claim file's mtime past the stale window.
        path = file_db._claim_path(work_dir, "y")
        old = time.time() - 999
        os.utime(path, (old, old))
        # Original holder is "dead" — stale_seconds=1 should let us reclaim.
        assert file_db.claim(work_dir, "y", stale_seconds=1) is True


def test_heartbeat_keeps_claim_fresh():
    with tempfile.TemporaryDirectory() as work_dir:
        assert file_db.claim(work_dir, "z") is True
        path = file_db._claim_path(work_dir, "z")
        # Backdate; heartbeat with short interval should bring it forward.
        os.utime(path, (time.time() - 999, time.time() - 999))
        hb = file_db.start_heartbeat(work_dir, "z", interval=0.05)
        try:
            time.sleep(0.3)
            age = time.time() - os.path.getmtime(path)
            assert age < 1, f"heartbeat did not refresh mtime (age={age:.2f}s)"
        finally:
            hb.stop()


def test_commit_atomic_writes_record_and_flag():
    with tempfile.TemporaryDirectory() as work_dir:
        rec = {"prediction": "42", "extra_records": {"k": "v"}}
        file_db.commit_atomic(work_dir, "abc", rec)
        assert "abc" in file_db.load_done_set(work_dir)
        loaded = file_db.load_record(work_dir, "abc")
        assert loaded == rec


def test_commit_atomic_replaces_existing_record():
    with tempfile.TemporaryDirectory() as work_dir:
        file_db.commit_atomic(work_dir, "k", {"v": 1})
        file_db.commit_atomic(work_dir, "k", {"v": 2})
        assert file_db.load_record(work_dir, "k") == {"v": 2}


def test_commit_atomic_extra_files():
    with tempfile.TemporaryDirectory() as work_dir:
        file_db.commit_atomic(
            work_dir, "h",
            {"prediction": "p"},
            extra_files={"report.html": "<html>ok</html>"},
        )
        html = os.path.join(file_db._record_dir(work_dir, "h"), "report.html")
        assert os.path.isfile(html)
        with open(html) as f:
            assert f.read() == "<html>ok</html>"


def test_iter_records_round_trips():
    with tempfile.TemporaryDirectory() as work_dir:
        for i in range(50):
            file_db.commit_atomic(work_dir, f"item_{i}", {"v": i})
        seen = dict(file_db.iter_records(work_dir))
        assert len(seen) == 50
        assert seen["item_7"] == {"v": 7}


def test_load_done_set_empty_when_dir_missing():
    with tempfile.TemporaryDirectory() as work_dir:
        # Don't call ensure_layout; _done/ shouldn't exist.
        assert file_db.load_done_set(work_dir) == set()


def test_load_fresh_claims_excludes_stale():
    with tempfile.TemporaryDirectory() as work_dir:
        file_db.claim(work_dir, "fresh")
        file_db.claim(work_dir, "stale")
        stale_path = file_db._claim_path(work_dir, "stale")
        os.utime(stale_path, (time.time() - 9999, time.time() - 9999))
        fresh = file_db.load_fresh_claims(work_dir, stale_seconds=10)
        assert "fresh" in fresh
        assert "stale" not in fresh


def test_stale_reclaim_race_n_threads_one_wins():
    """When N workers all see the same stale claim simultaneously, only
    one should win the reclaim. Without the O_EXCL stale-reclaim path,
    every worker would overwrite and all return True, leading to multiple
    workers thinking they own the same idx (commit collision)."""
    import json as _json, socket as _socket
    with tempfile.TemporaryDirectory() as work_dir:
        # Pre-stage a stale claim with a dead PID (forces _claim_is_stale=True
        # for every worker, regardless of mtime).
        os.makedirs(file_db._claim_dir(work_dir), exist_ok=True)
        path = file_db._claim_path(work_dir, "stale_idx")
        with open(path, "w") as f:
            _json.dump(
                {"pid": 999999999, "host": _socket.gethostname(),
                 "started_at": time.time()},
                f,
            )

        winners = []
        winners_lock = threading.Lock()
        ready = threading.Event()

        def worker():
            ready.wait()  # release all workers simultaneously
            if file_db.claim(work_dir, "stale_idx"):
                with winners_lock:
                    winners.append(threading.current_thread().name)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        ready.set()
        for t in threads:
            t.join()
        assert len(winners) == 1, f"expected exactly 1 winner, got {len(winners)}"


def test_stale_reclaim_then_unrelated_worker_cannot_take_over():
    """After a successful stale reclaim by worker A, a later worker B
    must see a fresh claim (not stale) and back off."""
    import json as _json, socket as _socket
    with tempfile.TemporaryDirectory() as work_dir:
        os.makedirs(file_db._claim_dir(work_dir), exist_ok=True)
        path = file_db._claim_path(work_dir, "k")
        # Stale claim with dead PID
        with open(path, "w") as f:
            _json.dump(
                {"pid": 999999999, "host": _socket.gethostname(),
                 "started_at": time.time()},
                f,
            )
        # Worker A reclaims successfully
        assert file_db.claim(work_dir, "k") is True
        # Worker B (this same PID, mid-test) sees the just-reclaimed lock —
        # it's NOT stale anymore (our PID is alive, mtime is fresh).
        assert file_db.claim(work_dir, "k") is False


def test_dead_pid_claim_is_reclaimable_immediately():
    """A claim whose recorded PID is no longer alive must be reclaimable
    even if the mtime is fresh — this is the kill-resume case."""
    import json as _json
    with tempfile.TemporaryDirectory() as work_dir:
        path = file_db._claim_path(work_dir, "k")
        os.makedirs(file_db._claim_dir(work_dir), exist_ok=True)
        # Write a claim whose PID is guaranteed dead but whose mtime is now.
        with open(path, "w") as f:
            _json.dump(
                {"pid": 999999999, "host": __import__("socket").gethostname(),
                 "started_at": time.time()},
                f,
            )
        # Despite fresh mtime, claim should be acquirable (PID is dead).
        assert file_db.claim(work_dir, "k") is True


def test_load_fresh_claims_excludes_dead_pid():
    import json as _json, socket as _socket
    with tempfile.TemporaryDirectory() as work_dir:
        os.makedirs(file_db._claim_dir(work_dir), exist_ok=True)
        # Two claims, both fresh mtime: one with my PID (alive), one with dead PID.
        for idx, pid in (("alive", os.getpid()), ("dead", 999999999)):
            with open(file_db._claim_path(work_dir, idx), "w") as f:
                _json.dump({"pid": pid, "host": _socket.gethostname(), "started_at": time.time()}, f)
        fresh = file_db.load_fresh_claims(work_dir)
        assert "alive" in fresh
        assert "dead" not in fresh


def test_atomic_write_str_and_json():
    with tempfile.TemporaryDirectory() as work_dir:
        p1 = os.path.join(work_dir, "a", "b", "x.txt")
        file_db.atomic_write(p1, "hello")
        with open(p1) as f:
            assert f.read() == "hello"
        p2 = os.path.join(work_dir, "y.json")
        file_db.atomic_write(p2, {"k": [1, 2, 3]})
        import json as _json
        with open(p2) as f:
            assert _json.load(f) == {"k": [1, 2, 3]}


def test_atomic_write_replaces_existing():
    with tempfile.TemporaryDirectory() as work_dir:
        p = os.path.join(work_dir, "z.html")
        file_db.atomic_write(p, "<html>v1</html>")
        file_db.atomic_write(p, "<html>v2</html>")
        with open(p) as f:
            assert f.read() == "<html>v2</html>"


def test_claim_and_heartbeat_context_manager():
    with tempfile.TemporaryDirectory() as work_dir:
        with file_db.claim_and_heartbeat(work_dir, "ctx") as got:
            assert got is True
            # Second concurrent claim should fail.
            assert file_db.claim(work_dir, "ctx") is False
        # After exit, claim file is released.
        assert file_db.claim(work_dir, "ctx") is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
