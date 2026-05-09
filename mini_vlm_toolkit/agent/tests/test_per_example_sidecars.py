"""End-to-end test for the per-example sidecar persistence path.

Verifies that infer_image_job:
  - writes per-example record.json + _done/<idx>.flag, not _supp.pkl
  - the aggregated xlsx contains all expected predictions
  - a second run is a no-op (result file present → short-circuit)
"""

import contextlib
import os
from pathlib import Path
import sys
import tempfile
import types

import pandas as pd

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]


def _track_progress_rich(func, tasks, save=None, keys=None, **_kwargs):
    """Synchronous stand-in that exercises the wrapper exactly like a thread
    pool would, but in test order (no concurrency)."""
    results = []
    for key, task in zip(keys, tasks):
        results.append(func(**task))
    if save is not None:
        from mini_vlm_toolkit.smp import dump
        dump({k: r for k, r in zip(keys, results)}, save)
    return results


utils_pkg.track_progress_rich = _track_progress_rich
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

# Stub reporting_utils — the goal here is to exercise the persistence and
# aggregation paths, not to render real HTML/full_trajectory artifacts.
reporting_stub = types.ModuleType("mini_vlm_toolkit.utils.reporting_utils")
reporting_stub.build_trajectory_records = lambda indices, data_all, **kwargs: [
    {"task_id": idx, "trajectory": (data_all[idx] or {}).get("extra_records", {}).get("trajectory")
     if isinstance(data_all[idx], dict) else None,
     "prediction": (data_all[idx] or {}).get("prediction") if isinstance(data_all[idx], dict) else None,
     "extra_records": (data_all[idx] or {}).get("extra_records", {}) if isinstance(data_all[idx], dict) else {},
     "dataset": kwargs.get("dataset_name"),
     "model_name": kwargs.get("model_name"),
     "agent_class": kwargs.get("agent_class")}
    for idx in indices
]
reporting_stub.write_input_snapshot_json = lambda *a, **kw: None
reporting_stub.write_html_reports = lambda *a, **kw: None
reporting_stub.write_full_trajectory_jsons = lambda *a, **kw: None
reporting_stub.write_raw_response_jsons = lambda *a, **kw: None
reporting_stub.write_raw_trajectory_jsons = lambda *a, **kw: None
reporting_stub.build_complete_messages_from_trajectory = lambda traj: None
reporting_stub.render_html_report = lambda rec: "<html>stub</html>"
reporting_stub._replace_data_uris_with_files = lambda *a, **kw: None
reporting_stub.select_records_for_report_limit = lambda records, limit: records[:limit] if limit and limit > 0 else records
sys.modules.setdefault("mini_vlm_toolkit.utils.reporting_utils", reporting_stub)

from mini_vlm_toolkit import inference_image  # noqa: E402
from mini_vlm_toolkit.utils import file_db  # noqa: E402


@contextlib.contextmanager
def _patch_env(updates):
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _Dataset:
    dataset_name = "ToySet"
    data = pd.DataFrame({"index": [10, 20, 30], "answer": ["a", "b", "c"]})

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item):
        return [{"type": "text", "value": f"q-{item['index']}"}]

    def dump_image(self, item):
        return []


class _Model:
    def generate(self, message=None, dataset=None, task_id=None, **_):
        return {"prediction": f"ans-{task_id}", "extra_records": {"trajectory": {"messages": []}}}


def test_per_example_sidecars_replace_supp_pkl():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        inference_image.infer_image_job(_Model(), str(work_dir), "M", _Dataset())

        ds_dir = work_dir / "ToySet"

        # Legacy artifacts MUST NOT exist.
        assert not (ds_dir / "M_supp.pkl").exists()
        assert not (ds_dir / "M_PREV.pkl").exists()
        assert not list(ds_dir.glob("0*.pkl"))  # per-rank shards

        # New layout artifacts MUST exist.
        assert (ds_dir / "_done").is_dir()
        assert (ds_dir / "results").is_dir()
        done_files = sorted(p.name for p in (ds_dir / "_done").iterdir())
        assert done_files == ["10.flag", "20.flag", "30.flag"]

        # Each record.json round-trips.
        for idx in (10, 20, 30):
            rec = file_db.load_record(str(ds_dir), idx)
            assert rec["prediction"] == f"ans-{idx}"

        # Aggregated xlsx contains all predictions.
        result = inference_image.load(str(ds_dir / "M.xlsx"))
        preds = dict(zip(result["index"], result["prediction"]))
        assert preds == {10: "ans-10", 20: "ans-20", 30: "ans-30"}


def test_second_run_short_circuits():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        inference_image.infer_image_job(_Model(), str(work_dir), "M", _Dataset())

        # Spy on the model: a second invocation must not call generate().
        class _Spy(_Model):
            calls = 0
            def generate(self, **kwargs):
                _Spy.calls += 1
                return super().generate(**kwargs)

        inference_image.infer_image_job(_Spy(), str(work_dir), "M", _Dataset())
        assert _Spy.calls == 0  # result file existed → no-op


def test_resume_after_partial_commit():
    """Simulate kill-resume: pre-populate _done/ for some idx, run, verify
    only the missing idx are re-inferred."""
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp) / "ToySet"
        work_dir.mkdir(parents=True)
        # Seed idx=20 as already done with a stale prediction.
        file_db.commit_atomic(
            str(work_dir), 20,
            {"prediction": "previous", "extra_records": {}},
        )

        class _Spy(_Model):
            seen = []
            def generate(self, message=None, dataset=None, task_id=None, **_):
                _Spy.seen.append(task_id)
                return {"prediction": f"new-{task_id}",
                        "extra_records": {"trajectory": {"messages": []}}}

        inference_image.infer_image_job(_Spy(), str(Path(tmp)), "M", _Dataset())

        # Only idx 10 and 30 should have been re-run.
        assert sorted(_Spy.seen) == [10, 30]
        # The pre-seeded idx=20 record survives.
        rec = file_db.load_record(str(work_dir), 20)
        assert rec["prediction"] == "previous"


if __name__ == "__main__":
    test_per_example_sidecars_replace_supp_pkl()
    test_second_run_short_circuits()
    test_resume_after_partial_commit()
    print("Per-example sidecar tests passed.")
