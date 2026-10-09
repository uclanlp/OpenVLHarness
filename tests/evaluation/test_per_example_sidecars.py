"""End-to-end test for the per-example sidecar persistence path.

Verifies that infer_image_job:
  - writes per-example record.json + _done/<idx>.flag, not _supp.pkl
  - the aggregated xlsx contains all expected predictions
  - a second run is a no-op (result file present → short-circuit)
"""

import contextlib
import os
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from evaluation_kit import runner
from evaluation_kit.utils import file_db

pytestmark = pytest.mark.usefixtures("synchronous_evaluation")


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
        runner.infer_image_job(_Model(), str(work_dir), "M", _Dataset())

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
        result = runner.load(str(ds_dir / "M.xlsx"))
        preds = dict(zip(result["index"], result["prediction"]))
        assert preds == {10: "ans-10", 20: "ans-20", 30: "ans-30"}


def test_second_run_short_circuits():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        runner.infer_image_job(_Model(), str(work_dir), "M", _Dataset())

        # Spy on the model: a second invocation must not call generate().
        class _Spy(_Model):
            calls = 0
            def generate(self, **kwargs):
                _Spy.calls += 1
                return super().generate(**kwargs)

        runner.infer_image_job(_Spy(), str(work_dir), "M", _Dataset())
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

        runner.infer_image_job(_Spy(), str(Path(tmp)), "M", _Dataset())

        # Only idx 10 and 30 should have been re-run.
        assert sorted(_Spy.seen) == [10, 30]
        # The pre-seeded idx=20 record survives.
        rec = file_db.load_record(str(work_dir), 20)
        assert rec["prediction"] == "previous"
