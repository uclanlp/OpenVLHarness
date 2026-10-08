"""End-to-end test using the real reporting utilities, exercising
the per-example sidecar wrapper through full_trajectories, htmls, and
tool_detections writes."""

import importlib.util
import pytest

if importlib.util.find_spec("openvlharness_eval") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "torch", "validators", "openpyxl", "xlsxwriter"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")


import contextlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from openvlharness_eval import inference_image
from openvlharness_eval.utils import file_db

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
    data = pd.DataFrame({"index": [10, 20, 30, 40], "answer": ["a", "b", "c", "d"]})

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item):
        return [{"type": "text", "value": f"q-{item['index']}"}]

    def dump_image(self, item):
        return []


class _Model:
    """Returns realistic structured records: prediction + extra_records with
    a trajectory dict containing input_messages, chat_history, messages."""

    def generate(self, message=None, dataset=None, task_id=None, **_):
        return {
            "prediction": f"ans-{task_id}",
            "extra_records": {
                "trajectory": {
                    "input_messages": message,
                    "chat_history": [
                        {
                            "user": {"content": "hi"},
                            "assistant": {"content": f"ans-{task_id}", "tool_calls": []},
                            "tool_responses": [],
                        }
                    ],
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": f"ans-{task_id}"},
                    ],
                    "final_answer": f"ans-{task_id}",
                },
                "tool_detections": [{"box": [1, 2, 3, 4], "score": 0.9}] if task_id == 20 else [],
            },
        }


def test_real_reporting_writes_full_trajectories():
    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        inference_image.infer_image_job(
            _Model(), str(work_dir), "M", _Dataset(),
            html_limit=2,  # only first-2 by completion order
            save_raw_trace=False,
        )

        ds = work_dir / "ToySet"

        # Per-task full_trajectories sidecars exist for ALL committed examples.
        full_files = sorted(p.name for p in (ds / "full_trajectories").iterdir() if p.is_file())
        assert full_files == ["10.json", "20.json", "30.json", "40.json"]
        for idx in (10, 20, 30, 40):
            with open(ds / "full_trajectories" / f"{idx}.json") as f:
                payload = json.load(f)
            assert payload["task_id"] == idx
            assert payload["prediction"] == f"ans-{idx}"
            assert payload["ground_truth"] == ("a" if idx == 10 else "b" if idx == 20 else "c" if idx == 30 else "d")
            assert payload["dataset"] == "ToySet"
            assert payload["agent_class"] == "_Model"
            assert isinstance(payload.get("messages"), list)

        # tool_detections only for idx=20 (the only one with non-empty list).
        td_dir = ds / "tool_detections"
        assert td_dir.is_dir()
        td_files = sorted(p.name for p in td_dir.iterdir())
        assert td_files == ["20.json"]
        with open(td_dir / "20.json") as f:
            assert json.load(f) == [{"box": [1, 2, 3, 4], "score": 0.9}]

        # htmls: only the first-2 by completion order (10, 20 in our sync stub).
        html_dir = ds / "htmls"
        html_files = sorted(p.name for p in html_dir.iterdir())
        assert html_files == ["10.html", "20.html"], f"got {html_files}"
        with open(html_dir / "10.html") as f:
            html = f.read()
        assert "Task 10" in html
        assert "ans-10" in html

        # No raw_traces (save_raw_trace=False).
        assert not (ds / "raw_traces").exists()

        # Final xlsx and per-example record.json all present.
        xlsx = inference_image.load(str(ds / "M.xlsx"))
        preds = dict(zip(xlsx["index"], xlsx["prediction"]))
        assert preds == {10: "ans-10", 20: "ans-20", 30: "ans-30", 40: "ans-40"}

        # No legacy artifacts.
        assert not (ds / "M_supp.pkl").exists()
        assert not (ds / "M_PREV.pkl").exists()
        assert not list(ds.glob("0*.pkl"))


def test_real_reporting_input_snapshot_lookup():
    """Verify the aggregation step can build an input_snapshot from sidecars
    when one exists in the trajectory."""
    class _ModelWithSnapshot(_Model):
        def generate(self, message=None, dataset=None, task_id=None, **_):
            r = super().generate(message=message, dataset=dataset, task_id=task_id)
            r["extra_records"]["trajectory"]["input_snapshot"] = {"sample": task_id}
            return r

    with _patch_env({"PRED_FORMAT": "xlsx"}), tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        inference_image.infer_image_job(
            _ModelWithSnapshot(), str(work_dir), "M", _Dataset(),
        )
        snap_path = work_dir / "ToySet" / "input_snapshot.json"
        assert snap_path.exists()
        with open(snap_path) as f:
            snap = json.load(f)
        # First example in iteration order has its snapshot picked.
        assert "sample" in snap
