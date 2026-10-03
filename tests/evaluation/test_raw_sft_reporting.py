"""Requested raw SFT exports fail visibly without losing ordinary predictions."""

import base64
import copy
import importlib.util
import io
import json
import logging

import pytest

if importlib.util.find_spec("mini_vlm_toolkit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "torch", "validators", "openpyxl", "xlsxwriter"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")

import pandas as pd
from PIL import Image

from mini_vlm_toolkit import inference_image
from mini_vlm_toolkit.utils import file_db, reporting_utils


def _record(task_id=7):
    return {
        "task_id": task_id,
        "model_name": "qwen",
        "trajectory": {"raw_sft": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Count."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "19"}]},
            ],
            "images": [],
        }},
    }


def _image_entry(filename="7_0.png"):
    image_bytes = io.BytesIO()
    Image.new("RGB", (3, 2), (21, 42, 84)).save(image_bytes, format="PNG")
    return {"filename": filename, "bytes_b64": base64.b64encode(image_bytes.getvalue()).decode("ascii")}


def test_text_export_preserves_successful_format_and_input(tmp_path):
    record = _record()
    before = copy.deepcopy(record)
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert json.loads((tmp_path / "7.json").read_text()) == {
        "task_id": 7, "model_name": "qwen", "messages": record["trajectory"]["raw_sft"]["messages"],
    }
    assert record == before
    assert summary["exported_records"] == 1
    assert summary["failed_records"] == 0
    assert summary["failure_details"] == []


def test_image_export_preserves_exact_bytes_and_relative_reference(tmp_path):
    record = _record()
    raw = record["trajectory"]["raw_sft"]
    image = _image_entry()
    raw["images"].append(image)
    raw["messages"][0]["content"].append({"type": "image", "image": "images/7_0.png"})
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert summary["exported_records"] == 1
    assert (tmp_path / "images" / image["filename"]).read_bytes() == base64.b64decode(image["bytes_b64"])
    assert json.loads((tmp_path / "7.json").read_text())["messages"] == raw["messages"]


@pytest.mark.parametrize("trajectory,code", [
    ({}, "raw_sft_missing"),
    (None, "raw_sft_missing"),
    ({"raw_sft_error": {"code": "missing_token_ids", "message": "Endpoint omitted completion token IDs."}}, "missing_token_ids"),
])
def test_missing_and_agent_failed_exports_have_persistent_errors(tmp_path, caplog, trajectory, code):
    record = _record()
    record["trajectory"] = trajectory
    with caplog.at_level(logging.ERROR):
        summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    error = json.loads((tmp_path / "errors" / "7.json").read_text())
    assert error["task_id"] == 7
    assert error["error"]["code"] == code
    if code == "missing_token_ids":
        assert error["error"] == trajectory["raw_sft_error"]
    assert summary["failed_records"] == 1
    assert summary["failure_details"][0]["code"] == code
    assert summary["failure_details"][0]["error_path"] == str(tmp_path / "errors" / "7.json")
    assert not (tmp_path / "7.json").exists()
    assert f"Raw SFT export failed for task 7 [{code}]" in caplog.text


@pytest.mark.parametrize("invalid_images", [
    None,
    {},
    [None],
    [{"bytes_b64": "abcd"}],
    [{"filename": "missing.png"}],
    [{"filename": "bad.png", "bytes_b64": "%%%"}],
    [{"filename": "text.png", "bytes_b64": base64.b64encode(b"not an image").decode("ascii")}],
    [_image_entry("../escape.png")],
    [_image_entry("/escape.png")],
    [_image_entry("..\\escape.png")],
    [_image_entry(), _image_entry()],
    [_image_entry(), {"filename": "broken.png", "bytes_b64": "%%%"}],
])
def test_invalid_images_fail_before_any_success_artifacts(tmp_path, invalid_images):
    record = _record()
    record["trajectory"]["raw_sft"]["images"] = invalid_images
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert summary["exported_records"] == 0
    assert summary["failed_records"] == 1
    assert summary["failure_details"][0]["code"] == "raw_sft_invalid"
    assert (tmp_path / "errors" / "7.json").exists()
    assert not (tmp_path / "7.json").exists()
    assert not list((tmp_path / "images").glob("*"))


@pytest.mark.parametrize("messages", [None, [], "conversation", ["assistant"]])
def test_invalid_messages_are_not_reported_as_exported(tmp_path, messages):
    record = _record()
    record["trajectory"]["raw_sft"]["messages"] = messages
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert summary["failed_records"] == 1
    assert summary["failure_details"][0]["code"] == "raw_sft_invalid"
    assert not (tmp_path / "7.json").exists()


def test_later_success_keeps_other_samples_failure_discoverable(tmp_path):
    failure = _record(7)
    failure["trajectory"] = {}
    reporting_utils.write_raw_trajectory_jsons([failure], tmp_path)
    reporting_utils.write_raw_trajectory_jsons([_record(8)], tmp_path)
    assert json.loads((tmp_path / "_summary.json").read_text())["failed_records"] == 0
    assert json.loads((tmp_path / "errors" / "7.json").read_text())["error"]["code"] == "raw_sft_missing"
    assert (tmp_path / "8.json").exists()
    # A successful retry of the same sample clears its obsolete failure marker.
    reporting_utils.write_raw_trajectory_jsons([_record(7)], tmp_path)
    assert not (tmp_path / "errors" / "7.json").exists()


@pytest.mark.parametrize("trajectory", [
    {},
    {"raw_sft_error": {"code": "missing_token_ids", "message": "No completion token IDs."}},
    {"raw_sft": {"messages": [], "images": []}},
])
def test_failed_reexport_removes_only_its_previous_success(tmp_path, trajectory):
    first = _record(7)
    first["trajectory"]["raw_sft"]["images"] = [_image_entry()]
    first["trajectory"]["raw_sft"]["messages"][0]["content"].append({"type": "image", "image": "images/7_0.png"})
    reporting_utils.write_raw_trajectory_jsons([first, _record(8)], tmp_path)
    other_export = (tmp_path / "8.json").read_bytes()
    failure = _record(7)
    failure["trajectory"] = trajectory
    summary = reporting_utils.write_raw_trajectory_jsons([failure], tmp_path)
    assert summary["failed_records"] == 1
    assert not (tmp_path / "7.json").exists()
    assert (tmp_path / "errors" / "7.json").exists()
    assert (tmp_path / "8.json").read_bytes() == other_export
    assert (tmp_path / "images" / "7_0.png").exists()


@pytest.mark.parametrize("reference", [None, [], "7_0.png", "images/missing.png", "images/../7_0.png"])
def test_dangling_message_image_references_are_rejected(tmp_path, reference):
    record = _record()
    raw = record["trajectory"]["raw_sft"]
    raw["images"] = [_image_entry()]
    raw["messages"][0]["content"].append({"type": "image", "image": reference})
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert summary["failed_records"] == 1
    assert summary["failure_details"][0]["code"] == "raw_sft_invalid"
    assert "image references" in summary["failure_details"][0]["message"]
    assert not (tmp_path / "7.json").exists()
    assert not list((tmp_path / "images").glob("*"))


def test_image_write_failure_is_reported_without_raising(tmp_path, monkeypatch):
    record = _record()
    record["trajectory"]["raw_sft"]["images"] = [_image_entry()]
    original_write = file_db.atomic_write

    def write(path, content, **kwargs):
        if str(path).endswith("7_0.png"):
            raise OSError("image storage unavailable")
        return original_write(path, content, **kwargs)

    monkeypatch.setattr(file_db, "atomic_write", write)
    summary = reporting_utils.write_raw_trajectory_jsons([record], tmp_path)
    assert summary["failed_records"] == 1
    assert json.loads((tmp_path / "errors" / "7.json").read_text())["error"] == {
        "code": "raw_sft_write_failed", "message": "image storage unavailable",
    }
    assert not (tmp_path / "7.json").exists()


def test_unavailable_raw_directory_still_returns_and_logs_failure(tmp_path, caplog):
    out_dir = tmp_path / "occupied"
    out_dir.write_text("This path is a file.")
    with caplog.at_level(logging.ERROR):
        summary = reporting_utils.write_raw_trajectory_jsons([_record()], out_dir)
    assert summary["failed_records"] == 1
    assert summary["failure_details"][0]["code"] == "raw_sft_write_failed"
    assert summary["failure_details"][0]["error_path"] is None
    assert summary["summary_path"] is None
    assert "Could not write raw SFT error sidecar for task 7" in caplog.text


@pytest.mark.usefixtures("synchronous_evaluation")
def test_raw_failures_preserve_ordinary_evaluation_sidecars(tmp_path, monkeypatch):
    class Dataset:
        dataset_name = "RawExportSet"
        data = pd.DataFrame({"index": [10, 20, 30, 40], "answer": ["19"] * 4})

        def __len__(self):
            return len(self.data)

        def build_prompt(self, item):
            return [{"type": "text", "value": "Count."}]

        def dump_image(self, item):
            return []

    class Model:
        def generate(self, message=None, dataset=None, task_id=None):
            trajectory = {
                "input_messages": message,
                "messages": [{"role": "user", "content": "Count."}, {"role": "assistant", "content": "19"}],
                "chat_history": [{"assistant": {"content": "19", "tool_calls": []}, "tool_responses": []}],
                "final_answer": "19",
            }
            if task_id == 20:
                trajectory["raw_sft_error"] = {"code": "missing_token_ids", "message": "No completion token IDs."}
            elif task_id in (30, 40):
                trajectory["raw_sft"] = _record(task_id)["trajectory"]["raw_sft"]
                if task_id == 30:
                    trajectory["raw_sft"]["images"] = [{"filename": "broken.png", "bytes_b64": "%%%"}]
            return {"prediction": "19", "extra_records": {
                "trajectory": trajectory,
                "tool_detections": [{"bbox": [1, 2, 3, 4], "score": 0.9}],
            }}

    monkeypatch.setenv("PRED_FORMAT", "tsv")
    monkeypatch.delenv("FULL_TRAJECTORIES_LIMIT", raising=False)
    for enabled in (False, True):
        inference_image.infer_image_job(Model(), str(tmp_path / str(enabled)), "M", Dataset(),
                                       html_limit=-1, save_raw_trajectory=enabled)
    disabled = tmp_path / "False" / "RawExportSet"
    enabled = tmp_path / "True" / "RawExportSet"
    pd.testing.assert_frame_equal(inference_image.load(str(disabled / "M.tsv")),
                                  inference_image.load(str(enabled / "M.tsv")))
    for task_id in (10, 20, 30, 40):
        assert file_db.load_record(str(enabled), task_id) == file_db.load_record(str(disabled), task_id)
        assert (enabled / "_done" / f"{task_id}.flag").exists()
        for directory, suffix in (("full_trajectories", "json"), ("htmls", "html"), ("tool_detections", "json")):
            path = f"{directory}/{task_id}.{suffix}"
            assert (enabled / path).read_bytes() == (disabled / path).read_bytes()
    raw_dir = enabled / "raw_trajectories"
    assert sorted(p.name for p in (raw_dir / "errors").iterdir()) == ["10.json", "20.json", "30.json"]
    assert (raw_dir / "40.json").exists()
    assert not (disabled / "raw_trajectories").exists()
