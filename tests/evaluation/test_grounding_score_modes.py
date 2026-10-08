"""Judge-config score modes must reach the persisted grounding metrics."""

import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("openvlharness_eval") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

for dependency in ("pandas", "torch", "validators"):
    pytest.importorskip(dependency, reason="Retained evaluator dependencies are optional")

import pandas as pd

from openvlharness_eval.dataset import eval_bridge, fscd, lvis, odinw
from openvlharness_eval.utils.judge_utils import load_judge_config


@pytest.fixture
def sidecars(tmp_path):
    path = tmp_path / "tool_detections"
    path.mkdir()
    (path / "val_1_1.json").write_text(json.dumps([
        {"boxes_xyxy_pixels": [[10, 10, 30, 30]], "scores": [0.25]},
    ]))
    return path


@pytest.mark.parametrize("env_mode", [None, "constant", " ORIGINAL "])
@pytest.mark.parametrize("has_sidecars", [False, True])
def test_unconfigured_modes_preserve_environment_and_auto_selection(
    tmp_path, monkeypatch, env_mode, has_sidecars,
):
    monkeypatch.delenv("SCORE_MODE", raising=False)
    if env_mode is not None:
        monkeypatch.setenv("SCORE_MODE", env_mode)
    path = tmp_path / "tool_detections"
    path.mkdir()
    if has_sidecars:
        (path / "row.json").write_text("[]")
    if env_mode is not None:
        mode = env_mode.strip().lower()
        expected = ([mode], mode)
    elif has_sidecars:
        expected = (["original", "constant"], "original")
    else:
        expected = (["constant"], "constant")
    assert eval_bridge._resolve_score_modes(str(path)) == expected


@pytest.mark.parametrize("mode", ["", "auto", "typo", False, ["constant"]])
def test_invalid_configured_mode_is_rejected(sidecars, monkeypatch, mode):
    monkeypatch.setenv("SCORE_MODE", "constant")
    with pytest.raises(ValueError, match="score_mode.*constant.*original"):
        eval_bridge._resolve_score_modes(str(sidecars), score_mode=mode)


@pytest.mark.parametrize("mode", ["constant", "original"])
def test_configured_mode_is_normalized(sidecars, mode):
    assert eval_bridge._resolve_score_modes(
        str(sidecars), score_mode=f" {mode.upper()} ",
    ) == ([mode], mode)


@pytest.mark.parametrize("dataset_cls,dataset_name,pred_kind", [
    (odinw.ODinWBBoxDataset, "ODinW13_bbox_val", "odinw"),
    (fscd.FSCD147BBoxDataset, "FSCD_147_bbox_val", "fscd"),
    (fscd.FSCDLVISBBoxDataset, "FSCD_LVIS_bbox_val", "fscd"),
    (lvis.LVISBBoxDataset, "LVIS_bbox_val", "lvis"),
])
@pytest.mark.parametrize("configured_mode,env_mode", [
    ("constant", None),
    ("constant", "original"),
    ("original", "constant"),
])
def test_judge_config_controls_primary_metrics_and_prediction_scores(
    tmp_path, sidecars, monkeypatch, dataset_cls, dataset_name, pred_kind,
    configured_mode, env_mode,
):
    pytest.importorskip("pycocotools")
    if pred_kind == "lvis":
        pytest.importorskip("lvis")
    monkeypatch.delenv("SCORE_MODE", raising=False)
    if env_mode is not None:
        monkeypatch.setenv("SCORE_MODE", env_mode)

    # Exercise real prediction conversion, COCO/LVIS metrics and output writing,
    # with tiny local annotations instead of downloading a benchmark.
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps({
        "info": {}, "licenses": [],
        "images": [{"id": 1, "width": 100, "height": 100,
                    "neg_category_ids": [], "not_exhaustive_category_ids": []}],
        "categories": [{"id": 1, "name": "object", "frequency": "f"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
    }))
    monkeypatch.setattr(odinw, "_build_odinw_gt_path_map", lambda domains: {("toy", "val"): str(gt_path)})
    monkeypatch.setattr(lvis, "_resolve_lvis_gt_path", lambda name: str(gt_path))
    dataset = object.__new__(dataset_cls)
    dataset.dataset_name = dataset_name
    dataset.data = pd.DataFrame([{
        "index": "val_1_1", "image_width": 100, "image_height": 100,
        "category_id": 1, "domain": "toy", "split": "val",
        "answer": "[[100, 100, 300, 300]]", "num_instances": 1,
    }])
    eval_file = tmp_path / "predictions.tsv"
    pd.DataFrame([{
        "index": "val_1_1", "prediction": "[[100, 100, 300, 300]]",
    }]).to_csv(eval_file, sep="\t", index=False)
    preset = Path(__file__).resolve().parents[2] / "configs" / "judge_no_judge.yaml"
    judge_kwargs = load_judge_config(str(preset))
    assert judge_kwargs["score_mode"] == "constant"
    judge_kwargs["score_mode"] = configured_mode

    result = dataset.evaluate(str(eval_file), **judge_kwargs)

    metrics = json.loads((tmp_path / "predictions_metrics.json").read_text())
    predictions = json.loads((tmp_path / f"predictions_{pred_kind}_pred.json").read_text())
    assert metrics["score_mode"] == configured_mode
    assert set(result["score_mode"]) == {configured_mode}
    assert len(predictions) == 1
    assert predictions[0]["score"] == (1.0 if configured_mode == "constant" else 0.25)
    assert list(tmp_path.glob("predictions_metrics_*.json")) == []
