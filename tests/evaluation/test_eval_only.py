"""Evaluation-only CLI uses saved identity and initializes no inference dependencies."""
import copy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from openvlharness import bootstrap
from openvlharness.services import models
from mini_vlm_toolkit import dataset as datasets

CLI = Path(__file__).resolve().parents[2] / "run_benchmark.py"


def saved_run(tmp_path, name="run_one", explicit_identity=False):
    root = tmp_path / "Qwen3-VL-8B-Instruct" / "mainagent" / name
    (root / "ToyEval").mkdir(parents=True)
    manifest = {"run_id": name, "agent_config_version": "paper_v1", "agent_config_hash": "saved-only",
        "judge_config_version": "direct", "created_at_utc": name,
        "config_snapshot_redacted": {
            "agent": {"agent_name": "MainAgent", "agent": {"model_service": "main"},
                      "services": {"models": {"main": {"alias": "removed-profile", "base": "${MISSING_ENDPOINT}"}}}},
            "judge": {"judge_config_version": "direct", "model_name": None, "score_mode": "constant"}},
        "resolved_model_profiles": {"main": {"profile": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"}}}}
    if explicit_identity:
        manifest.update(model_id="Qwen/Qwen3-VL-8B-Instruct", agent_class="MainAgent", prediction_format="tsv")
        del manifest["resolved_model_profiles"]
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame([{"index": "sample", "answer": "2", "prediction": "2"}]).to_csv(
        root / "ToyEval/Qwen3-VL-8B-Instruct.tsv", sep="\t", index=False)
    return root, manifest


@pytest.fixture
def no_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not initialize inference or make model/tool requests")
    monkeypatch.setattr(bootstrap, "build_agent", forbidden)
    monkeypatch.setattr(bootstrap, "load_agent_config", forbidden)
    monkeypatch.setattr(models, "load_profile", forbidden)
    monkeypatch.setattr(models.ModelAdapter, "load_tokenizer", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("PRED_FORMAT", "json")  # Saved TSV is authoritative.
    monkeypatch.delenv("MISSING_ENDPOINT", raising=False)


@pytest.mark.parametrize("explicit_identity", [False, True])
@pytest.mark.parametrize("override_judge", [False, True])
def test_eval_cli_needs_only_saved_outputs_and_optional_judge(tmp_path, monkeypatch, no_inference,
                                                            explicit_identity, override_judge):
    root, _ = saved_run(tmp_path, explicit_identity=explicit_identity)
    before = (root / "run_manifest.json").read_bytes()
    calls = []
    def evaluate(path, **kwargs):
        assert pd.read_csv(path, sep="\t")["prediction"].tolist() == [2]
        calls.append((path, kwargs))
    dataset = SimpleNamespace(MODALITY="IMAGE", data=pd.DataFrame([{"index": "sample"}]), evaluate=evaluate)
    monkeypatch.setattr(datasets, "build_dataset", lambda *args, **kwargs: dataset)
    argv = [str(CLI), "--mode", "eval", "--data", "ToyEval", "--run-dir", str(root),
            "--config", str(tmp_path / "absent-inference.yaml")]
    if override_judge:
        judge = tmp_path / "judge.yaml"
        judge.write_text(yaml.safe_dump({"judge_config_version": "new", "model_name": "judge-only",
                                       "api_base": "http://judge.invalid", "key": "fixture"}))
        argv += ["--judge-config", str(judge)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(CLI), run_name="__main__")
    assert len(calls) == 1
    assert calls[0][1]["model"] == ("judge-only" if override_judge else None)
    assert (root / "run_manifest.json").read_bytes() == before


@pytest.mark.parametrize("judge_model,needs_credentials", [
    (None, False), ("exact_matching", False), ("extract_matching", False), ("gpt-4o", True),
])
def test_eval_redacted_placeholder_key_allows_local_matching(tmp_path, monkeypatch, no_inference,
                                                           judge_model, needs_credentials):
    from mini_vlm_toolkit.utils.run_identity import redact_sensitive_fields

    root, manifest = saved_run(tmp_path)
    judge = {"judge_config_version": "matching", "model_name": judge_model, "key": "empty"}
    manifest["config_snapshot_redacted"]["judge"] = redact_sensitive_fields(judge)
    assert manifest["config_snapshot_redacted"]["judge"]["key"] == "<REDACTED>"
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    before = manifest_path.read_bytes()
    calls = []

    def evaluate(path, **kwargs):
        rows = pd.read_csv(path, sep="\t")
        assert (rows["prediction"] == rows["answer"]).all()
        calls.append(kwargs)

    dataset = SimpleNamespace(MODALITY="IMAGE", evaluate=evaluate)
    monkeypatch.setattr(datasets, "build_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(sys, "argv", [str(CLI), "--mode", "eval", "--data", "ToyEval",
                                     "--run-dir", str(root)])
    if needs_credentials:
        with pytest.raises(ValueError, match="Saved judge credentials are redacted; supply --judge-config"):
            runpy.run_path(str(CLI), run_name="__main__")
        assert not calls
    else:
        runpy.run_path(str(CLI), run_name="__main__")
        assert len(calls) == 1
        assert calls[0]["model"] == judge_model
    assert manifest_path.read_bytes() == before


def test_eval_discovery_uses_saved_metadata_and_latest_complete_predictions(tmp_path, monkeypatch, no_inference):
    module = runpy.run_path(str(CLI))
    saved_run(tmp_path, "run_1")
    latest, _ = saved_run(tmp_path, "run_2")
    args = SimpleNamespace(run_dir=None, config=None, work_dir=str(tmp_path), data=["ToyEval"], sample_filter="all")
    logger = SimpleNamespace(debug=lambda *args: None)
    assert module["_prepare_evaluation_run"](args, logger)[0] == str(latest)
    (latest / "ToyEval/Qwen3-VL-8B-Instruct.tsv").unlink()
    assert module["_prepare_evaluation_run"](args, logger)[0].endswith("run_1")


def test_explicit_eval_run_requires_requested_prediction_files(tmp_path, no_inference):
    module = runpy.run_path(str(CLI))
    root, manifest = saved_run(tmp_path)
    with pytest.raises(ValueError, match="Expected one saved prediction"):
        module["_saved_prediction_files"](root, manifest, ["MissingDataset"], "all")


def test_inference_manifest_checks_remain_strict(tmp_path, monkeypatch):
    from mini_vlm_toolkit.utils.run_identity import validate_manifest_compatibility
    _, manifest = saved_run(tmp_path)
    expected = copy.deepcopy(manifest)
    expected["agent_config_hash"] = "changed-profile-or-prompt"
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(manifest, expected)
