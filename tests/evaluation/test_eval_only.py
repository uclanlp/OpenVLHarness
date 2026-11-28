"""Evaluation-only CLI uses saved identity and initializes no inference dependencies."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from openvlharness import bootstrap
from openvlharness.services import models
from evaluation_kit import dataset as datasets

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
    from evaluation_kit.utils.run_identity import redact_sensitive_fields

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
    from evaluation_kit import cli as module
    saved_run(tmp_path, "run_1")
    latest, _ = saved_run(tmp_path, "run_2")
    args = SimpleNamespace(run_dir=None, config=None, work_dir=str(tmp_path), data=["ToyEval"], sample_filter="all")
    logger = SimpleNamespace(debug=lambda *args: None)
    assert module._prepare_evaluation_run(args, logger)[0] == str(latest)
    (latest / "ToyEval/Qwen3-VL-8B-Instruct.tsv").unlink()
    assert module._prepare_evaluation_run(args, logger)[0].endswith("run_1")


def test_eval_discovery_keeps_toolsmith_and_baseline_runs_apart(tmp_path, no_inference):
    from evaluation_kit import cli as module
    from openvlharness.toolsmith.catalog import catalog_fingerprint
    toolsmith, manifest = saved_run(tmp_path, "run_1")
    manifest["config_snapshot_redacted"]["agent"]["toolsmith"] = {"top_k": 10}
    manifest["agent_class"] = "ToolSmithAgent"
    manifest["toolsmith_catalog_sha256"] = catalog_fingerprint()
    (toolsmith / "run_manifest.json").write_text(json.dumps(manifest))
    baseline, _ = saved_run(tmp_path, "run_2")
    args = SimpleNamespace(run_dir=None, config=None, work_dir=str(tmp_path), data=["ToyEval"], sample_filter="all")
    logger = SimpleNamespace(debug=lambda *args: None)
    with pytest.raises(ValueError, match="multiple model/agent identities"):
        module._prepare_evaluation_run(args, logger)
    selector = {"agent_name": "MainAgent", "agent_config_version": "paper_v1", "agent": {"model_service": "main"},
                "services": {"models": {"main": {"alias": "removed-profile"}}}}
    for path, chosen, agent_class in ((tmp_path / "baseline.yaml", baseline, "MainAgent"),
                                      (tmp_path / "toolsmith.yaml", toolsmith, "ToolSmithAgent")):
        path.write_text(yaml.safe_dump({**selector, **({"toolsmith": {}} if agent_class == "ToolSmithAgent" else {})}))
        args.config = str(path)
        root, _, _, found_class, _ = module._prepare_evaluation_run(args, logger)
        assert (root, found_class) == (str(chosen), agent_class)


def test_eval_discovery_picks_only_runs_of_the_configured_catalog(tmp_path, no_inference):
    from evaluation_kit import cli as module
    from openvlharness.toolsmith.catalog import catalog_fingerprint
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "lookup.py").write_text("# a tool\n")
    roots = {}
    for name, digest in (("run_1", catalog_fingerprint(catalog)), ("run_2", "0" * 64), ("run_3", None)):
        roots[name], manifest = saved_run(tmp_path, name)  # run_2 and run_3 are newer, with other catalogs
        manifest["config_snapshot_redacted"]["agent"]["toolsmith"] = {"catalog": str(catalog)}
        manifest["agent_class"], manifest["created_at_utc"] = "ToolSmithAgent", f"2026-11-2{name[-1]}T00:00:00"
        if digest:
            manifest["toolsmith_catalog_sha256"] = digest
        (roots[name] / "run_manifest.json").write_text(json.dumps(manifest))
    selector = {"agent_name": "MainAgent", "agent_config_version": "paper_v1", "agent": {"model_service": "main"},
                "services": {"models": {"main": {"alias": "removed-profile"}}},
                "toolsmith": {"catalog": "${CATALOG_DIR}"}}
    (tmp_path / "toolsmith.yaml").write_text(yaml.safe_dump(selector))
    args = SimpleNamespace(run_dir=None, config=str(tmp_path / "toolsmith.yaml"), work_dir=str(tmp_path),
                           data=["ToyEval"], sample_filter="all")
    with patch.dict(os.environ, {"CATALOG_DIR": str(catalog)}):
        root = module._prepare_evaluation_run(args, SimpleNamespace(debug=lambda *args: None))[0]
    assert root == str(roots["run_1"])


@pytest.mark.parametrize("catalog,message", [
    ("${UNSET_CATALOG_DIR}", "is not a directory (it names an environment variable that is not set)"),
    ("/nonexistent/catalog", "is not a directory. Point it at the catalog the runs used")])
def test_eval_discovery_explains_a_toolsmith_catalog_it_cannot_read(tmp_path, no_inference, catalog, message):
    from evaluation_kit import cli as module
    saved_run(tmp_path, "run_1")
    selector = {"agent_name": "MainAgent", "agent_config_version": "paper_v1", "agent": {"model_service": "main"},
                "services": {"models": {"main": {"alias": "removed-profile"}}}, "toolsmith": {"catalog": catalog}}
    (tmp_path / "toolsmith.yaml").write_text(yaml.safe_dump(selector))
    args = SimpleNamespace(run_dir=None, config=str(tmp_path / "toolsmith.yaml"), work_dir=str(tmp_path),
                           data=["ToyEval"], sample_filter="all")
    with patch.dict(os.environ, {"UNSET_CATALOG_DIR": ""}), pytest.raises(SystemExit) as stop:
        os.environ.pop("UNSET_CATALOG_DIR")
        module._prepare_evaluation_run(args, SimpleNamespace(debug=lambda *args: None))
    assert str(stop.value).startswith(f"error: toolsmith.catalog {catalog!r} in {args.config}")
    assert message in str(stop.value)


def test_explicit_eval_run_requires_requested_prediction_files(tmp_path, no_inference):
    from evaluation_kit import cli as module
    root, manifest = saved_run(tmp_path)
    with pytest.raises(ValueError, match="Expected one saved prediction"):
        module._saved_prediction_files(root, manifest, ["MissingDataset"], "all")


def test_inference_manifest_checks_remain_strict(tmp_path, monkeypatch):
    from evaluation_kit.utils.run_identity import validate_manifest_compatibility
    _, manifest = saved_run(tmp_path)
    expected = copy.deepcopy(manifest)
    expected["agent_config_hash"] = "changed-profile-or-prompt"
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(manifest, expected)
