import importlib.util
import pytest

if importlib.util.find_spec("mini_vlm_toolkit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

import copy
import json
import re
from importlib.resources import files
from pathlib import Path

import yaml

from mini_vlm_toolkit.utils import run_identity
from openvlharness.services import models
from openvlharness.bootstrap import load_agent_config

from mini_vlm_toolkit.utils.run_identity import (  # noqa: E402
    build_agent_match_hash,
    build_run_id,
    build_run_manifest,
    validate_manifest_compatibility,
)


def _agent_cfg(version="v1"):
    return {
        "agent_name": "MainAgent",
        "agent_config_version": version,
        "agent": {"model_service": "orchestrator", "max_iterations": 15},
        "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}]},
        "capabilities": {"registries": ["paper"], "tools": {}},
        "services": {
            "models": {"orchestrator": {"alias": "qwen3-vl-8b-instruct", "base": "http://localhost:8222/v1", "key": "secret-value"}},
            "client": {"base_url": "http://localhost:8101"}, "server": {}, "tools": {},
        },
    }


def _judge_cfg(version="v1", model_name="Qwen/Qwen3-VL-8B-Instruct", key="secret-key", nproc=4):
    return {
        "judge_config_version": version,
        "model_name": model_name,
        "key": key,
        "nproc": nproc,
    }


def _manifest(config):
    return build_run_manifest(
        run_id="agent_v1__judge_v1", agent_config_version="v1", judge_config_version="v1",
        agent_config=config, judge_config=_judge_cfg(),
        config_path="configs/agent/qwen3_paper.yaml", judge_config_path="configs/judge.yaml",
    )


@pytest.fixture
def local_registry(tmp_path, monkeypatch):
    registry = {"profiles": {alias: models.load_profile(alias) for alias in
                             ("qwen3-vl-8b-instruct", "gpt-6-luna")}}
    path = tmp_path / "model_registry.yaml"
    path.write_text(yaml.safe_dump(registry))
    monkeypatch.setattr(models, "files", lambda package: tmp_path)
    return path, registry


def _config_with_auxiliary():
    config = _agent_cfg()
    config["services"]["models"]["auxiliary"] = {
        "alias": "gpt-6-luna", "base": "https://fixture.invalid/v1", "key": "aux-secret",
    }
    return config


@pytest.mark.parametrize("service", ["orchestrator", "auxiliary"])
@pytest.mark.parametrize("field", ["prompt_bundle", "continue_empty_answer", "context_policy"])
def test_manifest_rejects_changed_resolved_profile(local_registry, service, field):
    path, registry = local_registry
    config = _config_with_auxiliary()
    existing = json.loads(json.dumps(_manifest(config)))
    alias = config["services"]["models"][service]["alias"]
    profile = registry["profiles"][alias]
    original = copy.deepcopy(profile)
    if field == "prompt_bundle":
        profile[field] = "openai_v1" if profile[field] == "v8_instruct" else "v8_instruct"
    elif field == "context_policy":
        profile[field] = "provider_enforced" if profile[field] == "qwen_tokenizer" else "qwen_tokenizer"
    else:
        profile[field] = not profile[field]
    path.write_text(yaml.safe_dump(registry))
    expected = _manifest(config)
    assert existing["agent_config_hash"] != expected["agent_config_hash"]
    assert expected["agent_config_hash"] == build_agent_match_hash(config)
    assert existing["resolved_model_profiles"][service]["profile"] == original
    assert expected["resolved_model_profiles"][service]["profile"] == profile
    assert "aux-secret" not in json.dumps(expected)
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(existing, expected)


@pytest.mark.parametrize("filename", ["v8_instruct_system.txt", "openai_v1_system.txt", "user_prefix.txt"])
def test_manifest_rejects_prompt_content_change(tmp_path, monkeypatch, filename):
    resources = files("openvlharness.prompts")
    for name in ("v8_instruct_system.txt", "openai_v1_system.txt", "user_prefix.txt"):
        (tmp_path / name).write_bytes(resources.joinpath(name).read_bytes())
    monkeypatch.setattr(run_identity, "files", lambda package: tmp_path, raising=False)
    config = _config_with_auxiliary()
    existing = _manifest(config)
    (tmp_path / filename).write_text("Changed prompt content")
    expected = _manifest(config)
    assert existing["agent_config_hash"] != expected["agent_config_hash"]
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(existing, expected)


def test_unreferenced_registry_entries_do_not_change_identity(local_registry):
    path, registry = local_registry
    config = _agent_cfg()
    existing = _manifest(config)
    registry["profiles"]["unused-model"] = copy.deepcopy(registry["profiles"]["gpt-6-luna"])
    registry["profiles"]["gpt-6-luna"]["continue_empty_answer"] = True
    path.write_text(yaml.safe_dump(registry))
    validate_manifest_compatibility(existing, _manifest(config))


def test_manifest_rejects_legacy_config_only_identity():
    config = _agent_cfg()
    expected = _manifest(config)
    legacy = copy.deepcopy(expected)
    legacy.pop("resolved_model_profiles", None)
    legacy["agent_config_hash"] = run_identity.sha256_text(
        run_identity.canonical_json(run_identity.strip_sensitive_fields(config)))
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(legacy, expected)


def test_build_run_id_contains_prefixed_versions():
    run_id = build_run_id("v1_0215", "v1_0215")
    # Format: agent_{ver}_judge_{ver}_{YYYYMMDD}_{HHMMSS}_{tag}
    assert re.match(
        r"^agent_v1_0215_judge_v1_0215_\d{8}_\d{6}_default$", run_id
    ), f"Unexpected run_id format: {run_id}"


def test_build_run_id_custom_tag():
    run_id = build_run_id("v1", "v2", run_tag="debug")
    assert re.match(
        r"^agent_v1_judge_v2_\d{8}_\d{6}_debug$", run_id
    ), f"Unexpected run_id format: {run_id}"


def test_build_run_id_unique_timestamps():
    import time
    run_id_1 = build_run_id("v1", "v1")
    time.sleep(1)
    run_id_2 = build_run_id("v1", "v1")
    assert run_id_1 != run_id_2, "Two run_ids created at different times should differ"


def test_agent_hash_ignores_sensitive_fields():
    cfg_a = _agent_cfg()
    cfg_b = _agent_cfg()
    cfg_b["services"]["models"]["orchestrator"]["key"] = "different-key"
    cfg_b["services"]["models"]["orchestrator"]["api_key"] = "another-secret"

    assert build_agent_match_hash(cfg_a) == build_agent_match_hash(cfg_b)


def test_raw_export_effective_token_setting_is_saved_and_prevents_resume_mix():
    config = _agent_cfg()
    config["services"]["models"]["orchestrator"]["return_token_ids"] = False
    ordinary = _manifest(load_agent_config(config))
    raw = _manifest(load_agent_config(config, save_raw_trace=True))
    assert raw["config_snapshot_redacted"]["agent"]["services"]["models"]["orchestrator"]["return_token_ids"] is True
    assert raw["resolved_model_profiles"]["orchestrator"]["profile"]["raw_trace_format"] == "qwen_chatml"
    assert raw["agent_config_hash"] != ordinary["agent_config_hash"]
    with pytest.raises(ValueError, match="agent config mismatch"):
        validate_manifest_compatibility(ordinary, raw)


def test_manifest_compat_allows_judge_non_model_changes():
    agent_config = _agent_cfg()
    judge_v1 = _judge_cfg(version="v1", nproc=4, key="k1")
    judge_v2 = _judge_cfg(version="v1", nproc=64, key="k2")

    existing = build_run_manifest(
        run_id="agent_v1__judge_v1",
        agent_config_version="v1",
        judge_config_version="v1",
        agent_config=agent_config,
        judge_config=judge_v1,
        config_path="configs/agent/qwen3_paper.yaml",
        judge_config_path="configs/judge.yaml",
    )
    expected = build_run_manifest(
        run_id="agent_v1__judge_v1",
        agent_config_version="v1",
        judge_config_version="v1",
        agent_config=agent_config,
        judge_config=judge_v2,
        config_path="configs/agent/qwen3_paper.yaml",
        judge_config_path="configs/judge.yaml",
    )

    validate_manifest_compatibility(existing, expected)


def test_manifest_compat_rejects_judge_model_change():
    agent_config = _agent_cfg()
    existing = build_run_manifest(
        run_id="agent_v1__judge_v1",
        agent_config_version="v1",
        judge_config_version="v1",
        agent_config=agent_config,
        judge_config=_judge_cfg(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        config_path="configs/agent/qwen3_paper.yaml",
        judge_config_path="configs/judge.yaml",
    )
    expected = build_run_manifest(
        run_id="agent_v1__judge_v1",
        agent_config_version="v1",
        judge_config_version="v1",
        agent_config=agent_config,
        judge_config=_judge_cfg(model_name="Qwen/Qwen2.5-VL-7B-Instruct"),
        config_path="configs/agent/qwen3_paper.yaml",
        judge_config_path="configs/judge.yaml",
    )

    try:
        validate_manifest_compatibility(existing, expected)
    except ValueError as exc:
        assert "judge model mismatch" in str(exc)
        return
    raise AssertionError("Expected ValueError for judge model mismatch")
