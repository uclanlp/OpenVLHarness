import importlib.util
import pytest

if importlib.util.find_spec("mini_vlm_toolkit") is None:
    pytest.skip("Retained evaluator is available only from the research checkout", allow_module_level=True)

import re
from pathlib import Path


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
