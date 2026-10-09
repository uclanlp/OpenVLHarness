"""Every load_agent_config rejection names the offending option (one mutation per rule)."""
import copy

import pytest

from openvlharness.bootstrap import PAPER_CAPABILITIES, load_agent_config
from test_bootstrap import config_for

VALID = config_for(*PAPER_CAPABILITIES)


def put(path, value):
    def mutate(config):
        *parents, leaf = path
        for key in parents:
            config = config[key]
        config[leaf] = value
    return mutate


def drop(path):
    def mutate(config):
        *parents, leaf = path
        for key in parents:
            config = config[key]
        del config[leaf]
    return mutate


CASES = [
    (put(["agent_name"], "Other"), "agent_name: MainAgent"),
    (put(["extra"], {}), "Unknown agent-system sections"),
    (put(["memory"], []), "memory must be a mapping"),
    (put(["memory", "cache"], True), "Unknown memory options"),
    (put(["memory", "modules"], ["image"]), "memory.modules must be a list of named mappings"),
    (put(["memory", "modules"], [{"name": "url"}]), "must include image"),
    (put(["memory", "modules"], [{"name": "image", "limit": 1}]), "Unknown options for memory module image"),
    (put(["agent", "temperature"], 0), "Unknown agent options"),
    (put(["agent", "max_iterations"], 0), "max_iterations must be a positive integer"),
    (put(["agent", "max_iterations"], True), "max_iterations must be a positive integer"),
    (put(["agent", "paired_sampling_seed"], "yes"), "paired_sampling_seed must be boolean"),
    (put(["agent", "model_service"], "missing"), "agent.model_service must name an entry"),
    (put(["capabilities", "fallback"], True), "Unknown capability options"),
    (put(["capabilities", "registries"], ["extensions"]), "registries must include paper"),
    (put(["capabilities", "tools", "segmentation"], {"backend": "sam3_segment"}), "flat paper capability names"),
    (put(["capabilities", "tools", "ocr"], "ocr"), "capabilities.tools.ocr must be a mapping"),
    (put(["capabilities", "tools", "ocr"], {"backend": "ocr", "lang": "en"}), "accepts only backend selection"),
    (put(["capabilities", "tools", "grounding"], {"backend": "grounding_dino"}), "Unsupported grounding backend"),
    (put(["capabilities", "tools", "coding"], {"generator": "x", "executor": "python_executor_local"}),
     "coding requires python_coding_agent"),
    (put(["services", "cache"], {}), "Unknown service options"),
    (put(["services", "client"], None), "services.client must be a mapping"),
    (put(["services", "client"], {"timeout_seconds": 5}), "services.client requires base_url"),
    (put(["services", "server", "workers"], 2), "Unknown tool server options"),
    (put(["services", "server", "registration"], "all"), "registration must be configured_only"),
    (put(["services", "server", "port"], 70000), "port must be an integer in 1..65535"),
    (drop(["services", "tools", "ocr"]), "Missing services.tools.ocr configuration"),
    (put(["services", "tools", "ocr", "threshold"], 0.5), "algorithmic controls are service constants"),
    (put(["services", "tools", "ocr", "server_url"], ""), "services.tools.ocr.server_url is required"),
    (put(["services", "tools", "ocr", "server_url"], "http://raw.invalid/v1"), "without an endpoint suffix"),
    (put(["services", "tools", "serper_text_search", "model_service"], "nope"), "Missing model service 'nope'"),
    (put(["services", "tools", "python_coding_agent", "context"], "shared"), "requires context: fresh"),
    (put(["services", "tools", "python_coding_agent", "model_service"], "auxiliary"), "same backbone"),
    (put(["services", "tools", "python_executor_local", "transport"], "docker"), "local transport only"),
    (put(["services", "tools", "python_executor_local", "timeout_seconds"], float("inf")),
     "timeout_seconds must be positive and finite"),
    (put(["services", "models", "orchestrator", "base"], "${OPENVL_TEST_UNSET_VARIABLE}"),
     "Missing environment variable.*OPENVL_TEST_UNSET_VARIABLE"),
]


def test_valid_fixture_loads():
    assert load_agent_config(VALID)["agent_name"] == "MainAgent"


@pytest.mark.parametrize("mutate,pattern", CASES, ids=[m for _, m in CASES])
def test_invalid_config_is_rejected_with_actionable_message(mutate, pattern, monkeypatch):
    monkeypatch.delenv("OPENVL_TEST_UNSET_VARIABLE", raising=False)
    config = copy.deepcopy(VALID)
    mutate(config)
    with pytest.raises(ValueError, match=pattern):
        load_agent_config(config)
