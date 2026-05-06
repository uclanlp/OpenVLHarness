from pathlib import Path

from mini_vlm_toolkit.agent import APIAgent, build_agent
from mini_vlm_toolkit.agent import base as base_module


class _DummyGetResponse:
    status_code = 200
    text = '{"data":[]}'


def _patch_healthcheck():
    original_get = base_module.requests.get
    base_module.requests.get = lambda *args, **kwargs: _DummyGetResponse()
    return original_get


def test_api_agent_generate_uses_chat_completion_helper():
    captured = {}
    original_get = _patch_healthcheck()
    original_chat = base_module.chat_completion_request

    def fake_chat_completion_request(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    base_module.chat_completion_request = fake_chat_completion_request
    try:
        agent = APIAgent(
            base="https://api.openai.com",
            key="KEY",
            model="gpt-5-2025-08-07",
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            reasoning_effort="medium",
            verbose=False,
        )
        result = agent.generate(
            [{"type": "text", "value": "hello"}],
            min_p=0.05,
            dataset="ignored",
        )
    finally:
        base_module.requests.get = original_get
        base_module.chat_completion_request = original_chat

    assert result["prediction"] == "ok"
    assert captured["model"] == "gpt-5-2025-08-07"
    assert captured["base_url"] == "https://api.openai.com"
    assert captured["api_key"] == "KEY"
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 1.0
    assert captured["reasoning_effort"] == "medium"
    assert captured["extra_body"]["min_p"] == 0.05
    assert "dataset" not in captured.get("extra_body", {})


def test_build_agent_loads_api_agent_config():
    original_get = _patch_healthcheck()
    try:
        config_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "cf-testing"
            / "gpt_agent.yaml"
        )
        agent = build_agent(str(config_path))
    finally:
        base_module.requests.get = original_get

    assert isinstance(agent, APIAgent)
    assert agent.base == "https://api.openai.com"
    assert agent.model == "gpt-5-2025-08-07"


if __name__ == "__main__":
    test_api_agent_generate_uses_chat_completion_helper()
    test_build_agent_loads_api_agent_config()
    print("PASS: test_api_agent_generate_uses_chat_completion_helper")
    print("PASS: test_build_agent_loads_api_agent_config")
