import json
from pathlib import Path
import sys
import types

repo_root = Path(__file__).resolve().parents[3]
pkg_root = repo_root / "mini_vlm_toolkit"

# Avoid importing mini_vlm_toolkit/__init__.py (which pulls optional heavy deps).
mini_pkg = types.ModuleType("mini_vlm_toolkit")
mini_pkg.__path__ = [str(pkg_root)]
sys.modules.setdefault("mini_vlm_toolkit", mini_pkg)

agent_pkg = types.ModuleType("mini_vlm_toolkit.agent")
agent_pkg.__path__ = [str(pkg_root / "agent")]
sys.modules.setdefault("mini_vlm_toolkit.agent", agent_pkg)

smp_stub = types.ModuleType("mini_vlm_toolkit.smp")


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _get_logger(_name):
    return _Logger()


smp_stub.get_logger = _get_logger
smp_stub.encode_image_to_base64 = lambda _img: ""
smp_stub.parse_file = lambda value: (None, value)
sys.modules.setdefault("mini_vlm_toolkit.smp", smp_stub)

utils_pkg = types.ModuleType("mini_vlm_toolkit.utils")
utils_pkg.__path__ = [str(pkg_root / "utils")]
sys.modules.setdefault("mini_vlm_toolkit.utils", utils_pkg)

api_utils_stub = types.ModuleType("mini_vlm_toolkit.utils.api_utils")
api_utils_stub.chat_completion_request = lambda **kwargs: {}
api_utils_stub.filter_kwargs_for_endpoint = lambda kwargs, _endpoint: kwargs
api_utils_stub.tool_execute_request = lambda **kwargs: {}
sys.modules.setdefault("mini_vlm_toolkit.utils.api_utils", api_utils_stub)

import mini_vlm_toolkit.agent.base as base_module
from mini_vlm_toolkit.agent.base import BaseAgentWithTools


class _DummyAgent(BaseAgentWithTools):
    def __init__(self):
        # Intentionally skip parent init to avoid network calls in unit tests.
        pass


def _build_agent():
    agent = _DummyAgent()
    agent.model = "dummy-model"
    agent.api_base = "http://dummy/v1/chat/completions"
    agent.key = "EMPTY"
    agent.tool_base = "http://tool-server"
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "Python_Executor_Tool",
                "description": "Execute Python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                    },
                },
            },
        }
    ]
    agent.max_iterations = 3
    agent.system_prompt = agent._construct_system_message()
    agent.timeout = 10
    agent.fail_msg = "Error: test failure"
    agent.logger = _get_logger("BaseAgentNativeToolCallTest")
    return agent


def test_native_tool_call_single_text_tool():
    agent = _build_agent()
    state = {"chat_calls": 0}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        messages = kwargs["messages"]

        if state["chat_calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>I need arithmetic help.</perception>\n"
                                "<thinking>I should call the Python tool.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Python_Executor_Tool",
                                        "arguments": json.dumps({"code": "print(2+2)\nvalue=4"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            assistant_msgs = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
            assert len(assistant_msgs) >= 1
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert len(tool_msgs) >= 1

            tool_msg = tool_msgs[-1]
            assert tool_msg.get("tool_call_id") == "call_1"
            assert tool_msg.get("name") == "Python_Executor_Tool"
            assert isinstance(tool_msg.get("content"), str)

            payload = json.loads(tool_msg["content"])
            assert payload["ok"] is True
            assert payload["result"]["value"] == 4

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>The tool returned value=4.</perception>\n"
                                "<thinking>Now I can answer.</thinking>\n"
                                "4"
                            ),
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["tool_name"] == "Python_Executor_Tool"
        assert tool_call["tool_call_id"] == "call_1"
        assert tool_call["args"]["code"] == "print(2+2)\nvalue=4"
        return {
            "ok": True,
            "tool_name": "Python_Executor_Tool",
            "result": {"stdout": "4\n", "value": 4},
            "error": None,
            "tool_call_id": "call_1",
            "meta": {},
        }

    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Compute 2+2 with a tool."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert answer == "4"
    assert len(chat_history) == 1
    assert chat_history[0]["action_type"] == "tool_call"
    assert len(chat_history[0]["tool_responses"]) == 1
    assert chat_history[0]["tool_responses"][0]["ok"] is True


if __name__ == "__main__":
    test_native_tool_call_single_text_tool()
    print("PASS: test_native_tool_call_single_text_tool")
