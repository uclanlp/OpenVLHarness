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
api_utils_stub.local_image_to_data_url = lambda _path: "data:image/png;base64,"
api_utils_stub.tool_execute_request = lambda **kwargs: {}
sys.modules.setdefault("mini_vlm_toolkit.utils.api_utils", api_utils_stub)

import mini_vlm_toolkit.agent.base as base_module
from mini_vlm_toolkit.agent.base import BaseAgentWithTools
from mini_vlm_toolkit.agent.runtime_environment import ImageEnvironment


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
    agent.tool_choice = "auto"
    agent.system_prompt = agent._construct_system_message()
    agent.runtime_environment_emit_policy = "always"
    agent.runtime_environment_hooks = [ImageEnvironment(emit_policy="always")]
    agent.timeout = 10
    agent.fail_msg = "Error: test failure"
    agent.logger = _get_logger("BaseAgentNativeToolCallTest")
    agent.default_kwargs = {}
    agent._image_registry = {}
    agent._image_alias_to_ref = {}
    agent._image_url_to_ref = {}
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
    assert len(chat_history) == 2
    assert len(chat_history[0]["assistant"]["tool_calls"]) == 1
    assert chat_history[0]["assistant"]["perception"] == "I need arithmetic help."
    assert chat_history[0]["assistant"]["thinking"] == "I should call the Python tool."
    assert len(chat_history[0]["tool_responses"]) == 1
    assert chat_history[0]["tool_responses"][0]["ok"] is True
    assert chat_history[1]["final_answer"] == "4"


def test_native_tool_call_image_result_recursive_extraction():
    agent = _build_agent()
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "Image_Tool",
                "description": "Return image artifacts",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    state = {"chat_calls": 0}

    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
    )

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
                                "<perception>I should inspect masks.</perception>\n"
                                "<thinking>I need an image-producing tool.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_img_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Image_Tool",
                                        "arguments": json.dumps({"query": "show masks"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            tool_msg = tool_msgs[-1]
            assert tool_msg["tool_call_id"] == "call_img_1"
            assert tool_msg["name"] == "Image_Tool"
            assert isinstance(tool_msg["content"], list)

            env_updates = [
                m["content"]
                for m in messages
                if m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith("Environment Update")
            ]
            assert env_updates
            latest_update = env_updates[-1]
            assert "tool_call_img_1_image_1" in latest_update
            assert "tool_output:Image_Tool" in latest_update

            content_parts = tool_msg["content"]
            assert content_parts[0]["type"] == "text"
            image_parts = [p for p in content_parts[1:] if p.get("type") == "image_url"]
            assert len(image_parts) == 3
            assert all(p["image_url"]["url"].startswith("data:image/") for p in image_parts)

            compact_payload = json.loads(content_parts[0]["text"])
            compact_text = content_parts[0]["text"]
            assert tiny_png_b64 not in compact_text
            assert "data:image/" not in compact_text
            assert compact_payload["ok"] is True
            assert compact_payload["result"]["summary"] == "image artifacts returned"
            assert compact_payload["result"]["nested"]["other"] == 42

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>Received image artifacts.</perception>\n"
                                "<thinking>I can now answer directly.</thinking>\n"
                                "done"
                            ),
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["tool_name"] == "Image_Tool"
        assert tool_call["tool_call_id"] == "call_img_1"
        return {
            "ok": True,
            "tool_name": "Image_Tool",
            "result": {
                "summary": "image artifacts returned",
                "nested": {
                    "all_masks_viz_base64": tiny_png_b64,
                    "other": 42,
                },
                "artifacts": [
                    {"image_base64": tiny_png_b64},
                    {"note": "keep me"},
                ],
                "mask_gallery": {
                    "items": [
                        {"mask_png_b64": tiny_png_b64},
                    ]
                },
            },
            "error": None,
            "tool_call_id": "call_img_1",
            "meta": {},
        }

    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Return an image tool result."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert answer == "done"
    assert len(chat_history) == 2
    assert chat_history[0]["assistant"]["tool_calls"][0]["name"] == "Image_Tool"
    assert chat_history[1]["final_answer"] == "done"


def test_native_tool_call_resolves_hallucinated_image_path():
    agent = _build_agent()
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "SAM3_Segment_Tool",
                "description": "Segment with image input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "image_path": {"type": "string"},
                    },
                },
            },
        }
    ]
    state = {"chat_calls": 0, "tool_calls": 0}
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
    )
    expected_data_url = f"data:image/png;base64,{tiny_png_b64}"

    def fake_build_image_registry(_question):
        agent._image_registry = {
            "input_image_1": {
                "id": "input_image_1",
                "data_url": expected_data_url,
                "source": "user_input",
                "aliases": {"input_image_1", "image.jpg"},
            }
        }
        agent._image_alias_to_ref = {
            "input_image_1": "input_image_1",
            "image.jpg": "input_image_1",
        }
        agent._image_url_to_ref = {expected_data_url: "input_image_1"}
        return agent._image_registry

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        messages = kwargs["messages"]

        if state["chat_calls"] == 1:
            env_updates = [
                m["content"]
                for m in messages
                if m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith("Environment Update")
            ]
            assert env_updates
            assert "input_image_1" in env_updates[-1]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>I need segmentation.</perception>\n"
                                "<thinking>I will call SAM3.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_img_resolve_1",
                                    "type": "function",
                                    "function": {
                                        "name": "SAM3_Segment_Tool",
                                        "arguments": json.dumps(
                                            {
                                                "prompt": "children",
                                                "image_path": "image.jpg",
                                                "return_all_masks_viz": True,
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            payload = json.loads(tool_msgs[-1]["content"])
            assert payload["ok"] is True
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>Tool succeeded.</perception>\n"
                                "<thinking>I can answer now.</thinking>\n"
                                "done"
                            ),
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        state["tool_calls"] += 1
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["tool_name"] == "SAM3_Segment_Tool"
        assert tool_call["tool_call_id"] == "call_img_resolve_1"
        assert "image_path" not in tool_call["args"]
        assert tool_call["args"]["image_base64"] == expected_data_url
        return {
            "ok": True,
            "tool_name": "SAM3_Segment_Tool",
            "result": {"mask_count": 3},
            "error": None,
            "tool_call_id": "call_img_resolve_1",
            "meta": {},
        }

    agent._build_image_registry_from_inputs = fake_build_image_registry
    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Count the children."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert state["tool_calls"] == 1
    assert answer == "done"
    assert len(chat_history) == 2
    resolution_meta = chat_history[0]["tool_responses"][0]["meta"]["image_resolution"]
    assert len(resolution_meta["rewrites"]) == 1
    assert resolution_meta["errors"] == []


def test_native_tool_call_image_result_top_level_string_scrubbed():
    agent = _build_agent()
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "Image_String_Tool",
                "description": "Return image as a top-level data URL string",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    state = {"chat_calls": 0}
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
    )
    image_data_url = f"data:image/png;base64,{tiny_png_b64}"

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
                                "<perception>I should call the image tool.</perception>\n"
                                "<thinking>I need a visualization.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_img_string_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Image_String_Tool",
                                        "arguments": json.dumps({"query": "return image"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            tool_msg = tool_msgs[-1]
            assert tool_msg["tool_call_id"] == "call_img_string_1"
            assert tool_msg["name"] == "Image_String_Tool"
            assert isinstance(tool_msg["content"], list)

            content_parts = tool_msg["content"]
            assert content_parts[0]["type"] == "text"
            image_parts = [p for p in content_parts[1:] if p.get("type") == "image_url"]
            assert len(image_parts) == 1
            assert image_parts[0]["image_url"]["url"] == image_data_url

            compact_payload = json.loads(content_parts[0]["text"])
            compact_text = content_parts[0]["text"]
            assert tiny_png_b64 not in compact_text
            assert image_data_url not in compact_text
            assert compact_payload["ok"] is True
            assert compact_payload["result"] is None
            assert compact_payload["error"] is None

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>Received image output.</perception>\n"
                                "<thinking>I can provide the final answer.</thinking>\n"
                                "done"
                            ),
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["tool_name"] == "Image_String_Tool"
        assert tool_call["tool_call_id"] == "call_img_string_1"
        return {
            "ok": True,
            "tool_name": "Image_String_Tool",
            "result": image_data_url,
            "error": None,
            "tool_call_id": "call_img_string_1",
            "meta": {},
        }

    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Return image as top-level string."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert answer == "done"
    assert len(chat_history) == 2
    assert chat_history[0]["assistant"]["tool_calls"][0]["name"] == "Image_String_Tool"
    assert chat_history[1]["final_answer"] == "done"


def test_native_tool_call_forwards_tool_choice_and_extra_body():
    agent = _build_agent()
    state = {"chat_calls": 0}
    forced_tool_choice = {"type": "function", "function": {"name": "Python_Executor_Tool"}}
    extra_body = {"min_p": 0.05}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        assert kwargs.get("tool_choice") == forced_tool_choice
        assert kwargs.get("extra_body") == extra_body
        assert isinstance(kwargs.get("tools"), list)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<perception>Direct answer available.</perception>\n"
                            "<thinking>No tool execution needed.</thinking>\n"
                            "done"
                        ),
                    }
                }
            ]
        }

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer, chat_history = agent._run(
            [{"type": "text", "value": "Reply directly."}],
            tool_choice=forced_tool_choice,
            extra_body=extra_body,
        )
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert state["chat_calls"] == 1
    assert answer == "done"
    assert len(chat_history) == 1
    assert chat_history[0]["assistant"]["tool_calls"] == []


def test_native_tool_call_uses_agent_default_tool_choice():
    agent = _build_agent()
    agent.tool_choice = "none"
    state = {"chat_calls": 0}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        assert kwargs.get("tool_choice") == "none"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<perception>Ready.</perception>\n"
                            "<thinking>Tool use disabled.</thinking>\n"
                            "done"
                        ),
                    }
                }
            ]
        }

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Reply directly."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert state["chat_calls"] == 1
    assert answer == "done"
    assert len(chat_history) == 1


def test_native_tool_call_uses_agent_default_timeout_seconds():
    agent = _build_agent()
    agent.timeout = 37
    state = {"chat_calls": 0}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        assert kwargs.get("timeout_seconds") == 37
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<perception>Ready.</perception>\n"
                            "<thinking>Use default timeout.</thinking>\n"
                            "done"
                        ),
                    }
                }
            ]
        }

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Reply directly."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert state["chat_calls"] == 1
    assert answer == "done"
    assert len(chat_history) == 1


def test_native_tool_call_timeout_seconds_override_passthrough():
    agent = _build_agent()
    state = {"chat_calls": 0}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        assert kwargs.get("timeout_seconds") == 12
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<perception>Ready.</perception>\n"
                            "<thinking>Use runtime timeout override.</thinking>\n"
                            "done"
                        ),
                    }
                }
            ]
        }

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer = agent.generate([{"type": "text", "value": "Reply directly."}], timeout_seconds=12)
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert state["chat_calls"] == 1
    assert answer == "done"


def test_native_tool_call_invalid_tool_arguments_soft_fail():
    agent = _build_agent()
    state = {"chat_calls": 0, "tool_execute_calls": 0}

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
                                "<perception>I should call a tool.</perception>\n"
                                "<thinking>I will execute Python.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_bad_json_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Python_Executor_Tool",
                                        "arguments": '{"code": "print(2+2)",}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            payload = json.loads(tool_msgs[-1]["content"])
            assert payload["ok"] is False
            assert payload["error"]["code"] == "invalid_tool_arguments_json"
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>The previous call failed due malformed JSON.</perception>\n"
                                "<thinking>I can still conclude this test path works.</thinking>\n"
                                "handled"
                            ),
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        state["tool_execute_calls"] += 1
        raise AssertionError("tool_execute_request should not be called for invalid tool-call JSON")

    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Trigger malformed tool args."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert answer == "handled"
    assert state["tool_execute_calls"] == 0
    assert len(chat_history) == 2
    assert chat_history[0]["assistant"]["tool_calls"][0]["parse_error"]["code"] == "invalid_tool_arguments_json"
    assert chat_history[0]["tool_responses"][0]["ok"] is False
    assert chat_history[0]["tool_responses"][0]["error"]["code"] == "invalid_tool_arguments_json"


def test_native_tool_call_hard_fail_invalid_chat_response_shape():
    agent = _build_agent()

    def fake_chat_completion_request(**kwargs):
        _ = kwargs
        return {"id": "chatcmpl_dummy_without_choices"}

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        try:
            agent._run([{"type": "text", "value": "Invalid response shape."}])
            raise AssertionError("Expected RuntimeError for malformed chat response")
        except RuntimeError as exc:
            assert "missing choices" in str(exc).lower()
    finally:
        base_module.chat_completion_request = original_chat_completion_request


def test_native_tool_call_single_image_fallback_metadata():
    agent = _build_agent()
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "SAM3_Segment_Tool",
                "description": "Segment with image input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "image_path": {"type": "string"},
                    },
                },
            },
        }
    ]
    state = {"chat_calls": 0}
    tiny_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
    )
    expected_data_url = f"data:image/png;base64,{tiny_png_b64}"

    def fake_build_image_registry(_question):
        agent._image_registry = {
            "input_image_1": {
                "id": "input_image_1",
                "data_url": expected_data_url,
                "source": "user_input",
                "aliases": {"input_image_1"},
            }
        }
        agent._image_alias_to_ref = {"input_image_1": "input_image_1"}
        agent._image_url_to_ref = {expected_data_url: "input_image_1"}
        return agent._image_registry

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        if state["chat_calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>I need segmentation.</perception>\n"
                                "<thinking>I will call SAM3.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_single_fallback_1",
                                    "type": "function",
                                    "function": {
                                        "name": "SAM3_Segment_Tool",
                                        "arguments": json.dumps(
                                            {"prompt": "children", "image_path": "nonexistent_image_name.jpg"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<perception>Tool succeeded.</perception>\n"
                            "<thinking>I can answer now.</thinking>\n"
                            "done"
                        ),
                    }
                }
            ]
        }

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["args"]["image_base64"] == expected_data_url
        return {
            "ok": True,
            "tool_name": "SAM3_Segment_Tool",
            "result": {"mask_count": 1},
            "error": None,
            "tool_call_id": "call_single_fallback_1",
            "meta": {},
        }

    agent._build_image_registry_from_inputs = fake_build_image_registry
    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Count the children."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert answer == "done"
    rewrites = chat_history[0]["tool_responses"][0]["meta"]["image_resolution"]["rewrites"]
    assert len(rewrites) == 1
    assert rewrites[0]["fallback_mode"] == "single_image_registry_default"


def test_preprocess_message_with_role_no_mutation():
    agent = _build_agent()
    original = [
        {"role": "system", "value": "system-a"},
        {"role": "user", "type": "text", "value": "hello"},
    ]
    snapshot = json.loads(json.dumps(original))
    processed = agent.preprocess_message_with_role(original)

    assert original == snapshot
    assert processed == [{"type": "text", "value": "hello"}]
    assert "system-a" in (agent.system_prompt or "")


def test_run_returns_fail_msg_when_iterations_exhausted():
    agent = _build_agent()
    agent.max_iterations = 1

    def fake_chat_completion_request(**kwargs):
        _ = kwargs
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "No answer please."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert answer == agent.fail_msg
    assert len(chat_history) == 1


def test_normalize_execute_payload_strict_ok_parsing():
    agent = _build_agent()
    payload = agent._normalize_execute_payload(
        {"ok": "false", "result": None, "error": {"code": "bad", "message": "oops"}},
        tool_name="Demo_Tool",
        tool_call_id="call_123",
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad"

    payload2 = agent._normalize_execute_payload(
        {"detail": "tool failed unexpectedly"},
        tool_name="Demo_Tool",
        tool_call_id="call_124",
    )
    assert payload2["ok"] is False
    assert payload2["error"]["code"] == "tool_execution_error_payload"
    assert payload2["tool_call_id"] == "call_124"


def test_prompt_version_v2_prefix_and_system_prompt():
    agent = _build_agent()
    agent.prompt_version = "v2"
    agent.system_prompt = agent._construct_system_message()

    assert "Use the following format" in agent.system_prompt
    msgs = [{"role": "user", "content": "Count the objects."}]
    updated = agent._add_user_prompt_prefix(msgs)
    assert updated[-1]["content"] == "Question: Count the objects."


def test_extract_final_answer_from_v2_action_input():
    agent = _build_agent()
    agent.prompt_version = "v2"
    content = (
        "Question: What is shown?\n"
        "Observation: I can see the scene clearly.\n"
        "Thought: I have enough evidence.\n"
        "Action: Final Answer\n"
        "Action Input: There are three dogs on the beach."
    )
    answer = agent._extract_final_answer_from_content(content)
    assert answer == "There are three dogs on the beach."


def test_extract_final_answer_from_v1_scaffold_only_returns_empty():
    agent = _build_agent()
    content = (
        "<perception>I see multiple objects in the scene.</perception>\n"
        "<thinking>I should call a tool for exact counting.</thinking>"
    )
    answer = agent._extract_final_answer_from_content(content)
    assert answer == ""


def test_extract_final_answer_from_v2_scaffold_only_returns_empty():
    agent = _build_agent()
    agent.prompt_version = "v2"
    content = (
        "Question: How many objects are there?\n"
        "Observation: I can partially see the objects.\n"
        "Thought: I should use a counting tool."
    )
    answer = agent._extract_final_answer_from_content(content)
    assert answer == ""


def test_run_continues_after_v1_scaffold_only_response():
    agent = _build_agent()
    state = {"chat_calls": 0}

    def fake_chat_completion_request(**kwargs):
        state["chat_calls"] += 1
        if state["chat_calls"] == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<perception>I can see the scene, but need more evidence.</perception>\n"
                                "<thinking>I should reason a bit more before answering.</thinking>"
                            ),
                        }
                    }
                ]
            }
        if state["chat_calls"] == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Final answer: 42",
                        }
                    }
                ]
            }
        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    original_chat_completion_request = base_module.chat_completion_request
    base_module.chat_completion_request = fake_chat_completion_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Please solve it."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request

    assert state["chat_calls"] == 2
    assert answer == "42"
    assert len(chat_history) == 2
    assert chat_history[0]["final_answer"] == ""
    assert chat_history[1]["final_answer"] == "42"


def test_run_allows_finalization_after_last_iteration_tool_calls():
    agent = _build_agent()
    agent.max_iterations = 1
    state = {"chat_calls": 0, "tool_calls": 0}

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
                                "<perception>I need arithmetic.</perception>\n"
                                "<thinking>I should call Python tool first.</thinking>"
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_last_iter_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Python_Executor_Tool",
                                        "arguments": json.dumps({"code": "value=4"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        if state["chat_calls"] == 2:
            assert kwargs.get("tool_choice") == "none"
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Final answer: 4",
                        }
                    }
                ]
            }

        raise AssertionError(f"Unexpected number of chat calls: {state['chat_calls']}")

    def fake_tool_execute_request(*, base_url, tool_call, timeout):
        state["tool_calls"] += 1
        assert base_url == "http://tool-server"
        assert timeout == 10
        assert tool_call["tool_name"] == "Python_Executor_Tool"
        assert tool_call["tool_call_id"] == "call_last_iter_1"
        assert tool_call["args"]["code"] == "value=4"
        return {
            "ok": True,
            "tool_name": "Python_Executor_Tool",
            "result": {"value": 4},
            "error": None,
            "tool_call_id": "call_last_iter_1",
            "meta": {},
        }

    original_chat_completion_request = base_module.chat_completion_request
    original_tool_execute_request = base_module.tool_execute_request
    base_module.chat_completion_request = fake_chat_completion_request
    base_module.tool_execute_request = fake_tool_execute_request
    try:
        answer, chat_history = agent._run([{"type": "text", "value": "Compute 2+2."}])
    finally:
        base_module.chat_completion_request = original_chat_completion_request
        base_module.tool_execute_request = original_tool_execute_request

    assert state["chat_calls"] == 2
    assert state["tool_calls"] == 1
    assert answer == "4"
    assert len(chat_history) == 2
    assert chat_history[0]["assistant"]["tool_calls"][0]["name"] == "Python_Executor_Tool"
    assert chat_history[1]["final_answer"] == "4"


if __name__ == "__main__":
    test_native_tool_call_single_text_tool()
    test_native_tool_call_image_result_recursive_extraction()
    test_native_tool_call_resolves_hallucinated_image_path()
    test_native_tool_call_image_result_top_level_string_scrubbed()
    test_native_tool_call_forwards_tool_choice_and_extra_body()
    test_native_tool_call_uses_agent_default_tool_choice()
    test_native_tool_call_uses_agent_default_timeout_seconds()
    test_native_tool_call_timeout_seconds_override_passthrough()
    test_native_tool_call_invalid_tool_arguments_soft_fail()
    test_native_tool_call_hard_fail_invalid_chat_response_shape()
    test_native_tool_call_single_image_fallback_metadata()
    test_preprocess_message_with_role_no_mutation()
    test_run_returns_fail_msg_when_iterations_exhausted()
    test_normalize_execute_payload_strict_ok_parsing()
    test_prompt_version_v2_prefix_and_system_prompt()
    test_extract_final_answer_from_v2_action_input()
    test_extract_final_answer_from_v1_scaffold_only_returns_empty()
    test_extract_final_answer_from_v2_scaffold_only_returns_empty()
    test_run_continues_after_v1_scaffold_only_response()
    test_run_allows_finalization_after_last_iteration_tool_calls()
    print("PASS: test_native_tool_call_single_text_tool")
    print("PASS: test_native_tool_call_image_result_recursive_extraction")
    print("PASS: test_native_tool_call_resolves_hallucinated_image_path")
    print("PASS: test_native_tool_call_image_result_top_level_string_scrubbed")
    print("PASS: test_native_tool_call_forwards_tool_choice_and_extra_body")
    print("PASS: test_native_tool_call_uses_agent_default_tool_choice")
    print("PASS: test_native_tool_call_uses_agent_default_timeout_seconds")
    print("PASS: test_native_tool_call_timeout_seconds_override_passthrough")
    print("PASS: test_native_tool_call_invalid_tool_arguments_soft_fail")
    print("PASS: test_native_tool_call_hard_fail_invalid_chat_response_shape")
    print("PASS: test_native_tool_call_single_image_fallback_metadata")
    print("PASS: test_preprocess_message_with_role_no_mutation")
    print("PASS: test_run_returns_fail_msg_when_iterations_exhausted")
    print("PASS: test_normalize_execute_payload_strict_ok_parsing")
    print("PASS: test_prompt_version_v2_prefix_and_system_prompt")
    print("PASS: test_extract_final_answer_from_v2_action_input")
    print("PASS: test_extract_final_answer_from_v1_scaffold_only_returns_empty")
    print("PASS: test_extract_final_answer_from_v2_scaffold_only_returns_empty")
    print("PASS: test_run_continues_after_v1_scaffold_only_response")
    print("PASS: test_run_allows_finalization_after_last_iteration_tool_calls")
