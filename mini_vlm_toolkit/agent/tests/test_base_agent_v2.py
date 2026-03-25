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
from mini_vlm_toolkit.agent.base import BaseAgentWithToolsV2
from mini_vlm_toolkit.memory import PersistentStateMemory


class _DummyAgentV2(BaseAgentWithToolsV2):
    def __init__(self):
        # Skip parent init to avoid network calls.
        pass


def _build_agent():
    agent = _DummyAgentV2()
    agent.model = "dummy-model"
    agent.api_base = "http://dummy/v1/chat/completions"
    agent.key = "EMPTY"
    agent.tool_base = "http://tool-server"
    agent.json_schema = [
        {
            "type": "function",
            "function": {
                "name": "crop",
                "description": "Crop a region from an image given a bounding box.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Runtime image ID"},
                        "bbox": {"type": "array", "description": "[x1, y1, x2, y2] bounding box"},
                    },
                    "required": ["image_path", "bbox"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ocr",
                "description": "Extract text from an image using OCR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Runtime image ID"},
                    },
                    "required": ["image_path"],
                },
            },
        },
    ]
    agent.tools = {
        "crop": {"tool_name": "crop", "tool_description": "Crop a region from an image given a bounding box."},
        "ocr": {"tool_name": "ocr", "tool_description": "Extract text from an image using OCR."},
    }
    agent.prompt_version = "v3"
    agent.stop_tokens = ["</tool_call>"]
    agent.fail_msg = "Failed to obtain answer via BaseAgentWithToolsV2."
    agent.max_iterations = 15
    agent.verbose = False
    agent.logger = _get_logger("test")
    agent.timeout = 30
    agent.experience_file = None
    agent.tool_choice = "auto"
    agent._persistent_state_memory_factory = lambda: PersistentStateMemory()
    return agent


# ------------------------------------------------------------------ #
# Tests: _parse_tool_calls_from_text
# ------------------------------------------------------------------ #


def test_parse_tool_call_basic():
    agent = _build_agent()
    content = (
        '<perception>I see an image.</perception>\n'
        '<thinking>I should crop it.</thinking>\n'
        '<tool_call>\n{"name": "crop", "arguments": {"image_path": "image_1", "bbox": [10, 20, 200, 300]}}\n</tool_call>'
    )
    calls = agent._parse_tool_calls_from_text(content, turn_idx=0)
    assert len(calls) == 1
    assert calls[0]["name"] == "crop"
    assert calls[0]["args"]["image_path"] == "image_1"
    assert calls[0]["args"]["bbox"] == [10, 20, 200, 300]
    assert calls[0]["id"] == "call_1_1"
    assert calls[0]["parse_error"] is None
    # Check assistant_tool_call structure
    atc = calls[0]["assistant_tool_call"]
    assert atc["id"] == "call_1_1"
    assert atc["type"] == "function"
    assert atc["function"]["name"] == "crop"


def test_parse_tool_call_invalid_json():
    agent = _build_agent()
    content = "<tool_call>\nnot valid json\n</tool_call>"
    calls = agent._parse_tool_calls_from_text(content, turn_idx=0)
    assert len(calls) == 1
    assert calls[0]["parse_error"] is not None
    assert calls[0]["parse_error"]["code"] == "invalid_tool_call_json"


def test_parse_tool_call_missing_name():
    agent = _build_agent()
    content = '<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>'
    calls = agent._parse_tool_calls_from_text(content, turn_idx=0)
    assert len(calls) == 1
    assert calls[0]["parse_error"]["code"] == "invalid_tool_call_format"


def test_parse_tool_call_empty_content():
    agent = _build_agent()
    assert agent._parse_tool_calls_from_text("", turn_idx=0) == []
    assert agent._parse_tool_calls_from_text(None, turn_idx=0) == []


def test_parse_multiple_tool_call_blocks():
    agent = _build_agent()
    content = (
        '<tool_call>\n{"name": "crop", "arguments": {"image_path": "img1", "bbox": [0,0,10,10]}}\n</tool_call>\n'
        '<tool_call>\n{"name": "ocr", "arguments": {"image_path": "img2"}}\n</tool_call>'
    )
    calls = agent._parse_tool_calls_from_text(content, turn_idx=2)
    assert len(calls) == 2
    assert calls[0]["id"] == "call_3_1"
    assert calls[1]["id"] == "call_3_2"


# ------------------------------------------------------------------ #
# Tests: _extract_final_answer_from_content (inherited from parent, v3)
# ------------------------------------------------------------------ #


def test_extract_final_answer_action_pattern():
    agent = _build_agent()
    content = "Observation: I see the number.\nThought: The answer is clear.\nAction: Final Answer\nAction Input: 42"
    assert agent._extract_final_answer_from_content(content) == "42"


def test_extract_final_answer_no_action():
    agent = _build_agent()
    content = "Observation: I see things.\nThought: Let me think."
    # No "Action: Final Answer" pattern, no tool calls -> returns empty
    assert agent._extract_final_answer_from_content(content) == ""


def test_extract_empty():
    agent = _build_agent()
    assert agent._extract_final_answer_from_content("") == ""
    assert agent._extract_final_answer_from_content(None) == ""


# ------------------------------------------------------------------ #
# Tests: _build_tools_directory
# ------------------------------------------------------------------ #


def test_build_tools_directory_full_schemas():
    agent = _build_agent()
    directory = agent._build_tools_directory()
    # Should contain full JSON schemas, not just name+description
    assert '"name": "crop"' in directory
    assert '"name": "ocr"' in directory
    assert '"parameters"' in directory
    assert '"properties"' in directory
    # Verify it's valid JSON entries
    for entry in agent.json_schema:
        assert json.dumps(entry, ensure_ascii=False, indent=2) in directory


# ------------------------------------------------------------------ #
# Tests: parse_response (inherited from parent)
# ------------------------------------------------------------------ #


def test_parse_response_extracts_perception_thinking():
    agent = _build_agent()
    text = (
        "<perception>I see a dog.</perception>\n"
        "<thinking>It is brown.</thinking>\n"
        "Action: Final Answer\nAction Input: dog"
    )
    parsed = agent.parse_response(text)
    assert parsed["perception"] == "I see a dog."
    assert parsed["thinking"] == "It is brown."
    assert parsed["raw"] == text


# ------------------------------------------------------------------ #
# Tests: _get_prompt_templates
# ------------------------------------------------------------------ #


def test_get_prompt_templates_v3():
    agent = _build_agent()
    from mini_vlm_toolkit.agent.constant import DEFAULT_SYSTEM_PROMPT_V3, DEFAULT_USER_PROMPT_PREFIX_V3
    sys_prompt, user_prefix = agent._get_prompt_templates()
    assert sys_prompt is DEFAULT_SYSTEM_PROMPT_V3
    assert user_prefix is DEFAULT_USER_PROMPT_PREFIX_V3


# ------------------------------------------------------------------ #
# Tests: stop token recovery
# ------------------------------------------------------------------ #


def test_stop_token_recovery_appends_closing_tag():
    """When generation stops at </tool_call>, the closing tag is missing and should be re-appended."""
    agent = _build_agent()
    # Simulate content that was cut off by stop token
    content_without_close = (
        '<perception>Obs.</perception>\n<thinking>Think.</thinking>\n'
        '<tool_call>\n{"name": "crop", "arguments": {"image_path": "img1", "bbox": [0,0,10,10]}}'
    )
    # After recovery
    recovered = content_without_close + "</tool_call>"
    calls = agent._parse_tool_calls_from_text(recovered, turn_idx=0)
    assert len(calls) == 1
    assert calls[0]["name"] == "crop"


# ------------------------------------------------------------------ #
# Tests: Full loop with mocked tool execution
# ------------------------------------------------------------------ #


def test_run_loop_tool_then_final_answer():
    """Test a 2-turn loop: tool call -> final answer."""
    agent = _build_agent()
    agent.system_prompt = "You are a test agent."
    agent.default_kwargs = {}
    agent.allowed_types = ["text", "image_url"]
    agent.repetition_penalty = 1.0

    turn = [0]

    def mock_chat_completion(**kwargs):
        t = turn[0]
        turn[0] += 1
        if t == 0:
            return {
                "choices": [{
                    "message": {
                        "content": (
                            '<perception>I see an image.</perception>\n'
                            '<thinking>I should OCR it.</thinking>\n'
                            '<tool_call>\n{"name": "ocr", "arguments": {"image_path": "image_1"}}\n'
                        ),
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": (
                            'Observation: OCR returned text.\n'
                            'Thought: The answer is clear.\n'
                            'Action: Final Answer\n'
                            'Action Input: Hello World'
                        ),
                    }
                }]
            }

    def mock_tool_execute(**kwargs):
        return {"ok": True, "result": {"text": "Hello World"}, "tool_name": "ocr"}

    original_chat = base_module.chat_completion_request
    original_execute = base_module.tool_execute_request
    base_module.chat_completion_request = mock_chat_completion
    base_module.tool_execute_request = mock_tool_execute

    try:
        answer, history = agent._run(
            [{"type": "text", "value": "What does this say?"}],
        )
        assert answer == "Hello World"
        assert len(history) == 2
        # First turn should have a tool call
        assert len(history[0]["assistant"]["tool_calls"]) == 1
        assert history[0]["assistant"]["tool_calls"][0]["name"] == "ocr"
        # Second turn should have the final answer
        assert history[1].get("final_answer") == "Hello World"
    finally:
        base_module.chat_completion_request = original_chat
        base_module.tool_execute_request = original_execute


def test_run_loop_tool_results_use_role_tool():
    """Verify tool results are sent via role=tool (through memory), not role=user."""
    agent = _build_agent()
    agent.system_prompt = "You are a test agent."
    agent.default_kwargs = {}
    agent.allowed_types = ["text", "image_url"]
    agent.repetition_penalty = 1.0

    turn = [0]
    captured_msgs = []

    def mock_chat_completion(**kwargs):
        captured_msgs.append(list(kwargs.get("messages", [])))
        t = turn[0]
        turn[0] += 1
        if t == 0:
            return {
                "choices": [{
                    "message": {
                        "content": (
                            '<tool_call>\n{"name": "ocr", "arguments": {"image_path": "image_1"}}\n'
                        ),
                    }
                }]
            }
        else:
            return {
                "choices": [{
                    "message": {
                        "content": (
                            'Action: Final Answer\n'
                            'Action Input: done'
                        ),
                    }
                }]
            }

    def mock_tool_execute(**kwargs):
        return {"ok": True, "result": {"text": "some text"}, "tool_name": "ocr"}

    original_chat = base_module.chat_completion_request
    original_execute = base_module.tool_execute_request
    base_module.chat_completion_request = mock_chat_completion
    base_module.tool_execute_request = mock_tool_execute

    try:
        answer, history = agent._run(
            [{"type": "text", "value": "test"}],
        )
        assert answer == "done"

        # Check that the messages sent on turn 2 include a role=tool message
        msgs_turn2 = captured_msgs[1]
        tool_msgs = [m for m in msgs_turn2 if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1, "Tool results should use role=tool"

        # And the assistant message should include tool_calls
        assistant_msgs = [m for m in msgs_turn2 if m.get("role") == "assistant" and "tool_calls" in m]
        assert len(assistant_msgs) >= 1, "Assistant message should include tool_calls"
    finally:
        base_module.chat_completion_request = original_chat
        base_module.tool_execute_request = original_execute
