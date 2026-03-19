import json
import os

from client.client import ToolClient
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from tools.python_execution_common import prepare_execution_context
from tools.python_coder.tool import Python_Coder_Tool
from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def assert_contains_number(result, expected):
    output = result.get("printed_output") or ""
    variables = result.get("variables") or {}
    variable_text = " ".join(str(value) for value in variables.values())
    combined = f"{output} {variable_text}"
    assert str(expected) in combined, f"Expected {expected} in output or variables"


def test_direct_invocation():
    log_step("Python coder direct invocation")
    log_output(
        "LLM config",
        {
            "model": os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            "base_url": os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            "api_key_set": bool(os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)),
        },
    )
    tool = Python_Coder_Tool()
    result = tool.execute(
        query="Calculate 7 + 9 and print the result as a number."
    )
    log_output("Result", result)
    if "error" in result:
        raise AssertionError(f"Python coder error: {result['error']}")
    assert "execution_code" in result, "Expected execution_code in response"
    assert_contains_number(result, 16)


def test_execute_code_snippet_fenced_code_variants():
    log_step("Python coder execute_code_snippet fenced code variants")
    tool = Python_Coder_Tool()
    test_cases = [
        (
            "Please execute:\n```py\nprint(14)\nvalue = 14\n```\nDone.",
            "14",
        ),
        (
            "```python\nprint(18)\nvalue = 18\n```",
            "18",
        ),
        (
            "``` py\nprint(22)\nvalue = 22\n```",
            "22",
        ),
        (
            "``` python\nprint(26)\nvalue = 26\n```",
            "26",
        ),
    ]

    for code, expected in test_cases:
        result = tool.execute_code_snippet(code=code)
        if "error" in result:
            raise AssertionError(f"Python coder error: {result['error']}")
        assert result["printed_output"] == expected, f"Expected printed output {expected}"
        assert result["variables"]["value"] == expected, f"Expected value {expected}"
        assert "```" not in result["execution_code"], "Expected fenced markers to be stripped"


def test_execute_code_snippet_with_context():
    log_step("Python coder execute_code_snippet with context")
    tool = Python_Coder_Tool()
    result = tool.execute_code_snippet(
        code="value = scale(x)\nprint(value)",
        context_bindings={"x": 5},
        context_code="def scale(v):\n    return v * 6",
    )
    log_output("Result", result)
    if "error" in result:
        raise AssertionError(f"Python coder error: {result['error']}")
    assert result["printed_output"] == "30", "Expected printed output 30"
    assert result["variables"]["value"] == "30", "Expected value 30"


def test_execute_code_snippet_returns_full_stdout_and_all_variables():
    log_step("Python coder execute_code_snippet returns full stdout and all variables")
    tool = Python_Coder_Tool()
    long_text = "Y" * 8000
    assignments = "\n".join(f"v_{index} = {index}" for index in range(25))
    code = (
        f'text = "{long_text}"\n'
        "print(text)\n"
        f"{assignments}\n"
    )
    result = tool.execute_code_snippet(code=code)
    log_output(
        "Result summary",
        {
            "printed_output_length": len(result["printed_output"]),
            "variable_count": len(result["variables"]),
        },
    )
    if "error" in result:
        raise AssertionError(f"Python coder error: {result['error']}")
    assert result["printed_output"] == long_text, "Expected full printed output"
    assert result["variables"]["text"] == repr(long_text), "Expected full variable repr"
    assert "__truncated__" not in result["variables"], "Expected no dropped variables"
    for index in range(25):
        assert result["variables"][f"v_{index}"] == str(index), f"Expected v_{index}"


def test_execute_code_snippet_rejects_system_exit_in_context_code():
    log_step("Python coder execute_code_snippet rejects SystemExit in context_code")
    tool = Python_Coder_Tool()
    result = tool.execute_code_snippet(
        code="print('should not run')",
        context_code="import sys\nsys.exit(5)",
    )
    log_output("Result", result)
    assert "error" in result, "Expected error result"
    assert "SystemExit is not allowed" in result["error"], "Expected SystemExit guard"
    assert "5" in result["error"], "Expected exit code in error"


def test_execute_code_snippet_rejects_reserved_context_name():
    log_step("Python coder execute_code_snippet rejects reserved context name")
    tool = Python_Coder_Tool()
    result = tool.execute_code_snippet(
        code="print('should not run')",
        context_bindings={"__builtins__": "bad"},
    )
    log_output("Result", result)
    assert "error" in result, "Expected validation error"
    assert "reserved" in result["error"], "Expected reserved-name validation error"


def test_build_prompt_with_context_sections():
    log_step("Python coder prompt rendering with context sections")
    tool = Python_Coder_Tool()
    prompt = tool._build_prompt(
        query="Use scale(x) and print the result.",
        prepared_context=prepare_execution_context(
            context_bindings={"x": 3, "image_b64": "A" * 200},
            context_code="def scale(v):\n    return v * 10",
        ),
        context_annotations={
            "scale": "helper function already defined in context_code",
            "image_b64": "base64-encoded PNG image",
        },
    )
    log_output("Prompt", {"prompt": prompt})
    assert "Available executed context_code" in prompt, "Expected context_code section"
    assert "def scale(v):" in prompt, "Expected full context_code in prompt"
    assert "Available names from executed context_code" in prompt, "Expected context symbol section"
    assert "scale: function" in prompt, "Expected extracted context_code symbol"
    assert "image_b64: str base64-like len=200" in prompt, "Expected binding summary"
    assert "annotation: base64-encoded PNG image" in prompt, "Expected binding annotation"


def test_client_invocation():
    log_step("Python coder client invocation via server")
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Python_Code_Generator_Tool",
        {"query": "Calculate 12 * 4 and print the result."},
    )
    log_output("Response", response)
    result = response["result"]
    if "error" in result:
        raise AssertionError(f"Python coder error: {result['error']}")
    assert_contains_number(result, 48)


def main():
    test_execute_code_snippet_fenced_code_variants()
    test_execute_code_snippet_with_context()
    test_execute_code_snippet_returns_full_stdout_and_all_variables()
    test_execute_code_snippet_rejects_system_exit_in_context_code()
    test_execute_code_snippet_rejects_reserved_context_name()
    test_build_prompt_with_context_sections()
    if not ensure_tool_server(
        SERVER_URL, required_tools=["Python_Code_Generator_Tool"]
    ):
        return
    test_direct_invocation()
    test_client_invocation()
    print("Python coder tool test completed.")


if __name__ == "__main__":
    main()
