import json
import os

from client.client import ToolClient
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
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
    if not ensure_tool_server(
        SERVER_URL, required_tools=["Python_Code_Generator_Tool"]
    ):
        return
    test_direct_invocation()
    test_client_invocation()
    print("Python coder tool test completed.")


if __name__ == "__main__":
    main()
