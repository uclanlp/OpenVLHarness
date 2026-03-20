import json
import os

from client.client import ToolClient, ToolClientError
from client.json_tool_call_parser import parse_json_tool_call_args
from client.python_tool_call_parser import (
    parse_python_tool_call,
    parse_python_tool_call_args,
)
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.python_executor.tool import Python_Executor_Tool
from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(tool_name, tool_metadata, user_request, call_format="python"):
    if call_format == "json":
        return (
            "Generate a JSON tool call object.\n"
            "Use {\"tool_name\": \"ToolName\", \"args\": {...}} "
            "or {\"name\": \"ToolName\", \"arguments\": {...}}.\n"
            f"Tool name: {tool_name}\n"
            f"Tool metadata: {tool_metadata}\n"
            f"User request: {user_request}\n"
            "Return only the JSON object."
        )
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: ToolName(...).\n"
        "Use the tool name as the function name.\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("Direct invocation")
    tool = Python_Executor_Tool()
    result = tool.execute(code="print(2 + 3)\nvalue = 2 + 3")
    log_output("Result", result)
    assert "5" in result["printed_output"], "Expected printed output to contain 5"
    assert result["variables"]["value"] == "5", "Expected value to be 5"


def test_direct_invocation_nonstandard_outputs():
    log_step("Direct invocation with non-standard outputs")
    tool = Python_Executor_Tool()
    code = (
        "class BadRepr:\n"
        "    def __repr__(self):\n"
        "        raise ValueError('boom')\n"
        "\n"
        "class NonStringRepr:\n"
        "    def __repr__(self):\n"
        "        return 123\n"
        "\n"
        "bad = BadRepr()\n"
        "weird = NonStringRepr()\n"
        "value = {'ok': 1}\n"
        "print('ready')\n"
    )
    result = tool.execute(code=code)
    log_output("Result", result)
    assert "ready" in result["printed_output"], "Expected printed output to contain ready"
    assert result["variables"]["value"] == "{'ok': 1}", "Expected value to be the dict repr"
    assert result["variables"]["bad"].startswith(
        "<repr_error BadRepr"
    ), "Expected fallback repr for BadRepr"
    assert result["variables"]["weird"].startswith(
        "<repr_error NonStringRepr"
    ), "Expected fallback repr for NonStringRepr"


def test_direct_invocation_with_context_bindings():
    log_step("Direct invocation with context_bindings")
    tool = Python_Executor_Tool()
    result = tool.execute(
        code="result = x * 10\nprint(result)",
        context_bindings={"x": 3},
    )
    log_output("Result", result)
    assert result["printed_output"] == "30", "Expected printed output to be 30"
    assert result["variables"]["result"] == "30", "Expected result variable to be 30"
    assert result["variables"]["x"] == "3", "Expected injected binding to survive"


def test_direct_invocation_with_mutated_context_binding():
    log_step("Direct invocation with mutated context binding")
    tool = Python_Executor_Tool()
    context_bindings = {"data": {"count": 1}}
    result = tool.execute(
        code='data["count"] += 1\nprint(data["count"])',
        context_bindings=context_bindings,
    )
    log_output("Result", result)
    assert result["printed_output"] == "2", "Expected printed output to be 2"
    assert result["variables"]["data"] == "{'count': 2}", "Expected mutated binding"
    assert context_bindings == {"data": {"count": 1}}, "Expected caller-owned input to remain unchanged"


def test_direct_invocation_returns_full_stdout_and_variable_values():
    log_step("Direct invocation returns full stdout and variable values")
    tool = Python_Executor_Tool()
    long_text = "X" * 5000
    result = tool.execute(
        code=f'text = "{long_text}"\nprint(text)',
    )
    log_output(
        "Result summary",
        {
            "printed_output_length": len(result["printed_output"]),
            "variable_length": len(result["variables"]["text"]) - 2,
        },
    )
    assert result["printed_output"] == long_text, "Expected full printed output"
    assert result["variables"]["text"] == repr(long_text), "Expected full variable repr"


def test_direct_invocation_rejects_system_exit_in_main_code():
    log_step("Direct invocation rejects SystemExit in main code")
    tool = Python_Executor_Tool()
    try:
        tool.execute(code="import sys\nsys.exit(7)")
    except RuntimeError as exc:
        assert "SystemExit is not allowed" in str(exc), "Expected SystemExit guard"
        assert "7" in str(exc), "Expected exit code in error"
        return
    raise AssertionError("Expected SystemExit in main code to raise RuntimeError")


def test_direct_invocation_fenced_code_variants():
    log_step("Direct invocation with fenced code variants")
    tool = Python_Executor_Tool()
    test_cases = [
        (
            "Please execute this:\n```py\nprint(6)\nvalue = 6\n```\nThanks.",
            "6",
        ),
        (
            "```python\nprint(8)\nvalue = 8\n```",
            "8",
        ),
        (
            "``` py\nprint(10)\nvalue = 10\n```",
            "10",
        ),
        (
            "``` python\nprint(12)\nvalue = 12\n```",
            "12",
        ),
    ]

    for code, expected in test_cases:
        result = tool.execute(code=code)
        assert result["printed_output"] == expected, f"Expected printed output {expected}"
        assert result["variables"]["value"] == expected, f"Expected value to be {expected}"
        assert "```" not in result["execution_code"], "Expected fenced markers to be stripped"


def test_direct_invocation_rejects_reserved_context_name():
    log_step("Direct invocation rejects reserved context_bindings name")
    tool = Python_Executor_Tool()
    try:
        tool.execute(
            code="print('should not run')",
            context_bindings={"__builtins__": "bad"},
        )
    except ValueError as exc:
        assert "reserved" in str(exc), "Expected reserved-name validation error"
        return
    raise AssertionError("Expected reserved context_bindings name to raise ValueError")


def test_parse_python_tool_call_args_with_explanatory_text():
    log_step("Parse tool args with extra text")
    llm_output = (
        "Okay, let's see. The user wants a tool call.\n"
        "Python_Executor_Tool(code=\"print(3)\\nvalue = 3\")\n"
        "That should do it."
    )
    args = parse_python_tool_call_args(llm_output)
    log_output("Parsed args", args)
    assert "print(3)" in args["code"], "Expected code to include print(3)"


def test_parse_python_tool_call_args_with_fenced_block():
    log_step("Parse tool args with fenced code block")
    llm_output = (
        "Here is the code:\n"
        "```\n"
        "Python_Executor_Tool(code=\"print(4)\")\n"
        "```\n"
    )
    args = parse_python_tool_call_args(llm_output)
    log_output("Parsed args", args)
    assert args["code"] == "print(4)", "Expected code to match print(4)"


def test_parse_python_tool_call_args_with_think_block():
    log_step("Parse tool args with think block")
    llm_output = (
        "<think>\n"
        "The tool call should be: ToolName(...).\n"
        "</think>\n"
        "``` python\n"
        "Python_Executor_Tool(code=\"value = 7 + 9\\nprint(value)\")\n"
        "```\n"
    )
    args = parse_python_tool_call_args(llm_output)
    log_output("Parsed args", args)
    assert "value = 7 + 9" in args["code"], "Expected code to include the sum"


def test_parse_python_tool_call_with_tool_name():
    log_step("Parse tool call with tool name")
    llm_output = "Python_Executor_Tool(code=\"print(11)\")"
    parsed = parse_python_tool_call(llm_output)
    log_output("Parsed tool call", parsed)
    assert parsed["tool_name"] == "Python_Executor_Tool"
    assert parsed["args"]["code"] == "print(11)"


def test_client_invocation():
    log_step("Client invocation via server")
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Python_Executor_Tool",
        {"code": "print(10)\nvalue = 10"},
    )
    log_output("Response", response)
    result = response["result"]
    assert "10" in result["printed_output"], "Expected printed output to contain 10"
    assert result["variables"]["value"] == "10", "Expected value to be 10"


def test_client_invocation_with_context():
    log_step("Client invocation via server with context")
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Python_Executor_Tool",
        {
            "code": "value = x * 5\nprint(value)",
            "context_bindings": {"x": 4},
        },
    )
    log_output("Response", response)
    result = response["result"]
    assert result["printed_output"] == "20", "Expected printed output to be 20"
    assert result["variables"]["value"] == "20", "Expected value to be 20"


def test_client_invocation_system_exit_returns_error_and_server_stays_alive():
    log_step("Client invocation returns error for SystemExit and server stays alive")
    client = ToolClient(SERVER_URL)
    try:
        client.execute(
            "Python_Executor_Tool",
            {"code": "import sys\nsys.exit(11)"},
        )
    except ToolClientError as exc:
        assert exc.status_code == 500, "Expected server-side tool execution error"
        assert "SystemExit is not allowed" in str(exc), "Expected SystemExit guard"
    else:
        raise AssertionError("Expected ToolClientError for SystemExit")

    response = client.execute(
        "Python_Executor_Tool",
        {"code": "print(13)\nvalue = 13"},
    )
    log_output("Follow-up response", response)
    assert response["result"]["printed_output"] == "13", "Expected server to remain healthy"


def test_llm_invocation_python():
    log_step("LLM-assisted invocation via server (python)")
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    log_output(
        "LLM config",
        {
            "model": model,
            "base_url": base_url,
            "api_key_set": bool(api_key),
        },
    )
    engine = LLMEngine(model=model, base_url=base_url, api_key=api_key)

    client = ToolClient(SERVER_URL)
    tools = client.list_tools()
    metadata = tools["Python_Executor_Tool"]
    log_output("Tools", {"tool_names": sorted(tools.keys())})
    log_output("Python_Executor_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "Python_Executor_Tool",
        metadata,
        "Write Python code that prints 7 + 9 and stores the result in a variable named value.",
        call_format="python",
    )
    log_output("Prompt", prompt)
    generation_params = {
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    log_output("Generation params", generation_params)
    llm_output = engine.generate(prompt, **generation_params)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    log_output("Parsed args", args)
    response = client.execute("Python_Executor_Tool", args)
    log_output("Response", response)

    result = response["result"]
    assert "16" in result["printed_output"], "Expected printed output to contain 16"
    assert result["variables"]["value"] == "16", "Expected value to be 16"


def test_llm_invocation_json():
    log_step("LLM-assisted invocation via server (json)")
    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    log_output(
        "LLM config",
        {
            "model": model,
            "base_url": base_url,
            "api_key_set": bool(api_key),
        },
    )
    engine = LLMEngine(model=model, base_url=base_url, api_key=api_key)

    client = ToolClient(SERVER_URL)
    tools = client.list_tools()
    metadata = tools["Python_Executor_Tool"]
    log_output("Tools", {"tool_names": sorted(tools.keys())})
    log_output("Python_Executor_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "Python_Executor_Tool",
        metadata,
        "Write Python code that prints 7 + 9 and stores the result in a variable named value.",
        call_format="json",
    )
    log_output("Prompt", prompt)
    generation_params = {
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    log_output("Generation params", generation_params)
    llm_output = engine.generate(prompt, **generation_params)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    log_output("Parsed args", args)
    response = client.execute("Python_Executor_Tool", args)
    log_output("Response", response)

    result = response["result"]
    assert "16" in result["printed_output"], "Expected printed output to contain 16"
    assert result["variables"]["value"] == "16", "Expected value to be 16"


def main():
    log_step("Python executor tests")
    test_direct_invocation()
    test_direct_invocation_nonstandard_outputs()
    test_direct_invocation_with_context_bindings()
    test_direct_invocation_with_mutated_context_binding()
    test_direct_invocation_returns_full_stdout_and_variable_values()
    test_direct_invocation_rejects_system_exit_in_main_code()
    test_direct_invocation_fenced_code_variants()
    test_direct_invocation_rejects_reserved_context_name()
    test_parse_python_tool_call_args_with_explanatory_text()
    test_parse_python_tool_call_args_with_fenced_block()
    test_parse_python_tool_call_args_with_think_block()
    test_parse_python_tool_call_with_tool_name()
    if not ensure_tool_server(SERVER_URL, required_tools=["Python_Executor_Tool"]):
        print("Skipping server-backed python executor tests.")
        return
    test_client_invocation()
    test_client_invocation_with_context()
    test_client_invocation_system_exit_returns_error_and_server_stays_alive()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("All python executor tests passed.")


if __name__ == "__main__":
    main()
