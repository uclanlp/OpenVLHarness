import json
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.python_tool_call_parser import (
    parse_python_tool_call,
    parse_python_tool_call_args,
)
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.python_executor.tool import Python_Executor_Tool
from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(tool_name, tool_metadata, user_request):
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


def test_llm_invocation():
    log_step("LLM-assisted invocation via server")
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


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["Python_Executor_Tool"]):
        return
    log_step("Python executor tests")
    test_direct_invocation()
    test_direct_invocation_nonstandard_outputs()
    test_parse_python_tool_call_args_with_explanatory_text()
    test_parse_python_tool_call_args_with_fenced_block()
    test_parse_python_tool_call_args_with_think_block()
    test_parse_python_tool_call_with_tool_name()
    test_client_invocation()
    test_llm_invocation()
    print("All python executor tests passed.")


if __name__ == "__main__":
    main()
