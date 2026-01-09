import os

from mini_vlm_tool_kit.client.client import ToolClient
from mini_vlm_tool_kit.llm.engine import LLMEngine
from mini_vlm_tool_kit.tools.python_executor.tool import Python_Executor_Tool


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8001")


def test_direct_invocation():
    tool = Python_Executor_Tool()
    result = tool.execute(code="print(2 + 3)\nvalue = 2 + 3")
    assert "5" in result["printed_output"], "Expected printed output to contain 5"
    assert result["variables"]["value"] == "5", "Expected value to be 5"


def test_client_invocation():
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Python_Executor_Tool",
        {"code": "print(10)\nvalue = 10"},
    )
    result = response["result"]
    assert "10" in result["printed_output"], "Expected printed output to contain 10"
    assert result["variables"]["value"] == "10", "Expected value to be 10"


def test_llm_invocation():
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL is required for the LLM test")

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8003/v1")
    api_key = os.getenv("LLM_API_KEY", "local")
    engine = LLMEngine(model=model, base_url=base_url, api_key=api_key)

    client = ToolClient(SERVER_URL)
    tools = client.list_tools()
    metadata = tools["Python_Executor_Tool"]

    response = client.execute_with_llm(
        "Python_Executor_Tool",
        metadata,
        "Write Python code that prints 7 + 9 and stores the result in a variable named value.",
        engine,
    )

    result = response["result"]
    assert "16" in result["printed_output"], "Expected printed output to contain 16"
    assert result["variables"]["value"] == "16", "Expected value to be 16"


def main():
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("All python executor tests passed.")


if __name__ == "__main__":
    main()
