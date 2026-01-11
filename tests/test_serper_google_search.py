import json
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.serper_google_search.tool import (
    KEY_FILE_PATH,
    TOOL_NAME,
    Serper_Google_Search_Tool,
)
from tests.utils import ensure_tool_server


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(tool_name, tool_metadata, user_request):
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: execution = tool.execute(...).\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def require_key_file():
    if not os.path.exists(KEY_FILE_PATH):
        raise RuntimeError(
            f"Missing Serper API key file at {KEY_FILE_PATH}. "
            "Create the file and place the key on a single line."
        )


def assert_urls(result):
    assert isinstance(result, list), "Expected a list of URLs"
    assert result, "Expected at least one URL"
    assert all(isinstance(url, str) for url in result), "Expected string URLs"


def test_direct_invocation():
    log_step("Serper Google search direct invocation")
    require_key_file()
    tool = Serper_Google_Search_Tool()
    result = tool.execute(query="OpenAI homepage", num_results=5)
    log_output("Result", {"urls": result})
    assert_urls(result)


def test_client_invocation():
    log_step("Serper Google search client invocation via server")
    require_key_file()
    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {"query": "OpenAI homepage", "num_results": 5},
    )
    log_output("Response", response)
    assert_urls(response["result"])


def test_llm_invocation():
    log_step("Serper Google search LLM-assisted invocation via server")
    require_key_file()
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
    metadata = tools[TOOL_NAME]
    log_output("Tools", {"tool_names": sorted(tools.keys())})
    log_output("Serper tool metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Search for the OpenAI homepage and return 3 URLs.",
    )
    log_output("Prompt", prompt)
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.6, top_p=0.95)
    log_output("LLM output", llm_output)
    args = parse_tool_execute_args(llm_output)
    log_output("Parsed args", args)
    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    assert_urls(response["result"])


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("Serper Google search tool tests completed.")


if __name__ == "__main__":
    main()
