import json
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.json_tool_call_parser import parse_json_tool_call_args
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.llm_direct_generator.tool import (
    MODEL_BASE_URLS_ENV,
    TOOL_NAME,
    LLM_Direct_Generator_Tool,
)
from tests.utils import (
    ensure_model_available,
    ensure_tool_server,
    get_model_base_url_from_env,
    load_env_file,
)


MODEL_NAME = "Qwen/Qwen3-8B"

load_env_file()

SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def get_model_base_url():
    base_url = get_model_base_url_from_env(MODEL_NAME)
    if not base_url:
        print(f"{MODEL_BASE_URLS_ENV} missing {MODEL_NAME}; skipping tests.")
    return base_url


def ensure_direct_model_ready():
    base_url = get_model_base_url()
    if not base_url:
        return None
    if not ensure_model_available(base_url, MODEL_NAME):
        return None
    return base_url


def build_tool_call_prompt(tool_name, tool_metadata, user_request, call_format="python"):
    if call_format == "json":
        return (
            "Generate a JSON tool call object.\n"
            "Use {\"tool_name\": \"ToolName\", \"args\": {...}} "
            "or {\"name\": \"ToolName\", \"arguments\": {...}}.\n"
            "Use keyword arguments only.\n"
            f"Tool name: {tool_name}\n"
            f"Tool metadata: {tool_metadata}\n"
            f"User request: {user_request}\n"
            "Return only the JSON object."
        )
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: ToolName(...).\n"
        "Use the tool name as the function name.\n"
        "Use keyword arguments only.\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def assert_response_payload(result):
    assert isinstance(result, dict), "Expected a dict response"
    assert "choices" in result, "Expected choices in response"
    assert isinstance(result["choices"], list), "Expected choices to be a list"
    assert result["choices"], "Expected at least one choice"


def test_direct_invocation():
    log_step("Direct invocation")
    base_url = ensure_direct_model_ready()
    if not base_url:
        return

    tool = LLM_Direct_Generator_Tool()
    result = tool.execute(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=32,
        temperature=0.2,
    )
    log_output("Result", result)
    assert_response_payload(result)


def test_client_invocation():
    log_step("Client invocation via server")
    base_url = ensure_direct_model_ready()
    if not base_url:
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 32,
            "temperature": 0.2,
        },
    )
    log_output("Response", response)
    result = response["result"]
    assert_response_payload(result)


def test_llm_invocation_python():
    log_step("LLM-assisted invocation via server (python)")
    base_url = ensure_direct_model_ready()
    if not base_url:
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return

    engine = LLMEngine(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
    )

    client = ToolClient(SERVER_URL)
    tools = client.list_tools()
    metadata = tools[TOOL_NAME]
    log_output("Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        (
            "Send a short greeting to the configured model and keep the response brief. "
            "Use max_tokens=32 and temperature=0.2."
        ),
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("model", MODEL_NAME)
    args.setdefault("messages", [{"role": "user", "content": "Say hello."}])
    args.setdefault("max_tokens", 32)
    args.setdefault("temperature", 0.2)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert_response_payload(result)


def test_llm_invocation_json():
    log_step("LLM-assisted invocation via server (json)")
    base_url = ensure_direct_model_ready()
    if not base_url:
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return

    engine = LLMEngine(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        api_key=os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY),
    )

    client = ToolClient(SERVER_URL)
    tools = client.list_tools()
    metadata = tools[TOOL_NAME]
    log_output("Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        (
            "Send a short greeting to the configured model and keep the response brief. "
            "Use max_tokens=32 and temperature=0.2."
        ),
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("model", MODEL_NAME)
    args.setdefault("messages", [{"role": "user", "content": "Say hello."}])
    args.setdefault("max_tokens", 32)
    args.setdefault("temperature", 0.2)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert_response_payload(result)


def main():
    if not ensure_direct_model_ready():
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return

    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("LLM direct generator text-model tests completed.")


if __name__ == "__main__":
    main()
