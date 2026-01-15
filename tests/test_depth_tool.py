import json
import os

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.depth.tool import Depth_Tool, TOOL_NAME
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_depth_server,
    ensure_depth_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
DEPTH_SERVER_URL = os.getenv("DEPTH_ANYTHING_SERVER_URL", "http://127.0.0.1:8082")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(tool_name, tool_metadata, user_request, image_path):
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: ToolName(...).\n"
        "Use the tool name as the function name.\n"
        "Use keyword arguments only.\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"Image path to use: {image_path}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("Depth direct invocation")
    if not ensure_depth_server(DEPTH_SERVER_URL):
        return
    image_path = ensure_depth_test_image()
    if not image_path:
        print("Depth test image unavailable; skipping direct test.")
        return

    tool = Depth_Tool()
    result = tool.execute(image_path=image_path, server_url=DEPTH_SERVER_URL)
    log_output("Result", result)
    assert "depth_map_base64" in result, "Expected depth_map_base64 in response"

    depth_image = decode_base64_image(result["depth_map_base64"])
    save_test_output(depth_image, build_test_id(__file__, "test_direct_invocation"))


def test_client_invocation():
    log_step("Depth client invocation via server")
    if not ensure_depth_server(DEPTH_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_depth_test_image()
    if not image_path:
        print("Depth test image unavailable; skipping client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {
            "image_path": image_path,
            "server_url": DEPTH_SERVER_URL,
        },
    )
    log_output("Response", response)
    result = response["result"]
    assert "depth_map_base64" in result, "Expected depth_map_base64 in response"

    depth_image = decode_base64_image(result["depth_map_base64"])
    save_test_output(depth_image, build_test_id(__file__, "test_client_invocation"))


def test_llm_invocation():
    log_step("Depth LLM-assisted invocation via server")
    if not ensure_depth_server(DEPTH_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_depth_test_image()
    if not image_path:
        print("Depth test image unavailable; skipping LLM test.")
        return

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
    log_output("Depth metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Generate a depth map for the image.",
        image_path,
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.6)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("server_url", DEPTH_SERVER_URL)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert "depth_map_base64" in result, "Expected depth_map_base64 in response"

    depth_image = decode_base64_image(result["depth_map_base64"])
    save_test_output(depth_image, build_test_id(__file__, "test_llm_invocation"))


def main():
    if not ensure_depth_server(DEPTH_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("Depth tool tests completed.")


if __name__ == "__main__":
    main()
