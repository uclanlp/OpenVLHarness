import json
import os

from PIL import Image

from client.client import ToolClient
from client.json_tool_call_parser import parse_json_tool_call_args
from client.python_tool_call_parser import parse_python_tool_call_args
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.zoom_in_image_by_bbox.tool import ZoomInImageByBBox_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    encode_image_base64,
    ensure_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(tool_name, tool_metadata, user_request, image_path, call_format="python"):
    if call_format == "json":
        return (
            "Generate a JSON tool call object.\n"
            "Use {\"tool_name\": \"ToolName\", \"args\": {...}} "
            "or {\"name\": \"ToolName\", \"arguments\": {...}}.\n"
            "Use keyword arguments only.\n"
            f"Tool name: {tool_name}\n"
            f"Tool metadata: {tool_metadata}\n"
            f"Image path to use: {image_path}\n"
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
        f"Image path to use: {image_path}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("ZoomInImageByBBox direct invocation")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom direct test.")
        return

    tool = ZoomInImageByBBox_Tool()
    result = tool.execute(
        image_path=image_path,
        bbox=[0.2, 0.2, 0.4, 0.4],
        padding=0.05,
    )
    log_output("Result", result)

    cropped = decode_base64_image(result["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_direct_invocation"))
    base = Image.open(image_path)
    expected_width = int(round(0.5 * base.size[0]))
    expected_height = int(round(0.5 * base.size[1]))
    assert cropped.size == (expected_width, expected_height)


def test_client_invocation():
    log_step("ZoomInImageByBBox client invocation via server")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        "ZoomInImageByBBox_Tool",
        {
            "image_path": image_path,
            "bbox": [0.1, 0.1, 0.3, 0.3],
            "padding": 0.1,
        },
    )
    log_output("Response", response)
    cropped = decode_base64_image(response["result"]["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_client_invocation"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_direct_invocation_base64():
    log_step("ZoomInImageByBBox direct invocation (base64)")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom base64 direct test.")
        return

    image_base64 = encode_image_base64(image_path)
    tool = ZoomInImageByBBox_Tool()
    result = tool.execute(
        image_base64=image_base64,
        bbox=[0.2, 0.2, 0.4, 0.4],
        padding=0.05,
    )
    log_output("Result", result)

    cropped = decode_base64_image(result["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_direct_invocation_base64"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_client_invocation_base64():
    log_step("ZoomInImageByBBox client invocation via server (base64)")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom base64 client test.")
        return

    image_base64 = encode_image_base64(image_path)
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "ZoomInImageByBBox_Tool",
        {
            "image_base64": image_base64,
            "bbox": [0.1, 0.1, 0.3, 0.3],
            "padding": 0.1,
        },
    )
    log_output("Response", response)
    cropped = decode_base64_image(response["result"]["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_client_invocation_base64"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_llm_invocation_python():
    log_step("ZoomInImageByBBox LLM-assisted invocation via server (python)")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom LLM test.")
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
    metadata = tools["ZoomInImageByBBox_Tool"]
    log_output("ZoomInImageByBBox_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "ZoomInImageByBBox_Tool",
        metadata,
        "Zoom in on bbox [0.25, 0.25, 0.3, 0.3] with padding 0.05.",
        image_path,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("bbox", [0.25, 0.25, 0.3, 0.3])
    args.setdefault("padding", 0.05)

    response = client.execute("ZoomInImageByBBox_Tool", args)
    log_output("Response", response)
    cropped = decode_base64_image(response["result"]["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_llm_invocation_python"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_llm_invocation_json():
    log_step("ZoomInImageByBBox LLM-assisted invocation via server (json)")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping zoom LLM test.")
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
    metadata = tools["ZoomInImageByBBox_Tool"]
    log_output("ZoomInImageByBBox_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "ZoomInImageByBBox_Tool",
        metadata,
        "Zoom in on bbox [0.25, 0.25, 0.3, 0.3] with padding 0.05.",
        image_path,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("bbox", [0.25, 0.25, 0.3, 0.3])
    args.setdefault("padding", 0.05)

    response = client.execute("ZoomInImageByBBox_Tool", args)
    log_output("Response", response)
    cropped = decode_base64_image(response["result"]["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_llm_invocation_json"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def main():
    if not ensure_tool_server(
        SERVER_URL, required_tools=["ZoomInImageByBBox_Tool"]
    ):
        return
    test_direct_invocation()
    test_client_invocation()
    test_direct_invocation_base64()
    test_client_invocation_base64()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("Zoom in by bbox tool test completed.")


if __name__ == "__main__":
    main()
