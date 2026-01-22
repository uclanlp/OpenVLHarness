import json
import os

from PIL import Image

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.json_tool_call_parser import parse_json_tool_call_args
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.overlay_images.tool import OverlayImages_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    encode_image_base64,
    ensure_overlay_image,
    ensure_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def build_tool_call_prompt(
    tool_name, tool_metadata, user_request, base_image, overlay_image, call_format="python"
):
    if call_format == "json":
        return (
            "Generate a JSON tool call object.\n"
            "Use {\"tool_name\": \"ToolName\", \"args\": {...}} "
            "or {\"name\": \"ToolName\", \"arguments\": {...}}.\n"
            "Use keyword arguments only.\n"
            f"Tool name: {tool_name}\n"
            f"Tool metadata: {tool_metadata}\n"
            f"Base image path: {base_image}\n"
            f"Overlay image path: {overlay_image}\n"
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
        f"Base image path: {base_image}\n"
        f"Overlay image path: {overlay_image}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("Overlay direct invocation")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay direct test.")
        return

    tool = OverlayImages_Tool()
    result = tool.execute(
        base_image_path=base_image,
        overlay_image_path=overlay_image,
        alpha=0.6,
        bounding_box=[0.2, 0.2, 0.5, 0.5],
    )
    log_output("Result", result)

    blended = decode_base64_image(result["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_direct_invocation"))
    base = Image.open(base_image)
    assert blended.size == base.size


def test_client_invocation():
    log_step("Overlay client invocation via server")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        "OverlayImages_Tool",
        {
            "base_image_path": base_image,
            "overlay_image_path": overlay_image,
            "alpha": 0.6,
            "bounding_box": [0.1, 0.1, 0.4, 0.4],
        },
    )
    log_output("Response", response)
    blended = decode_base64_image(response["result"]["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_client_invocation"))
    assert blended.size[0] > 0 and blended.size[1] > 0


def test_direct_invocation_base64():
    log_step("Overlay direct invocation (base64)")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay base64 direct test.")
        return

    tool = OverlayImages_Tool()
    result = tool.execute(
        base_image_base64=encode_image_base64(base_image),
        overlay_image_base64=encode_image_base64(overlay_image),
        alpha=0.6,
        bounding_box=[0.2, 0.2, 0.5, 0.5],
    )
    log_output("Result", result)

    blended = decode_base64_image(result["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_direct_invocation_base64"))
    assert blended.size[0] > 0 and blended.size[1] > 0


def test_client_invocation_base64():
    log_step("Overlay client invocation via server (base64)")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay base64 client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        "OverlayImages_Tool",
        {
            "base_image_base64": encode_image_base64(base_image),
            "overlay_image_base64": encode_image_base64(overlay_image),
            "alpha": 0.6,
            "bounding_box": [0.1, 0.1, 0.4, 0.4],
        },
    )
    log_output("Response", response)
    blended = decode_base64_image(response["result"]["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_client_invocation_base64"))
    assert blended.size[0] > 0 and blended.size[1] > 0


def test_llm_invocation_python():
    log_step("Overlay LLM-assisted invocation via server (python)")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay LLM test.")
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
    metadata = tools["OverlayImages_Tool"]
    log_output("OverlayImages_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "OverlayImages_Tool",
        metadata,
        "Overlay the second image onto the top-left quarter with alpha 0.6.",
        base_image,
        overlay_image,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("base_image_path", base_image)
    args.setdefault("overlay_image_path", overlay_image)
    args.setdefault("alpha", 0.6)
    args.setdefault("bounding_box", [0.0, 0.0, 0.5, 0.5])

    response = client.execute("OverlayImages_Tool", args)
    log_output("Response", response)
    blended = decode_base64_image(response["result"]["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_llm_invocation_python"))
    assert blended.size[0] > 0 and blended.size[1] > 0


def test_llm_invocation_json():
    log_step("Overlay LLM-assisted invocation via server (json)")
    base_image = ensure_test_image()
    overlay_image = ensure_overlay_image()
    if not base_image or not overlay_image:
        print("Test images unavailable; skipping overlay LLM test.")
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
    metadata = tools["OverlayImages_Tool"]
    log_output("OverlayImages_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "OverlayImages_Tool",
        metadata,
        "Overlay the second image onto the top-left quarter with alpha 0.6.",
        base_image,
        overlay_image,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("base_image_path", base_image)
    args.setdefault("overlay_image_path", overlay_image)
    args.setdefault("alpha", 0.6)
    args.setdefault("bounding_box", [0.0, 0.0, 0.5, 0.5])

    response = client.execute("OverlayImages_Tool", args)
    log_output("Response", response)
    blended = decode_base64_image(response["result"]["blended_image_base64"])
    save_test_output(blended, build_test_id(__file__, "test_llm_invocation_json"))
    assert blended.size[0] > 0 and blended.size[1] > 0


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["OverlayImages_Tool"]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_direct_invocation_base64()
    test_client_invocation_base64()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("Overlay images tool test completed.")


if __name__ == "__main__":
    main()
