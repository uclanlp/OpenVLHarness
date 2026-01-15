import json
import os

from PIL import Image

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.crop.tool import Crop_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


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
    log_step("Crop direct invocation")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping crop direct test.")
        return

    tool = Crop_Tool()
    result = tool.execute(
        image_path=image_path,
        coords=[0.1, 0.2, 0.6, 0.7],
        coords_type="normalized_xyxy",
    )
    log_output("Result", result)

    cropped = decode_base64_image(result["cropped_image_base64"])
    save_test_output(
        cropped, build_test_id(__file__, "test_direct_invocation"), suffix="normalized"
    )
    base = Image.open(image_path)
    expected_width = int(round((0.6 - 0.1) * base.size[0]))
    expected_height = int(round((0.7 - 0.2) * base.size[1]))
    assert cropped.size == (expected_width, expected_height)

    pixel_coords = [10, 20, 60, 120]
    result_pixel = tool.execute(
        image_path=image_path,
        coords=pixel_coords,
        coords_type="pixel_xyxy",
    )
    log_output("Pixel Result", result_pixel)
    cropped_pixel = decode_base64_image(result_pixel["cropped_image_base64"])
    save_test_output(
        cropped_pixel, build_test_id(__file__, "test_direct_invocation"), suffix="pixel"
    )
    assert cropped_pixel.size == (pixel_coords[2] - pixel_coords[0], pixel_coords[3] - pixel_coords[1])


def test_client_invocation():
    log_step("Crop client invocation via server")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping crop client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Crop_Tool",
        {
            "image_path": image_path,
            "coords": [0.15, 0.25, 0.5, 0.6],
            "coords_type": "normalized_xyxy",
        },
    )
    log_output("Response", response)
    result = response["result"]
    cropped = decode_base64_image(result["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_client_invocation"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_llm_invocation():
    log_step("Crop LLM-assisted invocation via server")
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping crop LLM test.")
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
    metadata = tools["Crop_Tool"]
    log_output("Crop_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "Crop_Tool",
        metadata,
        "Crop the center area using coords [0.2, 0.2, 0.8, 0.8] in normalized_xyxy.",
        image_path,
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("coords", [0.2, 0.2, 0.8, 0.8])
    args.setdefault("coords_type", "normalized_xyxy")

    response = client.execute("Crop_Tool", args)
    log_output("Response", response)
    result = response["result"]
    cropped = decode_base64_image(result["cropped_image_base64"])
    save_test_output(cropped, build_test_id(__file__, "test_llm_invocation"))
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["Crop_Tool"]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("Crop tool test completed.")


if __name__ == "__main__":
    main()
