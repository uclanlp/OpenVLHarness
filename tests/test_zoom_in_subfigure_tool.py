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
from mini_vlm_toolkit.tools.zoom_in_subfigure.tool import ZoomInSubfigure_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
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


def ensure_quadrant_image():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, "quadrants.png")
    if os.path.isfile(image_path):
        return image_path

    image = Image.new("RGB", (200, 200), (255, 255, 255))
    pixels = image.load()
    for x in range(200):
        for y in range(200):
            if x < 100 and y < 100:
                pixels[x, y] = (255, 0, 0)
            elif x >= 100 and y < 100:
                pixels[x, y] = (0, 255, 0)
            elif x < 100 and y >= 100:
                pixels[x, y] = (0, 0, 255)
            else:
                pixels[x, y] = (255, 255, 0)
    image.save(image_path, format="PNG")
    return image_path


def assert_center_color(image, expected_rgb):
    width, height = image.size
    pixel = image.getpixel((width // 2, height // 2))
    assert pixel == expected_rgb


def test_direct_invocation():
    log_step("ZoomInSubfigure direct invocation")
    image_path = ensure_quadrant_image()

    tool = ZoomInSubfigure_Tool()
    result = tool.execute(image_path=image_path, description="top left subplot")
    log_output("Result", result)

    images = result["subplot_images_base64"]
    assert images and len(images) == 1
    crop = decode_base64_image(images[0]).convert("RGB")
    save_test_output(
        crop,
        build_test_id(__file__, "test_direct_invocation"),
        suffix="top_left",
    )
    assert crop.size == (100, 100)
    assert_center_color(crop, (255, 0, 0))


def test_client_invocation():
    log_step("ZoomInSubfigure client invocation via server")
    image_path = ensure_quadrant_image()
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "ZoomInSubfigure_Tool",
        {"image_path": image_path, "description": "bottom right"},
    )
    log_output("Response", response)
    images = response["result"]["subplot_images_base64"]
    crop = decode_base64_image(images[0]).convert("RGB")
    save_test_output(
        crop,
        build_test_id(__file__, "test_client_invocation"),
        suffix="bottom_right",
    )
    assert_center_color(crop, (255, 255, 0))


def test_llm_invocation():
    log_step("ZoomInSubfigure LLM-assisted invocation via server")
    image_path = ensure_quadrant_image()

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
    metadata = tools["ZoomInSubfigure_Tool"]
    log_output("ZoomInSubfigure_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "ZoomInSubfigure_Tool",
        metadata,
        "Return the top right subplot.",
        image_path,
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("description", "top right")

    response = client.execute("ZoomInSubfigure_Tool", args)
    log_output("Response", response)
    images = response["result"]["subplot_images_base64"]
    crop = decode_base64_image(images[0]).convert("RGB")
    save_test_output(
        crop,
        build_test_id(__file__, "test_llm_invocation"),
        suffix="top_right",
    )
    assert_center_color(crop, (0, 255, 0))


def main():
    if not ensure_tool_server(
        SERVER_URL, required_tools=["ZoomInSubfigure_Tool"]
    ):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("Zoom in subfigure tool test completed.")


if __name__ == "__main__":
    main()
