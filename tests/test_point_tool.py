import json
import os

from PIL import Image, ImageDraw

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.point.tool import Point_Tool
from tests.utils import (
    build_test_id,
    ensure_blank_image,
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


def save_points_preview(image_path, points, test_id, suffix=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for x_value, y_value in points:
        x_px = int(round(x_value * width))
        y_px = int(round(y_value * height))
        radius = 5
        draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            outline=(255, 0, 0),
            width=2,
        )
    save_test_output(image, test_id, suffix=suffix)


def test_direct_invocation():
    log_step("Point direct invocation")
    image_path = ensure_blank_image()

    tool = Point_Tool()
    result = tool.execute(
        image_path=image_path,
        points=[[0.25, 0.75]],
        coord_type="normalized",
    )
    log_output("Result", result)
    assert result["points"] == [[0.25, 0.75]]
    save_points_preview(
        image_path,
        result["points"],
        build_test_id(__file__, "test_direct_invocation"),
        suffix="normalized",
    )

    result_text = tool.execute(
        image_path=image_path,
        target="x=20 y=40",
    )
    log_output("Parsed Result", result_text)
    assert result_text["points"], "Expected points parsed from target text"
    save_points_preview(
        image_path,
        result_text["points"],
        build_test_id(__file__, "test_direct_invocation"),
        suffix="parsed",
    )


def test_client_invocation():
    log_step("Point client invocation via server")
    image_path = ensure_blank_image()
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "Point_Tool",
        {
            "image_path": image_path,
            "points": [[50, 100]],
            "coord_type": "pixel",
        },
    )
    log_output("Response", response)
    points = response["result"]["points"]
    assert points and len(points[0]) == 2
    save_points_preview(
        image_path,
        points,
        build_test_id(__file__, "test_client_invocation"),
    )


def test_llm_invocation():
    log_step("Point LLM-assisted invocation via server")
    image_path = ensure_blank_image()

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
    metadata = tools["Point_Tool"]
    log_output("Point_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "Point_Tool",
        metadata,
        "Return the point described as x=30 y=70 in the target string.",
        image_path,
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("target", "x=30 y=70")

    response = client.execute("Point_Tool", args)
    log_output("Response", response)
    points = response["result"]["points"]
    assert points and len(points[0]) == 2
    save_points_preview(
        image_path,
        points,
        build_test_id(__file__, "test_llm_invocation"),
    )


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["Point_Tool"]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("Point tool test completed.")


if __name__ == "__main__":
    main()
