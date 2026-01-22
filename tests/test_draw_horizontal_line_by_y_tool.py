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
from mini_vlm_toolkit.tools.draw_horizontal_line_by_y.tool import DrawHorizontalLineByY_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    encode_image_base64,
    ensure_blank_image,
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
    log_step("DrawHorizontalLineByY direct invocation")
    image_path = ensure_blank_image()

    tool = DrawHorizontalLineByY_Tool()
    result = tool.execute(image_path=image_path, y=0.5, coord_type="normalized")
    log_output("Result", result)

    edited = decode_base64_image(result["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_direct_invocation"))
    y_px = int(round(0.5 * edited.size[1]))
    pixel = edited.getpixel((2, y_px))
    assert pixel != (255, 255, 255)


def test_client_invocation():
    log_step("DrawHorizontalLineByY client invocation via server")
    image_path = ensure_blank_image()
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "DrawHorizontalLineByY_Tool",
        {
            "image_path": image_path,
            "y": 30,
            "coord_type": "pixel",
        },
    )
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_client_invocation"))
    pixel = edited.getpixel((5, 30))
    assert pixel != (255, 255, 255)


def test_direct_invocation_base64():
    log_step("DrawHorizontalLineByY direct invocation (base64)")
    image_path = ensure_blank_image()
    image_base64 = encode_image_base64(image_path)

    tool = DrawHorizontalLineByY_Tool()
    result = tool.execute(image_base64=image_base64, y=0.5, coord_type="normalized")
    log_output("Result", result)

    edited = decode_base64_image(result["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_direct_invocation_base64"))
    y_px = int(round(0.5 * edited.size[1]))
    pixel = edited.getpixel((2, y_px))
    assert pixel != (255, 255, 255)


def test_client_invocation_base64():
    log_step("DrawHorizontalLineByY client invocation via server (base64)")
    image_path = ensure_blank_image()
    image_base64 = encode_image_base64(image_path)
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "DrawHorizontalLineByY_Tool",
        {
            "image_base64": image_base64,
            "y": 30,
            "coord_type": "pixel",
        },
    )
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_client_invocation_base64"))
    pixel = edited.getpixel((5, 30))
    assert pixel != (255, 255, 255)


def test_llm_invocation_python():
    log_step("DrawHorizontalLineByY LLM-assisted invocation via server (python)")
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
    metadata = tools["DrawHorizontalLineByY_Tool"]
    log_output("DrawHorizontalLineByY_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "DrawHorizontalLineByY_Tool",
        metadata,
        "Draw a horizontal line at y=0.25 (normalized).",
        image_path,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("y", 0.25)
    args.setdefault("coord_type", "normalized")

    response = client.execute("DrawHorizontalLineByY_Tool", args)
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_llm_invocation_python"))
    y_px = int(round(0.25 * edited.size[1]))
    pixel = edited.getpixel((4, y_px))
    assert pixel != (255, 255, 255)


def test_llm_invocation_json():
    log_step("DrawHorizontalLineByY LLM-assisted invocation via server (json)")
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
    metadata = tools["DrawHorizontalLineByY_Tool"]
    log_output("DrawHorizontalLineByY_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "DrawHorizontalLineByY_Tool",
        metadata,
        "Draw a horizontal line at y=0.25 (normalized).",
        image_path,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("y", 0.25)
    args.setdefault("coord_type", "normalized")

    response = client.execute("DrawHorizontalLineByY_Tool", args)
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_llm_invocation_json"))
    y_px = int(round(0.25 * edited.size[1]))
    pixel = edited.getpixel((4, y_px))
    assert pixel != (255, 255, 255)


def main():
    if not ensure_tool_server(
        SERVER_URL, required_tools=["DrawHorizontalLineByY_Tool"]
    ):
        return
    test_direct_invocation()
    test_client_invocation()
    test_direct_invocation_base64()
    test_client_invocation_base64()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("Draw horizontal line tool test completed.")


if __name__ == "__main__":
    main()
