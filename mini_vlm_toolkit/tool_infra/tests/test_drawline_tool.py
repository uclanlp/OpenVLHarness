import json
import os

from client.client import ToolClient
from client.json_tool_call_parser import parse_json_tool_call_args
from client.python_tool_call_parser import parse_python_tool_call_args
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.drawline.tool import DrawLine_Tool
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
    log_step("DrawLine direct invocation")
    image_path = ensure_blank_image()

    tool = DrawLine_Tool()
    result = tool.execute(
        image_path=image_path,
        box=[10, 10, 90, 90],
        corner_a=1,
        corner_b=3,
    )
    log_output("Result", result)

    edited = decode_base64_image(result["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_direct_invocation"))
    pixel = edited.getpixel((12, 12))
    assert pixel != (255, 255, 255)


def test_client_invocation():
    log_step("DrawLine client invocation via server")
    image_path = ensure_blank_image()
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "DrawLine_Tool",
        {
            "image_path": image_path,
            "box": [20, 20, 80, 80],
            "corner_a": 2,
            "corner_b": 4,
        },
    )
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_client_invocation"))
    pixel = edited.getpixel((78, 22))
    assert pixel != (255, 255, 255)


def test_direct_invocation_base64():
    log_step("DrawLine direct invocation (base64)")
    image_path = ensure_blank_image()
    image_base64 = encode_image_base64(image_path)

    tool = DrawLine_Tool()
    result = tool.execute(
        image_base64=image_base64,
        box=[10, 10, 90, 90],
        corner_a=1,
        corner_b=3,
    )
    log_output("Result", result)

    edited = decode_base64_image(result["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_direct_invocation_base64"))
    pixel = edited.getpixel((12, 12))
    assert pixel != (255, 255, 255)


def test_client_invocation_base64():
    log_step("DrawLine client invocation via server (base64)")
    image_path = ensure_blank_image()
    image_base64 = encode_image_base64(image_path)
    client = ToolClient(SERVER_URL)
    response = client.execute(
        "DrawLine_Tool",
        {
            "image_base64": image_base64,
            "box": [20, 20, 80, 80],
            "corner_a": 2,
            "corner_b": 4,
        },
    )
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_client_invocation_base64"))
    pixel = edited.getpixel((78, 22))
    assert pixel != (255, 255, 255)


def test_llm_invocation_python():
    log_step("DrawLine LLM-assisted invocation via server (python)")
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
    metadata = tools["DrawLine_Tool"]
    log_output("DrawLine_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "DrawLine_Tool",
        metadata,
        "Draw a dashed line from corner 1 to corner 3 of box [15, 15, 85, 85].",
        image_path,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("box", [15, 15, 85, 85])
    args.setdefault("corner_a", 1)
    args.setdefault("corner_b", 3)

    response = client.execute("DrawLine_Tool", args)
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_llm_invocation_python"))
    pixel = edited.getpixel((17, 17))
    assert pixel != (255, 255, 255)


def test_llm_invocation_json():
    log_step("DrawLine LLM-assisted invocation via server (json)")
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
    metadata = tools["DrawLine_Tool"]
    log_output("DrawLine_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "DrawLine_Tool",
        metadata,
        "Draw a dashed line from corner 1 to corner 3 of box [15, 15, 85, 85].",
        image_path,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=2048, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("box", [15, 15, 85, 85])
    args.setdefault("corner_a", 1)
    args.setdefault("corner_b", 3)

    response = client.execute("DrawLine_Tool", args)
    log_output("Response", response)
    edited = decode_base64_image(response["result"]["edited_image_base64"]).convert("RGB")
    save_test_output(edited, build_test_id(__file__, "test_llm_invocation_json"))
    pixel = edited.getpixel((17, 17))
    assert pixel != (255, 255, 255)


def main():
    if not ensure_tool_server(SERVER_URL, required_tools=["DrawLine_Tool"]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_direct_invocation_base64()
    test_client_invocation_base64()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("Draw line tool test completed.")


if __name__ == "__main__":
    main()
