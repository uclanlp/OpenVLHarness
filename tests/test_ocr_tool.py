import json
import os

from PIL import Image, ImageDraw

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.json_tool_call_parser import parse_json_tool_call_args
from mini_vlm_toolkit.client.python_tool_call_parser import parse_python_tool_call_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.ocr.tool import OCR_Tool, TOOL_NAME
from tests.utils import (
    build_test_id,
    ensure_ocr_server,
    ensure_ocr_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
OCR_SERVER_URL = os.getenv("OCR_SERVER_URL", "http://127.0.0.1:8085")
OCR_LANGUAGES = ["ch_sim", "en"]


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


def annotate_ocr_boxes(image_path, results):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in results:
        box = item.get("box") if isinstance(item, dict) else None
        if not box or len(box) < 4:
            continue
        points = [(float(x), float(y)) for x, y in box]
        draw.line(points + [points[0]], fill="red", width=2)
    return image


def test_direct_invocation():
    log_step("OCR direct invocation")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR direct test.")
        return

    tool = OCR_Tool()
    result = tool.execute(
        image_path=image_path,
        languages=OCR_LANGUAGES,
        server_url=OCR_SERVER_URL,
    )
    log_output("Result", result)

    assert "texts" in result and isinstance(result["texts"], list)
    assert "results" in result and isinstance(result["results"], list)
    assert len(result["texts"]) > 0

    annotated = annotate_ocr_boxes(image_path, result.get("results", []))
    save_test_output(annotated, build_test_id(__file__, "test_direct_invocation"))


def test_client_invocation():
    log_step("OCR client invocation via server")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {
            "image_path": image_path,
            "languages": OCR_LANGUAGES,
            "server_url": OCR_SERVER_URL,
        },
    )
    log_output("Response", response)
    result = response["result"]
    assert "texts" in result and isinstance(result["texts"], list)
    annotated = annotate_ocr_boxes(image_path, result.get("results", []))
    save_test_output(annotated, build_test_id(__file__, "test_client_invocation"))


def test_llm_invocation_python():
    log_step("OCR LLM-assisted invocation via server (python)")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR LLM test.")
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
    log_output("OCR metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Extract any visible text from the image.",
        image_path,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args["languages"] = OCR_LANGUAGES
    args.setdefault("server_url", OCR_SERVER_URL)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert "texts" in result and isinstance(result["texts"], list)
    annotated = annotate_ocr_boxes(image_path, result.get("results", []))
    save_test_output(annotated, build_test_id(__file__, "test_llm_invocation_python"))


def test_llm_invocation_json():
    log_step("OCR LLM-assisted invocation via server (json)")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR LLM test.")
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
    log_output("OCR metadata", metadata)

    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Extract any visible text from the image.",
        image_path,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args["languages"] = OCR_LANGUAGES
    args.setdefault("server_url", OCR_SERVER_URL)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert "texts" in result and isinstance(result["texts"], list)
    annotated = annotate_ocr_boxes(image_path, result.get("results", []))
    save_test_output(annotated, build_test_id(__file__, "test_llm_invocation_json"))


def main():
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("OCR tool tests completed.")


if __name__ == "__main__":
    main()
