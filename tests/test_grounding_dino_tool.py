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
from mini_vlm_toolkit.tools.grounding_dino.tool import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    GroundingDINO_Tool,
    TOOL_NAME,
)
from tests.utils import (
    build_test_id,
    ensure_grounding_dino_server,
    ensure_test_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
GROUNDING_DINO_SERVER_URL = os.getenv(
    "GROUNDING_DINO_SERVER_URL", "http://127.0.0.1:8084"
)


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


def annotate_boxes(image_path, boxes):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        if max(x1, y1, x2, y2) <= 1.5:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image


def test_direct_invocation():
    log_step("GroundingDINO direct invocation")
    if not ensure_grounding_dino_server(GROUNDING_DINO_SERVER_URL):
        return
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping GroundingDINO direct test.")
        return

    query = os.getenv("GROUNDING_DINO_TEST_QUERY", "person")
    tool = GroundingDINO_Tool()
    result = tool.execute(
        image_path=image_path,
        query=query,
        box_threshold=DEFAULT_BOX_THRESHOLD,
        text_threshold=DEFAULT_TEXT_THRESHOLD,
        server_url=GROUNDING_DINO_SERVER_URL,
    )
    log_output("Result", result)

    assert "boxes" in result, "Expected boxes in GroundingDINO response"
    assert "scores" in result, "Expected scores in GroundingDINO response"
    assert "labels" in result, "Expected labels in GroundingDINO response"
    assert "size" in result and len(result["size"]) == 2, "Expected size in response"

    annotated = annotate_boxes(image_path, result.get("boxes", []))
    save_test_output(annotated, build_test_id(__file__, "test_direct_invocation"))


def test_client_invocation():
    log_step("GroundingDINO client invocation via server")
    if not ensure_grounding_dino_server(GROUNDING_DINO_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping GroundingDINO client test.")
        return

    query = os.getenv("GROUNDING_DINO_TEST_QUERY", "person")
    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {
            "image_path": image_path,
            "query": query,
            "box_threshold": DEFAULT_BOX_THRESHOLD,
            "text_threshold": DEFAULT_TEXT_THRESHOLD,
            "server_url": GROUNDING_DINO_SERVER_URL,
        },
    )
    log_output("Response", response)
    result = response["result"]
    assert "boxes" in result, "Expected boxes in GroundingDINO response"
    annotated = annotate_boxes(image_path, result.get("boxes", []))
    save_test_output(annotated, build_test_id(__file__, "test_client_invocation"))


def test_llm_invocation_python():
    log_step("GroundingDINO LLM-assisted invocation via server (python)")
    if not ensure_grounding_dino_server(GROUNDING_DINO_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping GroundingDINO LLM test.")
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
    log_output("GroundingDINO metadata", metadata)

    query = os.getenv("GROUNDING_DINO_TEST_QUERY", "person")
    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Detect the main subject in the image using GroundingDINO.",
        image_path,
        call_format="python",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.6)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("query", query)
    args.setdefault("box_threshold", DEFAULT_BOX_THRESHOLD)
    args.setdefault("text_threshold", DEFAULT_TEXT_THRESHOLD)
    args.setdefault("server_url", GROUNDING_DINO_SERVER_URL)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert "boxes" in result, "Expected boxes in GroundingDINO response"
    annotated = annotate_boxes(image_path, result.get("boxes", []))
    save_test_output(annotated, build_test_id(__file__, "test_llm_invocation_python"))


def test_llm_invocation_json():
    log_step("GroundingDINO LLM-assisted invocation via server (json)")
    if not ensure_grounding_dino_server(GROUNDING_DINO_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_test_image()
    if not image_path:
        print("Test image unavailable; skipping GroundingDINO LLM test.")
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
    log_output("GroundingDINO metadata", metadata)

    query = os.getenv("GROUNDING_DINO_TEST_QUERY", "person")
    prompt = build_tool_call_prompt(
        TOOL_NAME,
        metadata,
        "Detect the main subject in the image using GroundingDINO.",
        image_path,
        call_format="json",
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.6)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    args.setdefault("image_path", image_path)
    args.setdefault("query", query)
    args.setdefault("box_threshold", DEFAULT_BOX_THRESHOLD)
    args.setdefault("text_threshold", DEFAULT_TEXT_THRESHOLD)
    args.setdefault("server_url", GROUNDING_DINO_SERVER_URL)

    response = client.execute(TOOL_NAME, args)
    log_output("Response", response)
    result = response["result"]
    assert "boxes" in result, "Expected boxes in GroundingDINO response"
    annotated = annotate_boxes(image_path, result.get("boxes", []))
    save_test_output(annotated, build_test_id(__file__, "test_llm_invocation_json"))


def main():
    if not ensure_grounding_dino_server(GROUNDING_DINO_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("GroundingDINO tool tests completed.")


if __name__ == "__main__":
    main()
