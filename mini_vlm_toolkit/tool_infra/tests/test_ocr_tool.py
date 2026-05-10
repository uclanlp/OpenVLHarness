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
from tools.ocr.tool import OCR_Tool, TOOL_NAME
from tests.utils import (
    encode_image_base64,
    ensure_ocr_server,
    ensure_ocr_test_image,
    ensure_tool_server,
    run_json_error_server,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
OCR_SERVER_URL = os.getenv("OCR_SERVER_URL", "http://127.0.0.1:8085")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}")


def assert_ocr_result(result):
    assert isinstance(result, dict), f"expected dict, got {type(result)}"
    text = result.get("text")
    assert isinstance(text, str), f"expected text:str, got {type(text)}"
    assert text.strip(), "expected non-empty transcribed text"


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
        server_url=OCR_SERVER_URL,
    )
    log_output("Result", result)
    assert_ocr_result(result)


def test_subserver_error_propagation():
    log_step("OCR subserver error propagation")
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping error test.")
        return

    tool = OCR_Tool()
    payload = {"detail": "ocr dummy error"}
    with run_json_error_server(payload=payload) as server_url:
        try:
            tool.execute(
                image_path=image_path,
                server_url=server_url,
                timeout_seconds=5,
            )
        except Exception as exc:
            message = str(exc)
            assert "ocr dummy error" in message
            assert "400" in message
            return
    raise AssertionError("Expected OCR error but request succeeded")


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
            "server_url": OCR_SERVER_URL,
        },
    )
    log_output("Response", response)
    assert_ocr_result(response["result"])


def test_direct_invocation_base64():
    log_step("OCR direct invocation (base64)")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR base64 direct test.")
        return

    tool = OCR_Tool()
    result = tool.execute(
        image_base64=encode_image_base64(image_path),
        server_url=OCR_SERVER_URL,
    )
    log_output("Result", result)
    assert_ocr_result(result)


def test_client_invocation_base64():
    log_step("OCR client invocation via server (base64)")
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    image_path = ensure_ocr_test_image()
    if not image_path:
        print("OCR test image unavailable; skipping OCR base64 client test.")
        return

    client = ToolClient(SERVER_URL)
    response = client.execute(
        TOOL_NAME,
        {
            "image_base64": encode_image_base64(image_path),
            "server_url": OCR_SERVER_URL,
        },
    )
    log_output("Response", response)
    assert_ocr_result(response["result"])

def main():
    test_subserver_error_propagation()
    if not ensure_ocr_server(OCR_SERVER_URL):
        return
    if not ensure_tool_server(SERVER_URL, required_tools=[TOOL_NAME]):
        return
    test_direct_invocation()
    test_client_invocation()
    test_direct_invocation_base64()
    test_client_invocation_base64()
    print("OCR tool tests completed.")


if __name__ == "__main__":
    main()
