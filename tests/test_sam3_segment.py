import base64
import json
import os
import urllib.request

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.sam3_segment.tool import SAM3_Segment_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_sam3_server,
    ensure_tool_server,
    save_test_output,
)


SAM3_SERVER_URL = os.getenv("SAM3_SERVER_URL", "http://127.0.0.1:8083")
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
DEFAULT_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/test_image.jpg"
)


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def ensure_test_image():
    image_path = os.getenv("SAM3_TEST_IMAGE")
    if image_path:
        return image_path

    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, "sam3_demo.jpg")
    if os.path.isfile(image_path):
        return image_path

    image_url = os.getenv("SAM3_TEST_IMAGE_URL", DEFAULT_TEST_IMAGE_URL)
    log_step("Downloading SAM3 test image")
    print(f"Fetching {image_url}")
    try:
        urllib.request.urlretrieve(image_url, image_path)
    except Exception as exc:
        print(f"Failed to download SAM3 test image: {exc}")
        return None
    return image_path


def build_tool_call_prompt(tool_name, tool_metadata, user_request, image_path):
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: execution = tool.execute(...).\n"
        "Use keyword arguments only.\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"Image path to use: {image_path}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("SAM3 direct invocation")
    image_path = ensure_test_image()
    server_url = SAM3_SERVER_URL
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    tool = SAM3_Segment_Tool()
    result = tool.execute(
        prompt="basketball hoop",
        image_path=image_path,
        return_masks_png_b64=True,
        max_masks=3,
        server_url=server_url,
    )
    log_output("Result", result)

    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"

    # Reuse the notebook mask decoding snippet to validate base64 output.
    if result.get("masks_png_b64"):
        mask_png = base64.b64decode(result["masks_png_b64"][0])
        assert mask_png, "Expected non-empty decoded mask bytes"
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_direct_invocation"),
                suffix=f"mask_{idx}",
            )


def test_client_invocation():
    log_step("SAM3 client invocation via server")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 client test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    client = ToolClient(TOOL_SERVER_URL)
    response = client.execute(
        "SAM3_Segment_Tool",
        {
            "prompt": "basketball hoop",
            "image_path": image_path,
            "return_masks_png_b64": True,
            "max_masks": 3,
            "server_url": SAM3_SERVER_URL,
        },
    )
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_client_invocation"),
                suffix=f"mask_{idx}",
            )


def test_llm_invocation():
    log_step("SAM3 LLM-assisted invocation via server")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 LLM test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

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

    client = ToolClient(TOOL_SERVER_URL)
    tools = client.list_tools()
    metadata = tools["SAM3_Segment_Tool"]
    log_output("Tools", {"tool_names": sorted(tools.keys())})
    log_output("SAM3_Segment_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "SAM3_Segment_Tool",
        metadata,
        "Segment the basketball hoop in the image and return up to 2 masks.",
        image_path,
    )
    log_output("Prompt", prompt)
    generation_params = {
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    log_output("Generation params", generation_params)
    llm_output = engine.generate(prompt, **generation_params)
    log_output("LLM output", llm_output)
    args = parse_tool_execute_args(llm_output)
    log_output("Parsed args", args)
    if "server_url" not in args:
        args["server_url"] = SAM3_SERVER_URL
    response = client.execute("SAM3_Segment_Tool", args)
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_llm_invocation"),
                suffix=f"mask_{idx}",
            )


def main():
    if not ensure_tool_server(TOOL_SERVER_URL, required_tools=["SAM3_Segment_Tool"]):
        return
    if not ensure_sam3_server(SAM3_SERVER_URL):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("SAM3 segment tool test completed.")


if __name__ == "__main__":
    main()
