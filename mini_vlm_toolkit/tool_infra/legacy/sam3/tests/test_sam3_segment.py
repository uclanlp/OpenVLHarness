import base64
import json
import os
import urllib.request

from client.client import ToolClient
from client.json_tool_call_parser import parse_json_tool_call_args
from client.python_tool_call_parser import parse_python_tool_call_args
from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.sam3_segment.tool import SAM3_Segment_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_sam3_server,
    ensure_tool_server,
    run_json_error_server,
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


def encode_image_base64(image_path):
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


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
        prompt="shoe",
        image_path=image_path,
        return_masks_png_b64=True,
        server_url=server_url,
    )
    log_output("Result", result)

    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"

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
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_direct_invocation"),
            suffix="all_masks_viz",
        )


def test_direct_invocation_base64():
    log_step("SAM3 direct invocation (image_base64)")
    image_path = ensure_test_image()
    server_url = SAM3_SERVER_URL
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    image_base64 = encode_image_base64(image_path)
    tool = SAM3_Segment_Tool()
    result = tool.execute(
        prompt="shoe",
        image_base64=image_base64,
        return_masks_png_b64=True,
        server_url=server_url,
    )
    log_output("Result", result)

    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"

    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_direct_invocation_base64"),
                suffix=f"mask_{idx}",
            )
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_direct_invocation_base64"),
            suffix="all_masks_viz",
        )


def test_subserver_error_propagation():
    log_step("SAM3 subserver error propagation")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping error test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    tool = SAM3_Segment_Tool()
    payload = {"detail": "sam3 dummy error"}
    with run_json_error_server(payload=payload) as server_url:
        try:
            tool.execute(
                prompt="shoe",
                image_path=image_path,
                server_url=server_url,
                timeout_seconds=5,
            )
        except Exception as exc:
            message = str(exc)
            assert "sam3 dummy error" in message
            assert "400" in message
            return
    raise AssertionError("Expected SAM3 error but request succeeded")


def test_direct_invocation_viz_flags():
    log_step("SAM3 direct invocation (visualization flags)")
    image_path = ensure_test_image()
    server_url = SAM3_SERVER_URL
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    tool = SAM3_Segment_Tool()
    result_keep_viz = tool.execute(
        prompt="shoe",
        image_path=image_path,
        return_masks_png_b64=False,
        return_all_masks_viz=True,
        server_url=server_url,
    )
    log_output("Result (keep viz)", result_keep_viz)

    if "error" in result_keep_viz:
        raise AssertionError(f"SAM3 server error: {result_keep_viz['error']}")

    assert (
        "all_masks_viz_base64" in result_keep_viz
    ), "Expected visualization when return_all_masks_viz is True"
    assert (
        "masks_png_b64" not in result_keep_viz
    ), "Did not expect masks_png_b64 when return_masks_png_b64 is False"

    result_no_viz = tool.execute(
        prompt="shoe",
        image_path=image_path,
        return_masks_png_b64=True,
        return_all_masks_viz=False,
        server_url=server_url,
    )
    log_output("Result (no viz)", result_no_viz)

    if "error" in result_no_viz:
        raise AssertionError(f"SAM3 server error: {result_no_viz['error']}")

    assert (
        "all_masks_viz_base64" not in result_no_viz
    ), "Did not expect visualization when return_all_masks_viz is False"


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
            "prompt": "shoe",
            "image_path": image_path,
            "return_masks_png_b64": True,
            "server_url": SAM3_SERVER_URL,
        },
    )
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_client_invocation"),
                suffix=f"mask_{idx}",
            )
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_client_invocation"),
            suffix="all_masks_viz",
        )


def test_client_invocation_base64():
    log_step("SAM3 client invocation via server (image_base64)")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 client test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    image_base64 = encode_image_base64(image_path)
    client = ToolClient(TOOL_SERVER_URL)
    response = client.execute(
        "SAM3_Segment_Tool",
        {
            "prompt": "shoe",
            "image_base64": image_base64,
            "return_masks_png_b64": True,
            "server_url": SAM3_SERVER_URL,
        },
    )
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_client_invocation_base64"),
                suffix=f"mask_{idx}",
            )
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_client_invocation_base64"),
            suffix="all_masks_viz",
        )


def test_llm_invocation_python():
    log_step("SAM3 LLM-assisted invocation via server (python)")
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
        "Segment the shoe in the image.",
        image_path,
        call_format="python",
    )
    log_output("Prompt", prompt)
    generation_params = {
        "max_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    log_output("Generation params", generation_params)
    llm_output = engine.generate(prompt, **generation_params)
    log_output("LLM output", llm_output)
    args = parse_python_tool_call_args(llm_output)
    log_output("Parsed args", args)
    args.setdefault("return_all_masks_viz", True)
    if "server_url" not in args:
        args["server_url"] = SAM3_SERVER_URL
    response = client.execute("SAM3_Segment_Tool", args)
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_llm_invocation_python"),
                suffix=f"mask_{idx}",
            )
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_llm_invocation_python"),
            suffix="all_masks_viz",
        )


def test_llm_invocation_json():
    log_step("SAM3 LLM-assisted invocation via server (json)")
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
        "Segment the shoe in the image.",
        image_path,
        call_format="json",
    )
    log_output("Prompt", prompt)
    generation_params = {
        "max_tokens": 16384,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    log_output("Generation params", generation_params)
    llm_output = engine.generate(prompt, **generation_params)
    log_output("LLM output", llm_output)
    args = parse_json_tool_call_args(llm_output)
    log_output("Parsed args", args)
    args.setdefault("return_all_masks_viz", True)
    if "server_url" not in args:
        args["server_url"] = SAM3_SERVER_URL
    response = client.execute("SAM3_Segment_Tool", args)
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 server error: {result['error']}")

    assert "boxes" in result, "Expected boxes in SAM3 response"
    assert "scores" in result, "Expected scores in SAM3 response"
    assert "all_masks_viz_base64" in result, "Expected all-masks visualization in response"
    if result.get("masks_png_b64"):
        for idx, mask_b64 in enumerate(result["masks_png_b64"]):
            mask_image = decode_base64_image(mask_b64)
            save_test_output(
                mask_image,
                build_test_id(__file__, "test_llm_invocation_json"),
                suffix=f"mask_{idx}",
            )
    if result.get("all_masks_viz_base64"):
        viz_image = decode_base64_image(result["all_masks_viz_base64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_llm_invocation_json"),
            suffix="all_masks_viz",
        )


def main():
    test_subserver_error_propagation()
    if not ensure_tool_server(TOOL_SERVER_URL, required_tools=["SAM3_Segment_Tool"]):
        return
    if not ensure_sam3_server(SAM3_SERVER_URL):
        return
    test_direct_invocation()
    test_direct_invocation_base64()
    test_direct_invocation_viz_flags()
    test_client_invocation()
    test_client_invocation_base64()
    test_llm_invocation_python()
    test_llm_invocation_json()
    print("SAM3 segment tool test completed.")


if __name__ == "__main__":
    main()
