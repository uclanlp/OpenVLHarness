import base64
import json
import os
import urllib.request

from client.client import ToolClient
from tools.sam3_segment_new.tool import SAM3_Segment_New_Tool
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_sam3_server,
    ensure_tool_server,
    run_json_error_server,
    save_test_output,
)


SAM3_NEW_SERVER_URL = os.getenv("SAM3_NEW_SERVER_URL", "http://127.0.0.1:8086")
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8103")
CANONICAL_TOOL = "SAM3_Segment_New_Tool"
LEGACY_TOOL_ALIAS = "SAM3_Segment_Tool"
LEGACY_SNAKE_ALIAS = "sam3_segment"
DEFAULT_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/test_image.jpg"
)


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    # Truncate long base64 strings for readability
    def _truncate(obj):
        if isinstance(obj, dict):
            return {k: _truncate(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_truncate(v) for v in obj]
        if isinstance(obj, str) and len(obj) > 200:
            return obj[:80] + f"...({len(obj)} chars)"
        return obj

    print(f"{label}:\n{json.dumps(_truncate(payload), indent=2, sort_keys=True)}")


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


def test_direct_invocation():
    log_step("SAM3 New direct invocation (image_path)")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    tool = SAM3_Segment_New_Tool()
    result = tool.execute(
        prompt="shoe",
        image_path=image_path,
        server_url=SAM3_NEW_SERVER_URL,
    )
    log_output("Result", result)

    if "error" in result:
        raise AssertionError(f"SAM3 new server error: {result['error']}")

    assert "pred_boxes" in result, "Expected pred_boxes in response"
    assert "pred_scores" in result, "Expected pred_scores in response"
    assert "pred_masks" in result, "Expected pred_masks (RLE) in response"
    assert "rendered_image_b64" in result, "Expected rendered_image_b64 in response"
    assert result.get("num_masks", 0) > 0, "Expected at least one mask"

    # Verify RLE masks are strings (not dicts or binary)
    for i, rle in enumerate(result["pred_masks"]):
        assert isinstance(rle, str), f"pred_masks[{i}] should be an RLE string, got {type(rle)}"

    if result.get("rendered_image_b64"):
        viz_image = decode_base64_image(result["rendered_image_b64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_direct_invocation"),
            suffix="rendered",
        )
        print(f"Saved rendered image ({viz_image.size[0]}x{viz_image.size[1]})")


def test_direct_invocation_base64():
    log_step("SAM3 New direct invocation (image_base64)")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    image_base64 = encode_image_base64(image_path)
    tool = SAM3_Segment_New_Tool()
    result = tool.execute(
        prompt="shoe",
        image_base64=image_base64,
        server_url=SAM3_NEW_SERVER_URL,
    )
    log_output("Result", result)

    if "error" in result:
        raise AssertionError(f"SAM3 new server error: {result['error']}")

    assert "pred_boxes" in result, "Expected pred_boxes in response"
    assert "pred_masks" in result, "Expected pred_masks (RLE) in response"
    assert "rendered_image_b64" in result, "Expected rendered_image_b64 in response"
    assert result.get("num_masks", 0) > 0, "Expected at least one mask"

    if result.get("rendered_image_b64"):
        viz_image = decode_base64_image(result["rendered_image_b64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_direct_invocation_base64"),
            suffix="rendered",
        )


def test_subserver_error_propagation():
    log_step("SAM3 New subserver error propagation")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping error test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    tool = SAM3_Segment_New_Tool()
    payload = {"detail": "sam3 new dummy error"}
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
            assert "sam3 new dummy error" in message
            assert "400" in message
            return
    raise AssertionError("Expected SAM3 error but request succeeded")


def test_client_invocation():
    log_step("SAM3 New client invocation via tool server")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping client test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    client = ToolClient(TOOL_SERVER_URL)
    response = client.execute(
        CANONICAL_TOOL,
        {
            "prompt": "shoe",
            "image_path": image_path,
            "server_url": SAM3_NEW_SERVER_URL,
        },
    )
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 new server error: {result['error']}")

    assert "pred_boxes" in result, "Expected pred_boxes in response"
    assert "pred_masks" in result, "Expected pred_masks (RLE) in response"
    assert "rendered_image_b64" in result, "Expected rendered_image_b64 in response"

    if result.get("rendered_image_b64"):
        viz_image = decode_base64_image(result["rendered_image_b64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_client_invocation"),
            suffix="rendered",
        )


def test_client_invocation_base64():
    log_step("SAM3 New client invocation via tool server (image_base64)")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping client test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    image_base64 = encode_image_base64(image_path)
    client = ToolClient(TOOL_SERVER_URL)
    response = client.execute(
        CANONICAL_TOOL,
        {
            "prompt": "shoe",
            "image_base64": image_base64,
            "server_url": SAM3_NEW_SERVER_URL,
        },
    )
    log_output("Response", response)

    result = response["result"]
    if "error" in result:
        raise AssertionError(f"SAM3 new server error: {result['error']}")

    assert "pred_boxes" in result, "Expected pred_boxes in response"
    assert "pred_masks" in result, "Expected pred_masks (RLE) in response"
    assert "rendered_image_b64" in result, "Expected rendered_image_b64 in response"

    if result.get("rendered_image_b64"):
        viz_image = decode_base64_image(result["rendered_image_b64"])
        save_test_output(
            viz_image,
            build_test_id(__file__, "test_client_invocation_base64"),
            suffix="rendered",
        )


def test_alias_metadata_and_invocation():
    log_step("SAM3 New alias metadata and invocation")
    image_path = ensure_test_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping alias test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    client = ToolClient(TOOL_SERVER_URL)
    tools = client.list_tools()
    for name in (CANONICAL_TOOL, LEGACY_TOOL_ALIAS, LEGACY_SNAKE_ALIAS):
        assert name in tools, f"Expected {name} to be present in /tools"

    assert tools[CANONICAL_TOOL]["is_alias"] is False
    assert tools[CANONICAL_TOOL]["canonical_tool_name"] == CANONICAL_TOOL
    assert LEGACY_TOOL_ALIAS in tools[CANONICAL_TOOL].get("tool_aliases", [])
    assert LEGACY_SNAKE_ALIAS in tools[CANONICAL_TOOL].get("tool_aliases", [])
    assert tools[LEGACY_TOOL_ALIAS]["is_alias"] is True
    assert tools[LEGACY_TOOL_ALIAS]["canonical_tool_name"] == CANONICAL_TOOL
    assert tools[LEGACY_SNAKE_ALIAS]["is_alias"] is True
    assert tools[LEGACY_SNAKE_ALIAS]["canonical_tool_name"] == CANONICAL_TOOL

    for alias in (LEGACY_TOOL_ALIAS, LEGACY_SNAKE_ALIAS):
        response = client.execute(
            alias,
            {
                "prompt": "shoe",
                "image_path": image_path,
                "server_url": SAM3_NEW_SERVER_URL,
            },
        )
        log_output(f"Alias response ({alias})", response)
        assert response["tool_name"] == alias, "execute should echo requested alias name"
        result = response["result"]
        if "error" in result:
            raise AssertionError(f"SAM3 new server error ({alias}): {result['error']}")
        assert "pred_boxes" in result, "Expected pred_boxes in response"
        assert "pred_masks" in result, "Expected pred_masks (RLE) in response"
        assert "rendered_image_b64" in result, "Expected rendered_image_b64 in response"


def main():
    # Error propagation test runs without any server
    test_subserver_error_propagation()

    if not ensure_sam3_server(SAM3_NEW_SERVER_URL):
        print(f"SAM3 new server not available at {SAM3_NEW_SERVER_URL}; skipping live tests.")
        return

    test_direct_invocation()
    test_direct_invocation_base64()

    required_tools = [CANONICAL_TOOL, LEGACY_TOOL_ALIAS, LEGACY_SNAKE_ALIAS]
    if not ensure_tool_server(TOOL_SERVER_URL, required_tools=required_tools):
        print(f"Tool server not available at {TOOL_SERVER_URL}; skipping client tests.")
        return

    test_client_invocation()
    test_client_invocation_base64()
    test_alias_metadata_and_invocation()

    print("\n== All SAM3 Segment New tests passed. ==")


if __name__ == "__main__":
    main()
