import base64
import io
import json
import os

import numpy as np
from PIL import Image

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.tools.sam3_visualize.tool import (
    SAM3_VisualizeAllMasks_Tool,
    SAM3_VisualizeSingleMask_Tool,
)
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_blank_image,
    ensure_tool_server,
    save_test_output,
)


SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def _mask_to_png_base64(mask):
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_fake_predictions(image_size):
    width, height = image_size
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask1[40:140, 30:120] = 1
    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2[20:110, 90:170] = 1
    return {
        "masks_png_b64": [
            _mask_to_png_base64(mask1),
            _mask_to_png_base64(mask2),
        ],
        "boxes": [
            [30, 40, 120, 140],
            [90, 20, 170, 110],
        ],
    }


def test_direct_invocation():
    log_step("SAM3 visualize direct invocation")
    image_path = ensure_blank_image(filename="sam3_visualize_demo.png", size=(200, 200))
    image = Image.open(image_path)
    predictions = _build_fake_predictions(image.size)

    tool_all = SAM3_VisualizeAllMasks_Tool()
    result_all = tool_all.execute(
        image_path=image_path,
        predictions=predictions,
        predictions_format="sam3_server",
    )
    log_output("Result (all masks)", result_all)
    all_image = decode_base64_image(result_all["image_base64"])
    save_test_output(all_image, build_test_id(__file__, "test_direct_invocation"), suffix="all")

    tool_single = SAM3_VisualizeSingleMask_Tool()
    result_single = tool_single.execute(
        image_path=image_path,
        predictions=predictions,
        predictions_format="sam3_server",
        mask_index=0,
    )
    log_output("Result (single mask)", result_single)
    overlay = decode_base64_image(result_single["overlay_image_base64"])
    zoomed = decode_base64_image(result_single["zoomed_image_base64"])
    save_test_output(overlay, build_test_id(__file__, "test_direct_invocation"), suffix="single")
    save_test_output(zoomed, build_test_id(__file__, "test_direct_invocation"), suffix="zoomed")


def test_client_invocation():
    log_step("SAM3 visualize client invocation via server")
    image_path = ensure_blank_image(filename="sam3_visualize_demo.png", size=(200, 200))
    image = Image.open(image_path)
    predictions = _build_fake_predictions(image.size)

    client = ToolClient(SERVER_URL)
    response_all = client.execute(
        "SAM3_VisualizeAllMasks_Tool",
        {
            "image_path": image_path,
            "predictions": predictions,
            "predictions_format": "sam3_server",
        },
    )
    log_output("Response (all masks)", response_all)
    all_image = decode_base64_image(response_all["result"]["image_base64"])
    save_test_output(all_image, build_test_id(__file__, "test_client_invocation"), suffix="all")

    response_single = client.execute(
        "SAM3_VisualizeSingleMask_Tool",
        {
            "image_path": image_path,
            "predictions": predictions,
            "predictions_format": "sam3_server",
            "mask_index": 1,
        },
    )
    log_output("Response (single mask)", response_single)
    overlay = decode_base64_image(response_single["result"]["overlay_image_base64"])
    zoomed = decode_base64_image(response_single["result"]["zoomed_image_base64"])
    save_test_output(overlay, build_test_id(__file__, "test_client_invocation"), suffix="single")
    save_test_output(zoomed, build_test_id(__file__, "test_client_invocation"), suffix="zoomed")


def main():
    required = ["SAM3_VisualizeAllMasks_Tool", "SAM3_VisualizeSingleMask_Tool"]
    if not ensure_tool_server(SERVER_URL, required_tools=required):
        return
    test_direct_invocation()
    test_client_invocation()
    print("SAM3 visualize tool test completed.")


if __name__ == "__main__":
    main()
