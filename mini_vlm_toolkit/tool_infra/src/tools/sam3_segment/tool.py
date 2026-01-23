import mimetypes
import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail


TOOL_NAME = "SAM3_Segment_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8083"


class SAM3_Segment_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Call an external SAM3 segmentation server with a prompt and image."
            ),
            tool_version="0.1.0",
            input_types={
                "prompt": "str - text prompt to segment",
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "return_masks_png_b64": "bool - return PNG masks as base64",
                "return_all_masks_viz": "bool - return all masks visualization as base64",
                "max_masks": "int - optional limit on mask count",
            },
            output_type="dict - SAM3 server response (with optional visualization)",
            demo_commands=[
                {
                    "command": (
                        "SAM3_Segment_Tool(prompt=\"cow\", "
                        "image_path=\"/path/to/cow.jpg\")"
                    ),
                    "description": "Segment a cow image using the external SAM3 server.",
                }
            ],
            user_metadata={
                "server_env": "SAM3_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
            },
        )

    def sanity_check(self):
        server_url = os.getenv("SAM3_SERVER_URL", DEFAULT_SERVER_URL).rstrip("/")
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"SAM3 server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [f"SAM3 server unhealthy at {health_url}: {response.status_code}"]
        return []

    def execute(
        self,
        prompt,
        image_path=None,
        image_base64=None,
        return_masks_png_b64=True,
        return_all_masks_viz=True,
        max_masks=None,
        server_url=None,
        timeout_seconds=300,
    ):
        if not prompt:
            raise ValueError("prompt is required")
        if not image_path and not image_base64:
            raise ValueError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ValueError("Provide exactly one of image_path or image_base64")
        if image_path and not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("SAM3_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/segment"

        request_masks = bool(return_masks_png_b64) or bool(return_all_masks_viz)
        data = {
            "prompt": prompt,
            "return_masks_png_b64": str(request_masks).lower(),
        }
        if max_masks is not None:
            data["max_masks"] = str(max_masks)

        if image_path:
            content_type = (
                mimetypes.guess_type(image_path)[0] or "application/octet-stream"
            )
            with open(image_path, "rb") as handle:
                files = {
                    "image": (os.path.basename(image_path), handle, content_type),
                }
                response = requests.post(
                    endpoint, files=files, data=data, timeout=timeout_seconds
                )
                raise_for_status_with_detail(response, prefix="SAM3 server error")
                result = response.json()
        else:
            from tools.image_utils import decode_base64_image_data

            image_bytes, content_type = decode_base64_image_data(image_base64)
            files = {
                "image": (
                    "image",
                    image_bytes,
                    content_type or "application/octet-stream",
                ),
            }
            response = requests.post(
                endpoint, files=files, data=data, timeout=timeout_seconds
            )
            raise_for_status_with_detail(response, prefix="SAM3 server error")
            result = response.json()

        if "error" in result:
            return result

        if return_all_masks_viz:
            from tools.image_utils import (
                load_image,
                load_image_base64,
                pil_to_base64,
            )
            from tools.sam3_visualize.tool import (
                SAM3_VisualizeAllMasks_Tool,
            )

            if result.get("masks_png_b64"):
                viz_tool = SAM3_VisualizeAllMasks_Tool()
                viz_args = {
                    "predictions": result,
                    "predictions_format": "sam3_server",
                }
                if image_base64:
                    viz_args["image_base64"] = image_base64
                else:
                    viz_args["image_path"] = image_path
                viz = viz_tool.execute(**viz_args)
                result["all_masks_viz_base64"] = viz["image_base64"]
            else:
                image = (
                    load_image_base64(image_base64)
                    if image_base64
                    else load_image(image_path)
                )
                result["all_masks_viz_base64"] = pil_to_base64(image)

        if not return_masks_png_b64 and "masks_png_b64" in result:
            result.pop("masks_png_b64", None)

        return result
