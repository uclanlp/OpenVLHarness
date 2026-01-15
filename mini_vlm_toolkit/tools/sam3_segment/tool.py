import mimetypes
import os

import requests

from mini_vlm_toolkit.tools.base import BaseTool


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
                "return_masks_png_b64": "bool - return PNG masks as base64",
                "max_masks": "int - optional limit on mask count",
            },
            output_type="dict - SAM3 server response",
            demo_commands=[
                {
                    "command": (
                        "SAM3_Segment_Tool(prompt=\"cow\", "
                        "image_path=\"/path/to/cow.jpg\", max_masks=3)"
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
        image_path,
        return_masks_png_b64=True,
        max_masks=None,
        server_url=None,
        timeout_seconds=300,
    ):
        if not prompt:
            raise ValueError("prompt is required")
        if not image_path:
            raise ValueError("image_path is required")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("SAM3_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/segment"

        content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        data = {
            "prompt": prompt,
            "return_masks_png_b64": str(bool(return_masks_png_b64)).lower(),
        }
        if max_masks is not None:
            data["max_masks"] = str(max_masks)

        with open(image_path, "rb") as handle:
            files = {
                "image": (os.path.basename(image_path), handle, content_type),
            }
            response = requests.post(
                endpoint, files=files, data=data, timeout=timeout_seconds
            )
            response.raise_for_status()
            return response.json()
