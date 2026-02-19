import mimetypes
import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail
from tools.image_utils import decode_base64_image_data

TOOL_NAME = "Depth_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8082"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "colormap": {
            "type": "string",
            "description": "Output colormap",
            "enum": ["inferno", "grayscale"],
        },
    },
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class Depth_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Estimate a depth map from one image using a Depth-Anything server. "
                "Use this tool for relative near/far reasoning or depth visualization. "
                "Do not treat the output as calibrated metric depth without external "
                "camera calibration."
            ),
            tool_version="0.3.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "colormap": "str - optional output colormap (inferno or grayscale)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - depth_map_base64, colormap, model_id, device, size",
            demo_commands=[
                {
                    "command": (
                        "Depth_Tool(image_path=\"/path/to/image.jpg\")"
                    ),
                    "description": "Generate a depth map using the depth server.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "implemented",
                "server_env": "DEPTH_ANYTHING_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": "Wrapper around a Depth-Anything model server.",
            },
        )

    def sanity_check(self):
        server_url = os.getenv("DEPTH_ANYTHING_SERVER_URL", DEFAULT_SERVER_URL).rstrip(
            "/"
        )
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"Depth server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [f"Depth server unhealthy at {health_url}: {response.status_code}"]
        return []

    def execute(
        self,
        image_path=None,
        colormap=None,
        server_url=None,
        timeout_seconds=120,
        image_base64=None,
    ):
        if not image_path and not image_base64:
            raise ValueError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ValueError("Provide exactly one of image_path or image_base64")
        if image_path and not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("DEPTH_ANYTHING_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/depth"

        content_type = None
        data = {}
        if colormap:
            data["colormap"] = str(colormap)

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
                raise_for_status_with_detail(response, prefix="Depth server error")
                return response.json()

        image_bytes, content_type = decode_base64_image_data(image_base64)
        files = {
            "image": ("image", image_bytes, content_type or "application/octet-stream"),
        }
        response = requests.post(
            endpoint, files=files, data=data, timeout=timeout_seconds
        )
        raise_for_status_with_detail(response, prefix="Depth server error")
        return response.json()
