"""Produce metric depth maps and visualizations through the DA3 depth service.

The Depth_Estimation_Tool client uploads one image and returns encoded float32
depth values, camera metadata and a colorized PNG visualization.
"""
import mimetypes
import os

import requests

from openvlharness.services.tools.base import BaseTool
from openvlharness.services.tools.http_utils import raise_for_status_with_detail
from openvlharness.services.tools.image_utils import decode_base64_image_data

TOOL_NAME = "Depth_Estimation_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8087"
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
            "description": "Output colormap for visualization",
            "enum": ["inferno", "grayscale"],
        },
    },
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class Depth_Estimation_Tool(BaseTool):
    """DA3 depth estimation through the depth service."""
    def __init__(self):
        """Initialize tool metadata and its concrete input/output contract."""
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Estimate a monocular depth map from an image. Returns both raw float32 depth"
                "values (base64-encoded) and a colorized visualization. "
                "Use this tool for relative near/far reasoning, depth "
                "comparison between regions, or depth visualization. "
                # "Do not treat the output as calibrated metric depth "
                # "without external camera calibration."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "colormap": "str - optional output colormap (inferno or grayscale)",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - depth_meters_b64 (base64-encoded float32 array of metric "
                "depth in meters), depth_shape [H, W], "
                "depth_map_base64 (colorized PNG), colormap, model_id, device, size"
            ),
            demo_commands=[
                {
                    "command": (
                        'Depth_Estimation_Tool(image_path="/path/to/image.jpg")'
                    ),
                    "description": "Generate a depth map with raw depth values.",
                }
            ],
            user_metadata={
                "source": "DepthAnything3",
                "status": "implemented",
                "server_env": "DEPTH_ESTIMATION_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": "Wrapper around a Depth Anything 3 model server.",
            },
        )

    def sanity_check(self):
        """Probe the configured health endpoint and return issue strings, or an empty list."""
        server_url = os.getenv(
            "DEPTH_ESTIMATION_SERVER_URL", DEFAULT_SERVER_URL
        ).rstrip("/")
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"Depth estimation server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [
                f"Depth estimation server unhealthy at {health_url}: "
                f"{response.status_code}"
            ]
        return []

    def execute(
        self,
        image_path=None,
        colormap=None,
        server_url=None,
        timeout_seconds=120,
        image_base64=None,
    ):
        """Upload an image and return the depth service's JSON result.

        Args:
            image_path: Local image path, mutually exclusive with image_base64.
            colormap: Optional inferno or grayscale visualization selection.
            server_url: Service base; defaults to DEPTH_ESTIMATION_SERVER_URL or the
                built-in base.
            timeout_seconds: Per-request HTTP timeout.
            image_base64: Raw base64 or an image data URL.

        Returns:
            Depth metadata, a PNG visualization and float32 metric-depth bytes encoded
            as base64. depth_shape and size are [height, width] for the DA3 backend.

        Raises:
            ValueError: Image sources are missing, ambiguous or invalid.
            FileNotFoundError: A supplied path is not a file.
            RuntimeError: The depth service returns an HTTP error.
        """
        if not image_path and not image_base64:
            raise ValueError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ValueError("Provide exactly one of image_path or image_base64")
        if image_path and not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url
            or os.getenv("DEPTH_ESTIMATION_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/depth"

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
                raise_for_status_with_detail(
                    response, prefix="Depth estimation server error"
                )
                return response.json()

        image_bytes, content_type = decode_base64_image_data(image_base64)
        files = {
            "image": ("image", image_bytes, content_type or "application/octet-stream"),
        }
        response = requests.post(
            endpoint, files=files, data=data, timeout=timeout_seconds
        )
        raise_for_status_with_detail(
            response, prefix="Depth estimation server error"
        )
        return response.json()
