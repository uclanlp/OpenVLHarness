"""Detect objects named by text labels using the Grounding DINO service.

The GroundingDINO_Tool client uploads one image with fixed confidence cutoffs
and returns boxes, scores and labels. This backend does not produce masks.
"""
import mimetypes
import os

import requests

from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.http_utils import raise_for_status_with_detail
from openvlharness.services.tools.image_utils import decode_base64_image_data

TOOL_NAME = "GroundingDINO_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8084"
DEFAULT_BOX_THRESHOLD = 0.4
DEFAULT_TEXT_THRESHOLD = 0.3
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "query": {"type": "string", "description": "Text query for detection"},
    },
    "required": ["query"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class GroundingDINO_Tool(BaseTool):
    """Text-conditioned bounding-box detection through Grounding DINO."""
    def __init__(self):
        """Initialize tool metadata and its concrete input/output contract."""
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Run text-guided object detection and return bounding boxes. Use this "
                "tool when you need to locate objects from text prompts before "
                "zooming/cropping. Do not use it when you need precise pixel masks; "
                "use Visual_Grounding_Tool for segmentation."
            ),
            tool_version="0.3.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "query": "str - object names/phrases to detect (separate multiple phrases with '.')",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - boxes (pixel xyxy), boxes_normalized (0-1000 scale), scores, labels, size [H, W]"
            ),
            demo_commands=[
                {
                    "command": (
                        "GroundingDINO_Tool(image_path=\"/path/to/image.jpg\", "
                        "query=\"bus\")"
                    ),
                    "description": "Detect buses in an image via the GroundingDINO server.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "server_env": "GROUNDING_DINO_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": "Wrapper around the Hugging Face GroundingDINO server.",
            },
        )

    def execute(
        self,
        image_path=None,
        query=None,
        server_url=None,
        timeout_seconds=120,
        image_base64=None,
    ):
        """Detect query labels in one image using fixed confidence thresholds.

        Args:
            image_path: Local image path, exclusive with image_base64.
            query: Nonempty label query sent to the detector.
            server_url: Service base, otherwise resolved from GROUNDING_DINO_SERVER_URL.
            timeout_seconds: Per-request HTTP timeout.
            image_base64: Raw base64 or an image data URL.

        Returns:
            Backend JSON with pixel xyxy boxes, 0–1000 normalized boxes, aligned scores
            and labels, [height, width] size and model provenance.

        Raises:
            ValueError: Query or image inputs are invalid.
            FileNotFoundError: A supplied image file is missing.
            RuntimeError: The detector returns an HTTP error.
        """
        if not query:
            raise ToolArgumentError("query is required")
        if not image_path and not image_base64:
            raise ToolArgumentError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ToolArgumentError("Provide exactly one of image_path or image_base64")
        if image_path and not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("GROUNDING_DINO_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/grounding_dino"

        content_type = None
        data = {
            "query": query,
            "box_threshold": str(DEFAULT_BOX_THRESHOLD),
            "text_threshold": str(DEFAULT_TEXT_THRESHOLD),
        }
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
                    response, prefix="GroundingDINO server error"
                )
                return response.json()

        image_bytes, content_type = decode_base64_image_data(image_base64)
        files = {
            "image": ("image", image_bytes, content_type or "application/octet-stream"),
        }
        response = requests.post(
            endpoint, files=files, data=data, timeout=timeout_seconds
        )
        raise_for_status_with_detail(response, prefix="GroundingDINO server error")
        return response.json()
