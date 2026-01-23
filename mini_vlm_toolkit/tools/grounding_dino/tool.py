import mimetypes
import os

import requests

from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.http_utils import raise_for_status_with_detail
from mini_vlm_toolkit.tools.image_utils import decode_base64_image_data

TOOL_NAME = "GroundingDINO_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8084"
DEFAULT_BOX_THRESHOLD = 0.4
DEFAULT_TEXT_THRESHOLD = 0.3


class GroundingDINO_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Detect objects in an image using a GroundingDINO server and return "
                "bounding boxes."
            ),
            tool_version="0.2.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "query": "str - object names or short phrases to detect (separate with '.')",
                "box_threshold": "float - confidence threshold for boxes",
                "text_threshold": "float - confidence threshold for text",
                "server_url": "str - override GroundingDINO server URL",
                "timeout_seconds": "int - request timeout in seconds",
            },
            output_type=(
                "dict - boxes (pixel xyxy), boxes_normalized, scores, labels, size [H, W]"
            ),
            demo_commands=[
                {
                    "command": (
                        "GroundingDINO_Tool(image_path=\"/path/to/image.jpg\", "
                        "query=\"bus\", box_threshold=0.4, text_threshold=0.3)"
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

    def sanity_check(self):
        server_url = os.getenv("GROUNDING_DINO_SERVER_URL", DEFAULT_SERVER_URL).rstrip(
            "/"
        )
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"GroundingDINO server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [
                f"GroundingDINO server unhealthy at {health_url}: {response.status_code}"
            ]
        return []

    def execute(
        self,
        image_path=None,
        query=None,
        box_threshold=DEFAULT_BOX_THRESHOLD,
        text_threshold=DEFAULT_TEXT_THRESHOLD,
        server_url=None,
        timeout_seconds=120,
        image_base64=None,
    ):
        if not query:
            raise ValueError("query is required")
        if not image_path and not image_base64:
            raise ValueError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ValueError("Provide exactly one of image_path or image_base64")
        if image_path and not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("GROUNDING_DINO_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/grounding_dino"

        content_type = None
        data = {
            "query": query,
            "box_threshold": str(box_threshold),
            "text_threshold": str(text_threshold),
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
