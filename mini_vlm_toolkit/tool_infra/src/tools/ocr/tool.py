import json
import mimetypes
import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail
from tools.image_utils import decode_base64_image_data

TOOL_NAME = "OCR_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8085"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
    },
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class OCR_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Extract readable text from an image using an OCR server. Use this "
                "tool when the task is about words, numbers, or printed labels in "
                "the image. Do not use it for general object localization or "
                "segmentation."
            ),
            tool_version="0.3.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - texts, results, languages, gpu",
            demo_commands=[
                {
                    "command": 'OCR_Tool(image_path="/path/to/scan.png")',
                    "description": "Extract text from an image using the OCR server.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "server_env": "OCR_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": "Wrapper around a standalone EasyOCR server.",
            },
        )

    def sanity_check(self):
        server_url = os.getenv("OCR_SERVER_URL", DEFAULT_SERVER_URL).rstrip("/")
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"OCR server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [f"OCR server unhealthy at {health_url}: {response.status_code}"]
        return []

    def execute(
        self,
        image_path=None,
        languages=None,
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
            server_url or os.getenv("OCR_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/ocr"

        content_type = None
        data = {}
        if languages:
            if isinstance(languages, (list, tuple)):
                data["languages"] = json.dumps(list(languages))
            else:
                data["languages"] = str(languages)

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
                raise_for_status_with_detail(response, prefix="OCR server error")
                return response.json()

        image_bytes, content_type = decode_base64_image_data(image_base64)
        files = {
            "image": ("image", image_bytes, content_type or "application/octet-stream"),
        }
        response = requests.post(
            endpoint, files=files, data=data, timeout=timeout_seconds
        )
        raise_for_status_with_detail(response, prefix="OCR server error")
        return response.json()
