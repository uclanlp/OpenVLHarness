import json
import mimetypes
import os

import requests

from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "OCR_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8085"


class OCR_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Extract text strings from an image using an OCR server.",
            tool_version="0.2.0",
            input_types={
                "image_path": "str - local path to the image file",
                "languages": "list[str] | str - optional language list, e.g. ['en']",
                "server_url": "str - override OCR server URL",
                "timeout_seconds": "int - request timeout in seconds",
            },
            output_type="dict - texts, results, languages, gpu",
            demo_commands=[
                {
                    "command": (
                        "OCR_Tool(image_path=\"/path/to/scan.png\", "
                        "languages=[\"en\"])"
                    ),
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
        image_path,
        languages=None,
        server_url=None,
        timeout_seconds=120,
    ):
        if not image_path:
            raise ValueError("image_path is required")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("OCR_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/ocr"

        content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        data = {}
        if languages:
            if isinstance(languages, (list, tuple)):
                data["languages"] = json.dumps(list(languages))
            else:
                data["languages"] = str(languages)

        with open(image_path, "rb") as handle:
            files = {
                "image": (os.path.basename(image_path), handle, content_type),
            }
            response = requests.post(
                endpoint, files=files, data=data, timeout=timeout_seconds
            )
            response.raise_for_status()
            return response.json()
