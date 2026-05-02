import mimetypes
import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail


TOOL_NAME = "Visual_Grounding_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8086"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A short and simple object name or noun phrase, such as 'bottle', 'green ball', or 'wooden chair'. Avoid relational, comparative, or context-dependent descriptions such as 'the tallest man', 'something to eat', or 'the book on the left'."
            ),
        },
        "image_path": {
            "type": "string",
            "description": "Path to the input image file",
        },
    },
    "required": ["query", "image_path"],
}


class Visual_Grounding_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Locate and highlight objects in an image that match a text query. "
                "Returns an annotated image with the matched regions highlighted "
                "and numbered, along with bounding boxes and confidence scores."
            ),
            tool_version="0.3.0",
            input_types={
                "query": "str - a short and simple noun phrase",
                "image_path": "str - path to the input image file",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - contains rendered_image_b64 (annotated image), pred_boxes, pred_scores, and num_masks",
            demo_commands=[
                {
                    "command": (
                        'Visual_Grounding_Tool(query="cow", '
                        'image_path="/path/to/cow.jpg")'
                    ),
                    "description": "Find and highlight cows in an image.",
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
        query=None,
        image_path=None,
        image_base64=None,
        prompt=None,
        server_url=None,
        timeout_seconds=300,
    ):
        if query is None:
            query = prompt
        if not query:
            raise ValueError("query is required")
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

        data = {"prompt": query}

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

        return result
