import mimetypes
import os

import requests

from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.http_utils import raise_for_status_with_detail

TOOL_NAME = "Depth_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8082"


class Depth_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Estimate depth from an image using a Depth-Anything server."
            ),
            tool_version="0.2.0",
            input_types={
                "image_path": "str - local path to the image file",
                "colormap": "str - optional color map (inferno or grayscale)",
                "server_url": "str - override Depth-Anything server URL",
                "timeout_seconds": "int - request timeout in seconds",
            },
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
        image_path,
        colormap=None,
        server_url=None,
        timeout_seconds=120,
    ):
        if not image_path:
            raise ValueError("image_path is required")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        resolved_server_url = (
            server_url or os.getenv("DEPTH_ANYTHING_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/depth"

        content_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        data = {}
        if colormap:
            data["colormap"] = str(colormap)

        with open(image_path, "rb") as handle:
            files = {
                "image": (os.path.basename(image_path), handle, content_type),
            }
            response = requests.post(
                endpoint, files=files, data=data, timeout=timeout_seconds
            )
            raise_for_status_with_detail(response, prefix="Depth server error")
            return response.json()
