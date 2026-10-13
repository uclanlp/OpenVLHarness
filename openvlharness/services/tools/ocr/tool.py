"""Transcribe text from images using a PaddleOCR-VL HTTP endpoint.

The OCR_Tool backend accepts local or inline images, corrects EXIF orientation,
and returns plain transcription text with the served model ID.
"""
import os

import requests
from PIL import ImageOps

from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.http_utils import raise_for_status_with_detail
from openvlharness.services.tools.image_utils import load_image, load_image_base64, pil_to_base64

TOOL_NAME = "OCR_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8085"
DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
PROMPT = "OCR:"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
    },
    "required": [],
}


class OCR_Tool(BaseTool):
    """PaddleOCR-VL transcription through an HTTP model endpoint."""
    def __init__(self):
        """Initialize tool metadata and its concrete input/output contract."""
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Extract readable text from an image. Returns the "
                "transcribed text as a single string. Use this tool when the "
                "task is about words, numbers, or printed labels in the image. "
                "Do not use it for general object localization or segmentation."
            ),
            tool_version="0.5.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - text (str), model_id (str)",
            demo_commands=[
                {
                    "command": 'OCR_Tool(image_path="/path/to/scan.png")',
                    "description": "Transcribe text from an image using the OCR server.",
                }
            ],
            user_metadata={
                "source": DEFAULT_MODEL_ID,
                "status": "implemented",
                "server_env": "OCR_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": "Talks directly to a vLLM-served PaddleOCR-VL endpoint.",
            },
        )

    def execute(
        self,
        image_path=None,
        server_url=None,
        timeout_seconds=600,
        image_base64=None,
    ):
        """Transcribe one image after RGB conversion and EXIF orientation correction.

        Args:
            image_path: Local image path, mutually exclusive with image_base64.
            server_url: OCR service base; defaults to OCR_SERVER_URL or the built-in base.
            timeout_seconds: Per-request HTTP timeout.
            image_base64: Raw base64 or an image data URL.

        Returns:
            A mapping with stripped text and model_id.

        Raises:
            ValueError: Image sources are missing, ambiguous or invalid.
            OSError: A local image cannot be read.
            RuntimeError: The server reports an error or an unexpected response structure.
        """
        if not image_path and not image_base64:
            raise ToolArgumentError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ToolArgumentError("Provide exactly one of image_path or image_base64")

        if image_path:
            pil_image = load_image(image_path)
        else:
            pil_image = load_image_base64(image_base64)
        pil_image = ImageOps.exif_transpose(pil_image)

        data_url = "data:image/png;base64," + pil_to_base64(pil_image, image_format="PNG")

        resolved_server_url = (
            server_url or os.getenv("OCR_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/chat/completions"

        model_id = os.getenv("OCR_MODEL_ID", DEFAULT_MODEL_ID)
        max_new_tokens = 2048
        repetition_penalty = 1.05

        body = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        if repetition_penalty != 1.0:
            body["repetition_penalty"] = repetition_penalty

        response = requests.post(endpoint, json=body, timeout=timeout_seconds)
        raise_for_status_with_detail(response, prefix="OCR server error")

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OCR server returned unexpected response: {data}") from exc

        return {"text": (text or "").strip(), "model_id": model_id}
