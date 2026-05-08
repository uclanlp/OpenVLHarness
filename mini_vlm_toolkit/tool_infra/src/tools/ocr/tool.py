import base64
import io
import os

import requests
from PIL import ImageOps

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail
from tools.image_utils import load_image, load_image_base64, pil_to_base64

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
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Extract readable text from an image using PaddleOCR-VL "
                "(a vision-language OCR model served by vLLM). Returns the "
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
        server_url=None,
        timeout_seconds=600,
        image_base64=None,
    ):
        if not image_path and not image_base64:
            raise ValueError("image_path or image_base64 is required")
        if image_path and image_base64:
            raise ValueError("Provide exactly one of image_path or image_base64")

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
        max_new_tokens = int(os.getenv("OCR_MAX_NEW_TOKENS", "2048"))
        repetition_penalty = float(os.getenv("OCR_REPETITION_PENALTY", "1.05"))

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
