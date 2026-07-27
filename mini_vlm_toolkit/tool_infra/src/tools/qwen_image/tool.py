import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail

TOOL_NAME = "Image_Generation_Tool"
DEFAULT_SERVER_URL = "http://127.0.0.1:8088"
# Qwen-Image's recommended aspect ratios + corresponding (width, height) in
# pixels. The server side owns the mapping; tool-side only validates the
# enum label.
ASPECT_RATIO_ENUM = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": (
                "Text description of the image to generate. Supports both "
                "English and Chinese; the model is particularly strong at "
                "rendering legible text inside the image."
            ),
        },
        "negative_prompt": {
            "type": "string",
            "description": (
                "Optional negative prompt — content / style to avoid in the "
                "generated image."
            ),
        },
        "aspect_ratio": {
            "type": "string",
            "description": (
                "Output aspect ratio (label). The server maps each label to "
                "Qwen-Image's recommended pixel size."
            ),
            "enum": ASPECT_RATIO_ENUM,
        },
        "seed": {
            "type": "integer",
            "description": (
                "Random seed for reproducibility. Use a non-negative integer "
                "to fix the result; -1 (default) means a random seed."
            ),
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


class Image_Generation_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Generate a NEW image from a text prompt using Qwen-Image "
                "(diffusion). The output PNG is registered as "
                "``tool_generated_image_N`` and can be referenced by other "
                "tools (e.g. Image_Search_Tool) on subsequent turns.\n\n"
                "Use this tool to synthesize a reference image, visualize a "
                "hypothesis, or produce an illustrative figure when no "
                "existing image matches the need. The model has strong "
                "English+Chinese text rendering. For pure retrieval (find "
                "real-world images of a subject), prefer Text_Search_Tool "
                "with search_type='images' or Image_Search_Tool."
            ),
            tool_version="0.1.0",
            input_types={
                "prompt": "str - text description of the image to generate",
                "negative_prompt": "str - optional negative prompt",
                "aspect_ratio": "str - one of 1:1 / 16:9 / 9:16 / 4:3 / 3:4 / 3:2 / 2:3",
                "seed": "int - random seed (-1 = random)",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - image_base64 (PNG, framework-registered as "
                "tool_generated_image_N), width, height, aspect_ratio, "
                "seed_used, model_id"
            ),
            demo_commands=[
                {
                    "command": (
                        'Image_Generation_Tool(prompt="A cat sitting on a '
                        'wooden table, natural lighting", aspect_ratio="16:9")'
                    ),
                    "description": (
                        "Generate a 16:9 image from a text prompt."
                    ),
                },
                {
                    "command": (
                        'Image_Generation_Tool(prompt="A coffee shop '
                        'chalkboard reading \'Qwen Coffee \\$2 per cup\'", '
                        'aspect_ratio="4:3", seed=42)'
                    ),
                    "description": (
                        "Generate with a specific seed for reproducibility; "
                        "leverages Qwen-Image's strong in-image text rendering."
                    ),
                },
            ],
            user_metadata={
                "source": "Qwen/Qwen-Image",
                "status": "implemented",
                "server_env": "QWEN_IMAGE_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": (
                    "Wrapper around a Qwen-Image diffusion server "
                    "(HuggingFace diffusers, Apache 2.0)."
                ),
            },
        )

    def sanity_check(self):
        # Probe the SAME endpoint pool execute() uses (plural *_SERVER_URLS ->
        # singular -> default). The tool is usable if ANY replica is healthy, so a
        # data-parallel backend (or a remote DP node pool) passes the load-time check.
        urls = resolve_endpoints(
            "QWEN_IMAGE_SERVER_URLS", "QWEN_IMAGE_SERVER_URL", DEFAULT_SERVER_URL
        )
        errors = []
        for base in urls:
            health_url = f"{base.rstrip('/')}/health"
            try:
                response = requests.get(health_url, timeout=5)
            except Exception as exc:
                errors.append(f"{health_url}: {exc}")
                continue
            if response.status_code == 200:
                return []  # at least one replica healthy -> tool usable
            errors.append(f"{health_url}: HTTP {response.status_code}")
        return [f"Qwen-Image server(s) unreachable: {'; '.join(errors)}"]

    def execute(
        self,
        prompt=None,
        negative_prompt=None,
        aspect_ratio=None,
        seed=None,
        server_url=None,
        timeout_seconds=300,
        dump_dir=None,
    ):
        """Generate an image.

        ``dump_dir`` is an operator/ops parameter, NOT an agent-facing
        field: it is intentionally absent from INPUT_SCHEMA so the LLM
        never sees it and never decides where files land. Only callers
        in trusted code paths (manual curl-equivalent, an experiment
        runner, an explicit ops harness) should set it.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required and must be a non-empty string")
        if aspect_ratio is not None and aspect_ratio not in ASPECT_RATIO_ENUM:
            raise ValueError(
                f"aspect_ratio must be one of {ASPECT_RATIO_ENUM}, "
                f"got: {aspect_ratio!r}"
            )

        resolved_server_url = (
            server_url
            or os.getenv("QWEN_IMAGE_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/generate"

        payload = {"prompt": prompt.strip()}
        if negative_prompt:
            payload["negative_prompt"] = str(negative_prompt)
        if aspect_ratio:
            payload["aspect_ratio"] = str(aspect_ratio)
        if seed is not None:
            payload["seed"] = int(seed)
        if dump_dir:
            payload["dump_dir"] = str(dump_dir)

        response = requests.post(
            endpoint, json=payload, timeout=timeout_seconds
        )
        raise_for_status_with_detail(
            response, prefix="Qwen-Image server error"
        )
        return response.json()
