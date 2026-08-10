import os

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail

TOOL_NAME = "Image_Edit_Tool"
# NOTE: 8089 is Splunk's default management port — avoid it for tool servers
# (it collides on any host running Splunk/splunkd). Overridable via
# QWEN_IMAGE_EDIT_SERVER_URL.
DEFAULT_SERVER_URL = "http://127.0.0.1:8090"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "List of input image reference IDs to edit / combine, "
                "drawn from the Environment Update's 'Currently available "
                "image IDs' list (typically ``input_image_N`` or "
                "``tool_generated_image_N``). Multi-image inputs let the "
                "model compose / merge / transfer content across them; "
                "pass a single-element list for simple in-place edits. "
                "The framework resolves IDs to image bytes before "
                "dispatch — do NOT pass raw URLs, paths, or base64 here."
            ),
        },
        "edit_prompt": {
            "type": "string",
            "description": (
                "Plain-language description of the edit. Examples: "
                "'change the background to a sunset beach', "
                "'place the cat from image 1 onto the table in image 2', "
                "'remove the text in the upper-left corner'. Supports "
                "English and Chinese."
            ),
        },
        "negative_prompt": {
            "type": "string",
            "description": (
                "Optional negative prompt — content / style to avoid in "
                "the edited image."
            ),
        },
        "seed": {
            "type": "integer",
            "description": (
                "Random seed for reproducibility. Non-negative integer "
                "fixes the result; -1 (default) means a fresh random seed."
            ),
        },
    },
    "required": ["image_ids", "edit_prompt"],
    "additionalProperties": False,
}


class Image_Edit_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Edit / compose one or more input images according to a "
                "text instruction, using Qwen-Image-Edit-2509 (diffusion). "
                "Output PNG is registered as ``tool_generated_image_N`` and "
                "can be referenced by other tools (e.g. Image_Search_Tool) "
                "on subsequent turns.\n\n"
                "Use this tool when you have one or more existing images "
                "and want to MODIFY them (change the background, swap an "
                "object, merge two images, remove an element, etc.). For "
                "generating an image FROM SCRATCH (no input image), use "
                "Image_Generation_Tool instead. For retrieving real-world "
                "images, use Text_Search_Tool with search_type='images' "
                "or Image_Search_Tool.\n\n"
                "Multi-image input is supported (and is what the model "
                "was trained for): pass two refs to compose / transfer "
                "content across them. Pass a single ref for in-place edit."
            ),
            tool_version="0.1.0",
            input_types={
                "image_ids": "list[str] - input image refs (input_image_N / tool_generated_image_N)",
                "edit_prompt": "str - what to change in plain language",
                "negative_prompt": "str - optional negative prompt",
                "seed": "int - random seed (-1 = random)",
            },
            input_schema=INPUT_SCHEMA,
            output_type=(
                "dict - image_base64 (PNG, framework-registered as "
                "tool_generated_image_N), width, height, seed_used, model_id"
            ),
            demo_commands=[
                {
                    "command": (
                        'Image_Edit_Tool(image_ids=["input_image_1"], '
                        'edit_prompt="change the background to a sunset beach")'
                    ),
                    "description": (
                        "Single-image in-place edit. Keep the foreground, "
                        "replace the background per the instruction."
                    ),
                },
                {
                    "command": (
                        'Image_Edit_Tool(image_ids=["input_image_1", "input_image_2"], '
                        'edit_prompt="place the cat from the first image onto the wooden table in the second")'
                    ),
                    "description": (
                        "Two-image composition. Leverages Qwen-Image-Edit-2509's "
                        "multi-image strength."
                    ),
                },
            ],
            user_metadata={
                "source": "Qwen/Qwen-Image-Edit-2509",
                "status": "implemented",
                "server_env": "QWEN_IMAGE_EDIT_SERVER_URL",
                "default_server_url": DEFAULT_SERVER_URL,
                "notes": (
                    "Wrapper around a Qwen-Image-Edit-2509 diffusion server "
                    "(HuggingFace diffusers, QwenImageEditPlusPipeline)."
                ),
            },
        )

    def sanity_check(self):
        # Probe the same single endpoint execute() uses.
        server_url = os.getenv(
            "QWEN_IMAGE_EDIT_SERVER_URL", DEFAULT_SERVER_URL
        ).rstrip("/")
        health_url = f"{server_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
        except Exception as exc:
            return [f"Qwen-Image-Edit server unreachable at {health_url}: {exc}"]
        if response.status_code != 200:
            return [
                f"Qwen-Image-Edit server unhealthy at {health_url}: "
                f"{response.status_code}"
            ]
        return []

    def execute(
        self,
        image_base64s=None,
        edit_prompt=None,
        negative_prompt=None,
        seed=None,
        server_url=None,
        timeout_seconds=300,
        dump_dir=None,
        # Framework-injected ref marker, kept LAST so it never shifts the
        # positional order of the public parameters: the resolver rewrites the
        # LLM-facing ``image_ids`` -> ``image_base64s`` and preserves the
        # original refs here for provenance (see Camera_Trajectory_Tool,
        # test_spatial_image_resolution). Accept & ignore so execute() doesn't
        # raise "unexpected keyword argument '_image_ids'".
        _image_ids=None,
    ):
        """Edit images.

        The LLM-facing schema declares ``image_ids: List[str]``. The
        framework's ImageModule.before_tool_call (see
        memory/modules/image.py::_resolve_image_ids_in_args) rewrites
        ``image_ids`` → ``image_base64s`` (list of data URLs) before
        dispatch, so this method actually receives ``image_base64s``.
        Manual callers bypassing the framework must pre-resolve the IDs
        themselves.

        ``dump_dir`` is an operator/ops parameter, NOT an agent-facing
        field: it is intentionally absent from INPUT_SCHEMA so the LLM
        never sees it and never decides where files land.
        """
        if not edit_prompt or not isinstance(edit_prompt, str) or not edit_prompt.strip():
            raise ValueError("edit_prompt is required and must be a non-empty string")
        if not image_base64s or not isinstance(image_base64s, list) \
                or not all(isinstance(x, str) and x.strip() for x in image_base64s):
            raise ValueError(
                "image_base64s is required and must be a non-empty list of "
                "non-empty strings. If you called this directly bypassing "
                "the framework, pre-resolve image_ids to data URLs first."
            )

        resolved_server_url = (
            server_url
            or os.getenv("QWEN_IMAGE_EDIT_SERVER_URL", DEFAULT_SERVER_URL)
        ).rstrip("/")
        endpoint = f"{resolved_server_url}/v1/edit"

        payload = {
            "edit_prompt": edit_prompt.strip(),
            "image_base64s": list(image_base64s),
        }
        if negative_prompt:
            payload["negative_prompt"] = str(negative_prompt)
        if seed is not None:
            payload["seed"] = int(seed)
        if dump_dir:
            payload["dump_dir"] = str(dump_dir)

        response = requests.post(
            endpoint, json=payload, timeout=timeout_seconds
        )
        raise_for_status_with_detail(
            response, prefix="Qwen-Image-Edit server error"
        )
        return response.json()
