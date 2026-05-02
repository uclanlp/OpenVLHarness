from PIL import Image

from tools.base import BaseTool
from tools.image_utils import (
    clamp,
    normalized_xywh_to_pixels,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "OverlayImages_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "base_image_path": {"type": "string", "description": "Local base image path"},
        "overlay_image_path": {
            "type": "string",
            "description": "Local overlay image path",
        },
        "alpha": {
            "type": "number",
            "description": "Overlay alpha in [0, 1]",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "bounding_box": {
            "type": "array",
            "description": "Normalized overlay box [x, y, w, h] on the 0-1000 scale",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
    },
    "required": ["base_image_path", "overlay_image_path"],
    "additionalProperties": False,
}


class OverlayImages_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Overlay one image onto a base image with alpha blending inside a "
                "target region. Use this tool for visual composition or annotation "
                "mockups. Do not use it for segmentation, detection, or geometric "
                "alignment beyond simple resize-to-box behavior."
            ),
            tool_version="0.1.0",
            input_types={
                "base_image_path": "str - local path to the base image",
                "overlay_image_path": "str - local path to the overlay image",
                "alpha": "float - overlay transparency (default 0.6)",
                "bounding_box": "list[float] - normalized bbox [x, y, w, h] on the 0-1000 scale",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - blended_image_base64",
            demo_commands=[
                {
                    "command": "OverlayImages_Tool(base_image_path=\"/path/to/base.jpg\", overlay_image_path=\"/path/to/overlay.jpg\", alpha=0.6)",
                    "description": "Overlay images with transparency.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "implemented",
            },
        )

    def execute(
        self,
        base_image_path=None,
        overlay_image_path=None,
        alpha=0.6,
        bounding_box=None,
        base_image_base64=None,
        overlay_image_base64=None,
    ):
        base_image = resolve_image_input(
            image_path=base_image_path,
            image_base64=base_image_base64,
            image_label="base_image",
        )
        overlay_image = resolve_image_input(
            image_path=overlay_image_path,
            image_base64=overlay_image_base64,
            image_label="overlay_image",
        )

        if bounding_box is None:
            bounding_box = [0.0, 0.0, 1000.0, 1000.0]

        width, height = base_image.size
        left, top, right, bottom = normalized_xywh_to_pixels(
            bounding_box, width, height
        )
        box_width = right - left
        box_height = bottom - top
        if box_width <= 0 or box_height <= 0:
            raise ValueError("bounding_box results in empty overlay region")

        overlay_resized = overlay_image.resize(
            (box_width, box_height), Image.Resampling.LANCZOS
        )
        alpha_value = clamp(float(alpha), 0.0, 1.0)
        overlay_with_alpha = overlay_resized.copy().convert("RGBA")
        overlay_with_alpha.putalpha(int(255 * alpha_value))

        blended = Image.new("RGBA", base_image.size, (255, 255, 255, 255))
        blended.paste(base_image.convert("RGBA"), (0, 0))
        blended.paste(
            overlay_with_alpha,
            (left, top, right, bottom),
            overlay_with_alpha,
        )
        return {"blended_image_base64": pil_to_base64(blended.convert("RGB"))}
