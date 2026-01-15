from PIL import Image

from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.image_utils import (
    clamp,
    load_image,
    normalized_xywh_to_pixels,
    pil_to_base64,
)

TOOL_NAME = "OverlayImages_Tool"


class OverlayImages_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Overlay an image onto another with alpha blending.",
            tool_version="0.1.0",
            input_types={
                "base_image_path": "str - local path to the base image",
                "overlay_image_path": "str - local path to the overlay image",
                "alpha": "float - overlay transparency (default 0.6)",
                "bounding_box": "list[float] - normalized bbox [x, y, w, h]",
            },
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

    def execute(self, base_image_path, overlay_image_path, alpha=0.6, bounding_box=None):
        base_image = load_image(base_image_path)
        overlay_image = load_image(overlay_image_path)

        if bounding_box is None:
            bounding_box = [0.0, 0.0, 1.0, 1.0]

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
