from tools.base import BaseTool
from tools.image_utils import (
    coerce_xyxy,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "ZoomInImageByBBox_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "coords": {
            "type": "array",
            "description": (
                "Normalized bbox [x1, y1, x2, y2] where every value must be in [0, 1]. "
                "(x1, y1) is top-left and (x2, y2) is bottom-right."
            ),
            "items": {"type": "number"},
        },
        "padding": {
            "type": "number",
            "description": "Extra normalized margin on each side (>= 0.05).",
        },
    },
    "required": ["coords"],
}


class ZoomInImageByBBox_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Zoom into a region by cropping an image with a normalized bbox "
                "[x1, y1, x2, y2] plus padding, and return the cropped image. Use this "
                "tool for closer visual inspection after you already know the region. "
                "All coords values must be normalized in [0, 1]. "
                "Do not use it to find regions from text prompts; use "
                "GroundingDINO_Tool or SAM tools for region discovery/segmentation first."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "coords": "list[float] - normalized [x1, y1, x2, y2], each value in [0, 1]",
                "padding": "float - extra margin on each side (must be >= 0.05)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["image_zoom_in_tool"],
            output_type="dict - cropped_image_base64",
            demo_commands=[
                {
                    "command": 'ZoomInImageByBBox_Tool(image_path="/path/to/image.jpg", coords=[0.1, 0.2, 0.5, 0.6])',
                    "description": "Crop around a bbox with padding.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "implemented",
            },
        )

    def execute(
        self, image_path=None, coords=None, padding=0.05, image_base64=None
    ):
        if coords is None or len(coords) != 4:
            raise ValueError("coords must be a list of four values")
        padding = float(padding)
        if padding < 0.05:
            raise ValueError("padding must be at least 0.05")
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        x1, y1, x2, y2 = [float(v) for v in coords]
        # Apply padding
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding
        width, height = image.size
        left, top, right, bottom = coerce_xyxy(
            [x1, y1, x2, y2], width, height, "normalized_xyxy"
        )
        cropped = image.crop((left, top, right, bottom))
        return {"cropped_image_base64": pil_to_base64(cropped)}
