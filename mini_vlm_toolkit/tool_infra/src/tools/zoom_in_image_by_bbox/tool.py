from tools.base import BaseTool
from tools.image_utils import (
    crop_image_normalized,
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
        "bbox": {
            "type": "array",
            "description": (
                "Normalized bbox [x, y, w, h] where every value must be in [0, 1]. "
                "x,y is top-left and w,h are width/height ratios."
            ),
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "padding": {
            "type": "number",
            "description": "Extra normalized margin on each side (>= 0.05).",
            "minimum": 0.05,
            "default": 0.05,
        },
    },
    "required": ["bbox"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class ZoomInImageByBBox_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Zoom into a region by cropping an image with a normalized bbox "
                "[x, y, w, h] plus padding, and return the cropped image. Use this "
                "tool for closer visual inspection after you already know the region. "
                "All bbox values must be normalized in [0, 1]. "
                "Do not use it to find regions from text prompts; use "
                "GroundingDINO_Tool or SAM tools for region discovery/segmentation first."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "bbox": "list[float] - normalized [x, y, w, h], each value in [0, 1]",
                "padding": "float - extra margin on each side (must be >= 0.05)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["image_zoom_in_tool"],
            output_type="dict - cropped_image_base64",
            demo_commands=[
                {
                    "command": "ZoomInImageByBBox_Tool(image_path=\"/path/to/image.jpg\", bbox=[0.1, 0.2, 0.3, 0.4])",
                    "description": "Crop around a bbox with padding.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "implemented",
            },
        )

    def execute(
        self, image_path=None, bbox=None, padding=0.05, image_base64=None
    ):
        if bbox is None or len(bbox) != 4:
            raise ValueError("bbox must be a list of four values")
        padding = float(padding)
        if padding < 0.05:
            raise ValueError("padding must be at least 0.05")
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        x, y, w, h = [float(value) for value in bbox]
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        cropped = crop_image_normalized(image, x, y, w, h)
        return {"cropped_image_base64": pil_to_base64(cropped)}
