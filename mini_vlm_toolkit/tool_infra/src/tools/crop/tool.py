from tools.base import BaseTool
from tools.image_utils import (
    coerce_xyxy,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "Crop_Tool"
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
            "description": "Box coordinates [x1, y1, x2, y2]",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
        "coords_type": {
            "type": "string",
            "description": "Coordinate mode: normalized_xyxy or pixel_xyxy",
            "enum": ["normalized_xyxy", "pixel_xyxy"],
        },
    },
    "required": ["coords"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class Crop_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Crop an image using explicit box coordinates. Use this tool when "
                "you already know the target box and need the cropped region only. "
                "Do not use it to discover objects from text; use GroundingDINO_Tool "
                "or SAM tools for region discovery first."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "coords": "list[float] - box coordinates [x1, y1, x2, y2]",
                "coords_type": "str - normalized_xyxy (default) or pixel_xyxy",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - cropped_image_base64",
            demo_commands=[
                {
                    "command": "Crop_Tool(image_path=\"/path/to/image.jpg\", coords=[0.1, 0.2, 0.6, 0.7])",
                    "description": "Crop with normalized coordinates.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "coords_primary": "normalized_xyxy",
            },
        )

    def execute(
        self,
        image_path=None,
        coords=None,
        coords_type="normalized_xyxy",
        image_base64=None,
    ):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        width, height = image.size
        left, top, right, bottom = coerce_xyxy(coords, width, height, coords_type)
        cropped = image.crop((left, top, right, bottom))
        return {"cropped_image_base64": pil_to_base64(cropped)}
