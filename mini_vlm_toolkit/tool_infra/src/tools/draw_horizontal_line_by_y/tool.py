from PIL import ImageColor, ImageDraw

from tools.base import BaseTool
from tools.image_utils import (
    clamp,
    draw_dashed_line,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "DrawHorizontalLineByY_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": "A runtime image ID from Environment Update messages, e.g. input_image_1 or tool_generated_image_1.",
        },
        "y": {"type": "number", "description": "y coordinate"},
        "coord_type": {
            "type": "string",
            "description": "Coordinate mode: normalized (0-1000 scale) or pixel",
            "enum": ["normalized", "pixel"],
        },
    },
    "required": ["image_id", "y"],
    "additionalProperties": False,
}


class DrawHorizontalLineByY_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Draw a horizontal dashed line at a specified y coordinate. Use this "
                "tool for visual annotation or split-line marking. Do not use it for "
                "object detection or coordinate estimation."
            ),
            tool_version="0.1.0",
            input_types={
                "image_id": "str - runtime image ID from Environment Update (e.g. input_image_1)",
                "y": "float - y coordinate (normalized 0-1000 scale, or pixel)",
                "coord_type": "str - normalized (0-1000 scale) or pixel",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - edited_image_base64",
            demo_commands=[
                {
                    "command": "DrawHorizontalLineByY_Tool(image_id=\"input_image_1\", y=500)",
                    "description": "Draw a horizontal line at the vertical midpoint (y=500 on 0-1000 scale).",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "coord_type_default": "normalized",
            },
        )

    def execute(
        self, image_path=None, y=None, coord_type="normalized", image_base64=None
    ):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        width, height = image.size
        y_value = float(y)
        if coord_type == "normalized":
            y_value = clamp(y_value, 0.0, 1000.0) / 1000.0 * height
        elif coord_type == "pixel":
            y_value = clamp(y_value, 0.0, float(height))
        else:
            raise ValueError("coord_type must be normalized or pixel")

        draw = ImageDraw.Draw(image)
        color = ImageColor.getrgb("#e6194b")
        draw_dashed_line(
            draw,
            (0, y_value),
            (width, y_value),
            color=color,
            width=2,
        )
        return {"edited_image_base64": pil_to_base64(image)}
