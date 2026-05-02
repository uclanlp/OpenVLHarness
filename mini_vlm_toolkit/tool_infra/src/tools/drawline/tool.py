from PIL import ImageColor, ImageDraw

from tools.base import BaseTool
from tools.image_utils import (
    draw_dashed_line,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "DrawLine_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "box": {
            "type": "array",
            "description": "Pixel box [x1, y1, x2, y2]",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
        "corner_a": {
            "type": "integer",
            "description": "First corner index (1-4)",
            "minimum": 1,
            "maximum": 4,
        },
        "corner_b": {
            "type": "integer",
            "description": "Second corner index (1-4)",
            "minimum": 1,
            "maximum": 4,
        },
    },
    "required": ["image_path", "box", "corner_a", "corner_b"],
    "additionalProperties": False,
}


class DrawLine_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Draw a dashed annotation line between two chosen corners of a box. "
                "Use this tool to visualize spatial relations on an image. Do not use "
                "it for detection or measurement; it only draws overlays."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "box": "list[int] - pixel box [x1, y1, x2, y2]",
                "corner_a": "int - corner index (1-4)",
                "corner_b": "int - corner index (1-4)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - edited_image_base64",
            demo_commands=[
                {
                    "command": "DrawLine_Tool(image_path=\"/path/to/image.jpg\", box=[0, 10, 100, 200], corner_a=1, corner_b=3)",
                    "description": "Draw a line between box corners.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
            },
        )

    def execute(
        self,
        image_path=None,
        box=None,
        corner_a=None,
        corner_b=None,
        image_base64=None,
    ):
        if box is None or len(box) != 4:
            raise ValueError("box must be a list of four values")
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        x1, y1, x2, y2 = [float(value) for value in box]
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        points = {
            1: (left, top),
            2: (right, top),
            3: (right, bottom),
            4: (left, bottom),
        }
        if corner_a not in points or corner_b not in points:
            raise ValueError("corner_a and corner_b must be integers from 1 to 4")

        draw = ImageDraw.Draw(image)
        color = ImageColor.getrgb("#e6194b")
        draw_dashed_line(
            draw,
            points[corner_a],
            points[corner_b],
            color=color,
            width=2,
        )
        return {"edited_image_base64": pil_to_base64(image)}
