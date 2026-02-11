from PIL import ImageColor, ImageDraw

from tools.base import BaseTool
from tools.image_utils import (
    clamp,
    draw_dashed_line,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "DrawVerticalLineByX_Tool"


class DrawVerticalLineByX_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Draw a vertical dashed line at a specified x coordinate. Use this "
                "tool for visual annotation or split-line marking. Do not use it for "
                "object detection or coordinate estimation."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "x": "float - x coordinate (normalized or pixel)",
                "coord_type": "str - normalized or pixel",
            },
            output_type="dict - edited_image_base64",
            demo_commands=[
                {
                    "command": "DrawVerticalLineByX_Tool(image_path=\"/path/to/image.jpg\", x=0.5)",
                    "description": "Draw a vertical line at x=0.5.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "coord_type_default": "normalized",
            },
        )

    def execute(
        self, image_path=None, x=None, coord_type="normalized", image_base64=None
    ):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        width, height = image.size
        x_value = float(x)
        if coord_type == "normalized":
            x_value = clamp(x_value, 0.0, 1.0) * width
        elif coord_type == "pixel":
            x_value = clamp(x_value, 0.0, float(width))
        else:
            raise ValueError("coord_type must be normalized or pixel")

        draw = ImageDraw.Draw(image)
        color = ImageColor.getrgb("#e6194b")
        draw_dashed_line(
            draw,
            (x_value, 0),
            (x_value, height),
            color=color,
            width=2,
        )
        return {"edited_image_base64": pil_to_base64(image)}
