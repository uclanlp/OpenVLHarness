from PIL import ImageColor, ImageDraw

from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.image_utils import clamp, draw_dashed_line, load_image, pil_to_base64

TOOL_NAME = "DrawVerticalLineByX_Tool"


class DrawVerticalLineByX_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Draw a vertical line at the provided x coordinate.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "x": "float - x coordinate (normalized or pixel)",
                "coord_type": "str - normalized or pixel",
            },
            output_type="dict - edited_image_base64",
            demo_commands=[
                {
                    "command": "execution = tool.execute(image_path=\"/path/to/image.jpg\", x=0.5)",
                    "description": "Draw a vertical line at x=0.5.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "coord_type_default": "normalized",
            },
        )

    def execute(self, image_path, x, coord_type="normalized"):
        image = load_image(image_path)
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
