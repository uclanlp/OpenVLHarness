from PIL import ImageColor, ImageDraw

from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.image_utils import (
    draw_dashed_line,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "DrawLine_Tool"


class DrawLine_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Draw a dashed line between two corners of a bounding box.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "box": "list[int] - pixel box [x1, y1, x2, y2]",
                "corner_a": "int - corner index (1-4)",
                "corner_b": "int - corner index (1-4)",
            },
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
