from tools.base import BaseTool
from tools.image_utils import (
    crop_image_normalized,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "ZoomInImageByBBox_Tool"


class ZoomInImageByBBox_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Crop an image around a normalized bbox with optional padding.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "bbox": "list[float] - normalized bbox [x, y, w, h]",
                "padding": "float - padding added around the box",
            },
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
