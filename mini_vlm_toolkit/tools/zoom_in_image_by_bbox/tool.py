from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.image_utils import crop_image_normalized, load_image, pil_to_base64

TOOL_NAME = "ZoomInImageByBBox_Tool"


class ZoomInImageByBBox_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Crop an image around a normalized bbox with optional padding.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
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

    def execute(self, image_path, bbox, padding=0.05):
        if bbox is None or len(bbox) != 4:
            raise ValueError("bbox must be a list of four values")
        padding = float(padding)
        if padding < 0.05:
            raise ValueError("padding must be at least 0.05")
        image = load_image(image_path)
        x, y, w, h = [float(value) for value in bbox]
        x -= padding
        y -= padding
        w += 2 * padding
        h += 2 * padding
        cropped = crop_image_normalized(image, x, y, w, h)
        return {"cropped_image_base64": pil_to_base64(cropped)}
