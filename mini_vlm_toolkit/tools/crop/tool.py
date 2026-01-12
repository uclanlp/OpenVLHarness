from mini_vlm_toolkit.tools.base import BaseTool
from mini_vlm_toolkit.tools.image_utils import coerce_xyxy, load_image, pil_to_base64

TOOL_NAME = "Crop_Tool"


class Crop_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Crop an image using normalized or pixel coordinates.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "coords": "list[float] - bounding box coordinates",
                "coords_type": "str - normalized_xyxy (default) or pixel_xyxy",
            },
            output_type="dict - cropped_image_base64",
            demo_commands=[
                {
                    "command": "execution = tool.execute(image_path=\"/path/to/image.jpg\", coords=[0.1, 0.2, 0.6, 0.7])",
                    "description": "Crop with normalized coordinates.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
                "coords_primary": "normalized_xyxy",
            },
        )

    def execute(self, image_path, coords, coords_type="normalized_xyxy"):
        image = load_image(image_path)
        width, height = image.size
        left, top, right, bottom = coerce_xyxy(coords, width, height, coords_type)
        cropped = image.crop((left, top, right, bottom))
        return {"cropped_image_base64": pil_to_base64(cropped)}
