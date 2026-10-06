"""Crop selected image regions using normalized coordinates and padding.

Zoom_In_Tool expands a 0–1000 xyxy box by normalized padding, clips it to image
bounds and returns the crop as PNG base64.
"""
from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.image_utils import (
    coerce_xyxy,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "Zoom_In_Tool"
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
            "description": (
                "Normalized bbox [x1, y1, x2, y2] on a 0-1000 scale where every value "
                "must be in [0, 1000]. (x1, y1) is top-left and (x2, y2) is bottom-right."
            ),
            "items": {"type": "number"},
        },
        "padding": {
            "type": "number",
            "description": "Extra normalized margin on each side on the 0-1000 scale (>= 50).",
        },
    },
    "required": ["coords"],
}


class Zoom_In_Tool(BaseTool):
    """Local image cropping with normalized coordinates and padding."""
    def __init__(self):
        """Initialize tool metadata and its concrete input/output contract."""
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Zoom into a region by cropping an image with a normalized bbox "
                "[x1, y1, x2, y2] plus padding, and return the cropped image. Use this "
                "tool for closer visual inspection after you already know the region. "
                "All coords values must be normalized on the 0-1000 scale (each value in [0, 1000]). "
                "When trying ti count or ground objects, please try different prompts using Visual_Grounding_Tool first "
                "Only use this tool to ground objects when the Visual_Grounding_Tool does not work."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "coords": "list[float] - normalized [x1, y1, x2, y2] on 0-1000 scale, each value in [0, 1000]",
                "padding": "float - extra margin on each side on the 0-1000 scale (must be >= 50)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["image_zoom_in_tool"],
            output_type="dict - cropped_image_base64",
            demo_commands=[
                {
                    "command": 'Zoom_In_Tool(image_path="/path/to/image.jpg", coords=[100, 200, 500, 600])',
                    "description": "Crop around a bbox with padding.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "implemented",
            },
        )

    def execute(
        self, image_path=None, coords=None, padding=50, image_base64=None
    ):
        """Crop an image with a padded box on the 0–1000 coordinate scale.

        Args:
            image_path: Local image path, exclusive with image_base64.
            coords: Four normalized xyxy coordinates.
            padding: Margin added to each side in normalized units; at least 50.
            image_base64: Raw base64 or an image data URL.

        Returns:
            A mapping containing cropped_image_base64, encoded as PNG without a prefix.

        Raises:
            ValueError: Image source, coordinates, padding or the resulting crop is invalid.
        """
        if coords is None or len(coords) != 4:
            raise ToolArgumentError("coords must be a list of four values")
        padding = float(padding)
        if padding < 50:
            raise ToolArgumentError("padding must be at least 50")
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        x1, y1, x2, y2 = [float(v) for v in coords]
        # Apply padding
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding
        width, height = image.size
        left, top, right, bottom = coerce_xyxy(
            [x1, y1, x2, y2], width, height, "normalized_xyxy"
        )
        cropped = image.crop((left, top, right, bottom))
        return {"cropped_image_base64": pil_to_base64(cropped)}
