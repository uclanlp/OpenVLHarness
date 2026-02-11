from tools.base import BaseTool

TOOL_NAME = "SlidingWindowDetection_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "objects": {
            "type": "array",
            "description": "Object names to detect",
            "items": {"type": "string"},
        },
        "box_threshold": {"type": "number", "description": "Box confidence threshold"},
        "text_threshold": {"type": "number", "description": "Text confidence threshold"},
    },
    "required": ["objects"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class SlidingWindowDetection_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Placeholder tool intended for sliding-window small-object detection "
                "over overlapping crops. This tool is currently not implemented and "
                "raises NotImplementedError; do not call it in production flows."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "objects": "list[str] - objects to detect",
                "box_threshold": "float - confidence threshold for boxes",
                "text_threshold": "float - confidence threshold for text",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - patches, boxes",
            demo_commands=[
                {
                    "command": "SlidingWindowDetection_Tool(image_path=\"/path/to/image.jpg\", objects=[\"sign\"])",
                    "description": "Search for small objects using sliding windows.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "placeholder",
                "depends_on": "GroundingDINO_Tool",
            },
        )

    def execute(
        self,
        image_path=None,
        objects=None,
        box_threshold=0.35,
        text_threshold=0.25,
        image_base64=None,
    ):
        raise NotImplementedError("SlidingWindowDetection_Tool is not implemented yet.")
