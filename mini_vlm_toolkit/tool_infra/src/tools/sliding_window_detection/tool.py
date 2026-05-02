from tools.base import BaseTool

TOOL_NAME = "SlidingWindowDetection_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": "A runtime image ID from Environment Update messages, e.g. input_image_1 or tool_generated_image_1.",
        },
        "objects": {
            "type": "array",
            "description": "Object names to detect",
            "items": {"type": "string"},
        },
    },
    "required": ["image_id", "objects"],
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
            tool_version="0.2.0",
            input_types={
                "image_id": "str - runtime image ID from Environment Update (e.g. input_image_1)",
                "objects": "list[str] - objects to detect",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - patches, boxes",
            demo_commands=[
                {
                    "command": "SlidingWindowDetection_Tool(image_id=\"input_image_1\", objects=[\"sign\"])",
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
