from tools.base import BaseTool

TOOL_NAME = "SegmentAndMark_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": "A runtime image ID from Environment Update messages, e.g. input_image_1 or tool_generated_image_1.",
        },
        "alpha": {"type": "number", "description": "Mask alpha"},
    },
    "required": ["image_id"],
    "additionalProperties": False,
}


class SegmentAndMark_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Placeholder tool intended to segment a full image and annotate masks "
                "with numeric labels. This tool is currently not implemented and "
                "raises NotImplementedError; do not call it in production flows."
            ),
            tool_version="0.2.0",
            input_types={
                "image_id": "str - runtime image ID from Environment Update (e.g. input_image_1)",
                "alpha": "float - mask alpha",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - annotated_image_base64, bboxes",
            demo_commands=[
                {
                    "command": "SegmentAndMark_Tool(image_id=\"input_image_1\")",
                    "description": "Segment and label the image.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "placeholder",
            },
        )

    def execute(
        self,
        image_path=None,
        granularity=1.8,
        alpha=0.1,
        anno_mode=None,
        image_base64=None,
    ):
        raise NotImplementedError("SegmentAndMark_Tool is not implemented yet.")
