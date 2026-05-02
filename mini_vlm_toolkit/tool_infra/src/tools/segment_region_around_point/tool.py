from tools.base import BaseTool

TOOL_NAME = "SegmentRegionAroundPoint_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": "A runtime image ID from Environment Update messages, e.g. input_image_1 or tool_generated_image_1.",
        },
        "points": {
            "type": "array",
            "description": "Normalized points [[x, y], ...]",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        },
        "point_labels": {
            "type": "array",
            "description": "Optional labels (1 foreground, 0 background)",
            "items": {"type": "integer"},
        },
    },
    "required": ["image_id", "points"],
    "additionalProperties": False,
}


class SegmentRegionAroundPoint_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Placeholder tool intended to segment regions around user points. "
                "This tool is currently not implemented and raises "
                "NotImplementedError; do not call it in production flows."
            ),
            tool_version="0.1.0",
            input_types={
                "image_id": "str - runtime image ID from Environment Update (e.g. input_image_1)",
                "points": "list[list[float]] - normalized points [[x, y], ...]",
                "point_labels": "list[int] - optional labels (1 foreground, 0 background)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - masks or mask encodings",
            demo_commands=[
                {
                    "command": "SegmentRegionAroundPoint_Tool(image_id=\"input_image_1\", points=[[0.5, 0.5]])",
                    "description": "Segment the region around a point.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "placeholder",
                "backend": "SAM3",
            },
        )

    def execute(
        self,
        image_path=None,
        points=None,
        point_labels=None,
        image_base64=None,
    ):
        raise NotImplementedError("SegmentRegionAroundPoint_Tool is not implemented yet.")
