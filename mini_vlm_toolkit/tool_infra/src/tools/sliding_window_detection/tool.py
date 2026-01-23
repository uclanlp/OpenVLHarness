from tools.base import BaseTool

TOOL_NAME = "SlidingWindowDetection_Tool"


class SlidingWindowDetection_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Run detection on overlapping crops to find small objects.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "objects": "list[str] - objects to detect",
                "box_threshold": "float - confidence threshold for boxes",
                "text_threshold": "float - confidence threshold for text",
            },
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
