from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "GroundingDINO_Tool"


class GroundingDINO_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Detect objects in an image using GroundingDINO and return bounding boxes.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "query": "str - object names or short phrases to detect",
                "box_threshold": "float - confidence threshold for boxes",
                "text_threshold": "float - confidence threshold for text",
            },
            output_type="dict - boxes, logits, phrases, size",
            demo_commands=[
                {
                    "command": "execution = tool.execute(image_path=\"/path/to/image.jpg\", query=\"bus\")",
                    "description": "Detect buses in an image.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "placeholder",
                "notes": "Data-first output; annotations handled separately.",
            },
        )

    def execute(self, image_path, query, box_threshold=0.35, text_threshold=0.25):
        raise NotImplementedError("GroundingDINO_Tool is not implemented yet.")
