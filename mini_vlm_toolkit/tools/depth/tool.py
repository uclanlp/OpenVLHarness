from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Depth_Tool"


class Depth_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Estimate depth from an image and return a depth map.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
            },
            output_type="dict - depth_map_base64",
            demo_commands=[
                {
                    "command": "execution = tool.execute(image_path=\"/path/to/image.jpg\")",
                    "description": "Generate a depth map.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "placeholder",
            },
        )

    def execute(self, image_path):
        raise NotImplementedError("Depth_Tool is not implemented yet.")
