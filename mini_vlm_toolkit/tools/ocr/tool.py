from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "OCR_Tool"


class OCR_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Extract text strings from an image using OCR.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
            },
            output_type="list[str] - recognized text strings",
            demo_commands=[
                {
                    "command": "execution = tool.execute(image_path=\"/path/to/scan.png\")",
                    "description": "Extract text from an image.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "placeholder",
            },
        )

    def execute(self, image_path):
        raise NotImplementedError("OCR_Tool is not implemented yet.")
