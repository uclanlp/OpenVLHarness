from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "SegmentAndMark_Tool"


class SegmentAndMark_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Segment the full image and annotate each mask with a numeric label.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "granularity": "float - segmentation granularity",
                "alpha": "float - mask alpha",
                "anno_mode": "list[str] - annotation modes (Mask, Mark, Box)",
            },
            output_type="dict - annotated_image_base64, bboxes",
            demo_commands=[
                {
                    "command": "SegmentAndMark_Tool(image_path=\"/path/to/image.jpg\", granularity=1.8)",
                    "description": "Segment and label the image.",
                }
            ],
            user_metadata={
                "source": "VisualSketchpad",
                "status": "placeholder",
            },
        )

    def execute(self, image_path, granularity=1.8, alpha=0.1, anno_mode=None):
        raise NotImplementedError("SegmentAndMark_Tool is not implemented yet.")
