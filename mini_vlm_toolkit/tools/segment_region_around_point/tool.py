from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "SegmentRegionAroundPoint_Tool"


class SegmentRegionAroundPoint_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Segment a region around one or more points using SAM3.",
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "points": "list[list[float]] - normalized points [[x, y], ...]",
                "point_labels": "list[int] - optional labels (1 foreground, 0 background)",
            },
            output_type="dict - masks or mask encodings",
            demo_commands=[
                {
                    "command": "SegmentRegionAroundPoint_Tool(image_path=\"/path/to/image.jpg\", points=[[0.5, 0.5]])",
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
