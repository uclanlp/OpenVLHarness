import re

from tools.base import BaseTool
from tools.image_utils import (
    crop_image_normalized,
    pil_to_base64,
    resolve_image_input,
)

TOOL_NAME = "ZoomInSubfigure_Tool"


class ZoomInSubfigure_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Return coarse subfigure crops based on simple textual position "
                "heuristics (for example top-left, subplot 2, full image). Use this "
                "tool for quick region probing in multi-panel figures. Do not assume "
                "it performs learned subplot detection; selection is rule-based."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "description": "str - position hint like top-left, subplot 1, center, full",
            },
            output_type="dict - subplot_images_base64",
            demo_commands=[
                {
                    "command": "ZoomInSubfigure_Tool(image_path=\"/path/to/chart.png\", description=\"top-left subplot\")",
                    "description": "Select subplot images by description.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
            },
        )

    def execute(self, image_path=None, description=None, image_base64=None):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        boxes = _select_subfigure_boxes(description or "")
        crops = [crop_image_normalized(image, *bbox) for bbox in boxes]
        return {
            "subplot_images_base64": [pil_to_base64(crop) for crop in crops],
            "bboxes_normalized_xywh": boxes,
        }


def _select_subfigure_boxes(description):
    text = (description or "").lower()
    if any(token in text for token in ["full", "whole", "entire", "all"]):
        return [[0.0, 0.0, 1.0, 1.0]]

    grid = {
        "top_left": [0.0, 0.0, 0.5, 0.5],
        "top_right": [0.5, 0.0, 0.5, 0.5],
        "bottom_left": [0.0, 0.5, 0.5, 0.5],
        "bottom_right": [0.5, 0.5, 0.5, 0.5],
    }

    match = re.search(r"subplot\s*(\d+)", text)
    if match:
        idx = int(match.group(1))
        mapping = {
            1: "top_left",
            2: "top_right",
            3: "bottom_left",
            4: "bottom_right",
        }
        if idx in mapping:
            return [grid[mapping[idx]]]

    if any(token in text for token in ["center", "centre", "middle"]):
        return [[0.25, 0.25, 0.5, 0.5]]

    want_top = any(token in text for token in ["top", "upper"])
    want_bottom = any(token in text for token in ["bottom", "lower"])
    want_left = "left" in text
    want_right = "right" in text

    selected = []
    if want_top and want_left:
        selected.append(grid["top_left"])
    if want_top and want_right:
        selected.append(grid["top_right"])
    if want_bottom and want_left:
        selected.append(grid["bottom_left"])
    if want_bottom and want_right:
        selected.append(grid["bottom_right"])

    if not selected:
        if want_top:
            selected.extend([grid["top_left"], grid["top_right"]])
        elif want_bottom:
            selected.extend([grid["bottom_left"], grid["bottom_right"]])
        elif want_left:
            selected.extend([grid["top_left"], grid["bottom_left"]])
        elif want_right:
            selected.extend([grid["top_right"], grid["bottom_right"]])

    return selected or [[0.0, 0.0, 1.0, 1.0]]
