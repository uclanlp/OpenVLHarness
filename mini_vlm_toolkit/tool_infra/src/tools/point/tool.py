import re

from tools.base import BaseTool
from tools.image_utils import clamp, resolve_image_input

TOOL_NAME = "Point_Tool"


class Point_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Extract point coordinates from text or provided points and "
                "return normalized coordinates."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "target": "str - object or region description",
                "points": "list[list[float]] - optional point list",
                "coord_type": "str - normalized or pixel (used with points)",
            },
            output_type="dict - points (normalized x,y)",
            demo_commands=[
                {
                    "command": "Point_Tool(image_path=\"/path/to/image.jpg\", target=\"leftmost cup\")",
                    "description": "Predict a point for the target object.",
                }
            ],
            user_metadata={
                "source": "OpenThinkIMG",
                "status": "implemented",
            },
        )

    def execute(
        self,
        image_path=None,
        target=None,
        points=None,
        coord_type="normalized",
        image_base64=None,
    ):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        width, height = image.size

        normalized_points = []
        if points:
            for point in points:
                if isinstance(point, dict):
                    x_value = point.get("x")
                    y_value = point.get("y")
                else:
                    if len(point) != 2:
                        raise ValueError("Each point must have two values")
                    x_value, y_value = point
                if x_value is None or y_value is None:
                    raise ValueError("Point entries must include x and y")
                x_value = float(x_value)
                y_value = float(y_value)
                if coord_type == "normalized":
                    x_norm = clamp(x_value, 0.0, 1.0)
                    y_norm = clamp(y_value, 0.0, 1.0)
                elif coord_type == "pixel":
                    x_norm = clamp(x_value / float(width), 0.0, 1.0)
                    y_norm = clamp(y_value / float(height), 0.0, 1.0)
                else:
                    raise ValueError("coord_type must be normalized or pixel")
                normalized_points.append([x_norm, y_norm])

        if not normalized_points and target:
            normalized_points = _extract_points_from_text(target)

        if not normalized_points:
            raise ValueError(
                "No points found. Provide points or include x/y coordinates in target."
            )

        points_pixel = [
            [int(round(x * width)), int(round(y * height))]
            for x, y in normalized_points
        ]
        return {"points": normalized_points, "points_pixel": points_pixel}


def _extract_points_from_text(text):
    if not text:
        return []
    pattern = re.compile(
        r'(x\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?|'
        r'(y\d*)?=\s*"?([0-9]+(?:\.[0-9]+)?)"?'
    )
    values = {}
    for match in pattern.finditer(text):
        attr, x_val, y_attr, y_val = match.groups()
        if attr and "x" in attr:
            values[attr] = float(x_val)
        elif y_attr and "y" in y_attr:
            values[y_attr] = float(y_val)

    indices = sorted(
        {int(key[1:]) for key in values if key.startswith(("x", "y")) and key[1:].isdigit()}
    )
    if not indices:
        indices = [0] if ("x" in values or "y" in values) else []

    points = []
    for idx in indices:
        x_key = f"x{idx}" if f"x{idx}" in values else "x"
        y_key = f"y{idx}" if f"y{idx}" in values else "y"
        if x_key not in values or y_key not in values:
            continue
        x_value = values[x_key]
        y_value = values[y_key]
        if max(x_value, y_value) > 100:
            continue
        if max(x_value, y_value) > 1:
            x_value /= 100.0
            y_value /= 100.0
        points.append([clamp(x_value, 0.0, 1.0), clamp(y_value, 0.0, 1.0)])
    return points
