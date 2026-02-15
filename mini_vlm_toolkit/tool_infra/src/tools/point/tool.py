import re

from tools.base import BaseTool
from tools.image_utils import clamp, resolve_image_input

TOOL_NAME = "Point_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "points": {
            "type": "array",
            "description": (
                "Point list as [[x, y], ...] or [{x, y}, ...]. "
                "Values in [0,1] are treated as normalized; values >1 are treated as pixel."
            ),
            "items": {
                "oneOf": [
                    {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                    {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
    },
    "allOf": [
        {
            "oneOf": [
                {"required": ["image_path"]},
                {"required": ["image_base64"]},
            ]
        },
        {
            "required": ["points"]
        },
    ],
    "additionalProperties": False,
}


class Point_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Normalize point coordinates and return both normalized and pixel "
                "outputs. Use this tool when points are already provided (or written "
                "as x/y values in text) and you need canonical coordinates. Do not "
                "use it for vision-based point detection from image content."
            ),
            tool_version="0.2.1",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "points": "list[list[float]] | list[dict] - point list",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - points (normalized x,y)",
            demo_commands=[
                {
                    "command": (
                        "Point_Tool(image_path=\"/path/to/image.jpg\", "
                        "points=[[0.25, 0.6]])"
                    ),
                    "description": "Normalize provided point coordinates.",
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
        coord_type="auto",
        image_base64=None,
    ):
        image = resolve_image_input(image_path=image_path, image_base64=image_base64)
        width, height = image.size

        normalized_points = []
        if points:
            parsed_points = []
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
                parsed_points.append((float(x_value), float(y_value)))

            resolved_coord_type = _resolve_coord_type(coord_type, parsed_points)
            for x_value, y_value in parsed_points:
                if resolved_coord_type == "normalized":
                    x_norm = clamp(x_value, 0.0, 1.0)
                    y_norm = clamp(y_value, 0.0, 1.0)
                elif resolved_coord_type == "pixel":
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


def _resolve_coord_type(coord_type, parsed_points):
    if coord_type in (None, "auto"):
        has_values_over_one = any((x > 1.0 or y > 1.0) for x, y in parsed_points)
        return "pixel" if has_values_over_one else "normalized"
    if coord_type in ("normalized", "pixel"):
        return coord_type
    raise ValueError("coord_type must be auto, normalized, or pixel")


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
