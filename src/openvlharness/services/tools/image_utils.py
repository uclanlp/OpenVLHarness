"""Image decoding, coordinate conversion and drawing utilities for tool backends."""
import base64
import io
import math
import os

from PIL import Image


def clamp(value, min_value, max_value):
    """Clamp a scalar to the inclusive bounds min_value and max_value."""
    return max(min_value, min(value, max_value))


def load_image(image_path):
    """Open a local image as RGB, raising ValueError or FileNotFoundError for invalid paths."""
    if not image_path:
        raise ValueError("image_path is required")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image_path not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def decode_base64_image_data(image_base64):
    """Decode raw base64 or a data URL into bytes and an optional declared MIME type.

    Accepts strings or UTF-8 bytes. Raises ValueError for missing, malformed or
    empty input. This function decodes bytes but does not validate image content.
    """
    if image_base64 is None:
        raise ValueError("image_base64 is required")
    if isinstance(image_base64, bytes):
        raw = image_base64.decode("utf-8", errors="ignore")
    elif isinstance(image_base64, str):
        raw = image_base64
    else:
        raise ValueError("image_base64 must be a string")

    raw = raw.strip()
    if not raw:
        raise ValueError("image_base64 is empty")

    content_type = None
    if raw.startswith("data:"):
        if "," not in raw:
            raise ValueError("image_base64 data URL is missing a comma")
        header, raw = raw.split(",", 1)
        content_type = header[5:].split(";", 1)[0].strip() or None

    try:
        data = base64.b64decode(raw)
    except Exception as exc:
        raise ValueError("image_base64 must be valid base64 data") from exc

    if not data:
        raise ValueError("image_base64 decodes to empty content")
    return data, content_type


def infer_image_content_type_and_extension(image_bytes):
    """Inspect image bytes and return a supported MIME type and filename extension.

    Raises ValueError for invalid images or formats outside JPEG, PNG, WEBP, GIF,
    BMP and TIFF.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise ValueError(f"decoded base64 is not a valid image: {exc}") from exc

    content_type = Image.MIME.get(image_format)
    extension_map = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }
    extension = extension_map.get(image_format)

    if not content_type or not extension:
        raise ValueError(
            f"unsupported image format for upload: {image_format or 'unknown'}"
        )

    return content_type, extension


def load_image_base64(image_base64):
    """Decode inline image data as RGB, raising ValueError for invalid image bytes."""
    data, _ = decode_base64_image_data(image_base64)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"image_base64 is not a valid image: {exc}") from exc


def resolve_image_input(image_path=None, image_base64=None, image_label="image"):
    """Load exactly one local-path or base64 image source as RGB.

    image_label customizes error messages. Raises ValueError when both sources
    or neither source is supplied; file and image decoding errors propagate.
    """
    if image_path and image_base64:
        raise ValueError(
            f"Provide exactly one of {image_label}_path or {image_label}_base64"
        )
    if image_path:
        return load_image(image_path)
    if image_base64:
        return load_image_base64(image_base64)
    raise ValueError(f"{image_label}_path or {image_label}_base64 is required")


def pil_to_base64(image, image_format="PNG"):
    """Encode a Pillow image to base64 without a data-URL prefix; default format is PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def coerce_xyxy(coords, width, height, coords_type):
    # "normalized_xyxy" inputs are on a 0-1000 scale (Qwen-style); each
    # coord is divided by 1000 before being mapped to pixel space.
    """Convert a box to ordered, clamped integer pixel crop coordinates.

    Args:
        coords: Four xyxy coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        coords_type: normalized_xyxy for a 0–1000 scale, or pixel_xyxy.

    Returns:
        A (left, top, right, bottom) tuple.

    Raises:
        ValueError: Coordinates, coordinate type or the resulting crop are invalid.
    """
    if coords is None or len(coords) != 4:
        raise ValueError("coords must be a list of four values")
    x1, y1, x2, y2 = [float(value) for value in coords]
    if coords_type == "normalized_xyxy":
        x1 = clamp(x1, 0.0, 1000.0) / 1000.0 * width
        x2 = clamp(x2, 0.0, 1000.0) / 1000.0 * width
        y1 = clamp(y1, 0.0, 1000.0) / 1000.0 * height
        y2 = clamp(y2, 0.0, 1000.0) / 1000.0 * height
    elif coords_type == "pixel_xyxy":
        x1 = clamp(x1, 0.0, float(width))
        x2 = clamp(x2, 0.0, float(width))
        y1 = clamp(y1, 0.0, float(height))
        y2 = clamp(y2, 0.0, float(height))
    else:
        raise ValueError("coords_type must be normalized_xyxy or pixel_xyxy")

    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])
    left = int(round(left))
    right = int(round(right))
    top = int(round(top))
    bottom = int(round(bottom))
    if right <= left or bottom <= top:
        raise ValueError("coords result in empty crop")
    return left, top, right, bottom


def normalized_xywh_to_pixels(bbox, width, height):
    # bbox is expected on the 0-1000 scale (Qwen-style); divide by 1000
    # before mapping to pixel coordinates.
    """Convert a 0–1000 xywh box to a clamped pixel crop; reject empty crops."""
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be a list of four values")
    x, y, w, h = [float(value) for value in bbox]
    x = clamp(x, 0.0, 1000.0)
    y = clamp(y, 0.0, 1000.0)
    w = clamp(w, 0.0, 1000.0)
    h = clamp(h, 0.0, 1000.0)
    x2 = clamp(x + w, 0.0, 1000.0)
    y2 = clamp(y + h, 0.0, 1000.0)
    left = int(round(x / 1000.0 * width))
    top = int(round(y / 1000.0 * height))
    right = int(round(x2 / 1000.0 * width))
    bottom = int(round(y2 / 1000.0 * height))
    if right <= left or bottom <= top:
        raise ValueError("bbox results in empty crop")
    return left, top, right, bottom


def crop_image_normalized(image, x, y, width, height):
    """Return a Pillow crop for a 0–1000 normalized x, y, width and height."""
    image_w, image_h = image.size
    left, top, right, bottom = normalized_xywh_to_pixels(
        [x, y, width, height], image_w, image_h
    )
    return image.crop((left, top, right, bottom))


def draw_dashed_line(draw, start, end, color, width=2, dash_length=8, gap_length=6):
    """Draw a dashed line in place using pixel widths, dash lengths and gaps."""
    x1, y1 = start
    x2, y2 = end
    total_length = math.hypot(x2 - x1, y2 - y1)
    if total_length == 0:
        return
    step = dash_length + gap_length
    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length
    distance = 0.0
    while distance < total_length:
        dash_end = min(distance + dash_length, total_length)
        start_point = (x1 + dx * distance, y1 + dy * distance)
        end_point = (x1 + dx * dash_end, y1 + dy * dash_end)
        draw.line([start_point, end_point], fill=color, width=width)
        distance += step
