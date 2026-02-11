import base64
import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.base import BaseTool
from tools.image_utils import (
    clamp,
    load_image,
    load_image_base64,
    pil_to_base64,
)

try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None

TOOL_NAME_ALL = "SAM3_VisualizeAllMasks_Tool"
TOOL_NAME_SINGLE = "SAM3_VisualizeSingleMask_Tool"

_DEFAULT_ALPHA = 0.15
INPUT_SCHEMA_ALL = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "predictions": {"type": "object", "description": "Prediction payload"},
        "mask_alpha": {"type": "number", "description": "Mask alpha value"},
        "label_mode": {"type": "string", "description": "Label mode"},
        "predictions_format": {
            "type": "string",
            "description": "auto|grounding_agent|sam3_server",
        },
        "font_size_multiplier": {
            "type": "number",
            "description": "Text size multiplier",
        },
        "boarder_width_multiplier": {
            "type": "number",
            "description": "Box width multiplier",
        },
    },
    "required": ["predictions"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}
INPUT_SCHEMA_SINGLE = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Local image path"},
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (use instead of image_path)",
        },
        "predictions": {"type": "object", "description": "Prediction payload"},
        "mask_index": {"type": "integer", "description": "Mask index to visualize"},
        "mask_alpha": {"type": "number", "description": "Mask alpha value"},
        "zoom_mask_alpha": {"type": "number", "description": "Zoom mask alpha value"},
        "label_mode": {"type": "string", "description": "Label mode"},
        "predictions_format": {
            "type": "string",
            "description": "auto|grounding_agent|sam3_server",
        },
        "font_size_multiplier": {
            "type": "number",
            "description": "Text size multiplier",
        },
        "boarder_width_multiplier": {
            "type": "number",
            "description": "Box width multiplier",
        },
    },
    "required": ["predictions", "mask_index"],
    "oneOf": [
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}

_DEFAULT_COLOR_HEX = [
    "#ffff00",
    "#00ff00",
    "#00ffff",
    "#ff00ff",
    "#ff0000",
    "#ff7f00",
    "#7fff00",
    "#00ff7f",
    "#ff007f",
    "#7f00ff",
    "#c0ff00",
    "#ffc000",
    "#00ffc0",
    "#c000ff",
    "#ff00c0",
    "#ff4000",
    "#40ff00",
    "#00ff40",
    "#4000ff",
    "#ff0040",
]


class SAM3_VisualizeAllMasks_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME_ALL,
            tool_description=(
                "Overlay all provided masks on an image for inspection. Use this "
                "tool when you already have prediction masks (from SAM3 server or "
                "grounding-agent format) and want a visualization panel. Do not use "
                "it to generate new masks."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "predictions": "dict - grounding-agent predictions or SAM3 server output",
                "mask_alpha": "float - mask alpha",
                "label_mode": "str - label mode ('1' to show indices, '0' to hide)",
                "predictions_format": "str - auto|grounding_agent|sam3_server",
                "font_size_multiplier": "float - scales label text size",
                "boarder_width_multiplier": "float - scales box outline width",
            },
            input_schema=INPUT_SCHEMA_ALL,
            output_type="dict - image_base64, mask_count",
            demo_commands=[
                {
                    "command": (
                        "SAM3_VisualizeAllMasks_Tool(image_path=\"/path/to/image.jpg\", "
                        "predictions=sam3_output)"
                    ),
                    "description": "Overlay all SAM3 masks on the image.",
                }
            ],
            user_metadata={
                "source": "grounding_agent",
                "status": "implemented",
            },
        )

    def execute(
        self,
        image_path=None,
        image_base64=None,
        predictions=None,
        mask_alpha=_DEFAULT_ALPHA,
        label_mode="1",
        predictions_format="auto",
        font_size_multiplier=1.2,
        boarder_width_multiplier=0.0,
    ):
        if predictions is None:
            raise ValueError("predictions is required")
        image = _resolve_image(image_path, image_base64)
        masks, boxes = _parse_predictions(
            predictions, image, predictions_format=predictions_format
        )
        if not masks:
            raise ValueError("No masks found in predictions.")

        alpha = clamp(float(mask_alpha), 0.0, 1.0)
        colors = _palette_colors(len(masks))
        overlay = _overlay_masks(image, masks, colors, alpha)
        _draw_boxes_and_labels(
            overlay,
            masks,
            boxes,
            colors,
            label_mode=label_mode,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        return {"image_base64": pil_to_base64(overlay), "mask_count": len(masks)}


class SAM3_VisualizeSingleMask_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME_SINGLE,
            tool_description=(
                "Overlay one selected mask and return both full-image and zoomed "
                "views. Use this tool when you already have masks and need detailed "
                "inspection of one region. Do not use it to generate masks from "
                "scratch."
            ),
            tool_version="0.1.0",
            input_types={
                "image_path": "str - local path to the image file",
                "image_base64": "str - base64-encoded image (use instead of image_path)",
                "predictions": "dict - grounding-agent predictions or SAM3 server output",
                "mask_index": "int - index of the mask to visualize",
                "mask_alpha": "float - mask alpha",
                "zoom_mask_alpha": (
                    "float - optional mask alpha for zoomed view (defaults to mask_alpha)"
                ),
                "label_mode": "str - label mode ('1' to show indices, '0' to hide)",
                "predictions_format": "str - auto|grounding_agent|sam3_server",
                "font_size_multiplier": "float - scales label text size",
                "boarder_width_multiplier": "float - scales box outline width",
            },
            input_schema=INPUT_SCHEMA_SINGLE,
            output_type=(
                "dict - overlay_image_base64, zoomed_image_base64, color_hex, mask_index"
            ),
            demo_commands=[
                {
                    "command": (
                        "SAM3_VisualizeSingleMask_Tool(image_path=\"/path/to/image.jpg\", "
                        "predictions=sam3_output, mask_index=0)"
                    ),
                    "description": "Overlay a single mask and return a zoomed-in panel.",
                }
            ],
            user_metadata={
                "source": "grounding_agent",
                "status": "implemented",
            },
        )

    def execute(
        self,
        image_path=None,
        image_base64=None,
        predictions=None,
        mask_index=None,
        mask_alpha=_DEFAULT_ALPHA,
        zoom_mask_alpha=None,
        label_mode="1",
        predictions_format="auto",
        font_size_multiplier=1.2,
        boarder_width_multiplier=0.0,
    ):
        if predictions is None:
            raise ValueError("predictions is required")
        if mask_index is None:
            raise ValueError("mask_index is required")
        image = _resolve_image(image_path, image_base64)
        masks, boxes = _parse_predictions(
            predictions, image, predictions_format=predictions_format
        )
        if not masks:
            raise ValueError("No masks found in predictions.")
        idx = int(mask_index)
        if idx < 0 or idx >= len(masks):
            raise ValueError(f"mask_index {idx} is out of range (0..{len(masks)-1}).")

        alpha = clamp(float(mask_alpha), 0.0, 1.0)
        mask = masks[idx]
        color_rgb, color_hex = _pick_farthest_color(image, mask)

        overlay = _overlay_masks(image, [mask], [color_rgb], alpha)
        _draw_boxes_and_labels(
            overlay,
            [mask],
            _subset_boxes(boxes, idx),
            [color_rgb],
            label_mode=label_mode,
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
            mask_index_offset=idx,
        )

        zoom_alpha = alpha
        if zoom_mask_alpha is not None:
            zoom_alpha = clamp(float(zoom_mask_alpha), 0.0, 1.0)

        zoomed = _render_zoomed_view(
            image=image,
            mask=mask,
            color=color_rgb,
            mask_alpha=zoom_alpha,
        )

        return {
            "overlay_image_base64": pil_to_base64(overlay),
            "zoomed_image_base64": pil_to_base64(zoomed),
            "color_hex": color_hex,
            "mask_index": idx,
        }


def _parse_predictions(predictions, image, predictions_format="auto"):
    if predictions is None:
        raise ValueError("predictions is required")
    if predictions_format not in ("auto", "grounding_agent", "sam3_server"):
        raise ValueError("predictions_format must be auto, grounding_agent, or sam3_server")

    fmt = predictions_format
    if fmt == "auto":
        if "pred_masks" in predictions:
            fmt = "grounding_agent"
        elif "masks_png_b64" in predictions:
            fmt = "sam3_server"
        else:
            raise ValueError("Unable to infer predictions format from keys.")

    if fmt == "grounding_agent":
        return _parse_grounding_agent_predictions(predictions, image)
    return _parse_sam3_server_predictions(predictions, image)


def _resolve_image(image_path, image_base64):
    if image_path and image_base64:
        raise ValueError("Provide exactly one of image_path or image_base64")
    if image_base64:
        return load_image_base64(image_base64)
    if image_path:
        return load_image(image_path)
    raise ValueError("image_path or image_base64 is required")


def _parse_grounding_agent_predictions(predictions, image):
    orig_w, orig_h = image.size
    pred_h = int(predictions.get("orig_img_h", orig_h))
    pred_w = int(predictions.get("orig_img_w", orig_w))
    masks_raw = predictions.get("pred_masks", [])
    if not isinstance(masks_raw, list):
        raise ValueError("pred_masks must be a list")

    masks = [_decode_rle_mask(m, pred_h, pred_w) for m in masks_raw]
    _ensure_mask_shapes(masks, orig_h, orig_w)

    boxes = predictions.get("pred_boxes")
    boxes_xyxy = None
    if isinstance(boxes, list) and len(boxes) == len(masks):
        boxes_xyxy = [_normalized_xywh_to_xyxy(b, pred_w, pred_h) for b in boxes]
    return masks, boxes_xyxy


def _parse_sam3_server_predictions(predictions, image):
    masks_raw = predictions.get("masks_png_b64", [])
    if not isinstance(masks_raw, list):
        raise ValueError("masks_png_b64 must be a list")
    masks = [_decode_png_mask(m) for m in masks_raw]
    if not masks:
        return masks, None

    orig_w, orig_h = image.size
    _ensure_mask_shapes(masks, orig_h, orig_w)

    boxes = predictions.get("boxes")
    boxes_xyxy = None
    if isinstance(boxes, list) and len(boxes) == len(masks):
        boxes_xyxy = [_coerce_xyxy(b) for b in boxes]
    return masks, boxes_xyxy


def _decode_rle_mask(mask_repr, height, width):
    if mask_utils is None:
        raise ImportError(
            "pycocotools is required to decode grounding-agent RLE masks. "
            "Install requirements-server.txt to use this format."
        )
    if isinstance(mask_repr, dict):
        rle = mask_repr
    else:
        rle = {"counts": mask_repr, "size": [int(height), int(width)]}
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def _decode_png_mask(mask_b64):
    data = base64.b64decode(mask_b64)
    mask_img = Image.open(io.BytesIO(data)).convert("L")
    return (np.array(mask_img) > 0).astype(bool)


def _ensure_mask_shapes(masks, height, width):
    for mask in masks:
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image size {(height, width)}"
            )


def _normalized_xywh_to_xyxy(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("pred_boxes entries must be list[float] of length 4")
    x, y, w, h = [float(v) for v in box]
    x0 = clamp(x, 0.0, 1.0) * width
    y0 = clamp(y, 0.0, 1.0) * height
    x1 = clamp(x + w, 0.0, 1.0) * width
    y1 = clamp(y + h, 0.0, 1.0) * height
    return _coerce_xyxy([x0, y0, x1, y1])


def _coerce_xyxy(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]


def _palette_colors(count):
    colors = [_hex_to_rgb(h) for h in _DEFAULT_COLOR_HEX]
    return [colors[i % len(colors)] for i in range(count)]


def _hex_to_rgb(color_hex):
    color_hex = color_hex.lstrip("#")
    return tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(*color)


def _overlay_masks(image, masks, colors, alpha):
    base = np.array(image).astype(np.float32)
    for mask, color in zip(masks, colors):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        color_arr = np.array(color, dtype=np.float32)
        base[mask] = base[mask] * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(base.astype(np.uint8))


def _draw_boxes_and_labels(
    image,
    masks,
    boxes,
    colors,
    label_mode="1",
    font_size_multiplier=1.2,
    boarder_width_multiplier=0.0,
    mask_index_offset=0,
):
    show_labels = str(label_mode).lower() not in ("0", "none", "off", "false")
    draw_boxes = boarder_width_multiplier and boarder_width_multiplier > 0.0

    if not show_labels and not draw_boxes:
        return

    draw = ImageDraw.Draw(image)
    font = _load_font(image.size, font_size_multiplier)
    line_width = max(1, int(round(2 * float(boarder_width_multiplier))))

    for idx, mask in enumerate(masks):
        color = colors[idx]
        box = None
        if boxes and idx < len(boxes):
            box = boxes[idx]
        if box is None:
            box = _mask_bbox_xyxy(mask)

        if box and draw_boxes:
            draw.rectangle(box, outline=color, width=line_width)

        if show_labels:
            label = str(mask_index_offset + idx + 1)
            position = _label_position(mask, box, image.size)
            _draw_text_with_outline(draw, position, label, color, font=font)


def _load_font(image_size, font_size_multiplier):
    base = max(12, int(min(image_size) / 30))
    size = max(10, int(round(base * float(font_size_multiplier))))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_with_outline(draw, position, text, color, font):
    x, y = position
    outline = (0, 0, 0)
    draw.text((x + 1, y + 1), text, fill=outline, font=font)
    draw.text((x, y), text, fill=color, font=font)


def _label_position(mask, box, image_size):
    if box:
        x0, y0, x1, y1 = box
        x = x0 + 3
        y = y0 + 3
    else:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            x, y = 0, 0
        else:
            x = int(np.mean(xs))
            y = int(np.mean(ys))
    x = max(0, min(image_size[0] - 1, x))
    y = max(0, min(image_size[1] - 1, y))
    return x, y


def _mask_bbox_xyxy(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return [x0, y0, x1, y1]


def _mask_bbox_xywh(mask):
    xyxy = _mask_bbox_xyxy(mask)
    if not xyxy:
        return None
    x0, y0, x1, y1 = xyxy
    return [x0, y0, x1 - x0, y1 - y0]


def _mask_area(mask):
    return float(mask.astype(np.uint8).sum())


def _pick_farthest_color(image, mask):
    box = _mask_bbox_xyxy(mask)
    if not box:
        color = _hex_to_rgb(_DEFAULT_COLOR_HEX[0])
        return color, _rgb_to_hex(color)
    crop = image.crop(box)
    crop_np = np.array(crop).reshape(-1, 3)
    if crop_np.size == 0:
        color = _hex_to_rgb(_DEFAULT_COLOR_HEX[0])
        return color, _rgb_to_hex(color)

    palette = np.array([_hex_to_rgb(h) for h in _DEFAULT_COLOR_HEX], dtype=np.float32)
    crop_np = crop_np.astype(np.float32)
    distances = np.sqrt(((crop_np[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2))
    mean_distances = distances.mean(axis=0)
    idx = int(np.argmax(mean_distances))
    color = tuple(int(c) for c in palette[idx])
    return color, _rgb_to_hex(color)


def _get_shift(x, w, w_new, w_img):
    if w_new <= 0 or w_img <= 0:
        return 0.0
    shift = (w_new - w) / 2.0
    if x - shift + w_new > w_img:
        shift = x + w_new - w_img
    return min(x, shift)


def _get_zoom_in_box(mask_box_xywh, img_h, img_w, mask_area):
    area_large = 0.25
    area_medium = 0.05
    box_w, box_h = mask_box_xywh[2], mask_box_xywh[3]
    w_new = min(box_w + max(0.2 * box_w, 16), img_w)
    h_new = min(box_h + max(0.2 * box_h, 16), img_h)
    if w_new <= 0 or h_new <= 0:
        w_new, h_new = img_w, img_h

    mask_relative_area = mask_area / max(1.0, (w_new * h_new))

    w_new_large, h_new_large = w_new, h_new
    if mask_relative_area > area_large:
        ratio_large = np.sqrt(mask_relative_area / area_large)
        w_new_large = min(w_new * ratio_large, img_w)
        h_new_large = min(h_new * ratio_large, img_h)

    w_shift_large = _get_shift(mask_box_xywh[0], mask_box_xywh[2], w_new_large, img_w)
    h_shift_large = _get_shift(mask_box_xywh[1], mask_box_xywh[3], h_new_large, img_h)
    zoom_in_box = [
        mask_box_xywh[0] - w_shift_large,
        mask_box_xywh[1] - h_shift_large,
        w_new_large,
        h_new_large,
    ]

    w_new_medium, h_new_medium = w_new, h_new
    if mask_relative_area > area_medium:
        ratio_med = np.sqrt(mask_relative_area / area_medium)
        w_new_medium = min(w_new * ratio_med, img_w)
        h_new_medium = min(h_new * ratio_med, img_h)

    w_shift_medium = _get_shift(mask_box_xywh[0], mask_box_xywh[2], w_new_medium, img_w)
    h_shift_medium = _get_shift(mask_box_xywh[1], mask_box_xywh[3], h_new_medium, img_h)
    img_crop_box = [
        mask_box_xywh[0] - w_shift_medium,
        mask_box_xywh[1] - h_shift_medium,
        w_new_medium,
        h_new_medium,
    ]
    return zoom_in_box, img_crop_box


def _clip_box_xywh(box_xywh, img_w, img_h):
    x, y, w, h = box_xywh
    x0 = max(0.0, min(float(img_w), float(x)))
    y0 = max(0.0, min(float(img_h), float(y)))
    x1 = max(0.0, min(float(img_w), float(x + w)))
    y1 = max(0.0, min(float(img_h), float(y + h)))
    if x1 <= x0:
        x1 = min(float(img_w), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(img_h), y0 + 1.0)
    return [x0, y0, x1 - x0, y1 - y0]


def _xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [
        int(round(x)),
        int(round(y)),
        int(round(x + w)),
        int(round(y + h)),
    ]


def _render_zoomed_view(
    image,
    mask,
    color,
    mask_alpha=0.15,
):
    img_w, img_h = image.size
    bbox_xywh = _mask_bbox_xywh(mask)
    if bbox_xywh is None:
        return image.copy()

    mask_area = _mask_area(mask)
    zoom_in_box, img_crop_box = _get_zoom_in_box(bbox_xywh, img_h, img_w, mask_area)
    zoom_in_box = _clip_box_xywh(zoom_in_box, img_w, img_h)
    img_crop_box = _clip_box_xywh(img_crop_box, img_w, img_h)

    img_crop = image.crop(_xywh_to_xyxy(img_crop_box))
    zoom_crop = image.crop(_xywh_to_xyxy(zoom_in_box))

    draw = ImageDraw.Draw(img_crop)
    box_rel = [
        bbox_xywh[0] - img_crop_box[0],
        bbox_xywh[1] - img_crop_box[1],
        bbox_xywh[0] - img_crop_box[0] + bbox_xywh[2],
        bbox_xywh[1] - img_crop_box[1] + bbox_xywh[3],
    ]
    draw.rectangle(box_rel, outline=color, width=2)

    zoom_mask = _crop_mask(mask, zoom_in_box, img_w, img_h)
    zoom_overlay = _overlay_masks(zoom_crop, [zoom_mask], [color], mask_alpha)

    if img_crop.width < img_crop.height:
        combined = Image.new(
            "RGB",
            (img_crop.width + zoom_overlay.width, max(img_crop.height, zoom_overlay.height)),
            (255, 255, 255),
        )
        combined.paste(img_crop, (0, 0))
        combined.paste(zoom_overlay, (img_crop.width, 0))
    else:
        combined = Image.new(
            "RGB",
            (max(img_crop.width, zoom_overlay.width), img_crop.height + zoom_overlay.height),
            (255, 255, 255),
        )
        combined.paste(img_crop, (0, 0))
        combined.paste(zoom_overlay, (0, img_crop.height))

    return combined


def _crop_mask(mask, box_xywh, img_w, img_h):
    x0, y0, x1, y1 = _xywh_to_xyxy(box_xywh)
    left = int(round(max(0.0, min(float(img_w), x0))))
    top = int(round(max(0.0, min(float(img_h), y0))))
    right = int(round(max(0.0, min(float(img_w), x1))))
    bottom = int(round(max(0.0, min(float(img_h), y1))))
    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)
    return mask[top:bottom, left:right]


def _subset_boxes(boxes, idx):
    if not boxes or idx >= len(boxes):
        return None
    return [boxes[idx]]
