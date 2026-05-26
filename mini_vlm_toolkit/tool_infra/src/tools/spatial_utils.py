import base64
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def decode_depth_array(depth_result: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(depth_result, dict):
        return None
    encoded = depth_result.get("depth_meters_b64")
    shape = depth_result.get("depth_shape")
    if not isinstance(encoded, str) or not isinstance(shape, list) or len(shape) != 2:
        return None
    try:
        raw = base64.b64decode(encoded)
        return np.frombuffer(raw, dtype=np.float32).reshape(
            (int(shape[0]), int(shape[1]))
        )
    except Exception:
        return None


def image_size_from_grounding(result: Dict[str, Any]) -> Tuple[int, int]:
    width = result.get("orig_img_w")
    height = result.get("orig_img_h")
    if width and height:
        return max(int(width), 1), max(int(height), 1)

    size = result.get("size")
    if isinstance(size, list) and len(size) >= 2:
        return max(int(size[0]), 1), max(int(size[1]), 1)
    return 1, 1


def coarse_region(norm_center: List[float]) -> str:
    x, y = norm_center
    if x < 1 / 3:
        horizontal = "left"
    elif x > 2 / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    if y < 1 / 3:
        vertical = "top"
    elif y > 2 / 3:
        vertical = "bottom"
    else:
        vertical = "center"

    if horizontal == "center" and vertical == "center":
        return "center"
    return f"{horizontal}-{vertical}"


def _clip_box(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = int(max(0, min(width - 1, math.floor(x1))))
    y1 = int(max(0, min(height - 1, math.floor(y1))))
    x2 = int(max(0, min(width, math.ceil(x2))))
    y2 = int(max(0, min(height, math.ceil(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _decode_rle_mask(mask_rle: Any, height: int, width: int) -> Optional[np.ndarray]:
    if not mask_rle:
        return None
    try:
        from pycocotools import mask as mask_utils

        if isinstance(mask_rle, str):
            rle = {"size": [int(height), int(width)], "counts": mask_rle}
        elif isinstance(mask_rle, dict):
            rle = mask_rle
        else:
            return None
        decoded = mask_utils.decode(rle)
        if decoded is None:
            return None
        return np.asarray(decoded).astype(bool)
    except Exception:
        return None


def median_depth_for_region(
    depth: Optional[np.ndarray],
    box: List[float],
    image_width: int,
    image_height: int,
    mask_rle: Any = None,
) -> Optional[float]:
    if depth is None or depth.size == 0:
        return None
    depth_h, depth_w = depth.shape[:2]
    sx = depth_w / max(image_width, 1)
    sy = depth_h / max(image_height, 1)

    scaled_box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    x1, y1, x2, y2 = _clip_box(scaled_box, depth_w, depth_h)
    region = depth[y1:y2, x1:x2]

    mask = _decode_rle_mask(mask_rle, image_height, image_width)
    if mask is not None:
        try:
            ys, xs = np.nonzero(mask)
            if len(xs) > 0:
                scaled_xs = np.clip((xs * sx).astype(int), 0, depth_w - 1)
                scaled_ys = np.clip((ys * sy).astype(int), 0, depth_h - 1)
                region = depth[scaled_ys, scaled_xs]
        except Exception:
            pass

    valid = np.asarray(region, dtype=np.float32)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def extract_candidates(
    grounding_result: Dict[str, Any],
    depth: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(grounding_result, dict):
        return []
    boxes = grounding_result.get("pred_boxes") or grounding_result.get("boxes") or []
    scores = grounding_result.get("pred_scores") or grounding_result.get("scores") or []
    masks = grounding_result.get("pred_masks") or []
    width, height = image_size_from_grounding(grounding_result)

    candidates: List[Dict[str, Any]] = []
    for idx, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        norm_center = [
            min(max(center[0] / max(width, 1), 0.0), 1.0),
            min(max(center[1] / max(height, 1), 0.0), 1.0),
        ]
        score = safe_float(scores[idx], 0.0) if idx < len(scores) else 0.0
        mask_rle = masks[idx] if idx < len(masks) else None
        mask_area = None
        decoded_mask = _decode_rle_mask(mask_rle, height, width)
        if decoded_mask is not None:
            mask_area = int(decoded_mask.sum())

        candidates.append(
            {
                "object_id": idx + 1,
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round(center[0], 2), round(center[1], 2)],
                "normalized_center": [
                    round(norm_center[0], 4),
                    round(norm_center[1], 4),
                ],
                "region": coarse_region(norm_center),
                "depth_value": median_depth_for_region(
                    depth, [x1, y1, x2, y2], width, height, mask_rle
                ),
                "confidence": round(score or 0.0, 4),
                "mask_area": mask_area,
                "box_area": round(max(x2 - x1, 0.0) * max(y2 - y1, 0.0), 2),
            }
        )

    assign_relative_depth(candidates)
    return candidates


def assign_relative_depth(candidates: List[Dict[str, Any]]) -> None:
    with_depth = [
        item for item in candidates
        if isinstance(item.get("depth_value"), (int, float))
    ]
    if not with_depth:
        for item in candidates:
            item["relative_depth"] = "unknown"
        return
    if len(with_depth) == 1:
        with_depth[0]["relative_depth"] = "middle"
    else:
        values = sorted(float(item["depth_value"]) for item in with_depth)
        low = values[len(values) // 3]
        high = values[(2 * len(values)) // 3]
        for item in with_depth:
            depth_value = float(item["depth_value"])
            if depth_value <= low and low < high:
                item["relative_depth"] = "near"
            elif depth_value >= high and low < high:
                item["relative_depth"] = "far"
            else:
                item["relative_depth"] = "middle"
    for item in candidates:
        item.setdefault("relative_depth", "unknown")


def choose_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence") or 0.0))


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _inside_ratio(inner: List[float], outer: List[float]) -> float:
    ix1, iy1, ix2, iy2 = [float(v) for v in inner[:4]]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer[:4]]
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
    area = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    return 0.0 if area <= 0 else inter / area


def compute_relations(
    subject: Dict[str, Any],
    obj: Dict[str, Any],
    *,
    x_tolerance: float = 0.06,
    y_tolerance: float = 0.06,
) -> Dict[str, Any]:
    s_center = subject.get("normalized_center") or [0.5, 0.5]
    o_center = obj.get("normalized_center") or [0.5, 0.5]
    dx = float(s_center[0]) - float(o_center[0])
    dy = float(s_center[1]) - float(o_center[1])

    if dx < -x_tolerance:
        left_right = "subject_left_of_object"
    elif dx > x_tolerance:
        left_right = "subject_right_of_object"
    else:
        left_right = "roughly_same_horizontal_position"

    if dy < -y_tolerance:
        above_below = "subject_above_object"
    elif dy > y_tolerance:
        above_below = "subject_below_object"
    else:
        above_below = "roughly_same_vertical_position"

    s_depth = subject.get("depth_value")
    o_depth = obj.get("depth_value")
    front_back = "unclear"
    if isinstance(s_depth, (int, float)) and isinstance(o_depth, (int, float)):
        threshold = max(0.15, 0.08 * max(abs(float(s_depth)), abs(float(o_depth)), 1.0))
        if float(s_depth) < float(o_depth) - threshold:
            front_back = "subject_in_front_of_object"
        elif float(s_depth) > float(o_depth) + threshold:
            front_back = "subject_behind_object"
        else:
            front_back = "similar_depth"

    center_distance = math.sqrt(dx * dx + dy * dy)
    if center_distance <= 0.20:
        distance = "near"
    elif center_distance >= 0.45:
        distance = "far"
    else:
        distance = "moderate"

    iou = box_iou(subject["box"], obj["box"])
    subject_inside = _inside_ratio(subject["box"], obj["box"])
    object_inside = _inside_ratio(obj["box"], subject["box"])
    if iou >= 0.05:
        overlap_contact = "overlapping"
    else:
        sx1, sy1, sx2, sy2 = subject["box"]
        ox1, oy1, ox2, oy2 = obj["box"]
        gap_x = max(float(ox1) - float(sx2), float(sx1) - float(ox2), 0.0)
        gap_y = max(float(oy1) - float(sy2), float(sy1) - float(oy2), 0.0)
        max_extent = max(float(sx2) - float(sx1), float(sy2) - float(sy1), 1.0)
        overlap_contact = (
            "touching" if max(gap_x, gap_y) <= 0.03 * max_extent else "separate"
        )

    if subject_inside >= 0.85:
        containment = "subject_inside_object"
    elif object_inside >= 0.85:
        containment = "subject_around_object"
    else:
        containment = "none"

    return {
        "left_right": left_right,
        "above_below": above_below,
        "front_back": front_back,
        "distance": distance,
        "overlap_contact": overlap_contact,
        "containment": containment,
        "metrics": {
            "normalized_center_delta": [round(dx, 4), round(dy, 4)],
            "normalized_center_distance": round(center_distance, 4),
            "box_iou": round(iou, 4),
            "subject_inside_object_ratio": round(subject_inside, 4),
            "object_inside_subject_ratio": round(object_inside, 4),
        },
    }


_REGION_PROSE = {
    "left-top": "upper-left of the frame",
    "right-top": "upper-right of the frame",
    "left-bottom": "lower-left of the frame",
    "right-bottom": "lower-right of the frame",
    "left-center": "left side of the frame",
    "right-center": "right side of the frame",
    "center-top": "top of the frame",
    "center-bottom": "bottom of the frame",
    "center": "center of the frame",
}


_DEPTH_BAND_PROSE = {
    "near": "foreground",
    "middle": "mid-depth",
    "far": "background",
}


def _entity_position_clause(name: str, detection: Dict[str, Any]) -> str:
    region = _REGION_PROSE.get(detection.get("region") or "", "frame")
    depth_band = _DEPTH_BAND_PROSE.get(detection.get("relative_depth") or "")
    if depth_band:
        return f"the {name} is at the {region}, {depth_band}"
    return f"the {name} is at the {region}"


def _relation_phrases(object_name: str, relations: Dict[str, Any]) -> List[str]:
    phrases: List[str] = []
    containment = relations.get("containment")
    overlap = relations.get("overlap_contact")
    left_right = relations.get("left_right")
    above_below = relations.get("above_below")
    front_back = relations.get("front_back")
    distance = relations.get("distance")

    if containment == "subject_inside_object":
        phrases.append(f"inside the {object_name}")
    elif containment == "subject_around_object":
        phrases.append(f"around the {object_name}")
    else:
        if left_right == "subject_left_of_object":
            phrases.append(f"to the left of the {object_name}")
        elif left_right == "subject_right_of_object":
            phrases.append(f"to the right of the {object_name}")

        if above_below == "subject_above_object":
            phrases.append(f"above the {object_name}")
        elif above_below == "subject_below_object":
            phrases.append(f"below the {object_name}")

    if front_back == "subject_in_front_of_object":
        phrases.append(f"in front of the {object_name}")
    elif front_back == "subject_behind_object":
        phrases.append(f"behind the {object_name}")

    if overlap in {"overlapping", "touching"}:
        phrases.append(f"{overlap} the {object_name}")
    elif not phrases and distance in {"near", "far"}:
        phrases.append(f"{distance} from the {object_name}")

    if not phrases:
        if distance == "near":
            phrases.append(f"near the {object_name}")
        elif distance == "far":
            phrases.append(f"far from the {object_name}")
        else:
            phrases.append(f"at a moderate distance from the {object_name}")

    return phrases


def build_spatial_summary(
    subject_name: str,
    object_name: str,
    image_id: str,
    relations: Dict[str, Any],
    subject_detection: Optional[Dict[str, Any]] = None,
    object_detection: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = [
        "Image-frame measurement only — do not quote as a cardinal/world-frame answer."
    ]
    position_clauses: List[str] = []
    if isinstance(subject_detection, dict):
        position_clauses.append(_entity_position_clause(subject_name, subject_detection))
    if isinstance(object_detection, dict):
        position_clauses.append(_entity_position_clause(object_name, object_detection))
    if position_clauses:
        parts.append(f"In {image_id}: " + "; ".join(position_clauses) + ".")
    else:
        parts.append(f"In {image_id}.")

    relation_phrase = ", ".join(_relation_phrases(object_name, relations))
    parts.append(
        f"Image-frame relation: the {subject_name} is {relation_phrase}."
    )
    parts.append(
        "If the question needs cross-frame reasoning, do the composition yourself — "
        "call this tool once per frame and combine the results in prose, or use a "
        "coding tool against the per-frame data bindings."
    )
    return " ".join(parts)


def movement_direction(start: Dict[str, Any], end: Dict[str, Any]) -> Dict[str, Any]:
    s_center = start.get("normalized_center") or [0.5, 0.5]
    e_center = end.get("normalized_center") or [0.5, 0.5]
    dx = float(e_center[0]) - float(s_center[0])
    dy = float(e_center[1]) - float(s_center[1])
    magnitude_2d = math.sqrt(dx * dx + dy * dy)

    s_depth = start.get("depth_value")
    e_depth = end.get("depth_value")
    start_depth: Optional[float] = None
    end_depth: Optional[float] = None
    depth_delta: Optional[float] = None
    if isinstance(s_depth, (int, float)) and isinstance(e_depth, (int, float)):
        start_depth = round(float(s_depth), 3)
        end_depth = round(float(e_depth), 3)
        depth_delta = round(float(e_depth) - float(s_depth), 3)

    return {
        "dx": round(dx, 3),
        "dy": round(dy, 3),
        "magnitude_2d": round(magnitude_2d, 3),
        "start_depth": start_depth,
        "end_depth": end_depth,
        "depth_delta": depth_delta,
    }


_SUBJECT_RGB = (235, 64, 52)
_OBJECT_RGB = (52, 116, 235)


def _mask_for_selected(
    grounding_result: Dict[str, Any],
    selected: Optional[Dict[str, Any]],
    image_height: int,
    image_width: int,
) -> Optional[np.ndarray]:
    if not isinstance(grounding_result, dict) or not isinstance(selected, dict):
        return None
    masks = grounding_result.get("pred_masks") or []
    object_id = selected.get("object_id")
    if not isinstance(object_id, int) or object_id <= 0 or object_id > len(masks):
        return None
    return _decode_rle_mask(masks[object_id - 1], image_height, image_width)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> None:
    if mask.shape != canvas.shape[:2]:
        return
    color_arr = np.array(color, dtype=np.float32)
    region = canvas[mask].astype(np.float32)
    blended = region * (1.0 - alpha) + color_arr * alpha
    canvas[mask] = np.clip(blended, 0, 255).astype(np.uint8)


def render_subject_object_overlay(
    image_base64: str,
    *,
    subject_name: str,
    object_name: str,
    subject_grounding: Dict[str, Any],
    object_grounding: Dict[str, Any],
    subject_selected: Optional[Dict[str, Any]],
    object_selected: Optional[Dict[str, Any]],
) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
        from .image_utils import load_image_base64, pil_to_base64
    except Exception:
        return None

    try:
        image = load_image_base64(image_base64)
    except Exception:
        return None

    canvas = np.array(image.convert("RGB"))
    height, width = canvas.shape[:2]

    for selected, grounding, color in (
        (subject_selected, subject_grounding, _SUBJECT_RGB),
        (object_selected, object_grounding, _OBJECT_RGB),
    ):
        mask = _mask_for_selected(grounding, selected, height, width)
        if mask is not None:
            _paint_mask(canvas, mask, color)

    overlay = Image.fromarray(canvas)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for selected, color, role, name in (
        (subject_selected, _SUBJECT_RGB, "subject", subject_name),
        (object_selected, _OBJECT_RGB, "object", object_name),
    ):
        if not isinstance(selected, dict):
            continue
        box = selected.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{role}: {name}"
        text_xy = (x1 + 3, max(0.0, y1 - 14))
        draw.text(text_xy, label, fill=color, font=font)

    return pil_to_base64(overlay, image_format="PNG")


NOISE_FLOOR_2D = 0.02


def build_tracking_summary(target: str, movement: Dict[str, Any]) -> str:
    dx = float(movement.get("dx") or 0.0)
    dy = float(movement.get("dy") or 0.0)
    magnitude = float(movement.get("magnitude_2d") or 0.0)

    delta_clause = (
        f"Δx={dx:+.3f}, Δy={dy:+.3f}, magnitude={magnitude:.3f} "
        "in image-normalized coordinates"
    )

    if magnitude < NOISE_FLOOR_2D:
        parts: List[str] = [f"The {target} is tracked ({delta_clause})."]
    else:
        horizontal = "right" if dx > 0 else ("left" if dx < 0 else None)
        # screen-y grows downward, so dy<0 means the centroid moves up
        vertical = "up" if dy < 0 else ("down" if dy > 0 else None)
        direction_words = " and ".join(w for w in (horizontal, vertical) if w)
        parts = [
            f"The {target}'s centroid moves {direction_words} in the image "
            f"({delta_clause})."
        ]

    s_depth = movement.get("start_depth")
    e_depth = movement.get("end_depth")
    depth_delta = movement.get("depth_delta")
    if (
        isinstance(s_depth, (int, float))
        and isinstance(e_depth, (int, float))
        and isinstance(depth_delta, (int, float))
    ):
        parts.append(
            f"Centroid depth goes from {s_depth:.3f} m to "
            f"{e_depth:.3f} m (Δ={depth_delta:+.3f} m). Positive Δ = "
            "moving farther from the camera; negative Δ = moving "
            "toward it."
        )

    parts.append(
        "Reminder: if the scene is stationary, the camera moved opposite to these deltas."
    )
    return " ".join(parts)
