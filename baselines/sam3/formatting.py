"""Convert SAM3 server responses into the formats the evaluators expect."""

import json
from typing import Any, Dict, List


def _coerce_box(b: Any) -> List[float] | None:
    if not isinstance(b, (list, tuple)) or len(b) < 4:
        return None
    try:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except (TypeError, ValueError):
        return None


def boxes_pixel_to_norm1k(boxes_pixel: List[List[float]], orig_w: int, orig_h: int) -> List[List[float]]:
    if orig_w <= 0 or orig_h <= 0:
        return []
    out = []
    for b in boxes_pixel:
        coerced = _coerce_box(b)
        if coerced is None:
            continue
        x1, y1, x2, y2 = coerced
        out.append([
            max(0.0, min(1000.0, round(x1 / orig_w * 1000.0, 2))),
            max(0.0, min(1000.0, round(y1 / orig_h * 1000.0, 2))),
            max(0.0, min(1000.0, round(x2 / orig_w * 1000.0, 2))),
            max(0.0, min(1000.0, round(y2 / orig_h * 1000.0, 2))),
        ])
    return out


def format_grounding_prediction(sam3_response: Dict[str, Any]) -> str:
    """Return a JSON string of `[[x1,y1,x2,y2], ...]` in 0-1000 xyxy.

    The LVIS/ODinW evaluators parse this via `_parse_bbox_array_from_text`
    (mini_vlm_toolkit/dataset/lvis.py:58). Empty list -> "[]" is treated as
    "no detections" by the evaluator, which is the correct behaviour.
    """
    boxes_pixel = sam3_response.get("pred_boxes") or []
    orig_w = int(sam3_response.get("orig_img_w") or 0)
    orig_h = int(sam3_response.get("orig_img_h") or 0)
    norm = boxes_pixel_to_norm1k(boxes_pixel, orig_w, orig_h)
    return json.dumps(norm)


def format_counting_prediction(sam3_response: Dict[str, Any]) -> str:
    """Return the integer count as a string.

    The server already runs IoM-overlap removal at threshold 0.3 (see
    mini_vlm_toolkit/tool_infra/src/tools/sam3_segment/server.py:21), so
    `num_masks` is the post-dedup object count.
    """
    n = sam3_response.get("num_masks")
    if n is None:
        # Fall back to len(pred_boxes) if num_masks is missing for any reason.
        n = len(sam3_response.get("pred_boxes") or [])
    return str(int(n))


def build_sam3_raw_record(sam3_response: Dict[str, Any], sam3_prompt: str, task_id: str, dataset_name: str, image_path: str) -> Dict[str, Any]:
    """Full SAM3 response captured for offline inspection.

    Excludes `rendered_image_b64` to keep files small; can be re-rendered
    from boxes + masks if needed.
    """
    return {
        "task_id": task_id,
        "dataset": dataset_name,
        "image_path": image_path,
        "sam3_prompt": sam3_prompt,
        "pred_boxes_pixel": sam3_response.get("pred_boxes") or [],
        "pred_scores": sam3_response.get("pred_scores") or [],
        "pred_masks_rle": sam3_response.get("pred_masks") or [],
        "orig_img_w": sam3_response.get("orig_img_w"),
        "orig_img_h": sam3_response.get("orig_img_h"),
        "num_masks": sam3_response.get("num_masks", 0),
    }


def build_tool_detections_entry(sam3_response: Dict[str, Any], sam3_prompt: str) -> List[Dict[str, Any]]:
    """Mirror the schema written by `record_tool_detection`
    (mini_vlm_toolkit/memory/image_state_memory.py:91) so the existing
    LVIS/ODinW eval bridge picks up real SAM3 scores for AP.
    """
    boxes = []
    for b in sam3_response.get("pred_boxes") or []:
        coerced = _coerce_box(b)
        if coerced is not None:
            boxes.append(coerced)
    scores = [float(s) for s in (sam3_response.get("pred_scores") or [])]
    return [{
        "tool_call_idx": 0,
        "tool_name": "Visual_Grounding_Tool",
        "category_query": sam3_prompt,
        "boxes_xyxy_pixels": boxes,
        "scores": scores,
        "image_width": int(sam3_response.get("orig_img_w") or 0),
        "image_height": int(sam3_response.get("orig_img_h") or 0),
    }]
