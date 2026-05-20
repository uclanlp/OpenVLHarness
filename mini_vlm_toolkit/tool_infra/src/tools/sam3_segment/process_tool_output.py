"""SAM3-specific output post-processing for agent-visible messages."""

import json

PROCESSOR_MODE = "custom"


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def process_tool_output(payload, prepared, tool_metadata=None):
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE, "status": "fallback",
                "fallback_reason": "non_success_payload",
            }
        }

    result = payload.get("result")
    if not isinstance(result, dict):
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE, "status": "fallback",
                "fallback_reason": "result_not_dict",
            }
        }

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    pred_scores = result.get("pred_scores")
    if isinstance(pred_scores, list):
        mask_count = len(pred_scores)
    else:
        mask_count = _safe_int(result.get("num_masks"), default=0)

    tool_name = str(payload.get("tool_name") or "Visual_Grounding_Tool").strip()
    prompt = str(result.get("prompt") or "").strip()

    tool_detection = None
    if mask_count > 0 and image_ref:
        pred_boxes = result.get("pred_boxes", [])
        orig_w = result.get("orig_img_w", 1)
        orig_h = result.get("orig_img_h", 1)

        # Phase 2: stash per-call (boxes, scores) candidate set on the
        # orchestrator's eval sidecar. The orchestrator sees the
        # `tool_detection` key in this return and forwards it to
        # record_tool_detection — keeps process_fn free of memory access.
        if (
            isinstance(pred_scores, list)
            and pred_scores
            and isinstance(pred_boxes, list)
        ):
            tool_detection = {
                "tool_name": tool_name,
                "category_query": prompt,
                "boxes_xyxy_pixels": [
                    list(b) for b in pred_boxes
                    if isinstance(b, (list, tuple)) and len(b) >= 4
                ],
                "scores": list(pred_scores),
                "image_width": _safe_int(orig_w, default=0),
                "image_height": _safe_int(orig_h, default=0),
            }

        detections = [
            {
                "bbox_2d": [
                    min(max(round(box[0] / orig_w * 1000, 2), 0), 1000),
                    min(max(round(box[1] / orig_h * 1000, 2), 0), 1000),
                    min(max(round(box[2] / orig_w * 1000, 2), 0), 1000),
                    min(max(round(box[3] / orig_h * 1000, 2), 0), 1000),
                ],
                "label": prompt,
            }
            for box in pred_boxes
            if len(box) == 4
        ]
        inner = ",\n".join("\t" + json.dumps(d) for d in detections)
        boxes_block = f"```json\n[\n{inner}\n]```"
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}. "
            f"Their corresponding bounding boxes (on a 0-1000 scale, [x1, y1, x2, y2]) are: {boxes_block}. "
            f"The object numbers correspond to the labeled masks rendered in the following image, which you can refer to as {image_ref}."
        )
    else:
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}."
            + f"Please examine the image carefully to determine whether it contains {prompt!r}. If the image does contain {prompt!r} but {tool_name} did not successfully ground it, you may try calling {tool_name} again with an alternative / simpler prompt or locate it directly by yourself."
        )

    if image_parts:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": text}] + list(image_parts),
        }
    else:
        user_message = {"role": "user", "content": text}

    out = {
        "user_message": user_message,
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }
    if tool_detection is not None:
        out["tool_detection"] = tool_detection
    return out
