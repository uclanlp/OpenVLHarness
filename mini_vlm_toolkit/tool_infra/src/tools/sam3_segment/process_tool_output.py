"""SAM3-specific output post-processing for agent-visible messages."""

PROCESSOR_MODE = "custom"


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def process_tool_output(payload, memory, tool_metadata=None):
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "non_success_payload",
            },
        )

    result = payload.get("result")
    if not isinstance(result, dict):
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "result_not_dict",
            },
        )

    prepared = memory.prepare_tool_message_payload(payload)
    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None

    pred_scores = result.get("pred_scores")
    if isinstance(pred_scores, list):
        mask_count = len(pred_scores)
    else:
        mask_count = _safe_int(result.get("num_masks"), default=0)

    tool_name = str(payload.get("tool_name") or "Visual_Grounding_Tool").strip()
    prompt = str(result.get("prompt") or "").strip()

    if mask_count > 0 and image_ref:
        pred_boxes = result.get("pred_boxes", [])
        orig_w = result.get("orig_img_w", 1)
        orig_h = result.get("orig_img_h", 1)
        box_mapping = {
            f"object {i + 1}": [
                min(max(round(box[0] / orig_w * 1000, 2), 0), 1000),
                min(max(round(box[1] / orig_h * 1000, 2), 0), 1000),
                min(max(round(box[2] / orig_w * 1000, 2), 0), 1000),
                min(max(round(box[3] / orig_h * 1000, 2), 0), 1000),
            ]
            for i, box in enumerate(pred_boxes)
            if len(box) == 4
        }
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}. "
            f"Their corresponding bounding boxes (on a 0-1000 scale, [x1, y1, x2, y2]) are: {box_mapping}. "
            f"The object numbers correspond to the labeled masks rendered in the following image, which you can refer to as {image_ref}."
        )
    else:
        text = (
            f"{tool_name} grounded {mask_count} objects for the prompt {prompt!r}."
            + f"Please examine the image carefully to determine whether it contains {prompt!r}. If the image does contain {prompt!r} but {tool_name} did not successfully ground it, you may try calling {tool_name} again with an alternative / simpler prompt or locate it directly by yourself."
        )

    return memory.build_processed_tool_output(
        payload=payload,
        prepared_payload=prepared,
        text=text,
        processor_info={"mode": PROCESSOR_MODE, "status": "ok"},
    )
