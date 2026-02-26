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

    tool_name = str(payload.get("tool_name") or "SAM3_Segment_Tool").strip()
    prompt = str(result.get("prompt") or "").strip()

    if image_ref:
        if mask_count == 0:
            text = (
                f"{tool_name} generated 0 masks for prompt {prompt!r}. "
                f"A rendered result image is still provided below; refer to it as {image_ref}."
            )
        else:
            text = (
                f"{tool_name} generated {mask_count} masks"
                + (f" for prompt {prompt!r}" if prompt else "")
                + f". They are rendered on the following image, which shall be referred to as {image_ref}."
            )
    else:
        text = (
            f"{tool_name} generated {mask_count} masks"
            + (f" for prompt {prompt!r}" if prompt else "")
            + ". No rendered image was returned."
        )

    return memory.build_processed_tool_output(
        payload=payload,
        prepared_payload=prepared,
        text=text,
        processor_info={"mode": PROCESSOR_MODE, "status": "ok"},
    )
