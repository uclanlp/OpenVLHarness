"""Camera_Trajectory_Tool output post-processing for agent-visible messages."""

PROCESSOR_MODE = "custom"


def _build_bev_caption(reference_image_id, image_ref):
    frame = reference_image_id or "the reference image"
    return (
        f"Bird's-eye view of the camera trajectory on {frame}'s X-Z plane is "
        f"rendered in the following image - refer as {image_ref} "
        "(dots = camera positions, forward-pointing arrows = facing direction)."
    )


def _insert_after_preamble(text, caption):
    """Insert `caption` between the coordinate-system preamble (the first
    paragraph) and the rest of the text. Falls back to prepending if the text
    has no blank-line separator."""
    if "\n\n" in text:
        head, tail = text.split("\n\n", 1)
        return f"{head}\n\n{caption}\n\n{tail}"
    return f"{caption}\n\n{text}"


def process_tool_output(payload, prepared, tool_metadata=None):
    del tool_metadata
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    tool_name = str(payload.get("tool_name") or "Camera_Trajectory_Tool")
    text = result.get("text_summary")
    if not isinstance(text, str) or not text.strip():
        text = (
            f"{tool_name} produced no interpretable trajectory "
            "(text_summary missing)."
        )

    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []
    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = (
        image_refs[0]
        if isinstance(image_refs, list) and image_refs
        else None
    )
    if image_ref:
        caption = _build_bev_caption(result.get("reference_image_id"), image_ref)
        text = _insert_after_preamble(text, caption)

    if image_parts:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": text}] + list(image_parts),
        }
    else:
        user_message = {"role": "user", "content": text}

    return {
        "user_message": user_message,
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }
