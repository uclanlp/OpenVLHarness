"""Tool-specific output post-processing for Depth_Estimation_Tool."""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        # Delegate to default; tag the processor_info for traceability.
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    tool_name = str(payload.get("tool_name") or "Depth_Estimation_Tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    depth_shape = result.get("depth_shape", [])
    colormap = result.get("colormap", "unknown")

    lines = [
        f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:",
        f"Depth estimation complete. Shape: {depth_shape}, colormap: {colormap}.",
        "Metric depth in meters available as `depth_meters` (float32 ndarray).",
    ]
    if image_ref:
        lines.append(
            f"Colorized depth map rendered below; refer to it as {image_ref}."
        )
    text = "\n".join(lines)

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
