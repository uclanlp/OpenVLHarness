PROCESSOR_MODE = "custom"


def process_tool_output(payload, memory, tool_metadata=None):
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={"mode": PROCESSOR_MODE, "status": "fallback"},
        )

    prepared = memory.prepare_tool_message_payload(payload)
    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None

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

    return memory.build_processed_tool_output(
        payload=payload,
        prepared_payload=prepared,
        text="\n".join(lines),
        processor_info={"mode": PROCESSOR_MODE, "status": "ok"},
    )