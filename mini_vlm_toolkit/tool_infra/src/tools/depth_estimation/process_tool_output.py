PROCESSOR_MODE = "depth_estimation"


def process_tool_output(payload, memory, tool_metadata=None):
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={"mode": PROCESSOR_MODE, "status": "fallback"},
        )

    tool_name = str(payload.get("tool_name") or "tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    depth_shape = result.get("depth_shape", [])
    colormap = result.get("colormap", "unknown")
    model_id = result.get("model_id", "unknown")

    lines = [
        f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:",
        f"Depth estimation complete. Shape: {depth_shape}, colormap: {colormap}, model: {model_id}.",
        "Raw depth available as `depth_raw_b64` (float32, normalized 0-1).",
    ]

    return memory.build_processed_tool_output(
        payload=payload,
        text="\n".join(lines),
        processor_info={"mode": PROCESSOR_MODE, "status": "ok"},
    )
