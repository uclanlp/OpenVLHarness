"""Tracking_Tool output post-processing for agent-visible messages."""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    del tool_metadata
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    tool_name = str(payload.get("tool_name") or "Tracking_Tool")
    status = result.get("status", "unknown")
    summary = str(result.get("summary") or "").strip()
    if not summary:
        summary = f"{tool_name} finished with status `{status}`."

    text = f"{tool_name} status: {status}. {summary}"
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

