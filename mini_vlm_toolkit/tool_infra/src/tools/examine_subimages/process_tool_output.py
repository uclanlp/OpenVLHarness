"""Examine_Subimages-specific output post-processing for agent-visible messages.

Formats the aggregated yes/no summary into a single text line the agent can
read directly, rather than letting the orchestrator dump a raw JSON dict.
"""

import json

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    result = payload.get("result")
    if not isinstance(result, dict):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    if "error" in result:
        text = f"Examine_Subimages_Tool error: {result['error']}"
    else:
        summary = {
            "yes_count": result.get("yes_count", 0),
            "no_count": result.get("no_count", 0),
            "yes_objects": result.get("yes_objects", []),
            "no_objects": result.get("no_objects", []),
            "yes_boxes": result.get("yes_boxes", []),
            "no_boxes": result.get("no_boxes", []),
        }
        text = (
            "Here is the result summary provided by the "
            f"Examine_Subimages_Tool: {json.dumps(summary)}"
        )

    return {
        "user_message": {"role": "user", "content": text},
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }
