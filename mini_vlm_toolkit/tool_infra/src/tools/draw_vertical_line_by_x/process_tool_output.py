"""Tool-specific output post-processing — default placeholder."""

PROCESSOR_MODE = "default_placeholder"


def process_tool_output(payload, prepared, tool_metadata=None):
    """Return None to delegate to the orchestrator's default tool-return
    message construction. Tools needing a custom message build a dict
    ``{"user_message": ..., "processor_info": ...}``.
    """
    return None
