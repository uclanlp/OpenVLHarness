"""Tool-specific output post-processing for agent-visible messages (stage 1 placeholder)."""

PROCESSOR_MODE = "default_placeholder"


def process_tool_output(payload, memory, tool_metadata=None):
    return memory.process_tool_output_default(
        payload=payload,
        tool_metadata=tool_metadata,
        processor_info={"mode": PROCESSOR_MODE},
    )
