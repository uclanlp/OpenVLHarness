"""Tool-specific output post-processing for Python_Coding_Agent_Tool.

Formats the generated code for the agent's message history.
"""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, memory, tool_metadata=None):
    if not payload or not payload.get("ok"):
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={"mode": PROCESSOR_MODE, "status": "fallback"},
        )

    result = payload.get("result", {})
    if not isinstance(result, dict):
        return memory.process_tool_output_default(
            payload=payload,
            tool_metadata=tool_metadata,
            processor_info={"mode": PROCESSOR_MODE, "status": "fallback"},
        )

    generated_code = result.get("generated_code", "")
    printed_output = result.get("printed_output", "")
    # variables = result.get("variables", {})
    execution_error = result.get("execution_error", "")

    text_parts = []

    if generated_code:
        text_parts.append(
            f"Coding agent generated and executed the following Python code:\n"
            f"```python\n{generated_code}\n```\n\n"
        )
    else:
        text_parts.append("(No code was generated.)")

    if execution_error:
        text_parts.append(f"Execution error: {execution_error}")
    elif printed_output:
        text_parts.append(f"Execution output:\n{printed_output}")
    elif generated_code and not execution_error:
        text_parts.append("(Code executed successfully but produced no printed output.)")

    # if variables:
    #     # Show non-trivial computed variables
    #     var_lines = [f"  {k} = {v}" for k, v in variables.items()
    #                  if not k.startswith("_")]
    #     if var_lines:
    #         text_parts.append("Computed variables:\n" + "\n".join(var_lines))

    return memory.build_processed_tool_output(
        payload=payload,
        prepared_payload=memory.prepare_tool_message_payload(payload),
        text="\n\n".join(text_parts),
        processor_info={"mode": PROCESSOR_MODE, "status": "ok"},
    )
