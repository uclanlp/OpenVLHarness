"""Tool-specific output post-processing for Python_Coding_Agent_Tool.

Formats the generated code for the agent's message history.
"""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    if not payload or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    result = payload.get("result", {})
    if not isinstance(result, dict):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    generated_code = result.get("generated_code", "")
    printed_output = result.get("printed_output", "")
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

    text = "\n\n".join(text_parts)

    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

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
