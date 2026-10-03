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


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Python_Coding_Agent_Tool',
              'description': 'Write and run Python code to perform computations over results from '
                             'other tools (bounding boxes, depth maps, OCR text, etc.). Use this '
                             'tool when you need to combine, filter, measure, or calculate across '
                             'outputs from previous tool calls — for example, computing distances '
                             'using depth at detected object locations. Describe the computation '
                             'goal in plain language as the objective; all available data from '
                             'prior tools is automatically provided.',
              'parameters': {'type': 'object',
                             'properties': {'objective': {'type': 'string',
                                                          'description': 'Plain-language '
                                                                         'description of the '
                                                                         'computation to perform. '
                                                                         'Refer to the data you '
                                                                         'need by what it '
                                                                         "represents (e.g. 'the "
                                                                         'detected bounding '
                                                                         "boxes', 'the depth "
                                                                         "map'), not by variable "
                                                                         'names. Examples: '
                                                                         "'Compute the average "
                                                                         'depth at the center of '
                                                                         'each detected person '
                                                                         "bounding box', "}},
                             'required': ['objective'],
                             'additionalProperties': False}}}

from typing import TypedDict


class Output(TypedDict):
    generated_code: str | None
    printed_output: str
    variables: dict[str, str]
    execution_code: str | None
    execution_error: str | None
    images: list[str] | None


class Capability(BaseCapability):
    name = 'Python_Coding_Agent_Tool'
    backends = {'python_coding_agent': 'Python_Coding_Agent_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def translate_args(self, args, memory):
        if memory.state.bindings.descriptors and "tool_context" in memory.modules:
            args["binding_descriptors"] = memory.describe_bindings()
        return args

    def execute(self, call, args, memory):
        from ..services.tools.python_executor_local.executor import (
            prepare_execution_context, execute_python_code, CodeExecutionTimeoutError)
        payload = super().execute(call, args, memory)
        result = payload.get("result")
        if payload.get("ok") is True and isinstance(result, dict) and result.get("generated_code"):
            timeout = self.service_config.get("execution_timeout_seconds", 30)
            try:
                execution = execute_python_code(result["generated_code"],
                    prepare_execution_context(memory.copy_bindings_for_execution()), timeout_seconds=timeout)
                result.update(execution)
            except CodeExecutionTimeoutError:
                result["execution_error"] = f"Code execution timed out after {timeout}s"
            except Exception as exc:
                result["execution_error"] = str(exc)
        return payload

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        result = result if isinstance(result, dict) else {}
        return {"generated_code": result.get("generated_code"), "printed_output": result.get("printed_output", ""),
            "variables": result.get("variables", {}), "execution_code": result.get("execution_code"),
            "execution_error": result.get("execution_error"), "images": prepared["image_refs"] or None}
