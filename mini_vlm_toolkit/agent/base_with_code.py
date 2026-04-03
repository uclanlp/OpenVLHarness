"""Agent with orchestrator-side code execution for the coding agent tool.

``BaseAgentWithToolsAndCode`` extends ``BaseAgentWithTools`` to support the
``Python_Coding_Agent_Tool``: when the tool server returns generated code,
this agent executes it locally using the actual data bindings held in memory,
avoiding the need to send large payloads over HTTP.
"""

from typing import Any, Dict

from .base import BaseAgentWithTools
from .code_execution import (
    CodeExecutionTimeoutError,
    execute_python_code,
    prepare_execution_context,
)


class BaseAgentWithToolsAndCode(BaseAgentWithTools):
    """Extends BaseAgentWithTools with orchestrator-side code execution.

    When ``Python_Coding_Agent_Tool`` returns ``{"generated_code": "..."}``
    from the tool server, this agent intercepts the payload, executes the
    code locally with actual binding values from ``run_memory``, and merges
    the execution results back into the tool response before it reaches
    memory consumption.
    """

    _CODING_AGENT_TOOL_NAME = "Python_Coding_Agent_Tool"
    _CODE_EXECUTION_TIMEOUT = 30  # seconds

    def _post_process_tool_payload(
        self,
        normalized_payload: Dict[str, Any],
        tool_call: Dict[str, Any],
        run_memory,
    ) -> Dict[str, Any]:
        """Execute generated code from the coding agent tool locally."""
        tool_name = normalized_payload.get("tool_name", "")
        if tool_name != self._CODING_AGENT_TOOL_NAME:
            return normalized_payload

        if not normalized_payload.get("ok"):
            return normalized_payload

        result = normalized_payload.get("result")
        if not isinstance(result, dict):
            return normalized_payload

        generated_code = result.get("generated_code")
        if not generated_code:
            return normalized_payload

        # Get actual binding values from memory
        bindings = {}
        if hasattr(run_memory, "_tool_result_bindings"):
            bindings = dict(run_memory._tool_result_bindings)

        try:
            prepared = prepare_execution_context(bindings)
            exec_result = execute_python_code(
                code=generated_code,
                prepared_context=prepared,
                timeout_seconds=self._CODE_EXECUTION_TIMEOUT,
            )
        except CodeExecutionTimeoutError:
            normalized_payload["ok"] = False
            normalized_payload["error"] = {
                "code": "code_execution_timeout",
                "message": (
                    f"Generated code timed out after "
                    f"{self._CODE_EXECUTION_TIMEOUT}s"
                ),
            }
            normalized_payload["result"]["execution_error"] = "timeout"
            return normalized_payload
        except Exception as exc:
            normalized_payload["ok"] = False
            normalized_payload["error"] = {
                "code": "code_execution_error",
                "message": str(exc),
            }
            normalized_payload["result"]["execution_error"] = str(exc)
            return normalized_payload

        # Merge execution results into the payload
        result["printed_output"] = exec_result.get("printed_output", "")
        result["variables"] = exec_result.get("variables", {})
        result["execution_code"] = exec_result.get("execution_code", "")
        normalized_payload["result"] = result
        return normalized_payload
