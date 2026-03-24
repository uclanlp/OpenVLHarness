from tools.base import BaseTool
from tools.python_execution_common import (
    execute_python_code,
    prepare_execution_context,
)

TOOL_NAME = "Python_Executor_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute (raw code or fenced ```python/```py code block)",
        },
        "context_bindings": {
            "type": "object",
            "description": (
                "Optional JSON-serializable runtime bindings injected before executing code. "
                "Reserved for runtime. Do not pass in anything."
            ),
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


class Python_Executor_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Execute a provided Python code snippet and return stdout plus local variables. "
                "Important: if you need the numeric/text result explicitly, use print(...) and/or "
                "assign the value to a variable. Bare expressions are executed but their value is "
                "not returned automatically. "
                "Note: always include explicit import statements for any modules you use "
                "(e.g., `import math`, `import os`). No modules are pre-imported."
            ),
            tool_version="0.4.0",
            input_types={
                "code": (
                    "str - Python code to execute. Can be raw code or a fenced ```python/```py block. "
                    "Use print(...) and/or variable assignment if you need explicit values in the result."
                ),
                "context_bindings": (
                    "dict - optional JSON-serializable runtime bindings injected before execution. Reserved for runtime. Do not pass in anything."
                ),
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["python_executor", "code_interpreter"],
            output_type="dict - printed_output, variables, execution_code",
            demo_commands=[
                {
                    "command": 'Python_Executor_Tool(code="value = 2 + 2\\nprint(value)")',
                    "description": "Execute arithmetic and return both printed output and variable value."
                },
                {
                    "command": 'Python_Executor_Tool(code="for i in range(3):\\n    print(i)")',
                    "description": "Execute a short loop and return stdout."
                },
            ],
            user_metadata={
                "limitations": "No sandboxing. Code is executed as-is with exec()."
            },
        )

    def execute(self, code, context_bindings=None, timeout_seconds=5):
        if not code:
            raise ValueError("code is required")

        prepared_context = prepare_execution_context(
            context_bindings=context_bindings,
        )
        return execute_python_code(
            code=code,
            prepared_context=prepared_context,
            timeout_seconds=timeout_seconds,
        )
