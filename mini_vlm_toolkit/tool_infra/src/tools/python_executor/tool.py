import re
import signal
import threading
from contextlib import contextmanager
from io import StringIO
import contextlib

from tools.base import BaseTool

TOOL_NAME = "Python_Executor_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "Python code to execute (raw code or fenced ```python/```py code block)",
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    if not seconds or seconds <= 0:
        yield
        return

    if threading.current_thread() is not threading.main_thread():
        # Signal-based timeouts only work on the main thread.
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("Execution timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_code(text):
    match = re.search(
        r"```\s*(?:python|py)\b\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def safe_repr(value):
    try:
        representation = repr(value)
    except Exception as exc:
        return f"<repr_error {type(value).__name__}: {exc}>"
    if not isinstance(representation, str):
        return f"<repr_error {type(value).__name__}: non-string repr>"
    return representation


class Python_Executor_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Execute a provided Python code snippet and return stdout plus local variables. "
                "Important: if you need the numeric/text result explicitly, use print(...) and/or "
                "assign the value to a variable. Bare expressions are executed but their value is "
                "not returned automatically. Do not use with untrusted code. "
                "Note: always include explicit import statements for any modules you use "
                "(e.g., `import math`, `import os`). No modules are pre-imported."
            ),
            tool_version="0.3.1",
            input_types={
                "code": (
                    "str - Python code to execute. Can be raw code or a fenced ```python/```py block. "
                    "Use print(...) and/or variable assignment if you need explicit values in the result."
                )
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
                }
            ],
            user_metadata={
                "limitations": "No sandboxing. Code is executed as-is with exec()."
            },
        )

    def execute(self, code, timeout_seconds=5):
        if not code:
            raise ValueError("code is required")

        exec_code = extract_code(code)
        local_vars = {}
        stdout_buffer = StringIO()

        with timeout(timeout_seconds):
            with contextlib.redirect_stdout(stdout_buffer):
                exec(exec_code, {}, local_vars)

        printed_output = stdout_buffer.getvalue().strip()
        variables = {
            k: safe_repr(v)
            for k, v in local_vars.items()
            if not k.startswith("__")
        }

        return {
            "printed_output": printed_output,
            "variables": variables,
            "execution_code": exec_code,
        }
