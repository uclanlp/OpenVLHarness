import os
import re
import signal
import threading
from contextlib import contextmanager
from io import StringIO
import contextlib

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool

TOOL_NAME = "Python_Code_Generator_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Math-focused task description with all required inputs",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Execution timeout in seconds",
            "minimum": 1,
        },
        "model": {"type": "string", "description": "Override LLM model name"},
        "base_url": {"type": "string", "description": "Override LLM base URL"},
        "api_key": {"type": "string", "description": "Override LLM API key"},
    },
    "required": ["query"],
    "additionalProperties": False,
}

LIMITATION = f"""
The {TOOL_NAME} has several limitations:
1. Restricted to basic Python arithmetic operations and built-in mathematical functions.
2. Cannot use any external libraries or modules, including those in the Python standard library.
3. Limited to simple mathematical calculations and problems.
4. Cannot perform any string processing, data structure manipulation, or complex algorithms.
5. No access to any system resources, file operations, or network requests.
6. Cannot use 'import' statements.
7. All calculations must be self-contained within a single function or script.
8. Input must be provided directly in the query string.
9. Output is limited to numerical results or simple lists/tuples of numbers.
10. Output should be kept to a single numerical result or a simple list/tuple of numbers.
11. DO NOT generate loop output.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide clear and specific queries that describe the desired mathematical calculation.
2. Include all necessary numerical inputs directly in the query string.
3. Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.
4. Ensure all required numerical data is included in the query.
5. Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms.
6. Review generated code to ensure it only uses basic Python arithmetic operations and built-in math functions.
"""


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    if not seconds or seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
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


def extract_python_code(text):
    match = re.search(
        r"```\s*(?:python|py)\b\s*(.*?)\s*```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def truncate_string(text, max_length):
    if text is None:
        return ""
    if len(text) <= max_length:
        return text

    marker = " ... (truncated: middle content omitted) ... "
    if max_length <= len(marker) + 2:
        return text[:max_length]

    head_size = (max_length - len(marker)) // 2
    tail_size = max_length - len(marker) - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:]}"


def safe_repr(value, max_length=2000):
    try:
        representation = repr(value)
    except Exception as exc:
        return f"<repr_error {type(value).__name__}: {exc}>"
    if not isinstance(representation, str):
        return f"<repr_error {type(value).__name__}: non-string repr>"
    return truncate_string(representation, max_length)


def sanitize_code(code):
    for func in ("exit", "quit", "sys.exit"):
        pattern = rf"{re.escape(func)}\s*\([^)]*\)"
        code = re.sub(pattern, "None", code)
    return code


class Python_Coder_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Generate Python code from a math-focused query, execute it, and "
                "return stdout plus variables. Use this tool for arithmetic and "
                "lightweight math reasoning when the user provides all needed inputs. "
                "Do not use it for untrusted tasks, external I/O, or broad "
                "general-purpose programming."
            ),
            tool_version="0.1.0",
            input_types={
                "query": (
                    "str - math-focused task description with required numeric inputs"
                )
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - printed_output, variables, execution_code",
            demo_commands=[
                {
                    "command": 'Python_Code_Generator_Tool(query="Find the sum of prime numbers up to 50")',
                    "description": "Generate a Python code snippet to find the sum of prime numbers up to 50.",
                },
                {
                    "command": (
                        'query="Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '
                        'calculate the sum of squares of odd numbers"\n'
                        "Python_Code_Generator_Tool(query=query)"
                    ),
                    "description": (
                        "Generate a Python function for a specific mathematical operation"
                        " on a given list of numbers."
                    ),
                },
            ],
            user_metadata={
                "limitations": LIMITATION,
                "best_practices": BEST_PRACTICE,
            },
        )
        self.llm_engine = self._init_llm_engine()

    def _init_llm_engine(self, model=None, base_url=None, api_key=None):
        resolved_model = model or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
        resolved_api_key = api_key or os.getenv("LLM_API_KEY") or DEFAULT_LLM_API_KEY
        return LLMEngine(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            temperature=0.0,
            top_p=1.0,
        )

    def _build_prompt(self, query):
        task_description = (
            "Given a query, generate a Python code snippet that performs the specified "
            "operation on the provided data. Think step by step and keep the code simple. "
            "Ensure the final result is printed with a short descriptive label. "
            "Return only a single Python code block:\n\n"
            "```python\n"
            "<code snippet>\n"
            "```"
        )
        return f"Task:\n{task_description}\n\nQuery:\n{query}"

    def execute_code_snippet(
        self,
        code,
        timeout_seconds=10,
        max_output_length=7000,
        max_var_length=2000,
        max_vars=20,
    ):
        if not code:
            return {"error": "No code to execute"}

        exec_code = extract_python_code(code)
        exec_code = sanitize_code(exec_code)
        local_vars = {}
        stdout_buffer = StringIO()

        try:
            with timeout(timeout_seconds):
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(exec_code, {}, local_vars)
        except TimeoutError:
            return {"error": f"Execution timed out after {timeout_seconds} seconds"}
        except Exception as exc:
            return {"error": str(exc)}

        printed_output = truncate_string(stdout_buffer.getvalue().strip(), max_output_length)
        variables = {}
        for key, value in local_vars.items():
            if key.startswith("__"):
                continue
            if len(variables) >= max_vars:
                variables["__truncated__"] = (
                    f"... ({len(local_vars) - len(variables)} more variables omitted)"
                )
                break
            variables[key] = safe_repr(value, max_var_length)

        return {
            "printed_output": printed_output,
            "variables": variables,
            "execution_code": exec_code,
        }

    def execute(self, query, timeout_seconds=10, model=None, base_url=None, api_key=None):
        if not query:
            raise ValueError("query is required")

        engine = self.llm_engine
        if model or base_url or api_key:
            engine = self._init_llm_engine(model=model, base_url=base_url, api_key=api_key)

        prompt = self._build_prompt(query)
        try:
            response = engine.generate(prompt, temperature=0.0, top_p=1.0)
        except Exception as exc:
            return {"error": f"LLM generation failed: {exc}"}

        return self.execute_code_snippet(response, timeout_seconds=timeout_seconds)
