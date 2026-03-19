import os
import re

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool
from tools.python_execution_common import (
    TimeoutError,
    build_context_prompt_sections,
    execute_python_code,
    prepare_execution_context,
    validate_context_annotations,
)

TOOL_NAME = "Python_Code_Generator_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Task description for the Python code generator",
        },
        "context_bindings": {
            "type": "object",
            "description": (
                "Optional JSON-serializable runtime bindings injected before executing "
                "context_code and the generated code"
            ),
        },
        "context_code": {
            "type": "string",
            "description": (
                "Optional Python code executed before the generated code. Use it to "
                "define helper functions, classes, imports, or setup variables."
            ),
        },
        "context_annotations": {
            "type": "object",
            "description": (
                "Optional free-form prompt hints for names coming from context_bindings "
                "or context_code"
            ),
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
1. No sandboxing. Do not use it for untrusted tasks.
2. Best suited for lightweight Python reasoning and transformation tasks.
3. Prefer using provided context_bindings/context_code directly instead of recreating large constants or helpers.
4. Avoid unnecessary imports, external I/O, and network access.
5. Keep generated code short, direct, and easy to inspect.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide a clear task description.
2. Put reusable helper code in context_code when possible.
3. Put runtime values in context_bindings instead of asking the model to reconstruct them.
4. Use context_annotations when a binding is opaque or semantically important.
5. Review generated code before relying on it for important work.
"""


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
                "Generate Python code from a query, execute it, and "
                "return stdout plus variables. It can also use caller-provided "
                "runtime bindings and helper definitions through context_bindings "
                "and context_code. Do not use it for untrusted tasks, external I/O, "
                "or broad autonomous programming."
            ),
            tool_version="0.2.0",
            input_types={
                "query": (
                    "str - task description for the Python code generator"
                ),
                "context_bindings": (
                    "dict - optional JSON-serializable runtime bindings injected before execution"
                ),
                "context_code": (
                    "str - optional Python code executed before the generated code"
                ),
                "context_annotations": (
                    "dict[str, str] - optional free-form prompt hints for provided names"
                ),
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
                {
                    "command": (
                        'Python_Code_Generator_Tool('
                        'query="Use scale(x) and print the result.", '
                        'context_bindings={"x": 3}, '
                        'context_code="def scale(v):\\n    return v * 10", '
                        'context_annotations={"scale": "helper function already defined in context_code"}'
                        ")"
                    ),
                    "description": (
                        "Generate code that uses caller-provided runtime bindings and helper definitions."
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

    def _build_prompt(
        self,
        query,
        prepared_context=None,
        context_annotations=None,
    ):
        task_description = (
            "Given a query, generate a concise Python code snippet that solves the task. "
            "Use the available runtime context directly when it is relevant. "
            "Return only a single Python code block:\n\n"
            "```python\n"
            "<code snippet>\n"
            "```"
        )
        sections = [f"Task:\n{task_description}"]
        if prepared_context is not None:
            context_section = build_context_prompt_sections(
                prepared_context,
                context_annotations=context_annotations,
            )
            if context_section:
                sections.append(context_section)
        sections.append(f"Query:\n{query}")
        return "\n\n".join(sections)

    def execute_code_snippet(
        self,
        code,
        context_bindings=None,
        context_code=None,
        timeout_seconds=10,
        prepared_context=None,
    ):
        if not code:
            return {"error": "No code to execute"}

        try:
            if prepared_context is None:
                prepared_context = prepare_execution_context(
                    context_bindings=context_bindings,
                    context_code=context_code,
                )
            # Keep context_code semantics aligned with Python_Executor_Tool: caller-provided
            # prelude code is executed as-is, while only the model-generated main snippet
            # goes through the lightweight exit/quit sanitization.
            return execute_python_code(
                code=code,
                prepared_context=prepared_context,
                timeout_seconds=timeout_seconds,
                sanitize_main_code=sanitize_code,
            )
        except TimeoutError:
            return {"error": f"Execution timed out after {timeout_seconds} seconds"}
        except Exception as exc:
            return {"error": str(exc)}

    def execute(
        self,
        query,
        context_bindings=None,
        context_code=None,
        context_annotations=None,
        timeout_seconds=10,
        model=None,
        base_url=None,
        api_key=None,
    ):
        if not query:
            raise ValueError("query is required")

        prepared_context = prepare_execution_context(
            context_bindings=context_bindings,
            context_code=context_code,
        )
        validated_annotations = validate_context_annotations(context_annotations)

        engine = self.llm_engine
        if model or base_url or api_key:
            engine = self._init_llm_engine(model=model, base_url=base_url, api_key=api_key)

        prompt = self._build_prompt(
            query,
            prepared_context=prepared_context,
            context_annotations=validated_annotations,
        )
        try:
            response = engine.generate(prompt, temperature=0.0, top_p=1.0)
        except Exception as exc:
            return {"error": f"LLM generation failed: {exc}"}

        return self.execute_code_snippet(
            response,
            timeout_seconds=timeout_seconds,
            prepared_context=prepared_context,
        )
