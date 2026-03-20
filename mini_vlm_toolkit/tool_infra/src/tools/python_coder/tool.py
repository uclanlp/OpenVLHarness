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
)

TOOL_NAME = "Python_Code_Generator_Tool"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Task description: describe the task to generate and run code for with the corresponding input"
            ),
        },
        "context_bindings": {
            "type": "object",
            "description": (
                "Optional JSON-serializable runtime bindings injected before executing "
                "the generated code. "
                "Reserved for runtime. Do not pass in anything."
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
1. Restricted primarily to basic Python arithmetic operations and built-in mathematical functions.
2. Should not use any external libraries or modules.
3. Limited to simple mathematical calculations and problems.
4. Should not perform string processing, data structure manipulation, or complex algorithms.
5. Should not access system resources, file operations, or network requests.
6. Should not use 'import' statements.
7. All calculations must be self-contained within a single function or script.
8. The query should fully describe the meaning of any values the generated code needs.
9. Output should be kept to a single numerical result or a simple list/tuple of numbers.
10. DO NOT generate loop output.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Provide clear and specific queries that describe the desired mathematical calculation.
2. Include all necessary numerical inputs and semantic context directly in the query string.
3. Keep tasks focused on basic arithmetic, algebraic calculations, or simple mathematical algorithms.
4. Ensure the query explains what each required runtime value means if the harness has already injected it.
5. Verify that the query mainly involves mathematical operations and does not require external I/O or broad general-purpose programming.
6. Review generated code to ensure it mainly uses basic Python arithmetic operations and built-in math functions.
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
                "Generate Python code from a math-focused query, execute it, and "
                "return stdout plus variables. Use this tool for arithmetic and "
                "lightweight math reasoning when the user provides all needed task "
                "meaning in the query. Runtime-only bindings may already be injected "
                "by the harness, but agents should not populate that field themselves. "
                "Do not use it for untrusted tasks, external I/O, or broad "
                "general-purpose programming."
            ),
            tool_version="0.2.0",
            input_types={
                "query": (
                    "str - math-focused task description with all required inputs and task meaning described in the query"
                ),
                "context_bindings": (
                    "dict - runtime-only JSON-serializable bindings injected by the harness before execution. Reserved for runtime. Do not pass in anything."
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
    ):
        task_description = (
            "Given a query, generate a concise Python code snippet that solves the task. "
            "The query fully describes the meaning of any runtime bindings that may already "
            "be available. If the query refers to those existing names, use them directly. "
            "Return only a single Python code block:\n\n"
            "```python\n"
            "<code snippet>\n"
            "```"
        )
        sections = [f"Task:\n{task_description}"]
        if prepared_context is not None:
            context_section = build_context_prompt_sections(prepared_context)
            if context_section:
                sections.append(context_section)
        sections.append(f"Query:\n{query}")
        return "\n\n".join(sections)

    def execute_code_snippet(
        self,
        code,
        context_bindings=None,
        timeout_seconds=10,
        prepared_context=None,
    ):
        if not code:
            return {"error": "No code to execute"}

        try:
            if prepared_context is None:
                prepared_context = prepare_execution_context(
                    context_bindings=context_bindings,
                )
            # Only the model-generated main snippet goes through the lightweight
            # exit/quit sanitization. Runtime bindings are harness-provided input.
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
        timeout_seconds=10,
        model=None,
        base_url=None,
        api_key=None,
    ):
        if not query:
            raise ValueError("query is required")

        prepared_context = prepare_execution_context(
            context_bindings=context_bindings,
        )

        engine = self.llm_engine
        if model or base_url or api_key:
            engine = self._init_llm_engine(model=model, base_url=base_url, api_key=api_key)

        prompt = self._build_prompt(
            query,
            prepared_context=prepared_context,
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
