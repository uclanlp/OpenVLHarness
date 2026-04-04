"""Python Coding Agent Tool -- code generation only, no execution.

Receives a high-level code objective and lightweight binding descriptors
from the orchestrator.  A fresh-context coding LLM generates Python code
that references the described variables.  The generated code is returned
to the orchestrator which executes it locally with actual values from
memory (orchestrator-side execution pattern).

Configuration (env vars):
    CODING_AGENT_BASE_URL  - vLLM / OpenAI-compatible server URL
                             (e.g. "http://localhost:8004/v1")
    CODING_AGENT_API_KEY   - API key (default: "local" for local vLLM)
    CODING_AGENT_MODEL     - Model name served by the endpoint
                             (e.g. "Qwen/Qwen3-8B")
"""

import os

from tools.api_utils import chat_completion_request
from tools.base import BaseTool
from tools.python_execution_common import (
    build_rich_data_catalog,
    extract_python_code,
)

TOOL_NAME = "Python_Coding_Agent_Tool"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {
            "type": "string",
            "description": (
                "Plain-language description of the computation to perform. "
                "Refer to the data you need by what it represents "
                "(e.g. 'the detected bounding boxes', 'the depth map'), "
                "not by variable names. Examples: "
                "'Compute the average depth at the center of each detected "
                "person bounding box', "
            ),
        },
    },
    "required": ["objective"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a precise Python code generator.  Your task is to write Python code
that achieves a given objective using pre-loaded data variables.

RULES:
1. All data variables listed in the Data Catalog are already in scope.
   Reference them directly (e.g., grounding_person["boxes"]).
   Do NOT redefine or reassign them.
2. Use print() to output any final answers or intermediate results.
3. Assign important computed values to named variables so they appear
   in the result namespace.
4. You may import: numpy, math, json, base64, collections, itertools,
   functools, statistics, PIL (Image, ImageDraw), io, struct, re.
5. Do NOT use: os, sys, subprocess, requests, or any network/filesystem
   operations.
6. Write clean, readable code.  Add brief comments for non-obvious logic.
7. Return ONLY a single ```python code block.
8. If the objective is ambiguous, implement the most reasonable
   interpretation and add a comment noting your assumption.
"""


class Python_Coding_Agent_Tool(BaseTool):
    require_llm_engine = False

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Write and run Python code to perform computations over results "
                "from other tools (bounding boxes, depth maps, OCR text, etc.). "
                "Use this tool when you need to combine, filter, measure, or "
                "calculate across outputs from previous tool calls — for example, "
                "computing distances using depth at detected object locations. "
                "Describe the computation goal in plain language as the objective; "
                "all available data from prior tools is automatically provided."
            ),
            tool_version="0.1.0",
            input_types={
                "objective": "str - high-level code objective",
                "binding_descriptors": (
                    "object - structured metadata for available data bindings. "
                    "Reserved for runtime injection. Do not pass in anything."
                ),
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - generated_code",
            demo_commands=[
                {
                    "command": (
                        'Python_Coding_Agent_Tool(objective="Compute the '
                        "average depth at the center of each detected person's "
                        'bounding box")'
                    ),
                    "description": (
                        "Cross-reference detection boxes with depth data "
                        "to get per-object depth measurements."
                    ),
                },
                {
                    "command": (
                        'Python_Coding_Agent_Tool(objective="Filter the '
                        "detected objects to only those with confidence above "
                        "0.8 and compute the total area of their bounding "
                        'boxes in pixels")'
                    ),
                    "description": (
                        "Filter and aggregate detection results using "
                        "score thresholds and geometric calculations."
                    ),
                },
            ],
        )
        self._base_url = os.getenv("CODING_AGENT_BASE_URL", "")
        self._base_url = self._get_endpoint(self._base_url) if self._base_url else ""
        self._api_key = os.getenv("CODING_AGENT_API_KEY", "empty")
        self._model = os.getenv("CODING_AGENT_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    def _get_endpoint(self, base_url):
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/chat/completions"
        return f"{trimmed}/v1/chat/completions"

    def _validate_config(self):
        if not self._base_url:
            raise ValueError(
                "CODING_AGENT_BASE_URL env var is not set. "
                "Point it at a vLLM or OpenAI-compatible server "
                "(e.g. http://localhost:8004/v1)."
            )
        if not self._model:
            raise ValueError(
                "CODING_AGENT_MODEL env var is not set. "
                "Set it to the model name served by the endpoint "
                "(e.g. Qwen/Qwen3-8B)."
            )

    def _build_prompt(self, objective, binding_descriptors=None):
        """Build the user message for the coding LLM."""
        sections = []

        catalog = build_rich_data_catalog(binding_descriptors or {})
        if catalog:
            sections.append(f"Data Catalog:\n{catalog}")

        sections.append(f"Objective:\n{objective}")
        return "\n\n".join(sections)

    def execute(self, objective, binding_descriptors=None):
        if not objective:
            raise ValueError("objective is required")

        self._validate_config()

        user_prompt = self._build_prompt(objective, binding_descriptors)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = chat_completion_request(
                model=self._model,
                messages=messages,
                base_url=self._base_url,
                api_key=self._api_key,
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as exc:
            return {"error": f"LLM request failed: {exc}"}

        content = ""
        choices = response.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""

        generated_code = extract_python_code(content)
        if not generated_code:
            return {"error": "LLM returned empty code", "raw_response": content}

        return {"generated_code": generated_code}
