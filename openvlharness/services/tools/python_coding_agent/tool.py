"""Generate Python code for calculations over described tool-result bindings.

Python_Coding_Agent_Tool uses a fresh conversation with the configured model
and returns extracted code. Execution belongs to the calling capability.
"""


from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.python_coding_agent.catalog import build_rich_data_catalog
from openvlharness.services.tools.python_executor_local.executor import extract_python_code

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
    """Generate Python code from an objective and a catalog of available bindings."""
    require_llm_engine = False

    def __init__(self, model_adapter, inherit_options=False):
        """Attach the orchestrator-matched model adapter and initialize tool metadata.

        inherit_options sends code-generation requests with the adapter's configured
        options instead of the fixed 4,096-token, medium-effort request.
        """
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
        self.model_adapter = model_adapter
        self.inherit_options = inherit_options

    def _build_prompt(self, objective, binding_descriptors=None):
        """Build the user message for the coding LLM."""
        sections = []

        catalog = build_rich_data_catalog(binding_descriptors or {})
        if catalog:
            sections.append(f"Data Catalog:\n{catalog}")

        sections.append(f"Objective:\n{objective}")
        return "\n\n".join(sections)

    def execute(self, objective, binding_descriptors=None):
        """Generate code in a fresh conversation without executing it.

        Args:
            objective: Nonempty natural-language computation goal.
            binding_descriptors: Optional mapping describing available Python values.

        Returns:
            A mapping with generated_code on success, or error and optional raw_response
            when generation fails or returns no code. The capability executes the code.

        Raises:
            ValueError: The objective is empty.
        """
        if not objective:
            raise ToolArgumentError("objective is required")

        user_prompt = self._build_prompt(objective, binding_descriptors)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content = self.model_adapter.generate_text(messages, max_tokens=4096,
                                                       inherit_options=self.inherit_options)
        except Exception as exc:
            return {"error": f"LLM request failed: {exc}"}

        generated_code = extract_python_code(content)
        if not generated_code:
            return {"error": "LLM returned empty code", "raw_response": content}

        return {"generated_code": generated_code}
