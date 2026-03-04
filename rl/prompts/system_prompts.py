"""System prompt utilities for Hermes-style tool use training."""

from __future__ import annotations

import json
from typing import Iterable


def render_tools_block(tool_schemas: Iterable[dict]) -> str:
    """Render tool schemas into a compact prompt section."""
    blocks = []
    for schema in tool_schemas or []:
        try:
            blocks.append(json.dumps(schema, ensure_ascii=False))
        except Exception:
            blocks.append(str(schema))
    return "\n".join(blocks)


def build_hermes_system_prompt(tool_schemas: Iterable[dict] | None = None) -> str:
    """Build the Hermes `<tool_call>` instruction prompt.

    The prompt keeps the same reasoning style as the base agent while changing
    the action channel to Hermes tool call tags.
    """

    tools_block = render_tools_block(tool_schemas or [])
    tools_text = ""
    if tools_block:
        tools_text = (
            "\n\n# Tools\n"
            "You can call tools using the schemas below.\n"
            f"{tools_block}\n"
        )

    return (
        "You are a visual reasoning agent that answers questions about images.\n"
        "Use a Perception-Reasoning-Action process and be precise.\n"
        "You may call at most one tool per assistant turn.\n"
        "\n"
        "When you need a tool, output exactly one Hermes tool call block in this format:\n"
        '<tool_call>{"name":"tool_name","arguments":{"arg":"value"}}</tool_call>\n'
        "\n"
        "Rules:\n"
        "- Do not emit OpenAI tool_calls JSON objects directly in plain text.\n"
        "- Keep arguments valid JSON.\n"
        "- If tools need image input, use runtime image ids (for example image_1) in *_image_path fields.\n"
        "- If tool output is uninformative or errors, call a better tool or retry with corrected args.\n"
        "\n"
        "When you are not calling a tool, structure your response as:\n"
        "<perception>...</perception>\n"
        "<thinking>...</thinking>\n"
        "Then provide the final answer in plain text after </thinking>."
        f"{tools_text}"
    )
