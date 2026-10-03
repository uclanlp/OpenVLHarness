from typing import Any, Dict, List
from .run_state import MemoryRunState
from .images import _snapshot
from .bindings import _summarize_binding
from .urls import _WEBPAGE_URL_REF_PATTERN

_ENV_UPDATE_PREAMBLE = "Environment Update (not a new question):"

class EnvironmentUpdateBuilder:
    """Per-turn accumulator for environment-update sections contributed by modules.

    Lives on `state.scratch["env_update"]`. The orchestrator resets it before each
    `on_run_start` / `emit_after_tool_turn` pass and reads it after, coalescing
    all contributed sections into a single `role=user` Environment Update message.

    Each section is a block of lines. Sections are separated by a blank line.
    A leading `_ENV_UPDATE_PREAMBLE` header is prepended automatically.
    """

    SCRATCH_KEY = "env_update"

    def __init__(self) -> None:
        self.sections: List[List[str]] = []

    def add_section(self, lines: List[str]) -> None:
        if not lines:
            return
        self.sections.append(list(lines))

    def is_empty(self) -> bool:
        return not self.sections

    def build_message(self) -> Dict[str, Any]:
        body: List[str] = [_ENV_UPDATE_PREAMBLE]
        for i, section_lines in enumerate(self.sections):
            if i > 0:
                body.append("")
            body.extend(section_lines)
        return {"role": "user", "content": "\n".join(body)}

    @classmethod
    def get(cls, state: MemoryRunState) -> "EnvironmentUpdateBuilder":
        slot = state.scratch.get(cls.SCRATCH_KEY)
        if not isinstance(slot, EnvironmentUpdateBuilder):
            slot = cls()
            state.scratch[cls.SCRATCH_KEY] = slot
        return slot

    @classmethod
    def reset(cls, state: MemoryRunState) -> "EnvironmentUpdateBuilder":
        builder = cls()
        state.scratch[cls.SCRATCH_KEY] = builder
        return builder

def add_image_inventory(state):
    snapshot = _snapshot(state.images)
    builder = EnvironmentUpdateBuilder.get(state)
    lines: List[str] = ["Currently available image IDs for tool calls:"]
    if snapshot:
        for ref_id, source in snapshot:
            lines.append(f"- {ref_id}: {source}")
        lines.append(
            "When a tool needs image input, pass one of these IDs via the *_image_id field."
        )
        lines.append("Do NOT invent image file names or paths.")
    else:
        lines.append("- (none)")
        lines.append(
            "No images are available, but you can still use text-based tools "
            "(e.g. Text_Search_Tool, Webpage_Visit_Tool) to answer the question."
        )
    builder.add_section(lines)
    return []

def add_binding_inventory(state):
    if not state.bindings.bindings:
        return []
    builder = EnvironmentUpdateBuilder.get(state)
    lines = ["Available data bindings (auto-injected into Python_Coding_Agent_Tool):"]
    for key, value in state.bindings.bindings.items():
        summary = _summarize_binding(value)
        lines.append(f"- {key}: {summary}")
    builder.add_section(lines)
    return []

def add_url_inventory(state, webpage_env_update_limit=20):
    registry = state.urls.webpage_registry
    if not registry:
        return []
    entries = sorted(
        registry.items(),
        key=lambda kv: int(_WEBPAGE_URL_REF_PATTERN.match(kv[0]).group(1)),
    )
    if webpage_env_update_limit is not None:
        entries = entries[-webpage_env_update_limit:]
    lines: List[str] = ["Currently registered webpage URL refs:"]
    for ref_id, entry in entries:
        source = entry.get("source", "")
        domain = entry.get("domain", "")
        if domain:
            lines.append(f"- {ref_id}: {source} ({domain})")
        else:
            lines.append(f"- {ref_id}: {source}")
    lines.append(
        "When you want to visit a webpage, pass a webpage_url_N to Webpage_Visit_Tool.urls."
    )
    builder = EnvironmentUpdateBuilder.get(state)
    builder.add_section(lines)
    return []
