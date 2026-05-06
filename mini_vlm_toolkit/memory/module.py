from typing import Any, Dict, List, Tuple

from .run_state import MemoryRunState


class MemoryModule:
    """Base class for composable memory modules.

    Each hook is a no-op by default; modules implement only what they need.
    All hooks receive the shared `state: MemoryRunState` and the orchestrating
    `agent` reference. Per-module config flags are read from `agent.persistent_state_memory.options`.

    Hook lifecycle:
    - on_run_start:             per run, before first model call.
                                Set up registries from inputs (e.g., input images).
    - before_tool_call:         per tool call, before /execute.
                                Rewrite args. Returns (args, {"rewrites":[],"errors":[]}).
    - consume_tool_result:      per tool call, chained after the tool returned.
                                Modules contribute to the shared `prepared` dict
                                (image extraction, URL rewriting, etc.) and write
                                side-effect state (binding accumulation). Returns
                                the (possibly mutated) prepared dict.
                                The orchestrator builds the user message and ack
                                from the final prepared — modules do NOT build
                                messages here.
    - build_env_update_message: per turn, after all tool calls in the turn are
                                consumed. Modules add lines to the shared
                                EnvironmentUpdateBuilder.
    """

    name: str = ""

    def on_run_start(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
        state: MemoryRunState,
    ) -> List[Dict[str, Any]]:
        return []

    def before_tool_call(
        self,
        *,
        agent: Any,
        tool_name: str,
        args: Dict[str, Any],
        state: MemoryRunState,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return args, {"rewrites": [], "errors": []}

    def consume_tool_result(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_call: Dict[str, Any],
        payload: Dict[str, Any],
        prepared: Dict[str, Any],
        state: MemoryRunState,
    ) -> Dict[str, Any]:
        return prepared

    def build_env_update_message(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        state: MemoryRunState,
    ) -> List[Dict[str, Any]]:
        return []


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
