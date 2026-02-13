from typing import Any, Dict, List, Optional, Tuple


class RuntimeEnvironmentHook:
    """Hook interface for injecting runtime environment updates into chat messages."""

    hook_name = "base"

    def reset(self) -> None:
        """Reset per-run state."""
        return None

    def after_initial_user_message(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return extra messages after initial user input is prepared."""
        return []

    def after_tool_turn(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return extra messages after a tool-call turn completes."""
        return []


class ImageEnvironment(RuntimeEnvironmentHook):
    """Runtime environment hook that reports current image-registry status."""

    hook_name = "image"
    _ALLOWED_EMIT_POLICIES = ("always", "on_change")

    def __init__(self, emit_policy: str = "always"):
        policy = str(emit_policy).strip().lower()
        if policy not in self._ALLOWED_EMIT_POLICIES:
            raise ValueError(
                f"Invalid image environment emit_policy '{emit_policy}'. "
                f"Expected one of: {self._ALLOWED_EMIT_POLICIES}."
            )
        self.emit_policy = policy
        self._last_snapshot: Optional[Tuple[Tuple[str, str], ...]] = None

    def reset(self) -> None:
        self._last_snapshot = None

    def _snapshot(self, agent: Any) -> Tuple[Tuple[str, str], ...]:
        registry = getattr(agent, "_image_registry", {}) or {}
        rows = []
        for ref_id in sorted(registry.keys()):
            entry = registry.get(ref_id) or {}
            rows.append((ref_id, str(entry.get("source", "runtime"))))
        return tuple(rows)

    def _should_emit(self, snapshot: Tuple[Tuple[str, str], ...]) -> bool:
        if self.emit_policy == "always":
            return True
        return snapshot != self._last_snapshot

    def _build_environment_message(self, snapshot: Tuple[Tuple[str, str], ...]) -> Dict[str, Any]:
        lines = [
            "Environment Update (not a new question):",
            "Currently available image IDs for tool calls:",
        ]
        if snapshot:
            for ref_id, source in snapshot:
                lines.append(f"- {ref_id}: {source}")
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "When a tool needs image input, pass one of these IDs via the *_image_path field.",
                "Do NOT invent image file names or paths.",
            ]
        )
        return {"role": "user", "content": "\n".join(lines)}

    def _emit(self, *, agent: Any) -> List[Dict[str, Any]]:
        snapshot = self._snapshot(agent)
        should_emit = self._should_emit(snapshot)
        self._last_snapshot = snapshot
        if not should_emit:
            return []
        return [self._build_environment_message(snapshot)]

    def after_initial_user_message(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._emit(agent=agent)

    def after_tool_turn(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._emit(agent=agent)
