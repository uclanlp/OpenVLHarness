import copy as cp
import json
from typing import Any, Dict, List, Optional, Tuple


class PersistentStateMemory:
    """Base persistent state memory.

    Default behavior emits a text-only `role=user` message for each tool
    payload. Subclasses can override image/modality handling and turn-level
    state update emissions.

    Contract:
    - Instances are per-run only and must not be reused across runs.
    - Any mutable fields on subclasses are run-local.
    """

    memory_name = "base"

    def __init__(self, **kwargs):
        self.options = dict(kwargs)

    def reset(self) -> None:
        """Reset per-run state."""
        return None

    def configure_tool_schema(self, json_schema: Any) -> None:
        """Optional schema augmentation hook at agent init."""
        return None

    def on_run_start(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return extra messages before first model call for this run."""
        return []

    def emit_after_tool_turn(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return turn-level update messages after all tool calls in a turn."""
        return []

    def resolve_tool_call_args(
        self,
        *,
        agent: Any,
        args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Rewrite tool args before /execute (default: no-op)."""
        if not isinstance(args, dict):
            return {}, {"rewrites": [], "errors": []}
        return cp.deepcopy(args), {"rewrites": [], "errors": []}

    def prepare_tool_message_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare compact payload for text-only tool-result content."""
        compact_payload = {
            "ok": payload.get("ok"),
            "result": payload.get("result"),
            "error": payload.get("error"),
        }
        try:
            text_payload = json.dumps(compact_payload, ensure_ascii=False)
        except Exception:
            text_payload = str(compact_payload)
        return {"text_payload": text_payload}

    def build_tool_message_content(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build strict text-only tool-result content."""
        prepared = prepared_payload or self.prepare_tool_message_payload(payload)
        text_payload = prepared.get("text_payload")
        if isinstance(text_payload, str):
            return text_payload
        try:
            return json.dumps(text_payload, ensure_ascii=False)
        except Exception:
            return str(text_payload)

    def build_tool_message(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content = self.build_tool_message_content(payload, prepared_payload=prepared_payload)
        tool_name = str(payload.get("tool_name") or "tool")
        tool_call_id = str(payload.get("tool_call_id") or "unknown")
        return {
            "role": "user",
            "content": (
                f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:\n{content}"
            ),
        }

    def build_deferred_user_messages(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build user messages that should be appended after all tool messages."""
        return []

    def consume_tool_payload(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_call: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Consume one normalized payload and return emitted messages."""
        prepared_payload = self.prepare_tool_message_payload(payload)
        tool_message = self.build_tool_message(payload, prepared_payload=prepared_payload)
        deferred_messages = self.build_deferred_user_messages(payload, prepared_payload=prepared_payload)
        if not isinstance(deferred_messages, list):
            deferred_messages = []

        return {
            "prepared_payload": prepared_payload,
            "immediate_messages": [tool_message],
            "deferred_messages": deferred_messages,
        }
