import copy as cp
import json
from typing import Any, Dict, List, Optional, Tuple


class PersistentStateMemory:
    """Base persistent state memory.

    Default behavior emits one protocol `role=tool` message per tool payload.
    On success, it also queues a text-only `role=user` summary message to be
    appended after all tool messages in the turn. On failure, no user summary is
    emitted. Subclasses can override image/modality handling and turn-level
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
        tool_name: str = "",
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

    def build_protocol_tool_ack(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build strict OpenAI-style tool ack for transcript validity."""
        _ = prepared_payload  # reserved for subclasses that may need it
        return {
            "role": "tool",
            "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
            "name": str(payload.get("tool_name") or "tool"),
            "content": "ok",
        }

    def _is_success_payload(self, payload: Dict[str, Any]) -> bool:
        return isinstance(payload, dict) and payload.get("ok") is True

    def _build_failure_tool_content(self, payload: Dict[str, Any]) -> str:
        error = payload.get("error") if isinstance(payload, dict) else None
        code = ""
        message = ""
        if isinstance(error, dict):
            raw_code = error.get("code")
            raw_message = error.get("message")
            if isinstance(raw_code, str):
                code = raw_code.strip()
            elif raw_code is not None:
                code = str(raw_code).strip()
            if isinstance(raw_message, str):
                message = raw_message.strip()
            elif raw_message is not None:
                message = str(raw_message).strip()
        elif isinstance(error, str):
            message = error.strip()
        elif error not in (None, "", {}):
            message = str(error).strip()

        if not message:
            message = "Tool call failed."
        if code:
            return f"{code}: {message}"
        return message

    def build_failure_tool_message(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = prepared_payload
        return {
            "role": "tool",
            "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
            "name": str(payload.get("tool_name") or "tool"),
            "content": self._build_failure_tool_content(payload),
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
        del agent, turn_idx, tool_call  # reserved for subclasses
        prepared_payload = self.prepare_tool_message_payload(payload)

        if not self._is_success_payload(payload):
            return {
                "prepared_payload": prepared_payload,
                "immediate_messages": [
                    self.build_failure_tool_message(
                        payload,
                        prepared_payload=prepared_payload,
                    )
                ],
                "deferred_messages": [],
            }

        tool_message = self.build_tool_message(payload, prepared_payload=prepared_payload)
        deferred_messages = self.build_deferred_user_messages(payload, prepared_payload=prepared_payload)
        if not isinstance(deferred_messages, list):
            deferred_messages = []
        immediate_messages: List[Dict[str, Any]] = []
        if isinstance(tool_message, dict) and tool_message.get("role") == "tool":
            immediate_messages.append(tool_message)
        else:
            # NOTE: Strict OpenAI-native tool protocol requires a `role=tool`
            # message with matching `tool_call_id` after assistant `tool_calls`.
            immediate_messages.append(
                self.build_protocol_tool_ack(payload, prepared_payload=prepared_payload)
            )
            if isinstance(tool_message, dict):
                deferred_messages = [tool_message] + deferred_messages

        return {
            "prepared_payload": prepared_payload,
            "immediate_messages": immediate_messages,
            "deferred_messages": deferred_messages,
        }
