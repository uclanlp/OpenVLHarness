import copy as cp
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .module import EnvironmentUpdateBuilder, MemoryModule
from .modules import ImageModule, ToolContextModule, UrlModule
from .modules.image import _snapshot as _image_snapshot
from .modules.tool_context import get_binding_descriptors_payload
from .run_state import MemoryRunState, ToolOutputsState

_PRESET_NAME_ALIAS = {
    "base": "base",
    "image": "image",
    "tool_context": "tool_context",
    "url": "url",
}

# Per-tool process_tool_output module overrides for cases where name-based
# inference would resolve to the wrong directory (legacy public tool name vs.
# implementation directory).
_TOOL_OUTPUT_PROCESSOR_MODULE_OVERRIDES = {
    "Visual_Grounding_Tool": "tools.sam3_segment.process_tool_output",
}


class PersistentStateMemory:
    """Composable memory orchestrator.

    Owns the shared `MemoryRunState` and a list of `MemoryModule` instances.
    Per-call flow (success path):
      1. modules' `consume_tool_result` chain mutate the prepared dict and
         write side-effect state (binding accumulation, image registry, etc.).
      2. orchestrator builds the role=tool ack via `build_protocol_tool_ack`.
      3. orchestrator calls `_build_tool_return_message`, which looks up the
         per-tool `process_tool_output` plug-in and calls it with
         `(payload, prepared, tool_metadata=...)` — no `memory` parameter.
         If the plug-in returns None or doesn't exist, orchestrator builds
         the default placeholder message.

    Per-row eval sidecar (`task_id`, `tool_detections`, `tool_call_counter`)
    lives directly on this class — it's populated by tool postprocessors via
    `record_tool_detection(...)` and read by the benchmark evaluator.
    """

    memory_name = "base"

    def __init__(
        self,
        *,
        modules: Optional[List[MemoryModule]] = None,
        emit_policy: str = "always",
        url_rewrite_enabled: bool = True,
        webpage_env_update_limit: Optional[int] = None,
        **kwargs: Any,
    ):
        self.options: Dict[str, Any] = {
            "emit_policy": emit_policy,
            "url_rewrite_enabled": url_rewrite_enabled,
            "webpage_env_update_limit": webpage_env_update_limit,
            **kwargs,
        }
        self.modules: List[MemoryModule] = list(modules or [])
        self.state: MemoryRunState = MemoryRunState()
        # Eval sidecar (per-row, not in MemoryRunState).
        self.task_id: Optional[str] = None
        self.tool_detections: List[Dict[str, Any]] = []
        self._tool_call_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.state = MemoryRunState()
        self.task_id = None
        self.tool_detections = []
        self._tool_call_counter = 0

    def reset_for_row(self, task_id: Optional[str]) -> None:
        self.task_id = task_id
        self.tool_detections = []
        self._tool_call_counter = 0

    def record_tool_detection(
        self,
        *,
        tool_name: str,
        category_query: str,
        boxes_xyxy_pixels: List[List[float]],
        scores: List[float],
        image_width: int,
        image_height: int,
    ) -> None:
        try:
            self.tool_detections.append({
                "tool_call_idx": self._tool_call_counter,
                "tool_name": str(tool_name),
                "category_query": str(category_query),
                "boxes_xyxy_pixels": [list(map(float, b[:4])) for b in boxes_xyxy_pixels if len(b) >= 4],
                "scores": [float(s) for s in scores],
                "image_width": int(image_width),
                "image_height": int(image_height),
            })
            self._tool_call_counter += 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Agent-facing hooks
    # ------------------------------------------------------------------

    def on_run_start(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        EnvironmentUpdateBuilder.reset(self.state)
        for m in self.modules:
            m.on_run_start(agent=agent, question=question, messages=messages, state=self.state)
        for m in self.modules:
            m.build_env_update_message(
                agent=agent, turn_idx=-1, tool_calls=[], tool_payloads=[],
                messages=messages, state=self.state,
            )
        return self._coalesce_env_update(reset_change_signal=True)

    def resolve_tool_call_args(
        self,
        *,
        agent: Any,
        args: Dict[str, Any],
        tool_name: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(args, dict):
            return {}, {"rewrites": [], "errors": []}
        resolved = cp.deepcopy(args)
        info = {"rewrites": [], "errors": []}
        for m in self.modules:
            resolved, partial = m.before_tool_call(
                agent=agent, tool_name=tool_name, args=resolved, state=self.state,
            )
            if not isinstance(resolved, dict):
                resolved = {}
            if isinstance(partial, dict):
                info["rewrites"].extend(partial.get("rewrites") or [])
                info["errors"].extend(partial.get("errors") or [])
        return resolved, info

    def consume_tool_payload(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_call: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Failure short-circuit: orchestrator emits the failure ack; modules
        # don't run (no registry mutations on failure — preserves legacy
        # contract that env updates only refresh on at-least-one success).
        if not self._is_success_payload(payload):
            prepared = self._build_base_prepared_payload(payload)
            return {
                "prepared_payload": prepared,
                "immediate_messages": [
                    self.build_failure_tool_message(payload, prepared_payload=prepared)
                ],
                "deferred_messages": [],
                "processor_info": {
                    "mode": "failure_short_circuit",
                    "status": "failure",
                    "module": "",
                },
            }

        # Success path: single chain pass — modules contribute to prepared dict
        # and write side-effect state. Orchestrator owns the ack and calls the
        # per-tool process_tool_output to build the user message.
        prepared = self._build_base_prepared_payload(payload)
        for m in self.modules:
            result = m.consume_tool_result(
                agent=agent, turn_idx=turn_idx, tool_call=tool_call,
                payload=payload, prepared=prepared, state=self.state,
            )
            if isinstance(result, dict):
                prepared = result

        ack = self.build_protocol_tool_ack(payload, prepared_payload=prepared)
        built = self._build_tool_return_message(agent=agent, payload=payload, prepared=prepared)
        return {
            "prepared_payload": prepared,
            "immediate_messages": [ack],
            "deferred_messages": [built["user_message"]],
            "processor_info": built["processor_info"],
        }

    def emit_after_tool_turn(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if isinstance(tool_payloads, list):
            if not any(self._is_success_payload(p) for p in tool_payloads):
                return []
        EnvironmentUpdateBuilder.reset(self.state)
        for m in self.modules:
            m.build_env_update_message(
                agent=agent, turn_idx=turn_idx, tool_calls=tool_calls,
                tool_payloads=tool_payloads, messages=messages, state=self.state,
            )
        return self._coalesce_env_update(reset_change_signal=True)

    # ------------------------------------------------------------------
    # Env-update coalescing
    # ------------------------------------------------------------------

    def _should_emit_env_update(self) -> bool:
        policy = str(self.options.get("emit_policy") or "always").strip().lower()
        if policy == "always":
            return True
        new_snapshot = _image_snapshot(self.state.images)
        if new_snapshot != self.state.images.last_snapshot:
            return True
        if self.state.urls.webpage_counter != self.state.urls.last_webpage_counter:
            return True
        return False

    def _coalesce_env_update(self, *, reset_change_signal: bool) -> List[Dict[str, Any]]:
        builder = EnvironmentUpdateBuilder.get(self.state)
        should_emit = self._should_emit_env_update()
        if reset_change_signal:
            self.state.images.last_snapshot = _image_snapshot(self.state.images)
            self.state.urls.last_webpage_counter = self.state.urls.webpage_counter
        if not should_emit or builder.is_empty():
            return []
        return [builder.build_message()]

    # ------------------------------------------------------------------
    # Tool-return message construction
    # ------------------------------------------------------------------

    def _build_tool_return_message(
        self,
        *,
        agent: Any,
        payload: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Look up the per-tool process_tool_output plug-in and build the
        deferred role=user message. Falls back to a default placeholder if the
        plug-in is missing, returns None, or raises.
        """
        loader_record, tool_metadata = self._load_tool_output_processor(
            agent=agent, payload=payload,
        )
        module_path = str(loader_record.get("module_path") or "")
        base_info = {
            "mode": str(loader_record.get("mode") or "default_placeholder"),
            "status": "ok",
            "module": module_path,
        }
        process_fn = loader_record.get("process_fn")
        loader_error = loader_record.get("error")

        if callable(process_fn):
            try:
                custom = process_fn(payload, prepared, tool_metadata=tool_metadata)
            except Exception as exc:
                base_info = {
                    **base_info, "status": "fallback",
                    "fallback_reason": f"processor_exception:{type(exc).__name__}: {exc}",
                }
                custom = None

            if isinstance(custom, dict):
                # process_fn may declare a per-call detection sidecar entry to
                # be recorded (replaces direct `memory.record_tool_detection`
                # calls from old-contract process_fns).
                detection = custom.get("tool_detection")
                if isinstance(detection, dict):
                    try:
                        self.record_tool_detection(**detection)
                    except Exception:
                        pass

                if isinstance(custom.get("user_message"), dict):
                    user_msg = cp.deepcopy(custom["user_message"])
                    user_msg["role"] = "user"
                    info = dict(base_info)
                    custom_info = custom.get("processor_info")
                    if isinstance(custom_info, dict):
                        info.update(custom_info)
                    return {"user_message": user_msg, "processor_info": info}

                # Dict without user_message → fall through to default but
                # merge any processor_info the tool wanted to convey.
                custom_info = custom.get("processor_info")
                if isinstance(custom_info, dict):
                    base_info = {**base_info, **custom_info}

        # Default placeholder fallback.
        if loader_error and base_info.get("status") != "fallback":
            base_info = {
                **base_info, "status": "fallback", "fallback_reason": str(loader_error),
            }
        return self._build_default_tool_return_message(
            payload=payload, prepared=prepared, tool_metadata=tool_metadata, base_info=base_info,
        )

    def _build_default_tool_return_message(
        self,
        *,
        payload: Dict[str, Any],
        prepared: Dict[str, Any],
        tool_metadata: Optional[Dict[str, Any]],
        base_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_name = str(payload.get("tool_name") or "tool").strip()
        tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()
        refs = prepared.get("image_refs")
        text_payload = prepared.get("text_payload")
        if not isinstance(text_payload, str):
            try:
                text_payload = json.dumps(text_payload, ensure_ascii=False)
            except Exception:
                text_payload = str(text_payload)

        lines = [f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:"]
        if isinstance(refs, list) and refs:
            lines.append(f"Image refs: {', '.join(str(ref) for ref in refs)}")
        lines.append(text_payload)
        text = "\n".join(lines)

        image_parts = prepared.get("image_parts") if isinstance(prepared.get("image_parts"), list) else []
        user_msg = self.build_tool_result_user_message(text=text, image_parts=image_parts)

        info = dict(base_info)
        if isinstance(tool_metadata, dict):
            module_path = tool_metadata.get("tool_output_processor_module")
            if isinstance(module_path, str) and module_path:
                info.setdefault("module", module_path)
        return {"user_message": user_msg, "processor_info": info}

    # ------------------------------------------------------------------
    # Per-tool processor lookup (absorbed from old ImageModule)
    # ------------------------------------------------------------------

    def _ensure_tool_infra_src_import_path(self) -> Optional[str]:
        tool_outputs: ToolOutputsState = self.state.tool_outputs
        if tool_outputs.tool_infra_src_path_bootstrapped:
            return None
        tool_outputs.tool_infra_src_path_bootstrapped = True
        tool_infra_src = Path(__file__).resolve().parents[1] / "tool_infra" / "src"
        if not tool_infra_src.is_dir():
            return None
        tool_infra_src_str = str(tool_infra_src)
        if tool_infra_src_str not in sys.path:
            sys.path.insert(0, tool_infra_src_str)
        return tool_infra_src_str

    def _lookup_tool_metadata(self, agent: Any, tool_name: str) -> Dict[str, Any]:
        tools = getattr(agent, "tools", None) if agent is not None else None
        if not isinstance(tools, dict):
            return {}
        meta = tools.get(tool_name)
        if isinstance(meta, dict):
            return meta
        target = str(tool_name or "").casefold()
        if not target:
            return {}
        for name, candidate in tools.items():
            if str(name).casefold() == target and isinstance(candidate, dict):
                return candidate
        return {}

    def _infer_tool_output_processor_module(
        self,
        *,
        tool_name: str,
        tool_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if isinstance(tool_metadata, dict):
            tool_module_path = str(tool_metadata.get("tool_module_path") or "").strip()
            if tool_module_path:
                if tool_module_path.endswith(".tool"):
                    return f"{tool_module_path.rsplit('.', 1)[0]}.process_tool_output"
                return f"{tool_module_path}.process_tool_output"

        name = str(tool_name or "").strip()
        if not name:
            return ""
        override = _TOOL_OUTPUT_PROCESSOR_MODULE_OVERRIDES.get(name)
        if override:
            return override
        if name.endswith("_Tool"):
            name = name[: -len("_Tool")]
        name = name.replace("-", "_")
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"_+", "_", name)
        tool_dir = name.lower().strip("_")
        if not tool_dir:
            return ""
        return f"tools.{tool_dir}.process_tool_output"

    def _load_tool_output_processor(
        self,
        *,
        agent: Any,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tool_name = str(payload.get("tool_name") or "tool")
        tool_metadata = self._lookup_tool_metadata(agent, tool_name)
        module_path = ""
        if isinstance(tool_metadata, dict):
            module_path = str(tool_metadata.get("tool_output_processor_module") or "").strip()
        if not module_path:
            module_path = self._infer_tool_output_processor_module(
                tool_name=tool_name,
                tool_metadata=tool_metadata if isinstance(tool_metadata, dict) else None,
            )
        cache_key = module_path or f"__missing__:{tool_name}"
        cached = self.state.tool_outputs.processor_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached, tool_metadata

        record: Dict[str, Any] = {
            "module_path": module_path or "",
            "process_fn": None,
            "mode": "default_placeholder",
            "error": None,
        }
        if module_path:
            self._ensure_tool_infra_src_import_path()
            try:
                module = importlib.import_module(module_path)
                process_fn = getattr(module, "process_tool_output", None)
                if callable(process_fn):
                    record["process_fn"] = process_fn
                    mode = getattr(module, "PROCESSOR_MODE", None)
                    if isinstance(mode, str) and mode.strip():
                        record["mode"] = mode.strip()
                else:
                    record["error"] = (
                        f"module '{module_path}' missing callable process_tool_output"
                    )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["error"] = "tool metadata missing tool_output_processor_module"

        self.state.tool_outputs.processor_cache[cache_key] = record
        return record, tool_metadata

    # ------------------------------------------------------------------
    # Payload helpers (used by orchestrator + occasionally external callers)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_success_payload(payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("ok") is True

    def _build_base_prepared_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            "ok": payload.get("ok"),
            "result": payload.get("result"),
            "error": payload.get("error"),
        }
        try:
            text_payload = json.dumps(compact, ensure_ascii=False)
        except Exception:
            text_payload = str(compact)
        return {
            "text_payload": text_payload,
            "compact_payload": compact,
            "compact_result": None,
            "image_parts": [],
            "image_refs": [],
        }

    def prepare_tool_message_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Public helper: build prepared dict by chaining modules' consume_tool_result.

        External callers (test fixtures) use this to compute prepared from a
        payload without going through the full consume_tool_payload flow.
        Modules' consume_tool_result must tolerate `tool_call={}` (no call
        context) — binding accumulation already short-circuits when args are
        absent.
        """
        prepared = self._build_base_prepared_payload(payload)
        for m in self.modules:
            result = m.consume_tool_result(
                agent=None, turn_idx=-1, tool_call={},
                payload=payload, prepared=prepared, state=self.state,
            )
            if isinstance(result, dict):
                prepared = result
        return prepared

    def build_tool_message_content(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
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
        del prepared_payload
        return {
            "role": "tool",
            "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
            "name": str(payload.get("tool_name") or "tool"),
            "content": "ok",
        }

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
        del prepared_payload
        return {
            "role": "tool",
            "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
            "name": str(payload.get("tool_name") or "tool"),
            "content": self._build_failure_tool_content(payload),
        }

    def build_tool_result_user_message(
        self,
        *,
        text: str,
        image_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        safe_text = text if isinstance(text, str) else str(text)
        parts = image_parts if isinstance(image_parts, list) else []
        if parts:
            return {"role": "user", "content": [{"type": "text", "text": safe_text}] + parts}
        return {"role": "user", "content": safe_text}

    def get_binding_descriptors_payload(self) -> Dict[str, Any]:
        return get_binding_descriptors_payload(self.state)


def build_persistent_state_memory(
    name: str = "base",
    **kwargs: Any,
) -> PersistentStateMemory:
    """Factory: instantiate a PersistentStateMemory with a preset module composition."""
    preset = _PRESET_NAME_ALIAS.get(str(name).strip().lower(), "base")
    emit_policy = kwargs.get("emit_policy", "always")
    url_rewrite_enabled = kwargs.get("url_rewrite_enabled", True)
    webpage_env_update_limit = kwargs.get("webpage_env_update_limit", None)

    if preset == "base":
        modules: List[MemoryModule] = []
    elif preset == "image":
        modules = [ImageModule(emit_policy=emit_policy)]
    elif preset == "tool_context":
        modules = [ImageModule(emit_policy=emit_policy), ToolContextModule()]
    elif preset == "url":
        modules = [
            ImageModule(emit_policy=emit_policy),
            ToolContextModule(),
            UrlModule(
                url_rewrite_enabled=url_rewrite_enabled,
                webpage_env_update_limit=webpage_env_update_limit,
            ),
        ]
    else:
        modules = []
    instance = PersistentStateMemory(modules=modules, **kwargs)
    instance.memory_name = preset
    return instance
