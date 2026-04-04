import copy as cp
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.api_utils import local_image_to_data_url
from .persistent_state_memory import PersistentStateMemory


class ImageStateMemory(PersistentStateMemory):
    """Per-run image state memory that tracks runtime images and emits updates."""

    memory_name = "image"
    _DROP_VALUE = object()
    _ALLOWED_EMIT_POLICIES = ("always", "on_change")
    _INPUT_IMAGE_REF_PATTERN = re.compile(r"^input_image_(\d+)$")
    _TOOL_GENERATED_IMAGE_REF_PATTERN = re.compile(r"^tool_generated_image_(\d+)$")
    _IMAGE_REGISTRY_DEFAULT_ALIASES = ("image", "image.jpg", "image.jpeg", "image.png")
    _IMAGE_RESULT_KEY_HINTS = (
        "all_masks_viz_base64",
        "image_base64",
        "overlay_image_base64",
        "zoomed_image_base64",
        "blended_image_base64",
        "cropped_image_base64",
        "edited_image_base64",
        "annotated_image_base64",
        "depth_map_base64",
        "masks_png_b64",
        "subplot_images_base64",
    )
    _RAW_DATA_HINTS = ("raw", "meters")

    def __init__(self, emit_policy: str = "always", **kwargs):
        super().__init__(**kwargs)
        policy = str(emit_policy).strip().lower()
        if policy not in self._ALLOWED_EMIT_POLICIES:
            raise ValueError(
                f"Invalid image memory emit_policy '{emit_policy}'. "
                f"Expected one of: {self._ALLOWED_EMIT_POLICIES}."
            )
        self.emit_policy = policy
        self._last_snapshot: Optional[Tuple[Tuple[str, str], ...]] = None
        self._image_registry: Dict[str, Dict[str, Any]] = {}
        self._image_alias_to_ref: Dict[str, str] = {}
        self._image_url_to_ref: Dict[str, str] = {}
        self._tool_output_processor_cache: Dict[str, Dict[str, Any]] = {}
        self._tool_infra_src_path_bootstrapped = False
        self._tool_generated_image_counter = 0

    def reset(self) -> None:
        self._last_snapshot = None
        self._image_registry = {}
        self._image_alias_to_ref = {}
        self._image_url_to_ref = {}
        self._tool_output_processor_cache = {}
        self._tool_infra_src_path_bootstrapped = False
        self._tool_generated_image_counter = 0

    def configure_tool_schema(self, json_schema: Any) -> None:
        if not isinstance(json_schema, list):
            return
        suffix = " You may pass a runtime image ID from Environment Update messages instead of a filesystem path."
        for entry in json_schema:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function") or {}
            params = fn.get("parameters") or {}
            properties = params.get("properties") or {}
            if not isinstance(properties, dict):
                continue
            for key, spec in properties.items():
                if not isinstance(spec, dict):
                    continue
                key_lower = str(key).lower()
                if key_lower.endswith("_path") and "image" in key_lower:
                    desc = spec.get("description", "")
                    if suffix.strip() not in str(desc):
                        spec["description"] = f"{desc}{suffix}".strip()

    def on_run_start(
        self,
        *,
        agent: Any,
        question: Any,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._build_image_registry_from_inputs(question)
        return self._emit_environment_update()

    def emit_after_tool_turn(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_calls: List[Dict[str, Any]],
        tool_payloads: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del agent, turn_idx, tool_calls, messages  # reserved for future use
        if isinstance(tool_payloads, list):
            if not any(self._is_success_payload(payload) for payload in tool_payloads):
                return []
        return self._emit_environment_update()

    def _ensure_tool_infra_src_import_path(self) -> Optional[str]:
        if self._tool_infra_src_path_bootstrapped:
            return None
        self._tool_infra_src_path_bootstrapped = True
        tool_infra_src = Path(__file__).resolve().parents[1] / "tool_infra" / "src"
        if not tool_infra_src.is_dir():
            return None
        tool_infra_src_str = str(tool_infra_src)
        if tool_infra_src_str not in sys.path:
            sys.path.insert(0, tool_infra_src_str)
        return tool_infra_src_str

    def _lookup_tool_metadata(self, agent: Any, tool_name: str) -> Dict[str, Any]:
        tools = getattr(agent, "tools", None)
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
        cached = self._tool_output_processor_cache.get(cache_key)
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

        self._tool_output_processor_cache[cache_key] = record
        return record, tool_metadata

    def build_tool_result_user_message(
        self,
        *,
        text: str,
        image_parts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        safe_text = text if isinstance(text, str) else str(text)
        parts = image_parts if isinstance(image_parts, list) else []
        if parts:
            return {
                "role": "user",
                "content": [{"type": "text", "text": safe_text}] + parts,
            }
        return {"role": "user", "content": safe_text}

    def build_processed_tool_output(
        self,
        *,
        payload: Dict[str, Any],
        prepared_payload: Optional[Dict[str, Any]] = None,
        text: str,
        processor_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prepared = prepared_payload or self.prepare_tool_message_payload(payload)
        image_parts = prepared.get("image_parts")
        refs = prepared.get("image_refs")
        normalized_prepared = prepared if isinstance(prepared, dict) else {}
        normalized_info = dict(processor_info or {})
        if isinstance(refs, list) and refs:
            normalized_prepared.setdefault("image_refs", refs)
        return {
            "user_message": self.build_tool_result_user_message(
                text=text,
                image_parts=image_parts if isinstance(image_parts, list) else [],
            ),
            "prepared_payload": normalized_prepared,
            "processor_info": normalized_info,
        }

    def process_tool_output_default(
        self,
        *,
        payload: Dict[str, Any],
        tool_metadata: Optional[Dict[str, Any]] = None,
        processor_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prepared = self.prepare_tool_message_payload(payload)
        tool_name = str(payload.get("tool_name") or "tool").strip()
        tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()
        refs = prepared.get("image_refs")
        text_payload = prepared.get("text_payload")
        if not isinstance(text_payload, str):
            try:
                text_payload = json.dumps(text_payload, ensure_ascii=False)
            except Exception:
                text_payload = str(text_payload)

        lines = [
            f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:",
        ]
        if isinstance(refs, list) and refs:
            lines.append(f"Image refs: {', '.join(str(ref) for ref in refs)}")
        lines.append(text_payload)
        info = {
            "mode": "default_placeholder",
            "status": "ok",
            "module": "",
        }
        if isinstance(tool_metadata, dict):
            module_path = tool_metadata.get("tool_output_processor_module")
            if isinstance(module_path, str):
                info["module"] = module_path
        if isinstance(processor_info, dict):
            info.update(processor_info)

        return self.build_processed_tool_output(
            payload=payload,
            prepared_payload=prepared,
            text="\n".join(lines),
            processor_info=info,
        )

    def _normalize_processed_tool_output(
        self,
        *,
        payload: Dict[str, Any],
        processed: Any,
        default_processor_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(processed, dict):
            raise TypeError(
                f"process_tool_output must return dict, got {type(processed).__name__}"
            )

        user_message = processed.get("user_message")
        if not isinstance(user_message, dict):
            raise TypeError("process_tool_output return dict missing 'user_message' dict")
        user_message = cp.deepcopy(user_message)
        user_message["role"] = "user"

        prepared_payload = processed.get("prepared_payload")
        if not isinstance(prepared_payload, dict):
            prepared_payload = self.prepare_tool_message_payload(payload)

        processor_info = processed.get("processor_info")
        normalized_info = dict(default_processor_info)
        if isinstance(processor_info, dict):
            normalized_info.update(processor_info)

        return {
            "user_message": user_message,
            "prepared_payload": prepared_payload,
            "processor_info": normalized_info,
        }

    def _snapshot(self) -> Tuple[Tuple[str, str], ...]:
        rows = []
        for ref_id in sorted(self._image_registry.keys(), key=self._snapshot_sort_key):
            entry = self._image_registry.get(ref_id) or {}
            rows.append((ref_id, str(entry.get("source", "runtime"))))
        return tuple(rows)

    def _snapshot_sort_key(self, ref_id: str) -> Tuple[int, int, str]:
        match = self._INPUT_IMAGE_REF_PATTERN.match(ref_id)
        if match:
            return (0, int(match.group(1)), ref_id)

        match = self._TOOL_GENERATED_IMAGE_REF_PATTERN.match(ref_id)
        if match:
            return (1, int(match.group(1)), ref_id)

        return (2, 0, ref_id)

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

    def _emit_environment_update(self) -> List[Dict[str, Any]]:
        snapshot = self._snapshot()
        should_emit = self._should_emit(snapshot)
        self._last_snapshot = snapshot
        if not should_emit:
            return []
        return [self._build_environment_message(snapshot)]

    def _normalize_image_alias(self, alias: str) -> str:
        if not isinstance(alias, str):
            return ""
        return alias.strip().lower()

    def _is_likely_base64(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip()
        if len(value) < 64:
            return False
        if value.startswith("data:image/"):
            return True
        if value.startswith("base64:"):
            value = value.split(":", 1)[1]
        return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", value))

    def _register_image_data_url(
        self,
        ref_id: str,
        data_url: str,
        *,
        source: str,
        aliases: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not isinstance(data_url, str):
            return None
        data_url = data_url.strip()
        if not data_url.startswith("data:image/"):
            return None

        existing_ref = self._image_url_to_ref.get(data_url)
        if existing_ref and existing_ref in self._image_registry:
            entry = self._image_registry[existing_ref]
            for alias in aliases or []:
                norm_alias = self._normalize_image_alias(alias)
                if not norm_alias:
                    continue
                self._image_alias_to_ref[norm_alias] = existing_ref
                entry["aliases"].add(alias)
            return existing_ref

        entry = {
            "id": ref_id,
            "data_url": data_url,
            "source": source,
            "aliases": set(),
        }
        self._image_registry[ref_id] = entry
        self._image_url_to_ref[data_url] = ref_id

        for alias in [ref_id] + list(aliases or []):
            norm_alias = self._normalize_image_alias(alias)
            if not norm_alias:
                continue
            self._image_alias_to_ref[norm_alias] = ref_id
            entry["aliases"].add(alias)

        return ref_id

    def _build_image_registry_from_inputs(self, question: Any) -> Dict[str, Dict[str, Any]]:
        self._image_registry = {}
        self._image_alias_to_ref = {}
        self._image_url_to_ref = {}
        if not isinstance(question, list):
            return self._image_registry

        image_items = []
        for item in question:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and isinstance(item.get("value"), str):
                image_items.append(item.get("value").strip())

        for idx, image_value in enumerate(image_items, 1):
            ref_id = f"input_image_{idx}"
            aliases = [ref_id, f"image_{idx}"]
            data_url = None
            if image_value.startswith("data:image/"):
                data_url = image_value
            elif self._is_likely_base64(image_value):
                raw = image_value
                if raw.startswith("base64:"):
                    raw = raw.split(":", 1)[1]
                data_url = f"data:image/png;base64,{raw}"
            else:
                aliases.append(image_value)
                aliases.append(os.path.basename(image_value))
                try:
                    data_url = local_image_to_data_url(image_value)
                except Exception:
                    data_url = None

            registered = self._register_image_data_url(
                ref_id,
                data_url,
                source="user_input",
                aliases=aliases,
            )
            if registered and len(image_items) == 1:
                for alias in self._IMAGE_REGISTRY_DEFAULT_ALIASES:
                    self._image_alias_to_ref[self._normalize_image_alias(alias)] = registered
                    self._image_registry[registered]["aliases"].add(alias)

        return self._image_registry

    def _resolve_registry_image_data_url(self, image_ref: str):
        if not isinstance(image_ref, str):
            return None, None, None

        raw = image_ref.strip()
        if not raw:
            return None, None, None

        if raw.startswith("data:image/"):
            return raw, None, None

        if self._is_likely_base64(raw):
            if raw.startswith("base64:"):
                raw = raw.split(":", 1)[1]
            return f"data:image/png;base64,{raw}", None, None

        alias_candidates = [self._normalize_image_alias(raw)]
        alias_candidates.append(self._normalize_image_alias(os.path.basename(raw)))
        for alias in alias_candidates:
            ref_id = self._image_alias_to_ref.get(alias)
            if ref_id and ref_id in self._image_registry:
                return self._image_registry[ref_id]["data_url"], ref_id, None

        if os.path.isfile(raw):
            return None, None, None

        if len(self._image_registry) == 1:
            only_ref = next(iter(self._image_registry.keys()))
            return (
                self._image_registry[only_ref]["data_url"],
                only_ref,
                "single_image_registry_default",
            )

        return None, None, None

    def _is_image_path_key(self, key: str) -> bool:
        key_lower = str(key).lower()
        return key_lower.endswith("_path") and "image" in key_lower

    def _resolve_image_paths_in_args(
        self,
        value: Any,
        *,
        path: str,
        rewrites: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
    ) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                child_path = f"{path}.{key}"
                if isinstance(child, (dict, list)):
                    self._resolve_image_paths_in_args(
                        child,
                        path=child_path,
                        rewrites=rewrites,
                        errors=errors,
                    )

            for path_key in list(value.keys()):
                if not self._is_image_path_key(path_key):
                    continue
                base64_key = f"{path_key[:-5]}_base64"
                if value.get(base64_key):
                    continue

                raw_path = value.get(path_key)
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue

                data_url, matched_ref, fallback_mode = self._resolve_registry_image_data_url(raw_path)
                if data_url:
                    value[base64_key] = data_url
                    value.pop(path_key, None)
                    rewrite_item = {
                        "field": f"{path}.{path_key}",
                        "target_field": f"{path}.{base64_key}",
                        "original": raw_path,
                    }
                    if matched_ref:
                        rewrite_item["image_ref"] = matched_ref
                    if fallback_mode:
                        rewrite_item["fallback_mode"] = fallback_mode
                    rewrites.append(rewrite_item)
                elif not os.path.isfile(raw_path):
                    errors.append(
                        {
                            "field": f"{path}.{path_key}",
                            "value": raw_path,
                            "reason": "unresolved_image_reference",
                            "available_image_ids": list(self._image_registry.keys()),
                        }
                    )
            return

        if isinstance(value, list):
            for idx, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    self._resolve_image_paths_in_args(
                        child,
                        path=f"{path}[{idx}]",
                        rewrites=rewrites,
                        errors=errors,
                    )

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
        rewrites: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        self._resolve_image_paths_in_args(
            resolved,
            path="args",
            rewrites=rewrites,
            errors=errors,
        )
        return resolved, {"rewrites": rewrites, "errors": errors}

    def _next_tool_generated_image_ref_id(self) -> str:
        self._tool_generated_image_counter += 1
        return f"tool_generated_image_{self._tool_generated_image_counter}"

    def _register_tool_output_images(self, payload: Dict[str, Any], image_parts: List[Dict[str, Any]]) -> List[str]:
        refs = []
        tool_name = str(payload.get("tool_name") or "tool").strip() or "tool"
        for part in image_parts:
            if not isinstance(part, dict):
                continue
            image_url = (part.get("image_url") or {}).get("url")
            if not isinstance(image_url, str):
                continue
            image_url = image_url.strip()
            if not image_url:
                continue
            if not image_url.startswith("data:image/"):
                continue

            existing_ref = self._image_url_to_ref.get(image_url)
            if existing_ref and existing_ref in self._image_registry:
                refs.append(existing_ref)
                continue

            ref_id = self._next_tool_generated_image_ref_id()
            assigned_ref = self._register_image_data_url(
                ref_id,
                image_url,
                source=f"tool_output:{tool_name}",
                aliases=[],
            )
            if assigned_ref:
                refs.append(assigned_ref)
        return refs

    def _is_image_result_key(self, key: str) -> bool:
        k = str(key).lower()
        if any(hint in k for hint in self._RAW_DATA_HINTS):
            return False
        if k in self._IMAGE_RESULT_KEY_HINTS:
            return True
        if k.endswith("_base64") or k.endswith("_b64"):
            return any(
                hint in k
                for hint in (
                    "image",
                    "viz",
                    "mask",
                    "depth",
                    "overlay",
                    "edited",
                    "cropped",
                    "subplot",
                )
            )
        return False

    def _guess_image_mime(self, key: str) -> str:
        k = str(key).lower()
        if "jpg" in k or "jpeg" in k:
            return "image/jpeg"
        return "image/png"

    def _to_image_data_url(self, key: str, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if raw.startswith("data:image/"):
            return raw
        if raw.startswith("base64:"):
            raw = raw.split(":", 1)[1]
        if not self._is_likely_base64(raw):
            return None
        mime = self._guess_image_mime(key)
        return f"data:{mime};base64,{raw}"

    def _extract_images_and_scrub_result(
        self,
        value: Any,
        image_parts: List[Dict[str, Any]],
        *,
        key_hint: str = "",
        image_context: bool = False,
    ):
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for key, item in value.items():
                child_context = image_context or self._is_image_result_key(key)
                compact_item = self._extract_images_and_scrub_result(
                    item,
                    image_parts,
                    key_hint=str(key),
                    image_context=child_context,
                )
                if compact_item is not self._DROP_VALUE:
                    compact[key] = compact_item
            if not compact and image_context:
                return self._DROP_VALUE
            return compact

        if isinstance(value, list):
            compact_list = []
            for item in value:
                compact_item = self._extract_images_and_scrub_result(
                    item,
                    image_parts,
                    key_hint=key_hint,
                    image_context=image_context,
                )
                if compact_item is not self._DROP_VALUE:
                    compact_list.append(compact_item)
            if not compact_list and image_context:
                return self._DROP_VALUE
            return compact_list

        if isinstance(value, str):
            raw = value.strip()
            is_explicit_image = raw.startswith("data:image/")
            should_try_image = is_explicit_image or image_context or self._is_image_result_key(key_hint)
            if should_try_image:
                url = self._to_image_data_url(key_hint or "image", raw)
                if url:
                    image_parts.append({"type": "image_url", "image_url": {"url": url}})
                    return self._DROP_VALUE

        return value

    def _extract_image_parts(self, result: Any):
        image_parts: List[Dict[str, Any]] = []
        compact_result = self._extract_images_and_scrub_result(
            result,
            image_parts,
            key_hint="result",
            image_context=False,
        )
        if compact_result is self._DROP_VALUE:
            compact_result = None
        return image_parts, compact_result

    def prepare_tool_message_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = payload.get("result")
        image_parts, compact_result = self._extract_image_parts(result)
        image_refs = self._register_tool_output_images(payload, image_parts) if image_parts else []
        if image_parts:
            result_for_text = compact_result
        else:
            result_for_text = compact_result if isinstance(result, (dict, list)) else result
        compact_payload: Dict[str, Any] = {
            "ok": payload.get("ok"),
            "result": result_for_text,
            "error": payload.get("error"),
        }
        if image_refs:
            compact_payload["image_refs"] = image_refs

        try:
            text_payload = json.dumps(compact_payload, ensure_ascii=False)
        except Exception:
            text_payload = str(compact_payload)

        return {
            "text_payload": text_payload,
            "compact_payload": compact_payload,
            "compact_result": compact_result,
            "image_parts": image_parts,
            "image_refs": image_refs,
        }

    def build_deferred_user_messages(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        prepared = prepared_payload or self.prepare_tool_message_payload(payload)
        image_parts = prepared.get("image_parts")
        if not isinstance(image_parts, list) or not image_parts:
            return []

        image_refs = prepared.get("image_refs")
        refs = image_refs if isinstance(image_refs, list) else []
        tool_name = str(payload.get("tool_name") or "tool").strip()
        tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()
        lines = [
            f"Tool `{tool_name}` (tool_call_id={tool_call_id}) also returned the following image(s). Use these images for reasoning if helpful.",
            "See the accompanying tool-result summary for structured output.",
        ]
        if refs:
            lines.append(f"Image refs: {', '.join(refs)}")

        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "\n".join(lines)}] + image_parts,
            }
        ]

    def consume_tool_payload(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_call: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        del turn_idx, tool_call  # reserved for future processors

        if not self._is_success_payload(payload):
            # Failure payloads should be side-effect free: don't register any image
            # artifacts (they won't be surfaced via environment updates on all-failure turns).
            prepared_payload = super().prepare_tool_message_payload(payload)
            return {
                "prepared_payload": prepared_payload,
                "immediate_messages": [
                    self.build_failure_tool_message(
                        payload,
                        prepared_payload=prepared_payload,
                    )
                ],
                "deferred_messages": [],
                "processor_info": {
                    "mode": "failure_short_circuit",
                    "status": "failure",
                    "module": "",
                },
            }

        prepared_payload = self.prepare_tool_message_payload(payload)
        loader_record, tool_metadata = self._load_tool_output_processor(
            agent=agent,
            payload=payload,
        )
        module_path = str(loader_record.get("module_path") or "")
        default_info = {
            "mode": str(loader_record.get("mode") or "default_placeholder"),
            "status": "ok",
            "module": module_path,
        }

        process_fn = loader_record.get("process_fn")
        loader_error = loader_record.get("error")
        if not callable(process_fn):
            processed = self.process_tool_output_default(
                payload=payload,
                tool_metadata=tool_metadata,
                processor_info={
                    "mode": "default_placeholder",
                    "status": "fallback" if loader_error else "ok",
                    "module": module_path,
                    **({"fallback_reason": str(loader_error)} if loader_error else {}),
                },
            )
            normalized = self._normalize_processed_tool_output(
                payload=payload,
                processed=processed,
                default_processor_info=default_info,
            )
            # NOTE: Strict OpenAI-native tool protocol requires a `role=tool`
            # message with matching `tool_call_id` after assistant `tool_calls`.
            # Keep this terse ack immediate; queue rich user context so tool
            # replies remain contiguous when multiple tool calls are present.
            protocol_tool_ack = {
                "role": "tool",
                "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
                "name": str(payload.get("tool_name") or "tool"),
                "content": "ok",
            }
            return {
                "prepared_payload": normalized["prepared_payload"],
                "immediate_messages": [protocol_tool_ack],
                "deferred_messages": [normalized["user_message"]],
                "processor_info": normalized["processor_info"],
            }

        try:
            processed = process_fn(
                payload=payload,
                memory=self,
                tool_metadata=tool_metadata if isinstance(tool_metadata, dict) else None,
            )
            normalized = self._normalize_processed_tool_output(
                payload=payload,
                processed=processed,
                default_processor_info=default_info,
            )
        except Exception as exc:
            processed = self.process_tool_output_default(
                payload=payload,
                tool_metadata=tool_metadata,
                processor_info={
                    "mode": "default_placeholder",
                    "status": "fallback",
                    "module": module_path,
                    "fallback_reason": f"processor_exception:{type(exc).__name__}: {exc}",
                },
            )
            normalized = self._normalize_processed_tool_output(
                payload=payload,
                processed=processed,
                default_processor_info=default_info,
            )

        # NOTE: Strict OpenAI-native tool protocol requires a `role=tool`
        # message with matching `tool_call_id` after assistant `tool_calls`.
        # Keep this terse ack immediate; queue rich user context so tool
        # replies remain contiguous when multiple tool calls are present.
        protocol_tool_ack = {
            "role": "tool",
            "tool_call_id": str(payload.get("tool_call_id") or "unknown"),
            "name": str(payload.get("tool_name") or "tool"),
            "content": "ok",
        }
        return {
            "prepared_payload": normalized["prepared_payload"],
            "immediate_messages": [protocol_tool_ack],
            "deferred_messages": [normalized["user_message"]],
            "processor_info": normalized["processor_info"],
        }
