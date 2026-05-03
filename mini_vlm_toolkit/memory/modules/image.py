import copy as cp
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ...utils.api_utils import local_image_to_data_url
from ..module import EnvironmentUpdateBuilder, MemoryModule
from ..run_state import ImagesState, MemoryRunState

_DROP_VALUE = object()
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


def _normalize_image_alias(alias: Any) -> str:
    if not isinstance(alias, str):
        return ""
    return alias.strip().lower()


def _is_likely_base64(value: Any) -> bool:
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


def next_tool_generated_image_ref_id(images: ImagesState) -> str:
    images.tool_generated_counter += 1
    return f"tool_generated_image_{images.tool_generated_counter}"


def register_image_data_url(
    images: ImagesState,
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

    existing_ref = images.url_to_ref.get(data_url)
    if existing_ref and existing_ref in images.registry:
        entry = images.registry[existing_ref]
        for alias in aliases or []:
            norm_alias = _normalize_image_alias(alias)
            if not norm_alias:
                continue
            images.alias_to_ref[norm_alias] = existing_ref
            entry["aliases"].add(alias)
        return existing_ref

    entry: Dict[str, Any] = {
        "id": ref_id,
        "data_url": data_url,
        "source": source,
        "aliases": set(),
    }
    images.registry[ref_id] = entry
    images.url_to_ref[data_url] = ref_id

    for alias in [ref_id] + list(aliases or []):
        norm_alias = _normalize_image_alias(alias)
        if not norm_alias:
            continue
        images.alias_to_ref[norm_alias] = ref_id
        entry["aliases"].add(alias)
    return ref_id


_TRAILING_STRIP_FOR_LAZY = ".,;:!?'\"]}\\"


def register_image_url_lazy(
    images: ImagesState,
    url: str,
    *,
    source_tool: str = "",
    turn_idx: int = -1,
) -> Optional[str]:
    """Register an image URL as an unresolved registry entry (data_url=None).

    Bytes are fetched lazily on first consume by a visual tool via
    `_lazy_fetch_into_entry` (called from `resolve_registry_image_data_url`).
    Used by URL search-result rewriting in UrlModule.
    """
    del turn_idx
    if not isinstance(url, str):
        return None
    url = url.strip()
    while url and url[-1] in _TRAILING_STRIP_FOR_LAZY:
        url = url[:-1]
    if not url:
        return None

    existing_ref = images.url_to_ref.get(url)
    if existing_ref and existing_ref in images.registry:
        return existing_ref
    norm_alias = _normalize_image_alias(url)
    existing_alias_ref = images.alias_to_ref.get(norm_alias) if norm_alias else None
    if existing_alias_ref and existing_alias_ref in images.registry:
        return existing_alias_ref

    ref_id = next_tool_generated_image_ref_id(images)
    src_tool = (str(source_tool) if source_tool else "tool").strip() or "tool"
    entry: Dict[str, Any] = {
        "id": ref_id,
        "url": url,
        "data_url": None,
        "source": f"tool_output:{src_tool}",
        "aliases": {url, ref_id},
    }
    images.registry[ref_id] = entry
    images.url_to_ref[url] = ref_id
    ref_id_alias = _normalize_image_alias(ref_id)
    if ref_id_alias:
        images.alias_to_ref[ref_id_alias] = ref_id
    if norm_alias:
        images.alias_to_ref[norm_alias] = ref_id
    return ref_id


def _fetch_image_data_url(url: str) -> Optional[str]:
    import base64
    import requests

    try:
        resp = requests.get(url, timeout=30, stream=False)
        resp.raise_for_status()
    except Exception:
        return None

    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    if not content_type.startswith("image/"):
        return None

    b64 = base64.b64encode(resp.content).decode("ascii")
    return f"data:{content_type};base64,{b64}"


def _lazy_fetch_into_entry(images: ImagesState, ref_id: str, url: str) -> Optional[str]:
    data_url = _fetch_image_data_url(url)
    if not data_url:
        return None
    entry = images.registry.get(ref_id)
    if not isinstance(entry, dict):
        return None
    entry["data_url"] = data_url
    images.url_to_ref.setdefault(data_url, ref_id)
    return data_url


def _download_and_register_url(
    images: ImagesState, url: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    norm_alias = _normalize_image_alias(url)
    existing_ref = images.alias_to_ref.get(norm_alias)
    if existing_ref and existing_ref in images.registry:
        return images.registry[existing_ref].get("data_url"), existing_ref, None

    data_url = _fetch_image_data_url(url)
    if not data_url:
        return None, None, None

    ref_id = next_tool_generated_image_ref_id(images)
    assigned_ref = register_image_data_url(
        images,
        ref_id,
        data_url,
        source=f"downloaded:{url[:120]}",
        aliases=[url],
    )
    if assigned_ref:
        return data_url, assigned_ref, "http_download"
    return None, None, None


def resolve_registry_image_data_url(
    images: ImagesState, image_ref: Any
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(image_ref, str):
        return None, None, None
    raw = image_ref.strip()
    if not raw:
        return None, None, None
    if raw.startswith("data:image/"):
        return raw, None, None
    if _is_likely_base64(raw):
        if raw.startswith("base64:"):
            raw = raw.split(":", 1)[1]
        return f"data:image/png;base64,{raw}", None, None

    alias_candidates = [_normalize_image_alias(raw), _normalize_image_alias(os.path.basename(raw))]
    for alias in alias_candidates:
        ref_id = images.alias_to_ref.get(alias)
        if not ref_id or ref_id not in images.registry:
            continue
        entry = images.registry[ref_id]
        data_url = entry.get("data_url")
        if data_url:
            return data_url, ref_id, None
        url = entry.get("url")
        if isinstance(url, str) and url:
            fetched = _lazy_fetch_into_entry(images, ref_id, url)
            if fetched:
                return fetched, ref_id, "lazy_fetch"

    if raw.startswith("http://") or raw.startswith("https://"):
        return _download_and_register_url(images, raw)

    if os.path.isfile(raw):
        return None, None, None

    if len(images.registry) == 1:
        only_ref = next(iter(images.registry.keys()))
        return (
            images.registry[only_ref].get("data_url"),
            only_ref,
            "single_image_registry_default",
        )
    return None, None, None


def build_image_registry_from_inputs(images: ImagesState, question: Any) -> Dict[str, Dict[str, Any]]:
    images.registry.clear()
    images.alias_to_ref.clear()
    images.url_to_ref.clear()
    if not isinstance(question, list):
        return images.registry

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
        elif _is_likely_base64(image_value):
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

        registered = register_image_data_url(
            images, ref_id, data_url, source="user_input", aliases=aliases
        )
        if registered and len(image_items) == 1:
            for alias in _IMAGE_REGISTRY_DEFAULT_ALIASES:
                images.alias_to_ref[_normalize_image_alias(alias)] = registered
                images.registry[registered]["aliases"].add(alias)

    return images.registry


def _is_image_path_key(key: Any) -> bool:
    key_lower = str(key).lower()
    return key_lower.endswith("_path") and "image" in key_lower


def _resolve_image_paths_in_args(
    images: ImagesState,
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
                _resolve_image_paths_in_args(
                    images, child, path=child_path, rewrites=rewrites, errors=errors
                )

        for path_key in list(value.keys()):
            if not _is_image_path_key(path_key):
                continue
            base64_key = f"{path_key[:-5]}_base64"
            if value.get(base64_key):
                continue

            raw_path = value.get(path_key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue

            data_url, matched_ref, fallback_mode = resolve_registry_image_data_url(images, raw_path)
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
                        "available_image_ids": list(images.registry.keys()),
                    }
                )
        return

    if isinstance(value, list):
        for idx, child in enumerate(value):
            if isinstance(child, (dict, list)):
                _resolve_image_paths_in_args(
                    images, child, path=f"{path}[{idx}]", rewrites=rewrites, errors=errors
                )


def _is_image_result_key(key: Any) -> bool:
    k = str(key).lower()
    if any(hint in k for hint in _RAW_DATA_HINTS):
        return False
    if k in _IMAGE_RESULT_KEY_HINTS:
        return True
    if k.endswith("_base64") or k.endswith("_b64"):
        return any(
            hint in k
            for hint in (
                "image", "viz", "mask", "depth", "overlay",
                "edited", "cropped", "subplot",
            )
        )
    return False


def _guess_image_mime(key: Any) -> str:
    k = str(key).lower()
    if "jpg" in k or "jpeg" in k:
        return "image/jpeg"
    return "image/png"


def _to_image_data_url(key: Any, value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.startswith("data:image/"):
        return raw
    if raw.startswith("base64:"):
        raw = raw.split(":", 1)[1]
    if not _is_likely_base64(raw):
        return None
    mime = _guess_image_mime(key)
    return f"data:{mime};base64,{raw}"


def _extract_images_and_scrub_result(
    value: Any,
    image_parts: List[Dict[str, Any]],
    *,
    key_hint: str = "",
    image_context: bool = False,
):
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            child_context = image_context or _is_image_result_key(key)
            compact_item = _extract_images_and_scrub_result(
                item, image_parts, key_hint=str(key), image_context=child_context
            )
            if compact_item is not _DROP_VALUE:
                compact[key] = compact_item
        if not compact and image_context:
            return _DROP_VALUE
        return compact

    if isinstance(value, list):
        compact_list = []
        for item in value:
            compact_item = _extract_images_and_scrub_result(
                item, image_parts, key_hint=key_hint, image_context=image_context
            )
            if compact_item is not _DROP_VALUE:
                compact_list.append(compact_item)
        if not compact_list and image_context:
            return _DROP_VALUE
        return compact_list

    if isinstance(value, str):
        raw = value.strip()
        is_explicit_image = raw.startswith("data:image/")
        should_try_image = is_explicit_image or image_context or _is_image_result_key(key_hint)
        if should_try_image:
            url = _to_image_data_url(key_hint or "image", raw)
            if url:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
                return _DROP_VALUE
    return value


def _extract_image_parts(result: Any) -> Tuple[List[Dict[str, Any]], Any]:
    image_parts: List[Dict[str, Any]] = []
    compact_result = _extract_images_and_scrub_result(
        result, image_parts, key_hint="result", image_context=False
    )
    if compact_result is _DROP_VALUE:
        compact_result = None
    return image_parts, compact_result


def _register_tool_output_images(
    images: ImagesState, payload: Dict[str, Any], image_parts: List[Dict[str, Any]]
) -> List[str]:
    refs: List[str] = []
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

        existing_ref = images.url_to_ref.get(image_url)
        if existing_ref and existing_ref in images.registry:
            refs.append(existing_ref)
            continue

        ref_id = next_tool_generated_image_ref_id(images)
        assigned_ref = register_image_data_url(
            images, ref_id, image_url, source=f"tool_output:{tool_name}", aliases=[]
        )
        if assigned_ref:
            refs.append(assigned_ref)
    return refs


def _snapshot_sort_key(ref_id: str) -> Tuple[int, int, str]:
    match = _INPUT_IMAGE_REF_PATTERN.match(ref_id)
    if match:
        return (0, int(match.group(1)), ref_id)
    match = _TOOL_GENERATED_IMAGE_REF_PATTERN.match(ref_id)
    if match:
        return (1, int(match.group(1)), ref_id)
    return (2, 0, ref_id)


def _snapshot(images: ImagesState) -> Tuple[Tuple[str, str], ...]:
    rows = []
    for ref_id in sorted(images.registry.keys(), key=_snapshot_sort_key):
        entry = images.registry.get(ref_id) or {}
        rows.append((ref_id, str(entry.get("source", "runtime"))))
    return tuple(rows)


class ImageModule(MemoryModule):
    """Image domain: registry, arg resolution, tool-output extraction, env update."""

    name = "image"

    # No __init__: ImageModule has no per-instance config. The emit_policy
    # decision lives on the orchestrator (PersistentStateMemory) since it
    # consults cross-module state (image snapshot + webpage counter delta).

    # ---- on_run_start ----
    def on_run_start(self, *, agent, question, messages, state):
        build_image_registry_from_inputs(state.images, question)
        return []

    # ---- before_tool_call ----
    def before_tool_call(self, *, agent, tool_name, args, state):
        if not isinstance(args, dict):
            return {}, {"rewrites": [], "errors": []}
        resolved = cp.deepcopy(args)
        rewrites: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        _resolve_image_paths_in_args(
            state.images, resolved, path="args", rewrites=rewrites, errors=errors
        )
        return resolved, {"rewrites": rewrites, "errors": errors}

    # ---- consume_tool_result ----
    # Extract image_parts from result, register tool-output images as
    # tool_generated_image_N in state.images, rebuild text_payload using the
    # scrubbed compact_result. Tolerates `tool_call={}` (called from external
    # `prepare_tool_message_payload` with no call context).
    def consume_tool_result(self, *, agent, turn_idx, tool_call, payload, prepared, state):
        del agent, turn_idx, tool_call
        result = payload.get("result")
        image_parts, compact_result = _extract_image_parts(result)
        image_refs = (
            _register_tool_output_images(state.images, payload, image_parts)
            if image_parts
            else []
        )
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
        prepared.update({
            "text_payload": text_payload,
            "compact_payload": compact_payload,
            "compact_result": compact_result,
            "image_parts": image_parts,
            "image_refs": image_refs,
        })
        return prepared

    # ---- build_env_update_message ----
    # Modules unconditionally contribute lines to the EnvironmentUpdateBuilder.
    # The orchestrator decides whether to emit a coalesced message based on
    # emit_policy (always vs on_change) and cross-module state deltas.
    def build_env_update_message(self, *, agent, turn_idx, tool_calls, tool_payloads, messages, state):
        del agent, turn_idx, tool_calls, tool_payloads, messages
        snapshot = _snapshot(state.images)
        builder = EnvironmentUpdateBuilder.get(state)
        lines: List[str] = ["Currently available image IDs for tool calls:"]
        if snapshot:
            for ref_id, source in snapshot:
                lines.append(f"- {ref_id}: {source}")
            lines.append(
                "When a tool needs image input, pass one of these IDs via the *_image_path field."
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

    def snapshot(self, state: MemoryRunState) -> Tuple[Tuple[str, str], ...]:
        """Public snapshot helper used by orchestrator's on_change emit signal."""
        return _snapshot(state.images)
