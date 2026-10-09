"""Resolve capability image arguments and separate image evidence from text payloads.

Capabilities interpret backend result fields; memory stores image bytes and IDs.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from ..memory.run_state import ImagesState
from ..memory.images import (
    _is_likely_base64,
    resolve_registry_image_data_url,
    register_image_data_url,
    next_tool_generated_image_ref_id,
)

_DROP_VALUE = object()
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


def _is_image_id_key(key: Any) -> bool:
    """Recognize a public singular image-reference argument by its field name."""
    key_lower = str(key).lower()
    if key_lower.startswith("_"):
        return False
    return key_lower.endswith("_id") and "image" in key_lower


def _is_image_ids_key(key: Any) -> bool:
    """Recognize a public plural image-reference argument by its field name."""
    key_lower = str(key).lower()
    if key_lower.startswith("_"):
        return False
    return key_lower.endswith("_ids") and "image" in key_lower


def _resolve_image_ids_in_args(
    images: ImagesState,
    value: Any,
    *,
    path: str,
    rewrites: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> None:
    """Replace image ID fields with base64 fields in a nested argument structure.

    Mutates value in place, preserves source IDs under private keys, and appends
    resolution metadata or errors to the supplied lists. Existing base64 fields
    take precedence. Lazy references may trigger HTTP downloads.
    """
    if isinstance(value, dict):
        for key, child in list(value.items()):
            child_path = f"{path}.{key}"
            if isinstance(child, (dict, list)):
                _resolve_image_ids_in_args(
                    images, child, path=child_path, rewrites=rewrites, errors=errors
                )

        for id_key in list(value.keys()):
            if _is_image_ids_key(id_key):
                base64_key = f"{id_key[:-4]}_base64s"
                if value.get(base64_key):
                    continue

                raw_ids = value.get(id_key)
                if not isinstance(raw_ids, list):
                    continue

                resolved_urls = []
                unresolved = []
                for raw_id in raw_ids:
                    if not isinstance(raw_id, str) or not raw_id.strip():
                        unresolved.append(raw_id)
                        continue
                    data_url, matched_ref, fallback_mode = (
                        resolve_registry_image_data_url(images, raw_id)
                    )
                    if data_url:
                        resolved_urls.append(data_url)
                        rewrite_item = {
                            "field": f"{path}.{id_key}",
                            "target_field": f"{path}.{base64_key}",
                            "original": raw_id,
                        }
                        if matched_ref:
                            rewrite_item["image_ref"] = matched_ref
                        if fallback_mode:
                            rewrite_item["fallback_mode"] = fallback_mode
                        rewrites.append(rewrite_item)
                    else:
                        unresolved.append(raw_id)

                if unresolved:
                    errors.append(
                        {
                            "field": f"{path}.{id_key}",
                            "value": unresolved,
                            "reason": "unresolved_image_reference",
                            "available_image_ids": list(images.registry.keys()),
                        }
                    )
                    continue

                value[base64_key] = resolved_urls
                value[f"_{id_key}"] = list(raw_ids)
                value.pop(id_key, None)
                continue

            if not _is_image_id_key(id_key):
                continue
            base64_key = f"{id_key[:-3]}_base64"
            if value.get(base64_key):
                continue

            raw_id = value.get(id_key)
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue

            data_url, matched_ref, fallback_mode = resolve_registry_image_data_url(
                images, raw_id
            )
            if data_url:
                value[base64_key] = data_url
                value.pop(id_key, None)
                rewrite_item = {
                    "field": f"{path}.{id_key}",
                    "target_field": f"{path}.{base64_key}",
                    "original": raw_id,
                }
                if matched_ref:
                    rewrite_item["image_ref"] = matched_ref
                if fallback_mode:
                    rewrite_item["fallback_mode"] = fallback_mode
                rewrites.append(rewrite_item)
            elif not os.path.isfile(raw_id):
                errors.append(
                    {
                        "field": f"{path}.{id_key}",
                        "value": raw_id,
                        "reason": "unresolved_image_reference",
                        "available_image_ids": list(images.registry.keys()),
                    }
                )
        return

    if isinstance(value, list):
        for idx, child in enumerate(value):
            if isinstance(child, (dict, list)):
                _resolve_image_ids_in_args(
                    images,
                    child,
                    path=f"{path}[{idx}]",
                    rewrites=rewrites,
                    errors=errors,
                )


def _is_image_result_key(key: Any) -> bool:
    """Identify renderable image fields while excluding raw numeric data fields."""
    k = str(key).lower()
    if any(hint in k for hint in _RAW_DATA_HINTS):
        return False
    if k in _IMAGE_RESULT_KEY_HINTS:
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


def _guess_image_mime(key: Any) -> str:
    """Infer JPEG from a field name, defaulting other image fields to PNG."""
    k = str(key).lower()
    if "jpg" in k or "jpeg" in k:
        return "image/jpeg"
    return "image/png"


def _to_image_data_url(key: Any, value: Any) -> Optional[str]:
    """Normalize an image field to a data URL, or return None for unrecognized values."""
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
    """Copy a result tree while collecting image parts and omitting their encoded text.

    image_parts is appended in traversal order. Empty image-only subtrees return
    the private drop sentinel; other values remain in the compact result.
    """
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
        should_try_image = (
            is_explicit_image or image_context or _is_image_result_key(key_hint)
        )
        if should_try_image:
            url = _to_image_data_url(key_hint or "image", raw)
            if url:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
                return _DROP_VALUE
    return value


def _extract_image_parts(result: Any) -> Tuple[List[Dict[str, Any]], Any]:
    """Return ordered image content parts and a result copy without embedded image bytes."""
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
    """Register output image parts with tool provenance and return their memory IDs."""
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


def prepare_images(payload, prepared, state):
    """Extract, register and summarize image artifacts from a successful tool result.

    Args:
        payload: Execute envelope containing the backend result and public tool name.
        prepared: Mutable evidence mapping to populate with compact text, image
            parts and image references.
        state: Current question state receiving registered images.

    Returns:
        The updated prepared mapping. Numeric raw-data fields remain in text;
        recognized encoded image fields become separate multimodal content parts.
    """
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
    prepared.update(
        {
            "text_payload": text_payload,
            "compact_payload": compact_payload,
            "compact_result": compact_result,
            "image_parts": image_parts,
            "image_refs": image_refs,
        }
    )
    return prepared
