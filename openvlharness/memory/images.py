"""Image registration, alias resolution and lazy downloading for question memory."""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..media import local_image_to_data_url
from .run_state import ImagesState

_INPUT_IMAGE_REF_PATTERN = re.compile(r"^input_image_(\d+)$")
_TOOL_GENERATED_IMAGE_REF_PATTERN = re.compile(r"^tool_generated_image_(\d+)$")
_IMAGE_REGISTRY_DEFAULT_ALIASES = ("image", "image.jpg", "image.jpeg", "image.png")


def _normalize_image_alias(alias: Any) -> str:
    """Return a case-insensitive image alias, or an empty string for invalid input."""
    if not isinstance(alias, str):
        return ""
    return alias.strip().lower()


def _is_likely_base64(value: Any) -> bool:
    """Heuristically recognize inline image data without decoding its bytes."""
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
    """Increment the image counter and return the next tool_generated_image ID."""
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
    """Register image bytes and aliases, deduplicating identical data URLs.

    Args:
        images: Mutable image registry state.
        ref_id: Proposed ID, used only when the data URL is not registered.
        data_url: Image data URL to store.
        source: Provenance label displayed in inventories.
        aliases: Additional names for case-insensitive lookup.

    Returns:
        The assigned or existing ID, or None for a non-image data URL.
    """
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
    """Register a remote image URL for download on first resolution.

    Args:
        images: Mutable image registry state.
        url: Remote URL; trailing prose punctuation is removed.
        source_tool: Tool name recorded in the source label.
        turn_idx: Accepted for shared registration call sites; not stored.

    Returns:
        A deduplicated tool-generated image ID, or None for empty/non-string input.
        The entry's data_url remains None until a successful fetch.
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
    """Download a URL as image data; return None on HTTP errors or non-image MIME types."""
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
    """Fetch image bytes into an existing entry and update reverse lookup on success."""
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
    """Resolve a URL alias or download/register it, returning data, ID and source note."""
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
    """Resolve inline data, aliases, registry IDs or remote image URLs.

    Args:
        images: Mutable image registry, updated when a remote fetch succeeds.
        image_ref: Reference string supplied by a caller.

    Returns:
        A (data_url, registry_id, resolution_note) tuple. Inline data has no ID;
        notes identify downloads, lazy fetches or single-image fallback. Local
        files must already be registered. Unknown names may use the sole registry
        entry when exactly one exists. Unresolved values return None fields.
    """
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
    """Replace the image registry and alias indexes using normalized input items.

    Local files and inline data are registered as input_image_1, input_image_2,
    and so on. Invalid image items are skipped. Returns the mutable registry.
    """
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


def _snapshot_sort_key(ref_id: str) -> Tuple[int, int, str]:
    """Order input images before generated images, using numeric ID order."""
    match = _INPUT_IMAGE_REF_PATTERN.match(ref_id)
    if match:
        return (0, int(match.group(1)), ref_id)
    match = _TOOL_GENERATED_IMAGE_REF_PATTERN.match(ref_id)
    if match:
        return (1, int(match.group(1)), ref_id)
    return (2, 0, ref_id)

def _snapshot(images: ImagesState) -> Tuple[Tuple[str, str], ...]:
    """Return ordered image ID/source pairs for inventory rendering and change detection."""
    rows = []
    for ref_id in sorted(images.registry.keys(), key=_snapshot_sort_key):
        entry = images.registry.get(ref_id) or {}
        rows.append((ref_id, str(entry.get("source", "runtime"))))
    return tuple(rows)
