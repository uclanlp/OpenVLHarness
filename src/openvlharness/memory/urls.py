import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .run_state import MemoryRunState, UrlsState
from .images import register_image_url_lazy

_URL_REGEX = re.compile(
    r"""https?://(?:[^\s()\[\]\'\"<>\\]+|\([^)\s]*\))+"""
)
_WEBPAGE_URL_REF_PATTERN = re.compile(r"^\[?webpage_url_(\d+)\]?$")
_TRAILING_STRIP = ".,;:!?'\"]}\\"
_IMAGE_URL_EXT_PATTERN = re.compile(
    r"\.(jpe?g|png|webp|gif|bmp|svg|tiff?|ico)$", re.IGNORECASE
)
_IMG_KIND_SENTINEL = " <img>"


def _normalize_tool_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower()


def _safe_domain(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            return ""
        if parsed.port:
            return f"{host}:{parsed.port}"
        return host
    except Exception:
        return ""


def _infer_link_kind(url: str) -> str:
    if not isinstance(url, str):
        return "webpage"
    try:
        parsed = urlparse(url)
        if _IMAGE_URL_EXT_PATTERN.search(parsed.path or ""):
            return "image"
    except Exception:
        pass
    return "webpage"


def register_webpage_url(
    urls: UrlsState,
    url: str,
    *,
    source_tool: str = "",
    turn_idx: int = -1,
) -> Optional[str]:
    if not isinstance(url, str):
        return None
    url = url.strip()
    while url and url[-1] in _TRAILING_STRIP:
        url = url[:-1]
    if not url:
        return None
    existing = urls.url_to_ref.get(url)
    if existing:
        return existing
    urls.webpage_counter += 1
    ref_id = f"webpage_url_{urls.webpage_counter}"
    src_tool = (str(source_tool) if source_tool else "tool").strip() or "tool"
    urls.webpage_registry[ref_id] = {
        "url": url,
        "domain": _safe_domain(url),
        "source": f"tool_output:{src_tool}",
        "first_seen_turn": int(turn_idx) if isinstance(turn_idx, int) else -1,
    }
    urls.url_to_ref[url] = ref_id
    return ref_id


def rewrite_search_result_text(
    state: MemoryRunState,
    text: str,
    *,
    source_tool: str,
    turn_idx: int = -1,
) -> str:
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    sentinel = _IMG_KIND_SENTINEL
    sent_len = len(sentinel)

    out: List[str] = []
    i = 0
    while i < len(text):
        match = _URL_REGEX.search(text, i)
        if match is None:
            out.append(text[i:])
            break
        out.append(text[i:match.start()])
        raw = match.group(0)
        end = match.end()
        has_sentinel = text[end:end + sent_len] == sentinel

        tail = ""
        while raw and raw[-1] in _TRAILING_STRIP:
            tail = raw[-1] + tail
            raw = raw[:-1]
        if not raw:
            out.append(match.group(0))
            i = end
            continue

        if has_sentinel or _infer_link_kind(raw) == "image":
            ref = register_image_url_lazy(
                state.images, raw, source_tool=source_tool, turn_idx=turn_idx,
            )
        else:
            ref = register_webpage_url(
                state.urls, raw, source_tool=source_tool, turn_idx=turn_idx,
            )
        if ref:
            out.append(f"[{ref}]{tail}")
            i = end + sent_len if has_sentinel else end
        else:
            out.append(match.group(0))
            i = end
    return "".join(out)


def _resolve_urls_in_value(
    state: MemoryRunState,
    value: Any,
    *,
    rewrites: List[Dict[str, Any]],
    path: str,
) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        ref_match = _WEBPAGE_URL_REF_PATTERN.match(stripped)
        if ref_match:
            canonical = f"webpage_url_{ref_match.group(1)}"
            entry = state.urls.webpage_registry.get(canonical)
            if isinstance(entry, dict):
                resolved = entry.get("url", value)
                rewrites.append(
                    {
                        "field": path,
                        "original": value,
                        "resolved": resolved,
                        "webpage_url_ref": canonical,
                    }
                )
                return resolved
        return value
    if isinstance(value, list):
        return [
            _resolve_urls_in_value(state, v, rewrites=rewrites, path=f"{path}[{i}]")
            for i, v in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            k: _resolve_urls_in_value(state, v, rewrites=rewrites, path=f"{path}.{k}")
            for k, v in value.items()
        }
    return value


