import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..module import EnvironmentUpdateBuilder, MemoryModule
from ..run_state import MemoryRunState, UrlsState
from .image import register_image_url_lazy

_URL_REGEX = re.compile(
    r"""https?://(?:[^\s()\[\]\'\"<>\\]+|\([^)\s]*\))+"""
)
_WEBPAGE_URL_REF_PATTERN = re.compile(r"^\[?webpage_url_(\d+)\]?$")
_TRAILING_STRIP = ".,;:!?'\"]}\\"
_SEARCH_TOOL_NAMES = frozenset({
    "text_search_tool",
    "image_search_tool",
    "gemini_google_search_tool",
})
_WEBPAGE_VISIT_TOOL_NAMES = frozenset({
    "webpage_visit_tool",
})
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


class UrlModule(MemoryModule):
    """Webpage URL registry + URL rewriting on search-tool result text."""

    name = "url"

    def __init__(
        self,
        *,
        url_rewrite_enabled: bool = True,
        webpage_env_update_limit: Optional[int] = None,
    ) -> None:
        if webpage_env_update_limit is not None and (
            not isinstance(webpage_env_update_limit, int)
            or webpage_env_update_limit <= 0
        ):
            raise ValueError(
                "webpage_env_update_limit must be a positive int or None, "
                f"got {webpage_env_update_limit!r}."
            )
        self.url_rewrite_enabled = bool(url_rewrite_enabled)
        self.webpage_env_update_limit = webpage_env_update_limit

    # ---- before_tool_call ----
    def before_tool_call(self, *, agent, tool_name, args, state):
        if not isinstance(args, dict):
            return args, {"rewrites": [], "errors": []}
        rewrites: List[Dict[str, Any]] = []
        if (
            self.url_rewrite_enabled
            and _normalize_tool_name(tool_name) in _WEBPAGE_VISIT_TOOL_NAMES
        ):
            for url_key in ("urls", "url"):
                if url_key in args:
                    args[url_key] = _resolve_urls_in_value(
                        state, args[url_key], rewrites=rewrites, path=f"args.{url_key}",
                    )
        return args, {"rewrites": rewrites, "errors": []}

    # ---- consume_tool_result ----
    # URL rewriting on search-tool text_payload. Runs AFTER ImageModule's
    # consume_tool_result has produced text_payload via json.dumps so the
    # URLs scanned are exactly the bytes the agent will see. Image-kind URLs
    # are routed to register_image_url_lazy (state.images); webpage URLs go
    # to register_webpage_url (state.urls).
    def consume_tool_result(self, *, agent, turn_idx, tool_call, payload, prepared, state):
        del agent, turn_idx, tool_call
        if not self.url_rewrite_enabled:
            return prepared
        tool_name_norm = _normalize_tool_name(payload.get("tool_name"))
        if tool_name_norm not in _SEARCH_TOOL_NAMES:
            return prepared
        text = prepared.get("text_payload", "")
        if not isinstance(text, str) or not text:
            return prepared
        prepared["text_payload"] = rewrite_search_result_text(
            state, text, source_tool=str(payload.get("tool_name") or "tool"),
        )
        return prepared

    # ---- build_env_update_message ----
    def build_env_update_message(self, *, agent, turn_idx, tool_calls, tool_payloads, messages, state):
        del agent, turn_idx, tool_calls, tool_payloads, messages
        registry = state.urls.webpage_registry
        if not registry:
            return []
        entries = sorted(
            registry.items(),
            key=lambda kv: int(_WEBPAGE_URL_REF_PATTERN.match(kv[0]).group(1)),
        )
        if self.webpage_env_update_limit is not None:
            entries = entries[-self.webpage_env_update_limit:]
        lines: List[str] = ["Currently registered webpage URL refs:"]
        for ref_id, entry in entries:
            source = entry.get("source", "")
            domain = entry.get("domain", "")
            if domain:
                lines.append(f"- {ref_id}: {source} ({domain})")
            else:
                lines.append(f"- {ref_id}: {source}")
        lines.append(
            "When you want to visit a webpage, pass a webpage_url_N to Webpage_Visit_Tool.urls."
        )
        builder = EnvironmentUpdateBuilder.get(state)
        builder.add_section(lines)
        return []
