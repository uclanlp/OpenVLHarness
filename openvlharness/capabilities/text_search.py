"""Find web pages, images, news, scholarly articles, places or products from text.

Text_Search_Tool queries the selected Serper endpoint and presents summarized
evidence. Image searches can include fetched pixels; result links become memory IDs.
"""

import json
from typing import TypedDict

from ._base import BaseCapability

PROCESSOR_MODE = "custom"
MULTIMODAL_IMAGE_REF_PLACEHOLDER = "__MULTIMODAL_IMG_REF_PLACEHOLDER__"


def _recover_text_with_url_refs(prepared, fallback_text):
    """Read URL-rewritten result text, falling back when the prepared JSON is invalid."""
    text_payload = prepared.get("text_payload") if isinstance(prepared, dict) else None
    if isinstance(text_payload, str) and text_payload.strip():
        try:
            parsed = json.loads(text_payload)
        except (json.JSONDecodeError, ValueError):
            return fallback_text
        if isinstance(parsed, dict):
            result = parsed.get("result")
            if isinstance(result, dict):
                candidate = result.get("text")
                if isinstance(candidate, str):
                    return candidate
    return fallback_text


def _substitute_placeholders(text, image_refs):
    """Replace image placeholders in order; emit [] when references run out."""
    out_parts = []
    pos = 0
    ref_iter = iter(image_refs)
    while True:
        idx = text.find(MULTIMODAL_IMAGE_REF_PLACEHOLDER, pos)
        if idx < 0:
            out_parts.append(text[pos:])
            break
        out_parts.append(text[pos:idx])
        try:
            ref = next(ref_iter)
            out_parts.append(f"[{ref}]")
        except StopIteration:
            out_parts.append("[]")
        pos = idx + len(MULTIMODAL_IMAGE_REF_PLACEHOLDER)
    return "".join(out_parts)


def process_tool_output(payload, prepared, tool_metadata=None):
    """Render multimodal search evidence, using default rendering for text-only results.

    Args:
        payload: Execute envelope with the backend result.
        prepared: Evidence fields, including registered image parts and references.
        tool_metadata: Optional metadata accepted by the processor interface.

    Returns:
        A mapping with user_message and processor_info when custom evidence
        is available. None or a fallback record delegates to shared rendering.
    """
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {
            "processor_info": {
                "mode": PROCESSOR_MODE,
                "status": "fallback",
                "fallback_reason": "non_success_payload",
            }
        }

    result = payload.get("result")
    if not isinstance(result, dict) or "text" not in result:
        return None

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    if not isinstance(image_refs, list):
        image_refs = []
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    rendered_text = _recover_text_with_url_refs(
        prepared,
        fallback_text=result.get("text", ""),
    )
    rendered_text = _substitute_placeholders(rendered_text, image_refs)

    tool_name = str(payload.get("tool_name") or "Text_Search_Tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    lines = [f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:"]
    if image_refs:
        lines.append(f"Image refs: {', '.join(str(r) for r in image_refs)}")
    lines.append(rendered_text)
    text = "\n".join(lines)

    if image_parts:
        # Interleave the bare ``tool_generated_image_N`` ref name as a text
        # block before each image_part. No ``image `` prefix — agent should
        # copy the ref into tool call args verbatim, prefix-free. ``image_parts``
        # and ``image_refs`` are in the same kept-rank order.
        content_blocks = [{"type": "text", "text": text}]
        for idx, part in enumerate(image_parts):
            ref = image_refs[idx] if idx < len(image_refs) else "tool_generated_image_?"
            content_blocks.append({"type": "text", "text": ref})
            content_blocks.append(part)
        user_message = {"role": "user", "content": content_blocks}
    else:
        user_message = {"role": "user", "content": text}

    return {
        "user_message": user_message,
        "processor_info": {
            "mode": PROCESSOR_MODE,
            "status": "ok",
            "kept_image_count": len(image_refs),
        },
    }


PUBLIC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Text_Search_Tool",
        "description": "Search the caller-selected endpoint using search_type: general web search (the default), images, news, scholar, maps, places, shopping, or patents. Summarizes the retrieved evidence and returns retained, reranked results with source titles, snippets or metadata, and available links. Image search can include retained images and a multimodal summary. Use the summary and retained sources to plan follow-up searches; use Webpage_Visit_Tool to read a source page in detail.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string. Can be a natural-language "
                    "question, a keyword phrase, a product name, an academic paper "
                    "title, etc. Choose the appropriate search_type for your query, or "
                    "leave it as 'search' for general web results.",
                },
                "search_type": {
                    "type": "string",
                    "enum": [
                        "search",
                        "images",
                        "news",
                        "scholar",
                        "maps",
                        "places",
                        "shopping",
                        "patents",
                    ],
                    "default": "search",
                    "description": "Which search endpoint to use. Defaults to 'search' "
                    "(general web). Options:\n"
                    "  'search'   — general web search (titles, snippets, URLs) — use "
                    "for most queries\n"
                    "  'news'     — recent news articles, current events, "
                    "time-sensitive information\n"
                    "  'scholar'  — academic papers, citations, research "
                    "publications\n"
                    "  'images'   — find images matching a text description\n"
                    "  'maps'     — map locations with addresses, ratings, and contact "
                    "info\n"
                    "  'places'   — local businesses, restaurants, map results\n"
                    "  'shopping' — product listings with prices\n"
                    "  'patents'  — patent filings and inventions",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class Output(TypedDict):
    """Canonical search text, discovered URLs and optional registered result image IDs."""

    text: str | None
    urls: list[str] | None
    images: list[str] | None


class Capability(BaseCapability):
    """Expose Serper search and rewrite result URLs into question-memory references."""

    name = "Text_Search_Tool"
    backends = {"serper_text_search": "Text_Search_Tool"}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    rewrite_urls = True

    def translate_args(self, args, memory):
        """Validate that the selected backend supports the requested search type."""
        if (
            self.backend != "serper_text_search"
            and args.get("search_type", "search") != "search"
        ):
            raise ValueError(f"{self.backend} supports only search_type=search")
        return args

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        """Preserve search text and URLs while exposing image evidence as memory references."""
        import re

        text = (
            result
            if isinstance(result, str)
            else result.get("text")
            if isinstance(result, dict)
            else None
        )
        urls = result.get("urls") if isinstance(result, dict) else None
        if urls is None and text is not None:
            urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>"\\]+', text)))
        images = prepared["image_refs"] or (
            []
            if isinstance(result, dict)
            and isinstance(result.get("kept_image_data_urls"), list)
            else None
        )
        return {"text": text, "urls": urls, "images": images}
