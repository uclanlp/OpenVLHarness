"""Text_Search_Tool output post-processing for multimodal summarize.

When ``Text_Search_Tool.execute`` is called with ``search_type=images`` and
multimodal mode is on, ``execute`` may return a dict
``{"text": ..., "kept_image_data_urls": [...]}``. This processor mirrors the
logic in ``serper_image_search.process_tool_output``: substitute per-entry
placeholders with framework-assigned image refs and build a user_message
carrying both the rendered text and the fetched image bytes.

All other ``search_type`` values (search / news / scholar / ...) return string
results; for those we return ``None`` so the framework's default builder runs.
"""
import json

PROCESSOR_MODE = "custom"
MULTIMODAL_IMAGE_REF_PLACEHOLDER = "__MULTIMODAL_IMG_REF_PLACEHOLDER__"


def _recover_text_with_url_refs(prepared, fallback_text):
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
        prepared, fallback_text=result.get("text", ""),
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
