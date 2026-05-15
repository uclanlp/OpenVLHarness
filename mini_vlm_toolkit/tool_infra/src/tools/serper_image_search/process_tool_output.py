"""Image_Search_Tool output post-processing for multimodal summarize.

When ``Image_Search_Tool.execute`` returns a dict ``{"text": ..., "kept_image_data_urls": [...]}``
(SEARCH_MULTIMODAL_SUMMARIZE=true, at least one kept-rank image fetched), this
processor:

  1. Reads the URL-rewritten ``text`` from ``prepared['text_payload']`` (URLs in
     the rendered text have already been replaced with ``[webpage_url_N]`` refs
     by UrlModule).
  2. Substitutes each ``Image: __MULTIMODAL_IMG_REF_PLACEHOLDER__`` placeholder
     with ``Image: [tool_generated_image_N]`` in kept-rank order, sourcing refs
     from ``prepared['image_refs']``.
  3. Builds the agent-visible user_message: ``role=user`` content list with a
     text block (``Tool X returned:\\nImage refs: ...\\n<rendered text>``)
     followed by image_parts (the actual fetched bytes).

For text-only mode (result is a string) we return ``None`` to fall through to
the framework's default builder, which already handles the legacy text-only
shape correctly.
"""
import json

PROCESSOR_MODE = "custom"
MULTIMODAL_IMAGE_REF_PLACEHOLDER = "__MULTIMODAL_IMG_REF_PLACEHOLDER__"


def _recover_text_with_url_refs(prepared, fallback_text):
    """Parse ``prepared['text_payload']`` (JSON-encoded after UrlModule's URL
    rewriting) and extract ``result.text``. Falls back to ``fallback_text``
    when JSON parsing fails (agent then sees raw https URLs instead of refs —
    no information lost, just less convenient).
    """
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
    """Replace each placeholder occurrence with ``[tool_generated_image_N]``,
    consuming refs left-to-right. Excess placeholders (refs run out) emit a
    visible ``[]`` so a count mismatch is observable rather than silent.
    """
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
    # Text-only mode: defer to default builder (handles legacy string results).
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

    tool_name = str(payload.get("tool_name") or "Image_Search_Tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    lines = [f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:"]
    if image_refs:
        lines.append(f"Image refs: {', '.join(str(r) for r in image_refs)}")
    lines.append(rendered_text)
    text = "\n".join(lines)

    if image_parts:
        # Interleave the bare ``tool_generated_image_N`` ref name as a text
        # block BEFORE each image_part so the agent has a tight visual binding
        # between the rendered bytes and the ref it should pass to follow-up
        # tools. The label is just the ref name (no ``image `` prefix) so the
        # agent does not accidentally include extra prefix tokens when copying
        # the ref into a tool call argument. ``image_parts`` and ``image_refs``
        # are guaranteed to be in the same order (both kept-rank order); if
        # lengths disagree (shouldn't happen) we fall back to a generic label
        # so no image is silently dropped.
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
