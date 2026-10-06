"""Identify image subjects and discover related sources through reverse image search.

Image_Search_Tool sends a resolved image to Serper Lens and presents summarized
visual matches, source-page references and selected candidate images.
"""
import json

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
    """Render reverse-search evidence with ordered image references and image bytes.

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
    # Text-only results use the shared evidence renderer.
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


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Image_Search_Tool',
              'description': 'Identify objects, landmarks, artwork, text, products, plants, '
                             'animals, and more in an image using Google Lens reverse image '
                             'search. Returns a list of visual matches with titles, source URLs, '
                             'links, and thumbnails.\n'
                             '\n'
                             'Use this tool when you need to understand what is depicted in an '
                             'image — it can recognize specific places, products, artworks, logos, '
                             'plants, animals, celebrities, etc. Pass the target image via its '
                             '``image_id`` (``input_image_N`` or ``tool_generated_image_N``) — the '
                             'framework resolves the ID to image bytes before dispatch.\n'
                             '\n'
                             'This tool performs REVERSE image search (image → information). To '
                             'search for images by text description, use Text_Search_Tool with '
                             "search_type='images' instead.\n"
                             '\n'
                             'Search ONE image per call. To reverse-search several images, invoke '
                             'this tool once per image — results from successive calls remain '
                             'visible in your context, so you can correlate them.',
              'parameters': {'type': 'object',
                             'properties': {'image_id': {'type': 'string',
                                                         'description': 'Reference ID of the image '
                                                                        'to search, drawn from the '
                                                                        "Environment Update's "
                                                                        "'Currently available "
                                                                        "image IDs' list. "
                                                                        'Typically '
                                                                        '``input_image_N`` '
                                                                        '(user-provided) or '
                                                                        '``tool_generated_image_N`` '
                                                                        '(produced by a prior tool '
                                                                        'call). The framework '
                                                                        'resolves the ID to image '
                                                                        'bytes before dispatch — '
                                                                        'do NOT pass raw URLs, '
                                                                        'file paths, or base64 '
                                                                        'data here.'}},
                             'required': ['image_id'],
                             'additionalProperties': False}}}

from typing import TypedDict


class ImageMatch(TypedDict):
    """A reverse-image-search match with source links and optional image metadata.

    width and height describe image pixels; position is the provider's result
    rank. Missing titles, snippets, dimensions or links are represented by None.
    """
    title: str | None
    image_url: str | None
    source_url: str | None
    snippet: str | None
    source: str | None
    thumbnail: str | None
    position: int | None
    width: int | None
    height: int | None


class Output(TypedDict):
    """Canonical reverse-search text, optional match records and registered image IDs."""
    text: str | None
    matches: list[ImageMatch] | None
    images: list[str] | None


class Capability(BaseCapability):
    """Expose Serper Lens search and register result images and webpage references."""
    name = 'Image_Search_Tool'
    backends = {'serper_image_search': 'Image_Search_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    rewrite_urls = True

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        """Normalize match fields and expose registered image references alongside search text."""
        text = result if isinstance(result, str) else result.get("text") if isinstance(result, dict) else None
        records = result if isinstance(result, list) else result.get("matches") if isinstance(result, dict) else None
        matches = None
        if isinstance(records, list):
            fields = ("title", "image_url", "source_url", "snippet", "source", "thumbnail",
                      "position", "width", "height")
            matches = [{**{key: item.get(key) for key in fields},
                        "source_url": item.get("source_url", item.get("link"))}
                       for item in records if isinstance(item, dict)]
        images = prepared["image_refs"] or ([] if isinstance(result, dict) and
            isinstance(result.get("kept_image_data_urls"), list) else None)
        return {"text": text, "matches": matches, "images": images}
