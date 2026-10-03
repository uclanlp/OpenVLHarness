"""Zoom-In-specific output post-processing for agent-visible messages."""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    tool_name = str(payload.get("tool_name") or "Zoom_In_Tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    lines = [f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:"]
    if image_ref:
        lines.append(f"Zoomed in image rendered below; refer to it as {image_ref}.")
    else:
        lines.append("Zoom-in crop succeeded, but no image reference is available.")
    text = "\n".join(lines)

    if image_parts:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": text}] + list(image_parts),
        }
    else:
        user_message = {"role": "user", "content": text}

    return {
        "user_message": user_message,
        "processor_info": {"mode": PROCESSOR_MODE, "status": "ok"},
    }


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Zoom_In_Tool',
              'description': 'Zoom into a region by cropping an image with a normalized bbox [x1, '
                             'y1, x2, y2] plus padding, and return the cropped image. Use this '
                             'tool for closer visual inspection after you already know the region. '
                             'All coords values must be normalized on the 0-1000 scale (each value '
                             'in [0, 1000]). When trying ti count or ground objects, please try '
                             'different prompts using Visual_Grounding_Tool first Only use this '
                             'tool to ground objects when the Visual_Grounding_Tool does not work.',
              'parameters': {'type': 'object',
                             'properties': {'image_id': {'type': 'string',
                                                         'description': 'A runtime image ID from '
                                                                        'Environment Update '
                                                                        'messages, e.g. '
                                                                        'input_image_1 or '
                                                                        'tool_generated_image_1.'},
                                            'coords': {'type': 'array',
                                                       'description': 'Normalized bbox [x1, y1, '
                                                                      'x2, y2] on a 0-1000 scale '
                                                                      'where every value must be '
                                                                      'in [0, 1000]. (x1, y1) is '
                                                                      'top-left and (x2, y2) is '
                                                                      'bottom-right.',
                                                       'items': {'type': 'number'}},
                                            'padding': {'type': 'number',
                                                        'description': 'Extra normalized margin on '
                                                                       'each side on the 0-1000 '
                                                                       'scale (>= 50).'}},
                             'required': ['coords']}}}

from typing import TypedDict


class Output(TypedDict):
    image: str | None


class Capability(BaseCapability):
    name = 'Zoom_In_Tool'
    backends = {'zoom_in_image_by_bbox': 'Zoom_In_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        return {"image": next(iter(prepared["image_refs"]), None)}
