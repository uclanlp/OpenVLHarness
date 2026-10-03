"""Tool-specific output post-processing for Depth_Estimation_Tool."""

PROCESSOR_MODE = "custom"


def process_tool_output(payload, prepared, tool_metadata=None):
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not payload.get("ok"):
        # Delegate to default; tag the processor_info for traceability.
        return {"processor_info": {"mode": PROCESSOR_MODE, "status": "fallback"}}

    image_refs = prepared.get("image_refs") if isinstance(prepared, dict) else []
    image_ref = image_refs[0] if isinstance(image_refs, list) and image_refs else None
    image_parts = prepared.get("image_parts") if isinstance(prepared, dict) else []
    if not isinstance(image_parts, list):
        image_parts = []

    tool_name = str(payload.get("tool_name") or "Depth_Estimation_Tool").strip()
    tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()

    depth_shape = result.get("depth_shape", [])
    colormap = result.get("colormap", "unknown")

    lines = [
        f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:",
        f"Depth estimation complete. Shape: {depth_shape}, colormap: {colormap}.",
        "Metric depth in meters available as `depth_meters` (float32 ndarray).",
    ]
    if image_ref:
        lines.append(
            f"Colorized depth map rendered below; refer to it as {image_ref}."
        )
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
 'function': {'name': 'Depth_Estimation_Tool',
              'description': 'Estimate a monocular depth map from an image. Returns both raw '
                             'float32 depthvalues (base64-encoded) and a colorized visualization. '
                             'Use this tool for relative near/far reasoning, depth comparison '
                             'between regions, or depth visualization. ',
              'parameters': {'type': 'object',
                             'properties': {'image_id': {'type': 'string',
                                                         'description': 'A runtime image ID from '
                                                                        'Environment Update '
                                                                        'messages, e.g. '
                                                                        'input_image_1 or '
                                                                        'tool_generated_image_1.'},
                                            'colormap': {'type': 'string',
                                                         'description': 'Output colormap for '
                                                                        'visualization',
                                                         'enum': ['inferno', 'grayscale']}},
                             'additionalProperties': False,
                             'required': ['image_id']}}}

from ._base import store_selected_binding
FIELD_DESCRIPTIONS = {'depth_meters': 'per-pixel metric depth in meters',
 'depth_meters_b64': 'base64-encoded metric depth array (float32)',
 'depth_shape': '[height, width] of depth map',
 'colormap': 'colormap used for depth visualization',
 'size': '[width, height] of source image'}

from typing import TypedDict


class Output(TypedDict):
    binding: str | None  # depth_meters is float32 in this memory binding.
    depth_shape: list[int] | None
    units: str
    colormap: str | None
    size: list[int] | None
    model_id: str | None
    device: str | None
    image: str | None


class Capability(BaseCapability):
    name = 'Depth_Estimation_Tool'
    backends = {'depth_estimation': 'Depth_Estimation_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def store_result(self, call, result, memory):
        def decode(value):
            import base64
            import numpy as np
            if "depth_meters_b64" in value and "depth_shape" in value:
                try:
                    value["depth_meters"] = np.frombuffer(base64.b64decode(value["depth_meters_b64"]),
                        dtype=np.float32).reshape(value["depth_shape"])
                    del value["depth_meters_b64"]
                except Exception:
                    pass
            return value
        return store_selected_binding(memory, call, result, prefix="depth",
            fields=["depth_meters_b64", "depth_shape", "colormap", "size"],
            descriptions=FIELD_DESCRIPTIONS, convert=decode)

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        result = result if isinstance(result, dict) else {}
        return {"binding": next(iter(binding_refs), None), "depth_shape": result.get("depth_shape"),
            "units": "meters", "colormap": result.get("colormap"), "size": result.get("size"),
            "model_id": result.get("model_id"), "device": result.get("device"),
            "image": next(iter(prepared["image_refs"]), None)}
