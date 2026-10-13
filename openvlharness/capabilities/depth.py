"""Estimate per-pixel metric depth for spatial measurements in an image.

Depth_Estimation_Tool uses DA3, stores depth arrays in meters for Python
calculations, and returns a colorized visualization for inspection.
"""

import base64
import math
from typing import TypedDict

import numpy as np

from ._base import BaseCapability, store_selected_binding

PROCESSOR_MODE = "custom"


def _decode_depth(result):
    """Validate metric-depth metadata and decode a finite float32 [height, width] array.

    Raises ValueError for missing fields, malformed base64, invalid dimensions,
    mismatched byte counts or non-finite depth values. No artifacts are registered.
    """
    if not isinstance(result, dict):
        raise ValueError("Depth result must be an object")
    shape = result.get("depth_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dim) is not int or dim <= 0 for dim in shape)
    ):
        raise ValueError(
            "depth_shape must be [height, width] with positive integer dimensions"
        )
    size = result.get("size")
    if size is not None and (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(dim) is not int or dim <= 0 for dim in size)
    ):
        raise ValueError(
            "Depth size must be [height, width] with positive integer dimensions"
        )
    encoded = result.get("depth_meters_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Depth result requires nonempty depth_meters_b64")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("depth_meters_b64 must be valid base64") from exc
    if len(decoded) != math.prod(shape) * np.dtype(np.float32).itemsize:
        raise ValueError(
            "depth_meters_b64 byte count does not match float32 depth_shape"
        )
    depth = np.frombuffer(decoded, dtype=np.float32).reshape(shape)
    if not np.isfinite(depth).all():
        raise ValueError("depth_meters must contain only finite values")
    return depth


def process_tool_output(payload, prepared, tool_metadata=None):
    """Render a metric-depth explanation with the prepared visualization.

    Args:
        payload: Execute envelope with the backend result.
        prepared: Evidence fields, including registered image parts and references.
        tool_metadata: Optional metadata accepted by the processor interface.

    Returns:
        A mapping with user_message and processor_info when custom evidence
        is available. None or a fallback record delegates to shared rendering.
    """
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
        lines.append(f"Colorized depth map rendered below; refer to it as {image_ref}.")
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


PUBLIC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Depth_Estimation_Tool",
        "description": "Estimate a monocular depth map from an image. Returns both raw "
        "float32 depthvalues (base64-encoded) and a colorized visualization. Use this "
        "tool for relative near/far reasoning, depth comparison between regions, or "
        "depth visualization. ",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "A runtime image ID from Environment Update "
                    "messages, e.g. input_image_1 or tool_generated_image_1.",
                },
                "colormap": {
                    "type": "string",
                    "description": "Output colormap for visualization",
                    "enum": ["inferno", "grayscale"],
                },
            },
            "additionalProperties": False,
            "required": ["image_id"],
        },
    },
}

FIELD_DESCRIPTIONS = {
    "depth_meters": "per-pixel metric depth in meters",
    "depth_meters_b64": "base64-encoded metric depth array (float32)",
    "depth_shape": "[height, width] of depth map",
    "colormap": "colormap used for depth visualization",
    "size": "[height, width] of source image",
}


class Output(TypedDict):
    """Canonical metric-depth metadata and memory references.

    binding identifies a value containing float32 depth_meters with depth_shape
    [height, width]. The DA3 backend also reports size as [height, width]. image
    is a visualization ID. units is meters; colormap, model_id and device describe
    the backend output.
    """

    binding: str | None  # depth_meters is float32 in this memory binding.
    depth_shape: list[int] | None
    units: str
    colormap: str | None
    size: list[int] | None
    model_id: str | None
    device: str | None
    image: str | None


class Capability(BaseCapability):
    """Expose metric depth estimation with arrays stored in question memory."""

    name = "Depth_Estimation_Tool"
    backends = {"depth_estimation": "Depth_Estimation_Tool"}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    def validate_result(self, payload):
        """Reject unusable arrays before registering images, bindings or evidence."""
        _decode_depth(payload.get("result"))

    def store_result(self, call, result, memory):
        """Store a validated float32 depth array with its dimensions and metadata."""

        def decode(value):
            value["depth_meters"] = _decode_depth(value)
            del value["depth_meters_b64"]
            return value

        return store_selected_binding(
            memory,
            call,
            result,
            prefix="depth",
            fields=["depth_meters_b64", "depth_shape", "colormap", "size"],
            descriptions=FIELD_DESCRIPTIONS,
            convert=decode,
        )

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        """Return metric-depth metadata and references to the array binding and visualization."""
        result = result if isinstance(result, dict) else {}
        return {
            "binding": next(iter(binding_refs), None),
            "depth_shape": result.get("depth_shape"),
            "units": "meters",
            "colormap": result.get("colormap"),
            "size": result.get("size"),
            "model_id": result.get("model_id"),
            "device": result.get("device"),
            "image": next(iter(prepared["image_refs"]), None),
        }
