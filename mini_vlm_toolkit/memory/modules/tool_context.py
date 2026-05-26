import base64
import re
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..module import EnvironmentUpdateBuilder, MemoryModule
from ..run_state import BindingDescriptor, FieldDescriptor, MemoryRunState

_CODING_AGENT_TOOL_NAMES = frozenset({
    "Python_Coding_Agent_Tool",
})

_TOOL_NAME_TO_BINDING_PREFIX = {
    "GroundingDINO_Tool": "grounding",
    "Visual_Grounding_Tool": "segment",
    "Depth_Estimation_Tool": "depth",
    "Camera_Trajectory_Tool": "camera_trajectory",
    "OCR_Tool": "ocr",
    "Tracking_Tool": "tracking",
}

_TOOL_RESULT_FIELDS: Dict[str, Dict[str, Any]] = {
    "GroundingDINO_Tool": {
        "include": ["boxes", "boxes_normalized", "scores", "labels", "size"],
    },
    "Visual_Grounding_Tool": {
        "include": [
            "pred_boxes", "pred_scores", "pred_masks", "num_masks",
            "orig_img_h", "orig_img_w",
        ],
        "exclude": ["rendered_image_b64", "all_masks_viz_base64"],
    },
    "Depth_Estimation_Tool": {
        "include": ["depth_meters_b64", "depth_shape", "colormap", "size"],
        "exclude": ["depth_map_base64"],
    },
    "Camera_Trajectory_Tool": {
        "include": [
            "intrinsics", "extrinsics", "image_ids", "num_images",
            "cameras", "coordinate_system", "reference_image_id",
        ],
        "exclude": ["bird_eye_view_image_base64"],
    },
    "OCR_Tool": {
        "include": ["texts", "results"],
    },
    "Tracking_Tool": {
        "include": [
            "status", "target", "image_ids", "tracks", "movement",
            "ambiguities",
        ],
        "exclude": ["grounding_images"],
    },
}

_FIELD_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
    ("GroundingDINO_Tool", "boxes"): "bounding boxes [x1,y1,x2,y2] in pixels",
    ("GroundingDINO_Tool", "boxes_normalized"): "bounding boxes [x1,y1,x2,y2] on 0-1000 scale",
    ("GroundingDINO_Tool", "scores"): "detection confidence scores",
    ("GroundingDINO_Tool", "labels"): "detected object labels",
    ("GroundingDINO_Tool", "size"): "[width, height] of source image",
    ("Visual_Grounding_Tool", "pred_boxes"): "predicted bounding boxes",
    ("Visual_Grounding_Tool", "pred_scores"): "prediction confidence scores",
    ("Visual_Grounding_Tool", "pred_masks"): "predicted segmentation masks",
    ("Visual_Grounding_Tool", "num_masks"): "number of predicted masks",
    ("Visual_Grounding_Tool", "orig_img_h"): "original image height",
    ("Visual_Grounding_Tool", "orig_img_w"): "original image width",
    ("Depth_Estimation_Tool", "depth_meters"): "per-pixel metric depth in meters",
    ("Depth_Estimation_Tool", "depth_meters_b64"): "base64-encoded metric depth array (float32)",
    ("Depth_Estimation_Tool", "depth_shape"): "[height, width] of depth map",
    ("Depth_Estimation_Tool", "colormap"): "colormap used for depth visualization",
    ("Depth_Estimation_Tool", "size"): "[width, height] of source image",
    ("Camera_Trajectory_Tool", "intrinsics"): "per-image 3x3 camera intrinsic matrices (ndarray shape=(N,3,3))",
    ("Camera_Trajectory_Tool", "extrinsics"): "per-image 3x4 world-to-camera matrices (OpenCV/COLMAP), ndarray shape=(N,3,4)",
    ("Camera_Trajectory_Tool", "image_ids"): "ordered runtime image IDs aligned with intrinsics/extrinsics",
    ("Camera_Trajectory_Tool", "num_images"): "number of images processed",
    ("Camera_Trajectory_Tool", "cameras"): "per-image dicts (in reference frame): camera_center_delta, movement_summary, forward_direction, rotation_summary, yaw_deg, pitch_deg",
    ("Camera_Trajectory_Tool", "coordinate_system"): "human-readable coordinate convention used for cameras",
    ("Camera_Trajectory_Tool", "reference_image_id"): "image_id of the reference frame (always image_ids[0])",
    ("OCR_Tool", "texts"): "recognized text strings",
    ("OCR_Tool", "results"): "OCR results with text, confidence, and bbox",
    ("Tracking_Tool", "tracks"): "selected target detections across images",
    ("Tracking_Tool", "movement"): "2D movement and relative depth change summary",
}


def _sanitize_identifier_fragment(text: str, max_len: int = 30) -> str:
    fragment = text.lower().strip()
    fragment = re.sub(r"[^a-z0-9]+", "_", fragment)
    fragment = fragment.strip("_")
    if fragment and fragment[0].isdigit():
        fragment = f"q_{fragment}"
    return fragment[:max_len].rstrip("_")


def _make_binding_key(state: MemoryRunState, tool_name: str, tool_args: Dict[str, Any]) -> str:
    prefix = _TOOL_NAME_TO_BINDING_PREFIX.get(tool_name)
    if prefix is None:
        raw = tool_name
        if raw.endswith("_Tool"):
            raw = raw[:-5]
        raw = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw)
        raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
        prefix = re.sub(r"_+", "_", raw).lower().strip("_")

    query = ""
    for key in ("query", "prompt", "text"):
        val = tool_args.get(key)
        if isinstance(val, str) and val.strip():
            query = val.strip()
            break

    suffix = _sanitize_identifier_fragment(query) if query else "result"
    candidate = f"{prefix}_{suffix}" if suffix else prefix

    counters = state.bindings.binding_counters
    count = counters.get(candidate, 0) + 1
    counters[candidate] = count
    if count > 1:
        return f"{candidate}_{count}"
    return candidate


def _infer_dtype(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, np.ndarray):
        return f"ndarray shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, str):
        return f"str len={len(value)}"
    if isinstance(value, (list, tuple)):
        tname = "list" if isinstance(value, list) else "tuple"
        if not value:
            return f"{tname} len=0"
        first = value[0]
        if isinstance(first, list):
            inner = f"list[{type(first[0]).__name__}]" if first else "list"
            return f"{tname}[{inner}] len={len(value)}"
        return f"{tname}[{type(first).__name__}] len={len(value)}"
    if isinstance(value, dict):
        return f"dict len={len(value)}"
    return type(value).__name__


def _safe_preview(value: Any, max_len: int = 120) -> str:
    try:
        if isinstance(value, np.ndarray):
            if value.size > 0:
                return f"range=[{float(value.min()):.4g}, {float(value.max()):.4g}]"
            return "empty"
        r = repr(value)
    except Exception:
        return ""
    if len(r) > max_len:
        return r[: max_len - 3] + "..."
    return r


def _build_descriptor(
    binding_key: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    extracted: Dict[str, Any],
) -> BindingDescriptor:
    safe_args: Dict[str, Any] = {}
    for k, v in tool_args.items():
        if k in ("image_base64", "image_id", "binding_descriptors"):
            continue
        if isinstance(v, str) and len(v) > 200:
            continue
        safe_args[k] = v

    fields: Dict[str, FieldDescriptor] = {}
    for fname, fval in extracted.items():
        desc = _FIELD_DESCRIPTIONS.get((tool_name, fname), "")
        fields[fname] = FieldDescriptor(
            dtype=_infer_dtype(fval),
            description=desc,
            shape_hint="",
            preview=_safe_preview(fval),
        )
    return BindingDescriptor(
        key=binding_key,
        source_tool=tool_name,
        source_args=safe_args,
        fields=fields,
    )


def _extract_binding_value(
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    spec = _TOOL_RESULT_FIELDS.get(tool_name)
    if spec is None:
        return None

    include = spec.get("include")
    exclude = set(spec.get("exclude", []))

    extracted: Dict[str, Any] = {}
    if include:
        for key in include:
            if key in result and key not in exclude:
                extracted[key] = result[key]
    else:
        for key, value in result.items():
            if key not in exclude:
                extracted[key] = value

    if not extracted:
        return None

    if "depth_meters_b64" in extracted and "depth_shape" in extracted:
        try:
            raw = base64.b64decode(extracted["depth_meters_b64"])
            shape = extracted["depth_shape"]
            extracted["depth_meters"] = np.frombuffer(
                raw, dtype=np.float32,
            ).reshape(shape)
            del extracted["depth_meters_b64"]
        except Exception:
            pass

    if tool_name == "Camera_Trajectory_Tool":
        for key in ("intrinsics", "extrinsics"):
            if key in extracted:
                try:
                    extracted[key] = np.asarray(extracted[key], dtype=np.float32)
                except Exception:
                    pass

    return extracted


def _summarize_binding(value: Any) -> str:
    if isinstance(value, dict):
        keys_str = ", ".join(str(k) for k in value.keys())
        return f"dict (keys: {keys_str})"
    return type(value).__name__


def get_binding_descriptors_payload(state: MemoryRunState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, desc in state.bindings.descriptors.items():
        payload[key] = asdict(desc)
    return payload


class ToolContextModule(MemoryModule):
    """Tool-result binding accumulation + binding_descriptors injection for
    `Python_Coding_Agent_Tool`. The deprecated `Python_Code_Generator_Tool` /
    `Python_Executor_Tool` (which received raw transport-encoded
    `context_bindings`) are no longer supported.
    """

    name = "tool_context"

    # ---- before_tool_call ----
    def before_tool_call(self, *, agent, tool_name, args, state):
        if not isinstance(args, dict):
            return args, {"rewrites": [], "errors": []}
        if tool_name in _CODING_AGENT_TOOL_NAMES and state.bindings.descriptors:
            args["binding_descriptors"] = get_binding_descriptors_payload(state)
        return args, {"rewrites": [], "errors": []}

    # ---- consume_tool_result ----
    # Pure side effect: extract bindings from the raw result and store them in
    # `state.bindings`. Does not mutate `prepared`. Tolerates `tool_call={}`
    # (called from `prepare_tool_message_payload` with no call context) — when
    # tool_args is empty we skip accumulation to avoid attributing a binding
    # to no caller.
    def consume_tool_result(self, *, agent, turn_idx, tool_call, payload, prepared, state):
        del agent, turn_idx
        if not (isinstance(payload, dict) and payload.get("ok") is True):
            return prepared

        tool_args = tool_call.get("args") if isinstance(tool_call, dict) else None
        if not isinstance(tool_args, dict) or not tool_args:
            return prepared

        tool_name = str(payload.get("tool_name") or "")
        result = payload.get("result")
        binding_value = _extract_binding_value(tool_name, tool_args, result)
        if binding_value is not None:
            binding_key = _make_binding_key(state, tool_name, tool_args)
            state.bindings.bindings[binding_key] = binding_value
            state.bindings.descriptors[binding_key] = _build_descriptor(
                binding_key, tool_name, tool_args, binding_value,
            )
        return prepared

    # ---- build_env_update_message ----
    def build_env_update_message(self, *, agent, turn_idx, tool_calls, tool_payloads, messages, state):
        del agent, turn_idx, tool_calls, tool_payloads, messages
        if not state.bindings.bindings:
            return []
        builder = EnvironmentUpdateBuilder.get(state)
        lines = ["Available data bindings (auto-injected into Python_Coding_Agent_Tool):"]
        for key, value in state.bindings.bindings.items():
            summary = _summarize_binding(value)
            lines.append(f"- {key}: {summary}")
        builder.add_section(lines)
        return []
