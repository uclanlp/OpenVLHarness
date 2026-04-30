import base64
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .image_state_memory import ImageStateMemory


# ------------------------------------------------------------------ #
#  Descriptor dataclasses
# ------------------------------------------------------------------ #

@dataclass
class FieldDescriptor:
    """Lightweight metadata for a single field inside a binding."""
    dtype: str = ""            # e.g. "list[list[float]]", "ndarray(1080,1920,float32)"
    description: str = ""      # semantic meaning, e.g. "bounding boxes [x1,y1,x2,y2] px"
    shape_hint: str = ""       # e.g. "len=3, each len=4"
    preview: str = ""          # truncated repr of the value


@dataclass
class BindingDescriptor:
    """Provenance + field metadata for one tool-result binding."""
    key: str = ""                                   # e.g. "grounding_person"
    source_tool: str = ""                           # e.g. "GroundingDINO_Tool"
    source_args: Dict[str, Any] = field(default_factory=dict)  # e.g. {"query": "person"}
    fields: Dict[str, FieldDescriptor] = field(default_factory=dict)


class ToolContextMemory(ImageStateMemory):
    """Extends ImageStateMemory with structured tool result accumulation
    and auto-injection into coding tool context_bindings.

    After each successful tool call, structured metadata (bounding boxes,
    masks, depth arrays, OCR text, etc.) is accumulated in a per-run
    registry.  When a coding tool (Python_Code_Generator_Tool or
    Python_Executor_Tool) is dispatched, all accumulated results are
    automatically injected as ``context_bindings`` so the generated code
    can reference them directly.
    """

    memory_name = "tool_context"
    _TRANSPORT_TYPE_KEY = "__toolvision_type__"
    _TRANSPORT_NDARRAY_TYPE = "ndarray"

    _CODING_TOOL_NAMES = frozenset({
        "Python_Code_Generator_Tool",
        "Python_Executor_Tool",
    })

    # Tools that receive binding *descriptors* (not raw values) for
    # orchestrator-side code execution.
    _CODING_AGENT_TOOL_NAMES = frozenset({
        "Python_Coding_Agent_Tool",
    })

    _TOOL_NAME_TO_BINDING_PREFIX = {
        "GroundingDINO_Tool": "grounding",
        "Visual_Grounding_Tool": "segment",
        "Depth_Tool": "depth",
        "Depth_Estimation_Tool": "depth",
        "OCR_Tool": "ocr",
    }

    # Per-tool: which result keys to extract (include list) and which to
    # exclude.  If a tool is not listed here, its results are skipped.
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
        "Depth_Tool": {
            "include": ["depth_map_base64", "colormap", "size"],
        },
        "Depth_Estimation_Tool": {
            "include": ["depth_meters_b64", "depth_shape", "colormap", "size"],
            "exclude": ["depth_map_base64"],
        },
        "OCR_Tool": {
            "include": ["texts", "results"],
        },
    }

    # Semantic descriptions for known (tool_name, field_name) pairs.
    _FIELD_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
        # GroundingDINO
        ("GroundingDINO_Tool", "boxes"): "bounding boxes [x1,y1,x2,y2] in pixels",
        ("GroundingDINO_Tool", "boxes_normalized"): "bounding boxes [x1,y1,x2,y2] on 0-1000 scale",
        ("GroundingDINO_Tool", "scores"): "detection confidence scores",
        ("GroundingDINO_Tool", "labels"): "detected object labels",
        ("GroundingDINO_Tool", "size"): "[width, height] of source image",
        # Visual_Grounding
        ("Visual_Grounding_Tool", "pred_boxes"): "predicted bounding boxes",
        ("Visual_Grounding_Tool", "pred_scores"): "prediction confidence scores",
        ("Visual_Grounding_Tool", "pred_masks"): "predicted segmentation masks",
        ("Visual_Grounding_Tool", "num_masks"): "number of predicted masks",
        ("Visual_Grounding_Tool", "orig_img_h"): "original image height",
        ("Visual_Grounding_Tool", "orig_img_w"): "original image width",
        # Depth
        ("Depth_Tool", "depth_map_base64"): "base64-encoded depth map image",
        ("Depth_Tool", "colormap"): "colormap used for depth visualization",
        ("Depth_Tool", "size"): "[width, height] of source image",
        ("Depth_Estimation_Tool", "depth_meters"): "per-pixel metric depth in meters",
        ("Depth_Estimation_Tool", "depth_meters_b64"): "base64-encoded metric depth array (float32)",
        ("Depth_Estimation_Tool", "depth_shape"): "[height, width] of depth map",
        ("Depth_Estimation_Tool", "colormap"): "colormap used for depth visualization",
        ("Depth_Estimation_Tool", "size"): "[width, height] of source image",
        # OCR
        ("OCR_Tool", "texts"): "recognized text strings",
        ("OCR_Tool", "results"): "OCR results with text, confidence, and bbox",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tool_result_bindings: Dict[str, Any] = {}
        self._tool_result_binding_counters: Dict[str, int] = {}
        self._tool_result_descriptors: Dict[str, BindingDescriptor] = {}

    def reset(self) -> None:
        super().reset()
        self._tool_result_bindings = {}
        self._tool_result_binding_counters = {}
        self._tool_result_descriptors = {}

    # -------------------------------------------------------------- #
    #  Binding key generation
    # -------------------------------------------------------------- #

    @staticmethod
    def _sanitize_identifier_fragment(text: str, max_len: int = 30) -> str:
        """Convert free text into a valid Python identifier fragment."""
        fragment = text.lower().strip()
        fragment = re.sub(r"[^a-z0-9]+", "_", fragment)
        fragment = fragment.strip("_")
        if fragment and fragment[0].isdigit():
            fragment = f"q_{fragment}"
        return fragment[:max_len].rstrip("_")

    def _make_binding_key(
        self, tool_name: str, tool_args: Dict[str, Any],
    ) -> str:
        prefix = self._TOOL_NAME_TO_BINDING_PREFIX.get(tool_name)
        if prefix is None:
            # Derive prefix from tool name: strip _Tool suffix, snake_case
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

        suffix = self._sanitize_identifier_fragment(query) if query else "result"
        candidate = f"{prefix}_{suffix}" if suffix else prefix

        # Deduplicate
        count = self._tool_result_binding_counters.get(candidate, 0) + 1
        self._tool_result_binding_counters[candidate] = count
        if count > 1:
            return f"{candidate}_{count}"
        return candidate

    # -------------------------------------------------------------- #
    #  Descriptor helpers
    # -------------------------------------------------------------- #

    @staticmethod
    def _infer_dtype(value: Any) -> str:
        """Infer a human-readable dtype string for a value."""
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
            # Peek at first element for nested type hint
            first = value[0]
            if isinstance(first, list):
                inner = f"list[{type(first[0]).__name__}]" if first else "list"
                return f"{tname}[{inner}] len={len(value)}"
            return f"{tname}[{type(first).__name__}] len={len(value)}"
        if isinstance(value, dict):
            return f"dict len={len(value)}"
        return type(value).__name__

    @staticmethod
    def _safe_preview(value: Any, max_len: int = 120) -> str:
        """Truncated repr of a value for descriptor preview."""
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
        self,
        binding_key: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        extracted: Dict[str, Any],
    ) -> BindingDescriptor:
        """Build a BindingDescriptor from extracted tool result fields."""
        # Sanitize tool_args: drop image data and internal keys
        safe_args: Dict[str, Any] = {}
        for k, v in tool_args.items():
            if k in ("image_base64", "image_path", "context_bindings"):
                continue
            if isinstance(v, str) and len(v) > 200:
                continue
            safe_args[k] = v

        fields: Dict[str, FieldDescriptor] = {}
        for fname, fval in extracted.items():
            desc = self._FIELD_DESCRIPTIONS.get((tool_name, fname), "")
            fields[fname] = FieldDescriptor(
                dtype=self._infer_dtype(fval),
                description=desc,
                shape_hint="",
                preview=self._safe_preview(fval),
            )
        return BindingDescriptor(
            key=binding_key,
            source_tool=tool_name,
            source_args=safe_args,
            fields=fields,
        )

    @classmethod
    def _serialize_coding_context_for_transport(cls, value: Any) -> Any:
        """Convert coding context bindings to JSON-safe transport values."""
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                cls._TRANSPORT_TYPE_KEY: cls._TRANSPORT_NDARRAY_TYPE,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data_b64": base64.b64encode(array.tobytes()).decode("ascii"),
            }
        if isinstance(value, dict):
            return {
                key: cls._serialize_coding_context_for_transport(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._serialize_coding_context_for_transport(item)
                for item in value
            ]
        return value

    def get_binding_descriptors_payload(self) -> Dict[str, Any]:
        """Serialize all descriptors to a JSON-safe dict for transport."""
        payload: Dict[str, Any] = {}
        for key, desc in self._tool_result_descriptors.items():
            payload[key] = asdict(desc)
        return payload

    # -------------------------------------------------------------- #
    #  Result extraction
    # -------------------------------------------------------------- #

    def _extract_binding_value(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
    ) -> Optional[Dict[str, Any]]:
        """Extract structured data from a tool result.

        Returns ``None`` if the tool is not supported for binding extraction.
        """
        if not isinstance(result, dict):
            return None

        spec = self._TOOL_RESULT_FIELDS.get(tool_name)
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

        # Decode depth_meters_b64 to ndarray for direct use in coding tools
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

        return extracted

    # -------------------------------------------------------------- #
    #  Accumulation (override consume_tool_payload)
    # -------------------------------------------------------------- #

    def consume_tool_payload(
        self,
        *,
        agent: Any,
        turn_idx: int,
        tool_call: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        output = super().consume_tool_payload(
            agent=agent,
            turn_idx=turn_idx,
            tool_call=tool_call,
            payload=payload,
        )

        if self._is_success_payload(payload):
            tool_name = str(payload.get("tool_name") or "")
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            result = payload.get("result")

            binding_value = self._extract_binding_value(tool_name, tool_args, result)
            if binding_value is not None:
                binding_key = self._make_binding_key(tool_name, tool_args)
                self._tool_result_bindings[binding_key] = binding_value
                self._tool_result_descriptors[binding_key] = self._build_descriptor(
                    binding_key, tool_name, tool_args, binding_value,
                )

        return output

    # -------------------------------------------------------------- #
    #  Injection (override resolve_tool_call_args)
    # -------------------------------------------------------------- #

    def resolve_tool_call_args(
        self,
        *,
        agent: Any,
        args: Dict[str, Any],
        tool_name: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        resolved, resolution_info = super().resolve_tool_call_args(
            agent=agent, args=args, tool_name=tool_name,
        )

        if tool_name in self._CODING_TOOL_NAMES and self._tool_result_bindings:
            existing = resolved.get("context_bindings")
            if not isinstance(existing, dict):
                existing = {}
            merged = dict(self._tool_result_bindings)
            merged.update(existing)  # explicit overrides win
            # Direct Python coding tools cross the /execute JSON boundary.
            # Encode ndarray bindings only for that coding-only transport path;
            # other tools should keep using their normal request schemas.
            resolved["context_bindings"] = self._serialize_coding_context_for_transport(
                merged,
            )

        # Coding agent tools receive lightweight descriptors (not raw values).
        # Actual values are injected orchestrator-side at execution time.
        if tool_name in self._CODING_AGENT_TOOL_NAMES and self._tool_result_descriptors:
            resolved["binding_descriptors"] = self.get_binding_descriptors_payload()

        return resolved, resolution_info

    # -------------------------------------------------------------- #
    #  Environment Update extension
    # -------------------------------------------------------------- #

    @staticmethod
    def _summarize_binding(key: str, value: Any) -> str:
        if isinstance(value, dict):
            keys_str = ", ".join(str(k) for k in value.keys())
            return f"dict (keys: {keys_str})"
        return type(value).__name__

    def _build_environment_message(
        self, snapshot: Tuple[Tuple[str, str], ...],
    ) -> Dict[str, Any]:
        msg = super()._build_environment_message(snapshot)

        if self._tool_result_bindings:
            content = msg.get("content", "")
            lines = [
                "",
                "Available data bindings (auto-injected into coding tools):",
            ]
            for key, value in self._tool_result_bindings.items():
                summary = self._summarize_binding(key, value)
                lines.append(f"- {key}: {summary}")
            msg["content"] = content + "\n".join(lines)

        return msg
