"""Register structured tool values and describe them for Python code generation."""
import copy
import re
from dataclasses import asdict
from typing import Any, Dict
import numpy as np
from .run_state import BindingDescriptor, FieldDescriptor, MemoryRunState

def _sanitize_identifier_fragment(text: str, max_len: int = 30) -> str:
    """Return a bounded lowercase identifier fragment derived from free text."""
    fragment = text.lower().strip()
    fragment = re.sub(r"[^a-z0-9]+", "_", fragment)
    fragment = fragment.strip("_")
    if fragment and fragment[0].isdigit():
        fragment = f"q_{fragment}"
    return fragment[:max_len].rstrip("_")

def _infer_dtype(value: Any) -> str:
    """Describe a value's type and available shape or collection length."""
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
    """Return a bounded representation or array range, suppressing preview errors."""
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

def _summarize_binding(value: Any) -> str:
    """Summarize a binding by its dictionary keys or Python type name."""
    if isinstance(value, dict):
        keys_str = ", ".join(str(k) for k in value.keys())
        return f"dict (keys: {keys_str})"
    return type(value).__name__

def get_binding_descriptors_payload(state: MemoryRunState) -> Dict[str, Any]:
    """Return binding descriptors as independent JSON-compatible dictionaries."""
    payload: Dict[str, Any] = {}
    for key, desc in state.bindings.descriptors.items():
        payload[key] = asdict(desc)
    return payload

def store_binding(state, *, prefix, query, value, source_tool, source_args, descriptions):
    """Register a structured value with a unique, query-derived variable name.

    Args:
        state: Mutable state for the current question.
        prefix: Capability-specific variable-name prefix.
        query: Task text used to derive a readable suffix.
        value: Mapping of field names to values, stored without copying.
        source_tool: Public name of the producing tool.
        source_args: Tool arguments to summarize, excluding large/image fields.
        descriptions: Human-readable meanings keyed by field name.

    Returns:
        The unique binding key. Type descriptions and previews are stored beside
        the value; execution callers must request copies before mutating it.
    """
    suffix = _sanitize_identifier_fragment(query) if query else "result"
    candidate = f"{prefix}_{suffix}" if suffix else prefix
    count = state.bindings.binding_counters.get(candidate, 0) + 1
    state.bindings.binding_counters[candidate] = count
    key = f"{candidate}_{count}" if count > 1 else candidate
    safe_args = {}
    for name, item in source_args.items():
        if name in ("image_base64", "image_id", "binding_descriptors"):
            continue
        if isinstance(item, str) and len(item) > 200:
            continue
        safe_args[name] = item
    descriptor = BindingDescriptor(key=key, source_tool=source_tool, source_args=safe_args,
        fields={name: FieldDescriptor(dtype=_infer_dtype(item), description=descriptions.get(name, ""),
                shape_hint="", preview=_safe_preview(item)) for name, item in value.items()})
    state.bindings.bindings[key] = value
    state.bindings.descriptors[key] = descriptor
    return key


def copy_bindings_for_execution(state):
    """Deep-copy stored values so generated code cannot mutate question memory."""
    return copy.deepcopy(state.bindings.bindings)
