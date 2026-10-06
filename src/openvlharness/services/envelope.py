"""The router envelope and JSON normalization; arrays stay in local memory."""
import base64
import datetime
import json
import math

def _normalize_for_json(value, path="$", warnings=None):
    """Recursively convert values to JSON-safe data and append lossy-conversion warnings."""
    warnings = warnings if warnings is not None else []
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        warnings.append(f"{path}: non-finite float converted to string")
        return str(value)

    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, str):
                key_out = key
            else:
                key_out = str(key)
                warnings.append(f"{path}: non-string dict key converted to string")
            normalized[key_out] = _normalize_for_json(
                item,
                path=f"{path}.{key_out}",
                warnings=warnings,
            )
        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_for_json(item, path=f"{path}[{idx}]", warnings=warnings)
            for idx, item in enumerate(value)
        ]

    if isinstance(value, (set, frozenset)):
        warnings.append(f"{path}: set/frozenset converted to list")
        sortable = list(value)
        try:
            sortable = sorted(sortable)
        except Exception:
            sortable = sorted(sortable, key=lambda item: repr(item))
        return [
            _normalize_for_json(item, path=f"{path}[{idx}]", warnings=warnings)
            for idx, item in enumerate(sortable)
        ]

    if isinstance(value, (bytes, bytearray)):
        warnings.append(f"{path}: bytes converted to base64 string")
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"base64:{encoded}"

    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        warnings.append(f"{path}: date/time converted to ISO string")
        return value.isoformat()

    warnings.append(f"{path}: {type(value).__name__} converted to string")
    return str(value)

def _normalize_result_payload(result):
    """Return a JSON-safe result and metadata describing any normalization warnings."""
    warnings = []
    normalized = _normalize_for_json(result, path="result", warnings=warnings)
    try:
        json.dumps(normalized, allow_nan=False)
    except Exception:
        warnings.append("result: payload stringified after normalization failure")
        normalized = str(normalized)

    meta = {}
    if warnings:
        meta["normalization_warnings"] = warnings
    return normalized, meta

def _build_error_payload(tool_name, tool_call_id, code, message, detail=None):
    """Build a failed execute envelope with a code, message and optional detail."""
    error_payload = {"code": code, "message": str(message)}
    if detail is not None:
        error_payload["detail"] = detail
    return {
        "ok": False,
        "tool_name": tool_name,
        "result": None,
        "error": error_payload,
        "tool_call_id": tool_call_id,
        "meta": {},
    }

def _build_success_payload(tool_name, tool_call_id, result, meta=None):
    """Build a successful execute envelope from an already normalized result."""
    return {
        "ok": True,
        "tool_name": tool_name,
        "result": result,
        "error": None,
        "tool_call_id": tool_call_id,
        "meta": meta or {},
    }
