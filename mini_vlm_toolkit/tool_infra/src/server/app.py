import base64
import copy
import datetime
import json
import logging
import math
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tool_registry.loader import load_tools

app = FastAPI()
tool_instances = {}
tool_metadata = {}
logger = logging.getLogger("uvicorn.error")


def _format_input_types(input_types):
    if isinstance(input_types, dict):
        return ", ".join(f"{key}: {value}" for key, value in input_types.items())
    if isinstance(input_types, (list, tuple)):
        return ", ".join(str(item) for item in input_types)
    if input_types is None:
        return ""
    return str(input_types)


def _format_routes():
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        name = getattr(route, "name", None)
        if not path or not methods:
            continue
        method_list = ", ".join(sorted(m for m in methods if m))
        routes.append(f"{method_list} {path} ({name})")
    return sorted(routes)


def _infer_json_type(type_hint):
    hint = str(type_hint).lower()
    if "int" in hint:
        return "integer"
    if "float" in hint or "double" in hint or "number" in hint:
        return "number"
    if "bool" in hint:
        return "boolean"
    if "list" in hint or "array" in hint:
        return "array"
    if "dict" in hint or "object" in hint:
        return "object"
    return "string"


def _input_types_to_schema(input_types):
    properties = {}
    if isinstance(input_types, dict):
        for key, desc in input_types.items():
            properties[key] = {
                "type": _infer_json_type(desc),
                "description": str(desc),
            }
    elif isinstance(input_types, (list, tuple)):
        for idx, item in enumerate(input_types):
            key = f"arg{idx}"
            properties[key] = {
                "type": _infer_json_type(item),
                "description": str(item),
            }
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }


def _validate_input_schema(input_schema, tool_name):
    if input_schema is None:
        return
    if not isinstance(input_schema, dict):
        raise ValueError(f"{tool_name}: input_schema must be a dict")
    if input_schema.get("type") != "object":
        raise ValueError(f"{tool_name}: input_schema.type must be 'object'")

    properties = input_schema.get("properties", {})
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        raise ValueError(f"{tool_name}: input_schema.properties must be a dict")

    required = input_schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValueError(f"{tool_name}: input_schema.required must be a list[str]")

    missing = [item for item in required if item not in properties]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"{tool_name}: input_schema.required contains unknown properties: {missing_str}"
        )

    if "additionalProperties" in input_schema and not isinstance(
        input_schema["additionalProperties"], (bool, dict)
    ):
        raise ValueError(
            f"{tool_name}: input_schema.additionalProperties must be bool or dict"
        )

    if "oneOf" in input_schema and not isinstance(input_schema["oneOf"], list):
        raise ValueError(f"{tool_name}: input_schema.oneOf must be a list")


def _resolve_parameters_schema(metadata, tool_name):
    input_schema = metadata.get("input_schema")
    if input_schema is not None:
        _validate_input_schema(input_schema, tool_name)
        return copy.deepcopy(input_schema)
    return _input_types_to_schema(metadata.get("input_types"))


def _count_tools():
    canonical_count = 0
    alias_count = 0
    for metadata in tool_metadata.values():
        if metadata.get("is_alias"):
            alias_count += 1
        else:
            canonical_count += 1
    return canonical_count, alias_count


def _normalize_for_json(value, path="$", warnings=None):
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
    return {
        "ok": True,
        "tool_name": tool_name,
        "result": result,
        "error": None,
        "tool_call_id": tool_call_id,
        "meta": meta or {},
    }


def _resolve_execute_tool(requested_tool_name):
    # Keep exact match semantics first for deterministic behavior.
    tool = tool_instances.get(requested_tool_name)
    if tool is not None:
        return tool, None

    requested_casefold = requested_tool_name.casefold()
    casefold_matches = [
        name for name in tool_instances if name.casefold() == requested_casefold
    ]
    if not casefold_matches:
        return None, None

    canonical_matches = {}
    for matched_name in casefold_matches:
        metadata = tool_metadata.get(matched_name) or {}
        canonical_name = metadata.get("canonical_tool_name", matched_name)
        canonical_matches.setdefault(canonical_name, []).append(matched_name)

    if len(canonical_matches) > 1:
        details = {
            "requested_tool_name": requested_tool_name,
            "case_insensitive_matches": sorted(casefold_matches),
            "canonical_tool_names": sorted(canonical_matches),
        }
        return None, _build_error_payload(
            tool_name=requested_tool_name,
            tool_call_id=None,
            code="tool_name_ambiguous_case",
            message=(
                "Tool name is ambiguous under case-insensitive matching: "
                f"{requested_tool_name}"
            ),
            detail=details,
        )

    matched_names = next(iter(canonical_matches.values()))
    resolved_name = sorted(matched_names)[0]
    for matched_name in matched_names:
        metadata = tool_metadata.get(matched_name) or {}
        if metadata.get("is_alias") is False:
            resolved_name = matched_name
            break

    return tool_instances.get(resolved_name), None


@app.on_event("startup")
def load_tool_registry():
    global tool_instances, tool_metadata
    try:
        tool_instances, tool_metadata = load_tools()
        if tool_metadata:
            routes = _format_routes()
            if routes:
                print("Endpoints:")
                for route in routes:
                    print(f"  - {route}")
            print("Serving tools:")
            for tool_name in sorted(tool_metadata):
                metadata = tool_metadata[tool_name]
                _validate_input_schema(metadata.get("input_schema"), tool_name)
                canonical_tool_name = metadata.get("canonical_tool_name", tool_name)
                if metadata.get("is_alias"):
                    print(f"  - {tool_name} (alias for {canonical_tool_name})")
                    continue

                inputs = _format_input_types(metadata.get("input_types")) or "none"
                output_type = metadata.get("output_type") or "none"
                aliases = metadata.get("tool_aliases") or []
                alias_suffix = f" | aliases: {aliases}" if aliases else ""
                print(
                    f"  - {tool_name} (inputs: {inputs}) -> {output_type}{alias_suffix}"
                )
    except Exception as exc:
        print(f"Tool startup failed: {exc}")
        raise SystemExit(1)


class ExecuteRequest(BaseModel):
    tool_name: str
    args: dict
    tool_call_id: Optional[str] = None


@app.get("/tools")
def list_tools():
    return tool_metadata


@app.get("/health")
def health_check():
    canonical_count, alias_count = _count_tools()
    return {
        "status": "ok",
        "tool_count": len(tool_metadata),
        "canonical_tool_count": canonical_count,
        "alias_tool_count": alias_count,
        "tools": sorted(tool_metadata),
    }


@app.get("/jsonschema")
def list_tools_jsonschema():
    json_schemas = []
    for tool_name in sorted(tool_metadata):
        metadata = tool_metadata[tool_name]
        json_schemas.append(
            {
                "function":
                    {
                    "name": metadata.get("tool_name", tool_name),
                    "description": metadata.get("tool_description", ""),
                    "parameters": _resolve_parameters_schema(metadata, tool_name),
                },
                "type": "function"
            }
            
        )
    return json_schemas


def execute_tool_request(request: ExecuteRequest):
    tool, resolution_error = _resolve_execute_tool(request.tool_name)
    if resolution_error is not None:
        resolution_error["tool_call_id"] = request.tool_call_id
        return resolution_error, 404
    if tool is None:
        return (
            _build_error_payload(
                tool_name=request.tool_name,
                tool_call_id=request.tool_call_id,
                code="tool_not_found",
                message=f"Tool not found: {request.tool_name}",
            ),
            404,
        )

    try:
        result = tool.execute(**request.args)
    except Exception as exc:
        logger.exception(
            "Tool execution failed: tool_name=%s tool_call_id=%s exception_type=%s arg_keys=%s",
            request.tool_name,
            request.tool_call_id,
            type(exc).__name__,
            sorted(request.args.keys()),
        )
        return (
            _build_error_payload(
                tool_name=request.tool_name,
                tool_call_id=request.tool_call_id,
                code="tool_execution_error",
                message=str(exc),
                detail={"exception_type": type(exc).__name__},
            ),
            500,
        )

    normalized_result, meta = _normalize_result_payload(result)
    return (
        _build_success_payload(
            tool_name=request.tool_name,
            tool_call_id=request.tool_call_id,
            result=normalized_result,
            meta=meta,
        ),
        200,
    )


@app.post("/execute")
def execute_tool(request: ExecuteRequest):
    payload, status_code = execute_tool_request(request)
    if status_code >= 400:
        return JSONResponse(status_code=status_code, content=payload)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)
