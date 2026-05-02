import base64
import copy
import datetime
import json
import logging
import math
from typing import Optional
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re
from tool_registry.loader import load_tools

app = FastAPI()
tool_instances = {}
tool_metadata = {}
logger = logging.getLogger("uvicorn.error")

ENV_SNAPSHOT_KEYS = (
    "HOST",
    "PORT",
    "TOOL_MODULES",
    "LIGHT_TOOL_MODULES",
    "HEAVY_TOOL_MODULES",
    "SAM3_SERVER_URL",
    "GROUNDING_DINO_SERVER_URL",
    "OCR_SERVER_URL",
    "OCR_LANGS",
    "DEPTH_ANYTHING_SERVER_URL",
    "DEPTH_ANYTHING_COLORMAP",
    "SUBAGENT_CONFIG_DIR",
    "STRICT_TOOL_LOADING",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_DIRECT_MODEL_BASE_URLS",
    "LLM_DIRECT_MODEL_API_KEYS",
    "LLM_DIRECT_API_KEY",
    "WEBPAGE_VISIT_LLM_MODEL",
    "WEBPAGE_VISIT_LLM_BASE_URL",
    "WEBPAGE_VISIT_LLM_API_KEY",
    "TOOL_SERVER_URL",
)
SENSITIVE_ENV_PATTERNS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
)


def _is_sensitive_env_key(key):
    key_upper = str(key).upper()
    return any(pattern in key_upper for pattern in SENSITIVE_ENV_PATTERNS)


def _redact_env_value(key, value):
    if _is_sensitive_env_key(key):
        return "<REDACTED>"
    return value


def get_runtime_env_snapshot():
    """Return an allowlisted runtime env snapshot for run reproducibility."""
    snapshot = {
        key: _redact_env_value(key, value)
        for key in ENV_SNAPSHOT_KEYS
        if (value := os.environ.get(key)) is not None
    }
    snapshot["_SOURCE"] = "tool_server_runtime"
    snapshot["_REDACTION"] = "allowlisted keys only; sensitive values redacted"
    return snapshot


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


def _load_skills_for_tool(tool_name: str) -> str:
    """Load SKILLS.md for a tool, or auto-generate a schema doc if not found."""
    metadata = tool_metadata.get(tool_name)
    if metadata is None:
        # Try case-insensitive lookup
        for name, meta in tool_metadata.items():
            if name.casefold() == tool_name.casefold():
                metadata = meta
                tool_name = name
                break

    if metadata is not None:
        module_path = metadata.get("tool_module_path", "")
        if module_path:
            # e.g. "tools.sam3_segment.tool" -> "tools/sam3_segment/"
            parts = module_path.rsplit(".", 1)
            rel_dir = parts[0].replace(".", os.sep) if len(parts) > 1 else module_path.replace(".", os.sep)
            tools_base = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
            )
            # rel_dir is like "tools/sam3_segment", strip leading "tools/"
            tool_subdir = rel_dir.split(os.sep, 1)[1] if os.sep in rel_dir else rel_dir
            skills_path = os.path.join(tools_base, tool_subdir, "SKILLS.md")
            if os.path.isfile(skills_path):
                with open(skills_path, "r", encoding="utf-8") as f:
                    return f.read()

    # Auto-generate from metadata
    return _auto_generate_skills(tool_name, metadata)


def _auto_generate_skills(tool_name: str, metadata: Optional[dict]) -> str:
    """Auto-generate a schema document when SKILLS.md is not available."""
    if metadata is None:
        return f"# {tool_name}\nNo schema information available for this tool."

    description = metadata.get("tool_description", "No description available.")
    schema = _project_parameters_for_llm(
        _resolve_parameters_schema(metadata, tool_name)
    )
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [f"# {tool_name} Usage Guide", "", description, "", "## Parameters"]
    if properties:
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            req_label = "required" if param_name in required else "optional"
            lines.append(f"- `{param_name}` ({param_type}, {req_label}): {param_desc}")
    else:
        lines.append("No parameters documented.")

    lines.append("")
    lines.append("## Call Format")
    example_args = {}
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        if param_type == "integer":
            example_args[param_name] = 0
        elif param_type == "number":
            example_args[param_name] = 0.0
        elif param_type == "boolean":
            example_args[param_name] = True
        elif param_type == "array":
            example_args[param_name] = []
        elif param_type == "object":
            example_args[param_name] = {}
        else:
            example_args[param_name] = f"<{param_name}>"

    call_obj = {"type": "tool_call", "name": tool_name, "arguments": example_args}
    lines.append(json.dumps(call_obj, ensure_ascii=False, indent=2))

    return "\n".join(lines)


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


def _normalize_tool_lookup_name(name: str) -> str:
    # Normalize a tool name into a compact comparison form used only for
    # execute-time matching. We intentionally drop separators so common model
    # variants like:
    #   "Visual_Grounding_Tool"
    #   "Visual Grounding Tool"
    #   "Visual-Grounding-Tool"
    # all compare as the same key: "visualgroundingtool".
    text = str(name or "").casefold()
    text = re.sub(r"[\s_-]+", "", text)
    return text


def _tool_lookup_keys(name: str) -> set[str]:
    # Build the exact normalized keys accepted for a single registry/tool name.
    #
    # We always include the fully normalized key, e.g.:
    #   Visual_Grounding_Tool -> {"visualgroundingtool"}
    #
    # If that normalized form ends with "tool", also add a suffix-stripped
    # stem key so requests like "visualgrounding" can resolve without fuzzy
    # matching:
    #   Visual_Grounding_Tool -> {"visualgroundingtool", "visualgrounding"}
    normalized = _normalize_tool_lookup_name(name)
    keys = {normalized}
    if normalized.endswith("tool"):
        stem = normalized[: -len("tool")]
        if stem:
            keys.add(stem)
    return {key for key in keys if key}


def _canonical_registry_name(name: str) -> str:
    # Resolve a matched registry entry back to the canonical registry name so
    # aliases execute through the same underlying tool identity.
    metadata = tool_metadata.get(name) or {}
    return metadata.get("canonical_tool_name", name)


def _resolve_execute_tool(requested_tool_name):
    # Resolution order is intentionally deterministic and gets progressively
    # more tolerant:
    # 1. exact raw-name lookup
    # 2. exact case-insensitive lookup
    # 3. exact normalized lookup (separator-insensitive, optional "tool" suffix)
    # 4. otherwise not found
    #
    # We never do substring guessing or fuzzy scoring here.

    # Stage 1: exact raw match.
    # This preserves current behavior for canonical names and exact aliases.
    tool = tool_instances.get(requested_tool_name)
    if tool is not None:
        return tool, None

    # Stage 2: exact case-insensitive match.
    # This accepts variants like "visual_grounding_tool" when the registry
    # contains "Visual_Grounding_Tool", but still requires the same characters
    # and separators.
    requested_casefold = requested_tool_name.casefold()
    casefold_matches = [
        name for name in tool_instances if name.casefold() == requested_casefold
    ]
    if casefold_matches:
        # Use the first matched registry entry and resolve aliases back to the
        # canonical registry name before dispatch.
        resolved_name = _canonical_registry_name(casefold_matches[0])
        return tool_instances.get(resolved_name), None

    # Stage 3: normalized exact-key match.
    # This is where we accept separator-insensitive forms such as:
    #   "Visual Grounding Tool"
    #   "visualgroundingtool"
    #   "visualgrounding"
    requested_lookup_keys = _tool_lookup_keys(requested_tool_name)
    normalized_matches = {}
    for name in tool_instances:
        candidate_keys = _tool_lookup_keys(name)
        shared_keys = requested_lookup_keys & candidate_keys
        if not shared_keys:
            continue
        # Keep the matching key(s) for debugging and ambiguity reporting.
        normalized_matches[name] = sorted(shared_keys)

    # No match at all across exact, case-insensitive, or normalized lookup.
    if not normalized_matches:
        return None, None

    # Pick the first normalized match and resolve aliases back to the canonical
    # registry name before dispatch.
    resolved_name = _canonical_registry_name(next(iter(normalized_matches)))
    return tool_instances.get(resolved_name), None


@app.on_event("startup")
def load_tool_registry():
    global tool_instances, tool_metadata
    try:
        tool_instances, tool_metadata = load_tools()
        # Inject the schema loader into Get_Tool_Schema so it can
        # serve schemas for all registered tools at runtime.
        gts_instance = tool_instances.get("Get_Tool_Schema")
        if gts_instance is not None and hasattr(gts_instance, "configure"):
            gts_instance.configure(load_fn=_load_skills_for_tool)
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


@app.get("/env")
def env_snapshot():
    return get_runtime_env_snapshot()


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


_LLM_IMAGE_ID_DESCRIPTION = (
    "A runtime image ID from Environment Update messages, "
    "e.g. input_image_1 or tool_generated_image_1."
)


def _project_required_for_llm(required, rename_map):
    if not isinstance(required, list):
        return required
    return [rename_map.get(item, item) for item in required if not _is_image_base64_key(item)]


def _is_image_base64_key(key):
    key_lower = str(key).lower()
    return key_lower.endswith("_base64") and "image" in key_lower


def _is_image_path_key(key):
    key_lower = str(key).lower()
    return key_lower.endswith("_path") and "image" in key_lower


def _project_oneof_branches_for_llm(branches, rename_map):
    if not isinstance(branches, list):
        return branches
    projected = []
    for branch in branches:
        if not isinstance(branch, dict):
            projected.append(branch)
            continue
        branch_required = branch.get("required")
        if isinstance(branch_required, list) and any(
            _is_image_base64_key(name) for name in branch_required
        ):
            # Drop the entire branch when it's the base64 alternative.
            continue
        new_branch = copy.deepcopy(branch)
        new_branch["required"] = _project_required_for_llm(branch_required, rename_map)
        projected.append(new_branch)
    return projected


def _project_parameters_for_llm(parameters):
    """Project a wire-contract parameters block into the agent-facing view.

    Rules:
    - Drop properties whose name matches *image*_base64 (the agent never sets
      these; the resolver populates them from runtime image IDs).
    - Rename *image*_path properties to *image*_id and rewrite their
      description to the canonical "runtime image ID" wording.
    - Update required, oneOf, and allOf to use the renamed names and drop
      base64-only branches. If oneOf collapses to a single branch, fold it
      into required and drop oneOf.
    """
    if not isinstance(parameters, dict):
        return parameters

    projected = copy.deepcopy(parameters)
    properties = projected.get("properties", {})
    if not isinstance(properties, dict):
        return projected

    rename_map = {}
    new_properties = {}
    for name, spec in properties.items():
        if _is_image_base64_key(name):
            continue
        if _is_image_path_key(name):
            new_name = name[: -len("_path")] + "_id"
            rename_map[name] = new_name
            new_spec = spec if isinstance(spec, dict) else {"type": "string"}
            new_spec = copy.deepcopy(new_spec)
            new_spec["description"] = _LLM_IMAGE_ID_DESCRIPTION
            new_properties[new_name] = new_spec
        else:
            new_properties[name] = spec
    projected["properties"] = new_properties

    if "required" in projected:
        projected["required"] = _project_required_for_llm(
            projected["required"], rename_map
        )

    if "oneOf" in projected:
        projected["oneOf"] = _project_oneof_branches_for_llm(
            projected["oneOf"], rename_map
        )
        # If only one branch survives, fold it into required and drop oneOf.
        if len(projected["oneOf"]) == 1 and isinstance(projected["oneOf"][0], dict):
            sole = projected["oneOf"][0]
            sole_required = sole.get("required")
            if isinstance(sole_required, list):
                existing = projected.get("required", [])
                if not isinstance(existing, list):
                    existing = []
                merged = list(existing) + [
                    item for item in sole_required if item not in existing
                ]
                projected["required"] = merged
                del projected["oneOf"]

    if "allOf" in projected and isinstance(projected["allOf"], list):
        rebuilt_allof = []
        for clause in projected["allOf"]:
            if not isinstance(clause, dict):
                rebuilt_allof.append(clause)
                continue
            new_clause = copy.deepcopy(clause)
            if "oneOf" in new_clause:
                new_clause["oneOf"] = _project_oneof_branches_for_llm(
                    new_clause["oneOf"], rename_map
                )
                if len(new_clause["oneOf"]) == 1 and isinstance(
                    new_clause["oneOf"][0], dict
                ):
                    sole_required = new_clause["oneOf"][0].get("required")
                    if isinstance(sole_required, list):
                        existing = projected.get("required", [])
                        if not isinstance(existing, list):
                            existing = []
                        merged = list(existing) + [
                            item for item in sole_required if item not in existing
                        ]
                        projected["required"] = merged
                        # Skip emitting this clause; it's been folded in.
                        continue
            if "required" in new_clause:
                new_clause["required"] = _project_required_for_llm(
                    new_clause["required"], rename_map
                )
            rebuilt_allof.append(new_clause)
        if rebuilt_allof:
            projected["allOf"] = rebuilt_allof
        else:
            del projected["allOf"]

    return projected


@app.get("/llm-schema")
def list_tools_llm_schema():
    """Agent-facing projection of /jsonschema.

    Hides the wire-level base64 inputs (the agent's resolver populates those
    from runtime image IDs) and renames image_path / *_image_path properties
    to image_id / *_image_id so the LLM-facing field names match what the
    model is actually asked to produce.
    """
    projected = []
    for entry in list_tools_jsonschema():
        if not isinstance(entry, dict):
            projected.append(entry)
            continue
        new_entry = copy.deepcopy(entry)
        function = new_entry.get("function")
        if isinstance(function, dict) and "parameters" in function:
            function["parameters"] = _project_parameters_for_llm(function["parameters"])
        projected.append(new_entry)
    return projected

def resolve_tool(tool_name):
    """Resolve a tool by name with case-insensitive fallback.
    
    Args:
        tool_name: The name of the tool to resolve
        
    Returns:
        The tool instance if found, None otherwise
    """
    tool = tool_instances.get(tool_name, None)
    if tool is None:
        for k, v in tool_instances.items():
            if k.lower() == tool_name.lower():
                return v
    return tool

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
