"""Generate verl tool_config.yaml from integrate-verl tool server schemas."""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

import requests
import yaml

CURATED_V1_TOOLS = [
    "SAM3_Segment_Tool",
]


def _request_json(url: str, timeout_sec: float) -> Any:
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _sanitize_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}

    cleaned = copy.deepcopy(parameters)
    cleaned["type"] = "object"

    properties = cleaned.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    sanitized_props = {}
    for key, value in properties.items():
        if not isinstance(value, dict):
            sanitized_props[str(key)] = {"type": "string"}
            continue
        prop = copy.deepcopy(value)
        if "type" not in prop or not isinstance(prop.get("type"), str):
            prop["type"] = "string"
        sanitized_props[str(key)] = prop
    cleaned["properties"] = sanitized_props

    required = cleaned.get("required", [])
    if not isinstance(required, list):
        required = []
    required = [item for item in required if isinstance(item, str) and item in sanitized_props]
    cleaned["required"] = required

    return cleaned


def _build_tool_schema(schema_item: dict[str, Any]) -> dict[str, Any]:
    function = schema_item.get("function") if isinstance(schema_item, dict) else {}
    name = function.get("name")
    description = function.get("description")
    parameters = _sanitize_parameters(function.get("parameters"))

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(description or ""),
            "parameters": parameters,
        },
    }


def generate_tool_config(tool_base: str, tool_names: list[str], timeout_sec: float) -> dict[str, Any]:
    tool_base = tool_base.rstrip("/")
    tools_metadata = _request_json(f"{tool_base}/tools", timeout_sec)
    schemas = _request_json(f"{tool_base}/jsonschema", timeout_sec)

    if not isinstance(tools_metadata, dict):
        raise RuntimeError("/tools must return a JSON object")
    if not isinstance(schemas, list):
        raise RuntimeError("/jsonschema must return a JSON list")

    canonical_names = set()
    for name, metadata in tools_metadata.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("is_alias"):
            continue
        canonical_names.add(str(metadata.get("canonical_tool_name") or name))

    schema_map = {}
    for item in schemas:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        # Prefer first entry so canonical definitions win over any duplicate alias rows.
        schema_map.setdefault(name, item)

    requested = tool_names or CURATED_V1_TOOLS
    missing = [name for name in requested if name not in canonical_names]
    if missing:
        raise RuntimeError(
            "Requested canonical tool(s) missing from /tools: " + ", ".join(sorted(missing))
        )

    tools = []
    for name in requested:
        if name not in schema_map:
            raise RuntimeError(f"Tool schema for {name} not found in /jsonschema")

        tools.append(
            {
                "class_name": "rl.tools.tool_server_adapter.MiniVLMHTTPTool",
                "config": {
                    "type": "native",
                    "tool_name": name,
                    "tool_base": tool_base,
                    "request_timeout_sec": 120,
                },
                "tool_schema": _build_tool_schema(schema_map[name]),
            }
        )

    return {"tools": tools}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verl tool config from integrate-verl tool server.")
    parser.add_argument("--tool-base", default="http://127.0.0.1:8101", help="Tool server base URL")
    parser.add_argument(
        "--tools",
        default=",".join(CURATED_V1_TOOLS),
        help="Comma-separated canonical tool names",
    )
    parser.add_argument(
        "--output",
        default="rl/configs/tool_config.yaml",
        help="Output YAML path",
    )
    parser.add_argument("--timeout-sec", type=float, default=10.0, help="HTTP timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_names = [item.strip() for item in args.tools.split(",") if item.strip()]
    config = generate_tool_config(args.tool_base, tool_names, args.timeout_sec)

    output = args.output
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(json.dumps({"output": output, "tool_count": len(config["tools"]), "tools": tool_names}, indent=2))


if __name__ == "__main__":
    main()
