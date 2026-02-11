#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_HOST = "127.0.0.1"
DEFAULT_TIMEOUT_SEC = 5.0
VALID_ENDPOINTS = {"tools", "jsonschema"}


def _normalize_endpoint(endpoint):
    normalized = endpoint.strip().lower()
    if normalized.startswith("/"):
        normalized = normalized[1:]
    if normalized not in VALID_ENDPOINTS:
        raise ValueError(
            f"Unsupported endpoint: {endpoint}. Use /tools or /jsonschema."
        )
    return normalized


def _fetch_json(host, port, endpoint, timeout_sec):
    url = f"http://{host}:{port}/{endpoint}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    return url, json.loads(payload)


def _clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def _stringify(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _format_input_types(input_types):
    if input_types is None:
        return "none"
    if isinstance(input_types, dict):
        if not input_types:
            return "none"
        parts = []
        for key in sorted(input_types):
            parts.append(f"{key}: {_stringify(input_types[key])}")
        return ", ".join(parts)
    if isinstance(input_types, list):
        if not input_types:
            return "none"
        return ", ".join(_stringify(item) for item in input_types)
    return _stringify(input_types)


def _print_tools(payload):
    if not isinstance(payload, dict):
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return

    tool_names = sorted(payload.keys())
    print(f"Tool count: {len(tool_names)}")
    for idx, tool_name in enumerate(tool_names, start=1):
        metadata = payload.get(tool_name) or {}
        canonical_name = metadata.get("canonical_tool_name", tool_name)
        is_alias = bool(metadata.get("is_alias"))
        description = _clean_text(metadata.get("tool_description", ""))
        aliases = metadata.get("tool_aliases") or []
        output_type = metadata.get("output_type") or "none"
        inputs = _format_input_types(metadata.get("input_types"))

        print(f"{idx}. {tool_name}")
        if is_alias:
            print(f"   alias_for: {canonical_name}")
            continue

        if canonical_name != tool_name:
            print(f"   canonical_name: {canonical_name}")
        print(f"   inputs: {inputs}")
        print(f"   output: {output_type}")
        print(f"   description: {description or '(none)'}")
        if aliases:
            print(f"   aliases: {', '.join(str(item) for item in aliases)}")


def _print_jsonschema(payload):
    if not isinstance(payload, list):
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return

    print(f"Tool schema count: {len(payload)}")
    for idx, item in enumerate(payload, start=1):
        function_meta = item.get("function", {}) if isinstance(item, dict) else {}
        tool_name = function_meta.get("name", f"<unnamed_{idx}>")
        description = _clean_text(function_meta.get("description", ""))
        parameters = function_meta.get("parameters", {})

        print(f"{idx}. {tool_name}")
        print(f"   description: {description or '(none)'}")
        print("   parameters:")
        parameters_dump = json.dumps(
            parameters, indent=2, ensure_ascii=False, sort_keys=True
        )
        for line in parameters_dump.splitlines():
            print(f"     {line}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Request /tools or /jsonschema from the tool server and pretty-print "
            "the tool metadata."
        )
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        required=True,
        help="Tool server port (for example: 8101).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Tool server host (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default="/tools",
        help="Endpoint to request: /tools or /jsonschema (default: /tools).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SEC}).",
    )
    args = parser.parse_args()

    try:
        endpoint = _normalize_endpoint(args.endpoint)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        url, payload = _fetch_json(args.host, args.port, endpoint, args.timeout)
    except urllib.error.HTTPError as exc:
        print(f"Request failed: HTTP {exc.code} for {exc.url}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc.reason}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Response was not valid JSON: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"Endpoint: {url}")
    if endpoint == "tools":
        _print_tools(payload)
    else:
        _print_jsonschema(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
