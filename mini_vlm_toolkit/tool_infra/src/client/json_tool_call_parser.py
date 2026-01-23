"""Parser for JSON tool-call payloads."""

import json
import re


def _strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_code_blocks(text):
    matches = re.findall(
        r"```\s*(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```",
        text,
        re.DOTALL,
    )
    return [match.strip() for match in matches if match.strip()]


def _extract_json_blocks(text):
    matches = re.findall(r"```\s*json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return [match.strip() for match in matches if match.strip()]


def _coerce_args(value, key_name):
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{key_name} must be a JSON object")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key_name} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{key_name} must be a JSON object")
    return value


def _normalize_tool_call(payload):
    if isinstance(payload, list):
        if len(payload) == 0:
            raise ValueError("tool calls list is empty")
        if len(payload) > 1:
            raise ValueError("Multiple tool calls found")
        return _normalize_tool_call(payload[0])

    if not isinstance(payload, dict):
        return None

    if "tool_name" in payload or "args" in payload:
        if "tool_name" not in payload or "args" not in payload:
            raise ValueError("tool_name and args are required for JSON tool calls")
        tool_name = payload["tool_name"]
        args = _coerce_args(payload["args"], "args")
        key_name = "tool_name"
    elif "name" in payload or "arguments" in payload:
        if "name" not in payload or "arguments" not in payload:
            raise ValueError("name and arguments are required for JSON tool calls")
        tool_name = payload["name"]
        args = _coerce_args(payload["arguments"], "arguments")
        key_name = "name"
    elif "tool_call" in payload:
        return _normalize_tool_call(payload["tool_call"])
    elif "tool_calls" in payload:
        tool_calls = payload["tool_calls"]
        if not isinstance(tool_calls, list):
            raise ValueError("tool_calls must be a list")
        if len(tool_calls) == 0:
            raise ValueError("tool_calls is empty")
        if len(tool_calls) > 1:
            raise ValueError("Multiple tool calls found")
        return _normalize_tool_call(tool_calls[0])
    elif "function" in payload:
        return _normalize_tool_call(payload["function"])
    else:
        return None

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError(f"{key_name} must be a non-empty string")
    return {"tool_name": tool_name, "args": args}


def _parse_json_tool_call_from_text(text):
    decoder = json.JSONDecoder()
    idx = 0
    tool_calls = []

    while idx < len(text):
        brace_index = text.find("{", idx)
        if brace_index == -1:
            break
        try:
            payload, end = decoder.raw_decode(text[brace_index:])
        except json.JSONDecodeError:
            idx = brace_index + 1
            continue
        try:
            tool_call = _normalize_tool_call(payload)
        except ValueError as exc:
            raise exc
        if tool_call:
            tool_calls.append(tool_call)
            if len(tool_calls) > 1:
                raise ValueError("Multiple tool calls found")
        idx = brace_index + end

    if not tool_calls:
        raise ValueError("No JSON tool call found")
    return tool_calls[0]


def _should_abort_on_error(exc):
    return isinstance(exc, ValueError) and str(exc) == "Multiple tool calls found"


def parse_json_tool_call(text):
    cleaned = _strip_think_blocks(text)
    candidates = []
    json_blocks = _extract_json_blocks(cleaned)
    if json_blocks:
        candidates.extend(json_blocks)
    for block in _extract_code_blocks(cleaned):
        if block not in candidates:
            candidates.append(block)
    candidates.append(cleaned)

    last_error = None
    parsed = None
    for candidate in candidates:
        try:
            parsed = _parse_json_tool_call_from_text(candidate)
            break
        except ValueError as exc:
            if _should_abort_on_error(exc):
                raise
            last_error = exc

    if parsed is None:
        if last_error:
            raise last_error
        raise ValueError("No JSON tool call found")

    try:
        _parse_json_tool_call_from_text(cleaned)
    except ValueError as exc:
        if _should_abort_on_error(exc):
            raise

    return parsed


def parse_json_tool_call_args(text):
    return parse_json_tool_call(text)["args"]
