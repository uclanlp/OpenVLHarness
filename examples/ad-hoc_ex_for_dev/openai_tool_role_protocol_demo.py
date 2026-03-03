#!/usr/bin/env python3
"""
Demonstrate strict native tool-calling protocol behavior on OpenAI endpoints.

This script runs:
1) First model call with tools enabled and a prompt that usually triggers
   Python_Executor_Tool.
2) Local tool execution via the tool server (/execute).
3) Two second model calls:
   - Protocol-correct: assistant tool_calls -> role=tool reply.
   - Protocol-incorrect: assistant tool_calls -> role=user summary.

On strict OpenAI endpoints, the protocol-incorrect second call is expected to
fail because each tool_call must be followed by a matching role=tool message.

Prerequisites:
- Tool server running (default: http://127.0.0.1:8101) with Python_Executor_Tool loaded.
- OpenAI API key available.

Example:
  OPENAI_API_KEY=... python examples/ad-hoc_ex_for_dev/openai_tool_role_protocol_demo.py
"""

import argparse
import json
import os
import uuid
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests


TOOL_NAME = "Python_Executor_Tool"

DEFAULT_SYSTEM_PROMPT = (
    "You are a math assistant. "
    "Use Python_Executor_Tool for arithmetic before answering."
)

DEFAULT_USER_PROMPT = (
    "Use Python_Executor_Tool to compute 137*19+845. "
    "Return the exact integer result."
)


def _normalize_chat_endpoint(base_url: str) -> str:
    trimmed = str(base_url).rstrip("/")
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/v1/chat/completions"


def _looks_like_openai_host(endpoint: str) -> bool:
    try:
        parsed = urlparse(endpoint)
    except Exception:
        return False
    return (parsed.netloc or "").lower() == "api.openai.com"


def _is_gpt5(model: str) -> bool:
    return str(model or "").strip().lower().startswith("gpt-5")


def _truncate(value: Any, max_chars: int = 320) -> str:
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    value = value.replace("\n", "\\n")
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... (truncated)"


def _extract_error_detail(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return _truncate(body)


def _build_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Execute Python code and return printed output and variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }


def _build_chat_payload(
    *,
    endpoint: str,
    model: str,
    messages: list[Dict[str, Any]],
    tools: Optional[list[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    max_tokens: int = 512,
    temperature: float = 1.0,
    reasoning_effort: Optional[str] = "medium",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "messages": messages}
    is_openai = _looks_like_openai_host(endpoint)
    is_openai_gpt5 = is_openai and _is_gpt5(model)

    if is_openai_gpt5:
        payload["max_completion_tokens"] = max_tokens
        if float(temperature) == 1.0:
            payload["temperature"] = temperature
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            payload["reasoning_effort"] = reasoning_effort.strip()
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    if isinstance(tools, list) and tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    return payload


def _post_chat(
    *,
    endpoint: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Tuple[int, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = response.text
    return int(response.status_code), body


def _extract_first_tool_call(first_response_body: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(first_response_body, dict):
        return None, None
    choices = first_response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None, None
    message = choice0.get("message")
    if not isinstance(message, dict):
        return None, None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return message, None
    first_tool_call = tool_calls[0]
    if not isinstance(first_tool_call, dict):
        return message, None
    return message, first_tool_call


def _extract_turn_summary(body: Any) -> Dict[str, Any]:
    summary = {"finish_reason": None, "usage": None}
    if not isinstance(body, dict):
        return summary
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        summary["finish_reason"] = choices[0].get("finish_reason")
    usage = body.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage
    return summary


def _execute_tool(
    *,
    tool_server: str,
    tool_call: Dict[str, Any],
    timeout: float,
) -> Tuple[str, str, str, int, Any]:
    fn = tool_call.get("function") or {}
    tool_name = str(fn.get("name") or TOOL_NAME)
    tool_call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}")

    raw_arguments = fn.get("arguments")
    if isinstance(raw_arguments, dict):
        args = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            args = parsed if isinstance(parsed, dict) else {}
        except Exception:
            args = {}
    else:
        args = {}

    execute_payload = {
        "tool_name": tool_name,
        "args": args,
        "tool_call_id": tool_call_id,
    }
    execute_url = f"{tool_server.rstrip('/')}/execute"
    response = requests.post(execute_url, json=execute_payload, timeout=timeout)
    try:
        execute_body = response.json()
    except Exception:
        execute_body = {"ok": False, "error": {"message": response.text}, "result": None}

    compact = {
        "ok": execute_body.get("ok") if isinstance(execute_body, dict) else False,
        "result": execute_body.get("result") if isinstance(execute_body, dict) else None,
        "error": execute_body.get("error") if isinstance(execute_body, dict) else {"message": str(execute_body)},
    }
    tool_content = json.dumps(compact, ensure_ascii=False)
    return tool_name, tool_call_id, tool_content, int(response.status_code), execute_body


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OpenAI second-turn behavior with role=tool vs role=user after tool_calls."
    )
    parser.add_argument("--model-base", default="https://api.openai.com")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--tool-server", default="http://127.0.0.1:8101")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument(
        "--force-tool-choice",
        action="store_true",
        help="Force Python_Executor_Tool on turn 1 (useful if your model skips tool calls).",
    )
    args = parser.parse_args()

    if not isinstance(args.api_key, str) or not args.api_key.strip():
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    endpoint = _normalize_chat_endpoint(args.model_base)
    tools = [_build_tool_schema()]
    first_tool_choice: Any = "auto"
    if args.force_tool_choice:
        first_tool_choice = {"type": "function", "function": {"name": TOOL_NAME}}

    system_message = {"role": "system", "content": args.system_prompt}
    user_message = {"role": "user", "content": args.user_prompt}

    print("=== Setup ===")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {args.model}")
    print(f"Tool server: {args.tool_server}")
    print(f"First-turn tool_choice: {_truncate(first_tool_choice)}")
    print(f"User prompt: {_truncate(args.user_prompt)}")

    first_payload = _build_chat_payload(
        endpoint=endpoint,
        model=args.model,
        messages=[system_message, user_message],
        tools=tools,
        tool_choice=first_tool_choice,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )
    first_status, first_body = _post_chat(
        endpoint=endpoint,
        api_key=args.api_key,
        payload=first_payload,
        timeout=args.timeout_seconds,
    )

    print("\n=== Turn 1 (model call with tools) ===")
    print(f"HTTP {first_status}")
    if first_status != 200:
        print(f"Error: {_extract_error_detail(first_body)}")
        return 1
    turn1_summary = _extract_turn_summary(first_body)
    print(f"Finish reason: {turn1_summary.get('finish_reason')}")
    if turn1_summary.get("usage") is not None:
        print(f"Usage preview: {_truncate(turn1_summary['usage'])}")

    assistant_message, first_tool_call = _extract_first_tool_call(first_body)
    if not isinstance(assistant_message, dict):
        print("Failed to parse assistant message from turn 1 response.")
        print(f"Turn-1 raw response preview: {_truncate(first_body, max_chars=1200)}")
        return 1
    print(f"Assistant content preview: {_truncate(assistant_message.get('content'))}")

    if not isinstance(first_tool_call, dict):
        print("No tool_call returned on turn 1.")
        print(f"Turn-1 raw response preview: {_truncate(first_body, max_chars=1200)}")
        if turn1_summary.get("finish_reason") == "length":
            print(
                "Hint: completion stopped at length before emitting tool_calls. "
                "Try higher --max-tokens and/or lower --reasoning-effort."
            )
        else:
            print(
                "Hint: try --force-tool-choice or a more tool-requiring user prompt."
            )
        return 2

    tool_id = first_tool_call.get("id")
    fn = first_tool_call.get("function") or {}
    print(f"Tool call id: {tool_id}")
    print(f"Tool call name: {fn.get('name')}")
    print(f"Tool arguments preview: {_truncate(fn.get('arguments'))}")

    tool_name, tool_call_id, tool_content, exec_status, exec_body = _execute_tool(
        tool_server=args.tool_server,
        tool_call=first_tool_call,
        timeout=args.timeout_seconds,
    )
    print("\n=== Tool execution (/execute) ===")
    print(f"HTTP {exec_status}")
    print(f"Tool response preview: {_truncate(exec_body)}")

    assistant_tool_turn: Dict[str, Any] = {"role": "assistant", "tool_calls": [first_tool_call]}
    if "content" in assistant_message:
        assistant_tool_turn["content"] = assistant_message.get("content")
    conversation_prefix = [system_message, user_message, assistant_tool_turn]

    protocol_good_messages = conversation_prefix + [
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_content,
        }
    ]
    protocol_bad_messages = conversation_prefix + [
        {
            "role": "user",
            "content": f"Tool `{tool_name}` (tool_call_id={tool_call_id}) returned:\n{tool_content}",
        }
    ]

    def run_second_turn(messages: list[Dict[str, Any]], label: str) -> Tuple[int, Any]:
        payload = _build_chat_payload(
            endpoint=endpoint,
            model=args.model,
            messages=messages,
            tools=tools,
            tool_choice="none",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )
        status, body = _post_chat(
            endpoint=endpoint,
            api_key=args.api_key,
            payload=payload,
            timeout=args.timeout_seconds,
        )
        print(f"\n=== Turn 2 ({label}) ===")
        print(f"HTTP {status}")
        if status == 200:
            if isinstance(body, dict):
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                print(f"Assistant reply preview: {_truncate(content)}")
            else:
                print(f"Body preview: {_truncate(body)}")
        else:
            print(f"Error: {_extract_error_detail(body)}")
        return status, body

    good_status, _ = run_second_turn(protocol_good_messages, "protocol-correct role=tool")
    bad_status, bad_body = run_second_turn(protocol_bad_messages, "protocol-incorrect role=user")

    print("\n=== Summary ===")
    print(f"Correct path status: {good_status}")
    print(f"Incorrect path status: {bad_status}")
    if bad_status != 200:
        print("Observed strict behavior: role=user replacement for tool result was rejected.")
        return 0
    print("Incorrect path was accepted by this endpoint; strict protocol rejection was not observed here.")
    print(f"Unexpected acceptance body preview: {_truncate(bad_body)}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
