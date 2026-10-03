"""HTTP errors and transient retry policy shared by model and tool clients."""
import json
import random
import re
import time
import requests
DEFAULT_ERROR_BODY_LIMIT = 2048

def _stringify_detail(detail):
    if detail is None:
        return None
    if isinstance(detail, (dict, list)):
        try:
            return json.dumps(detail, ensure_ascii=True)
        except Exception:
            return str(detail)
    return str(detail)

def extract_error_detail(response, max_chars=DEFAULT_ERROR_BODY_LIMIT):
    payload = None
    detail = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is None:
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error.get("detail")
        if detail is None:
            detail = payload.get("message")

    if detail is None:
        text = response.text.strip()
        if text:
            detail = text

    detail = _stringify_detail(detail)
    if detail and max_chars and len(detail) > max_chars:
        detail = f"{detail[:max_chars]}... (truncated)"
    return detail, payload

def raise_for_status_with_detail(response, prefix=None, max_chars=DEFAULT_ERROR_BODY_LIMIT):
    if response.ok:
        return
    detail, _payload = extract_error_detail(response, max_chars=max_chars)
    message = f"{response.status_code} {detail or response.reason}"
    if prefix:
        message = f"{prefix}: {message}"
    raise RuntimeError(message)

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
    max_delay: float = 30,
):
    """Retry a function with exponential backoff for transient errors."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                message = str(exc).lower()
                status_code = None
                matches = re.findall(r"\b([1-5]\d{2})\b", message)
                if matches:
                    try:
                        status_code = int(matches[0])
                    except ValueError:
                        status_code = None

                transient_statuses = {408, 429, 500, 502, 503, 504}
                transient = False
                if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                    transient = True
                elif status_code in transient_statuses:
                    transient = True
                else:
                    transient = any(
                        token in message
                        for token in (
                            "rate limit",
                            "timed out",
                            "timeout",
                            "too many requests",
                            "internal",
                            "connection",
                        )
                    )
                if not transient:
                    raise
                num_retries += 1
                if num_retries > max_retries:
                    raise
                delay *= exponential_base * (1 + (jitter and (random.random())))
                delay = min(delay, max_delay)
                time.sleep(delay)

    return wrapper

def _tool_execute_once(
    *,
    base_url,
    tool_call,
    timeout = 120,
):
    response = requests.post(f"{base_url}/execute", json=tool_call, timeout=timeout)

    payload = None
    try:
        payload = response.json()
    except ValueError:
        payload = None

    # Preserve structured execute envelopes, including non-2xx error payloads.
    if isinstance(payload, dict) and ("ok" in payload) and ("result" in payload or "error" in payload):
        envelope = dict(payload)
        if not envelope.get("tool_name"):
            envelope["tool_name"] = tool_call.get("tool_name")
        if not envelope.get("tool_call_id"):
            envelope["tool_call_id"] = tool_call.get("tool_call_id")
        meta = envelope.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("http_status", response.status_code)
        envelope["meta"] = meta

        # Server/protocol mismatch guard: non-2xx should not claim success.
        if not response.ok and bool(envelope.get("ok")):
            envelope["ok"] = False
            err = envelope.get("error")
            if isinstance(err, dict):
                err = dict(err)
                err.setdefault("code", "execute_protocol_error")
                err.setdefault("message", "Execute endpoint returned non-2xx with ok=true payload.")
            else:
                err = {
                    "code": "execute_protocol_error",
                    "message": "Execute endpoint returned non-2xx with ok=true payload.",
                }
            envelope["error"] = err
        return envelope

    raise_for_status_with_detail(response, prefix="Tool execution request failed")
    try:
        if payload is not None:
            return payload
        return response.json()
    except ValueError:
        text_preview = response.text
        if isinstance(text_preview, str) and len(text_preview) > 500:
            text_preview = text_preview[:500] + "...(truncated)"
        content_type = ""
        headers = getattr(response, "headers", {}) or {}
        if isinstance(headers, dict):
            content_type = headers.get("Content-Type", "")
        return {
            "ok": False,
            "tool_name": tool_call.get("tool_name"),
            "result": None,
            "error": {
                "code": "execute_non_json_response",
                "message": "Execute endpoint returned non-JSON response.",
                "detail": {
                    "content_type": content_type,
                    "response_text_preview": text_preview,
                },
            },
            "tool_call_id": tool_call.get("tool_call_id"),
            "meta": {
                "warning": "non_json_response_from_execute",
                "http_status": response.status_code,
            },
        }
