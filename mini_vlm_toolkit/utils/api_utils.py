"""
Lightweight helpers for calling OpenAI-compatible HTTP endpoints (OpenAI, or vLLM servers) with a consistent interface. 
Designed for use across tools that need chat/embeddings with optional images and tools.
"""

import base64
import copy
import json
import mimetypes
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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

# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #

MODEL_BASE_URLS_ENV = "LLM_DIRECT_MODEL_BASE_URLS"
MODEL_API_KEYS_ENV = "LLM_DIRECT_MODEL_API_KEYS"
DEFAULT_API_KEY_ENV = "LLM_DIRECT_API_KEY"
DEFAULT_API_KEY_VALUE = "EMPTY"

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


def _parse_mapping(raw_value: str, env_name: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not raw_value:
        return mapping
    for item in raw_value.split(","):
        entry = item.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"{env_name} entry '{entry}' must be name:value")
        key, value = entry.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{env_name} entry '{entry}' must include name and value")
        mapping[key] = value
    return mapping


def get_model_base_url(model: str, fallback_base_url: Optional[str] = None) -> str:
    raw_base_urls = os.getenv(MODEL_BASE_URLS_ENV, "")
    mapping = _parse_mapping(raw_base_urls, MODEL_BASE_URLS_ENV)
    base_url = mapping.get(model) or fallback_base_url
    if not base_url:
        raise RuntimeError(
            f"No base URL configured for model '{model}'. "
            f"Set {MODEL_BASE_URLS_ENV} or pass fallback_base_url."
        )
    return base_url.rstrip("/")


def get_model_api_key(model: str, fallback_api_key: Optional[str] = None) -> Optional[str]:
    raw_api_keys = os.getenv(MODEL_API_KEYS_ENV, "")
    mapping = _parse_mapping(raw_api_keys, MODEL_API_KEYS_ENV)
    api_key = mapping.get(model)
    if api_key is None:
        api_key = fallback_api_key or os.getenv(DEFAULT_API_KEY_ENV, DEFAULT_API_KEY_VALUE)
    if _is_empty_key(api_key):
        return None
    return api_key


def _is_empty_key(value: Optional[str]) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    if not value.strip():
        return True
    return value.strip().upper() == DEFAULT_API_KEY_VALUE


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #

def local_image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    # check if file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def attach_images(messages: List[Dict[str, Any]], image_paths: List[str]) -> List[Dict[str, Any]]:
    if not image_paths:
        return messages
    updated = copy.deepcopy(messages)
    if not updated or updated[-1].get("role") != "user":
        updated.append({"role": "user", "content": []})

    # Normalize last user message content to list form for multimodal payloads
    if isinstance(updated[-1].get("content"), str):
        text = updated[-1]["content"]
        updated[-1]["content"] = [{"type": "text", "text": text}]
    elif updated[-1].get("content") is None:
        updated[-1]["content"] = []

    if not isinstance(updated[-1]["content"], list):
        updated[-1]["content"] = [{"type": "text", "text": str(updated[-1]["content"])}]

    for image_path in image_paths:
        updated[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": local_image_to_data_url(image_path)}}
        )
    return updated


# --------------------------------------------------------------------------- #
# Endpoint helpers
# --------------------------------------------------------------------------- #

def _normalize_chat_endpoint(
    base_url: str,
) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    elif trimmed.endswith("completions"):
        return trimmed
    return f"{trimmed}/v1/chat/completions"


def _normalize_embeddings_endpoint(
    base_url: str,
) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return f"{trimmed}/embeddings"
    return f"{trimmed}/v1/embeddings"

# Filter out extra kwargs that are not valid for the endpoint to avoid request failures
def filter_kwargs_for_endpoint(kwargs: Dict[str, Any], endpoint_type: str) -> Dict[str, Any]:
    valid_keys = {
        "chat": {
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "extra_body",
            "image_paths",
            "timeout_seconds",
        },
        "embeddings": set(),
    }
    allowed_keys = valid_keys.get(endpoint_type, set())
    updated_kwargs = {}
    for k, v in kwargs.items():
        if k in allowed_keys:
            updated_kwargs[k] = v
        else:
            if "extra_body" not in updated_kwargs:
                updated_kwargs["extra_body"] = {}
            updated_kwargs["extra_body"][k] = v
    return updated_kwargs

# --------------------------------------------------------------------------- #
# Core request helpers
# --------------------------------------------------------------------------- #

@retry_with_exponential_backoff
def chat_completion_request(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = 4096,
    top_p: Optional[float] = None,
    stop: Optional[Iterable[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Union[str, Dict[str, Any], None] = "auto", # can be "auto", "none", or a specific tool
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 120,
    extra_body: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send a chat/completions request to an OpenAI-compatible endpoint.
    - Set base_url to a vLLM host (e.g., http://localhost:8000) or leave None
      to default to https://api.openai.com.
    """
    resolved_base_url = base_url or "https://api.openai.com"
    resolved_api_key = api_key or get_model_api_key(model)
    payload: Dict[str, Any] = {"model": model, "messages": attach_images(messages, image_paths or [])}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if stop is not None:
        payload["stop"] = stop
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    if response_format:
        payload["response_format"] = response_format
    if extra_body:
        payload.update(extra_body)

    endpoint = _normalize_chat_endpoint(resolved_base_url)
    headers = {"Content-Type": "application/json"}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
    raise_for_status_with_detail(response, prefix=f"Chat request failed at endpoint {endpoint}")
    return response.json()

@retry_with_exponential_backoff
def embeddings_request(
    *,
    model: str,
    input_text: Union[str, List[str]],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: float = 60,
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    texts = [input_text] if isinstance(input_text, str) else input_text
    resolved_base_url = base_url or "https://api.openai.com"
    resolved_api_key = api_key or get_model_api_key(model)
    payload: Dict[str, Any] = {"model": model, "input": texts}
    if extra_body:
        payload.update(extra_body)

    endpoint = _normalize_embeddings_endpoint(resolved_base_url)
    headers = {"Content-Type": "application/json"}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
    raise_for_status_with_detail(response, prefix="Embedding request failed")
    data = response.json().get("data", [])
    return [item["embedding"] for item in data]


@retry_with_exponential_backoff
def tool_execute_request(
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
