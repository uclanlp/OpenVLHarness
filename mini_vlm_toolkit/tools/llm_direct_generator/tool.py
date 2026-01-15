import os
from typing import Any, Dict

import requests

from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "LLM_Direct_Generator_Tool"
MODEL_BASE_URLS_ENV = "LLM_DIRECT_MODEL_BASE_URLS"
MODEL_API_KEYS_ENV = "LLM_DIRECT_MODEL_API_KEYS"
DEFAULT_API_KEY_ENV = "LLM_DIRECT_API_KEY"
DEFAULT_API_KEY_VALUE = "EMPTY"
CHAT_REQUEST_TYPE = "chat"
EMBEDDINGS_REQUEST_TYPE = "embeddings"


def _parse_mapping(raw_value, env_name):
    mapping = {}
    if not raw_value:
        return mapping
    for item in raw_value.split(","):
        entry = item.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"{env_name} entry '{entry}' must be formatted as name:value"
            )
        key, value = entry.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"{env_name} entry '{entry}' must include both name and value"
            )
        mapping[key] = value
    return mapping


def _is_empty_key(value):
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    if not value.strip():
        return True
    return value.strip().upper() == DEFAULT_API_KEY_VALUE


class LLM_Direct_Generator_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Send OpenAI-compatible chat or embeddings requests to a configured "
                "server and return the full JSON payload."
            ),
            tool_version="0.2.0",
            input_types={
                "model": "str - model name (must map to LLM_DIRECT_MODEL_BASE_URLS)",
                "request_type": "str - chat (default) or embeddings",
                "messages": "list[dict] - OpenAI-style chat messages (chat only)",
                "input": "str | list[str] - embedding inputs (embeddings only)",
                "temperature": "float (optional)",
                "top_p": "float (optional)",
                "max_tokens": "int (optional)",
                "stop": "str or list[str] (optional)",
                "extra_body": "dict (optional) - extra request fields",
                "request_timeout_seconds": "float (optional, default 60)",
            },
            output_type="dict - raw OpenAI-compatible response payload",
            demo_commands=[
                {
                    "command": (
                        "LLM_Direct_Generator_Tool("
                        "model=\"Qwen/Qwen3-8B\", "
                        "messages=[{\"role\": \"user\", \"content\": \"Hi\"}], "
                        "max_tokens=64"
                        ")"
                    ),
                    "description": "Generate a short chat completion.",
                },
                {
                    "command": (
                        "LLM_Direct_Generator_Tool("
                        "model=\"Qwen/Qwen3-Embedding-8B\", "
                        "request_type=\"embeddings\", "
                        "input=[\"Hello world\", \"Embedding example\"]"
                        ")"
                    ),
                    "description": "Request embeddings for a list of inputs.",
                },
            ],
            user_metadata={
                "env": {
                    MODEL_BASE_URLS_ENV: "required",
                    MODEL_API_KEYS_ENV: "optional",
                    DEFAULT_API_KEY_ENV: f"optional (default {DEFAULT_API_KEY_VALUE})",
                },
                "notes": "Streaming responses are not supported.",
            },
        )
        self._base_url_map = None
        self._api_key_map = None
        self._default_api_key = None
        self._sessions = {}

    def _load_config(self):
        if self._base_url_map is not None:
            return
        raw_base_urls = os.getenv(MODEL_BASE_URLS_ENV)
        if not raw_base_urls or not raw_base_urls.strip():
            raise RuntimeError(
                f"{MODEL_BASE_URLS_ENV} is required (format: model:http://host:port/v1)"
            )
        self._base_url_map = _parse_mapping(raw_base_urls, MODEL_BASE_URLS_ENV)
        raw_api_keys = os.getenv(MODEL_API_KEYS_ENV, "")
        self._api_key_map = _parse_mapping(raw_api_keys, MODEL_API_KEYS_ENV)
        self._default_api_key = os.getenv(DEFAULT_API_KEY_ENV, DEFAULT_API_KEY_VALUE)

    def _get_base_url(self, model):
        self._load_config()
        base_url = self._base_url_map.get(model)
        if not base_url:
            raise ValueError(
                f"No base URL configured for model '{model}'. "
                f"Add it to {MODEL_BASE_URLS_ENV}."
            )
        return base_url

    def _get_api_key(self, model):
        self._load_config()
        api_key = self._api_key_map.get(model, self._default_api_key)
        if _is_empty_key(api_key):
            return None
        return api_key

    def _get_endpoint(self, base_url):
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/chat/completions"
        return f"{trimmed}/v1/chat/completions"

    def _get_embeddings_endpoint(self, base_url):
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/embeddings"
        return f"{trimmed}/v1/embeddings"

    def _normalize_request_type(self, request_type):
        if request_type is None:
            return CHAT_REQUEST_TYPE
        if not isinstance(request_type, str):
            raise TypeError("request_type must be a string")
        normalized = request_type.strip().lower()
        if normalized in ("chat", "chat.completions", "chat-completions"):
            return CHAT_REQUEST_TYPE
        if normalized in ("embeddings", "embedding"):
            return EMBEDDINGS_REQUEST_TYPE
        raise ValueError("request_type must be 'chat' or 'embeddings'")

    def _get_session(self, base_url):
        session = self._sessions.get(base_url)
        if session is None:
            session = requests.Session()
            self._sessions[base_url] = session
        return session

    def sanity_check(self):
        try:
            self._load_config()
        except Exception as exc:
            return [str(exc)]
        return []

    def execute(
        self,
        model,
        messages=None,
        input=None,
        request_type=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        stop=None,
        extra_body=None,
        request_timeout_seconds=60,
        stream=False,
        **kwargs,
    ):
        if not model:
            raise ValueError("model is required")
        request_type = self._normalize_request_type(request_type)
        if stream:
            raise ValueError("streaming is not supported")

        if extra_body is not None and not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dict")
        if extra_body and extra_body.get("stream"):
            raise ValueError("streaming is not supported")

        base_url = self._get_base_url(model)
        payload: Dict[str, Any] = {"model": model}
        if request_type == CHAT_REQUEST_TYPE:
            if messages is None:
                raise ValueError("messages are required for chat requests")
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            if not isinstance(messages, list):
                raise TypeError("messages must be a list of dicts or a string")
            if not messages:
                raise ValueError("messages must be non-empty")
            payload["messages"] = messages
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if stop is not None:
                payload["stop"] = stop
        else:
            if input is not None:
                payload["input"] = input
            elif not (extra_body and "input" in extra_body):
                raise ValueError("input is required for embeddings requests")

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = value
        if extra_body is not None:
            payload.update(extra_body)

        if request_type == CHAT_REQUEST_TYPE:
            endpoint = self._get_endpoint(base_url)
        else:
            endpoint = self._get_embeddings_endpoint(base_url)
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key(model)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        session = self._get_session(base_url)
        response = session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=request_timeout_seconds,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Non-JSON response from {endpoint}: {exc}"
            ) from exc
