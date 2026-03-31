import os
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from config import VLLM_QWEN3_32B_URL, VLLM_QWEN3_8B_URL


class LLMEngine:
    def __init__(
        self,
        model,
        base_url=None,
        api_key=None,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        extra_body: Optional[Dict[str, Any]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        if not model:
            raise ValueError("model is required")
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        # Merge enable_thinking into extra_body when explicitly set
        if enable_thinking is not None:
            extra_body = dict(extra_body or {})
            chat_kwargs = extra_body.get("chat_template_kwargs", {})
            chat_kwargs["enable_thinking"] = enable_thinking
            extra_body["chat_template_kwargs"] = chat_kwargs
        self.extra_body = extra_body
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @classmethod
    def from_env(cls, model, base_url_env="LLM_BASE_URL", api_key_env="LLM_API_KEY"):
        base_url = os.getenv(base_url_env)
        api_key = os.getenv(api_key_env)
        return cls(model=model, base_url=base_url, api_key=api_key)

    @classmethod
    def vllm_qwen3_32b(cls, model, api_key="local"):
        return cls(model=model, base_url=VLLM_QWEN3_32B_URL, api_key=api_key)

    @classmethod
    def vllm_qwen3_8b(cls, model, api_key="local"):
        return cls(model=model, base_url=VLLM_QWEN3_8B_URL, api_key=api_key)

    def generate(
        self,
        prompt_or_messages,
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any,
    ):
        messages = self._build_messages(prompt_or_messages, system_prompt=system_prompt)
        temperature = generation_kwargs.pop("temperature", self.temperature)
        top_p = generation_kwargs.pop("top_p", self.top_p)
        max_tokens = generation_kwargs.pop("max_tokens", self.max_tokens)
        extra_body = generation_kwargs.pop("extra_body", self.extra_body)
        # OpenAI reasoning models (o1, o3, o4, gpt-5-mini) only accept
        # max_completion_tokens and do not support temperature/top_p.
        # Other OpenAI models (gpt-4o, etc.) accept both param names.
        _REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
        # Strip provider prefix (e.g. "openai/o3-mini" -> "o3-mini")
        bare_model = self.model.rsplit("/", 1)[-1]
        is_reasoning = any(bare_model.startswith(p) for p in _REASONING_PREFIXES)

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            **generation_kwargs,
        }
        if is_reasoning:
            request_kwargs["max_completion_tokens"] = max_tokens
            # reasoning models ignore temperature/top_p; omit them
        else:
            request_kwargs["max_tokens"] = max_tokens
            request_kwargs["temperature"] = temperature
            request_kwargs["top_p"] = top_p
        if extra_body is not None:
            request_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **generation_kwargs: Any,
    ):
        return self.generate(
            messages,
            system_prompt=system_prompt,
            **generation_kwargs,
        )

    def _build_messages(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(prompt_or_messages, str):
            return [{"role": "user", "content": prompt_or_messages}]

        if not isinstance(prompt_or_messages, list):
            raise TypeError("prompt_or_messages must be a string or a list of messages")

        if system_prompt and not any(
            isinstance(item, dict) and item.get("role") == "system"
            for item in prompt_or_messages
        ):
            return [{"role": "system", "content": system_prompt}] + prompt_or_messages

        return prompt_or_messages
