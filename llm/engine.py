import os

from openai import OpenAI

VLLM_QWEN3_32B_URL = "http://localhost:8003/v1"
VLLM_QWEN3_8B_URL = "http://localhost:8004/v1"


class LLMEngine:
    def __init__(self, model, base_url=None, api_key=None, temperature=0.0):
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

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content
