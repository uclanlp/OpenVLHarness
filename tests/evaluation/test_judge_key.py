"""Judge clients read OPENAI_API_KEY for the 'empty' placeholder only on api.openai.com."""
import pytest

from evaluation_kit.dataset.api.gpt import openai_judge_key


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)


@pytest.mark.parametrize("key,api_base,expected", [
    (None, "http://127.0.0.1:8000/v1/chat/completions", "env-key"),        # unchanged: missing key reads the env
    ("empty", "https://api.openai.com/v1/chat/completions", "env-key"),    # placeholder on OpenAI
    ("EMPTY", None, "env-key"),                                            # placeholder, default endpoint
    ("empty", "http://127.0.0.1:8000/v1/chat/completions", "empty"),       # placeholder elsewhere stays
    ("sk-config", "https://api.openai.com/v1/chat/completions", "sk-config"),  # explicit key wins
])
def test_openai_judge_key(key, api_base, expected):
    assert openai_judge_key(key, api_base) == expected
