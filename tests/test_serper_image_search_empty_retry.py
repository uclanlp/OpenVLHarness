"""Unit tests for Image_Search_Tool's empty-result retry.

Serper Lens sometimes answers HTTP 200 with ``organic: []`` for an image it
matches on a later call. ``SerperClient`` retries only transient HTTP status
codes, so an empty 200 never triggered it and the tool reported NO VISUAL
MATCHES on the first drop. These tests run offline: the Serper client is stubbed.
"""
import importlib
import os

import pytest

os.environ.setdefault("SERPER_API_KEY", "test-key-not-used")

import openvlharness.services.tools.serper_image_search.tool as ist  # noqa: E402
from openvlharness.services.tools.serper_common import SerperRequestError  # noqa: E402


def _match(i):
    return {
        "title": f"Match {i}",
        "link": f"https://example.com/page{i}",
        "source": "example.com",
        "imageUrl": f"https://example.com/img{i}.jpg",
        "thumbnailUrl": f"https://example.com/thumb{i}.jpg",
    }


class _StubClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def lens(self, url):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


EMPTY = {"organic": []}
FULL = {"organic": [_match(i) for i in range(1, 4)]}
UNPARSEABLE = {"organic": ["not-a-dict", 42, None]}


def _build(script, max_attempts):
    tool = ist.Image_Search_Tool(llm_engine=None)
    tool._llm_summarize = False
    tool._multimodal_summarize = False
    tool._resolve_image_url = lambda b64: "https://example.com/uploaded.jpg"
    client = _StubClient(script)
    tool._get_client = lambda: client
    ist.LENS_EMPTY_MAX_ATTEMPTS = max_attempts
    ist.LENS_EMPTY_BACKOFF_BASE_SEC = 0.0
    ist.LENS_EMPTY_BACKOFF_MAX_SEC = 0.0
    return tool, client


def _text(out):
    return out["text"] if isinstance(out, dict) else out


def test_recovers_when_serper_drops_then_answers():
    tool, client = _build([EMPTY, EMPTY, FULL], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "NO VISUAL MATCHES" not in out, out[:300]
    assert "Match 1" in out
    assert client.calls == 3


def test_same_stub_fails_with_the_old_single_attempt_behaviour():
    tool, client = _build([EMPTY, EMPTY, FULL], max_attempts=1)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "NO VISUAL MATCHES" in out, out[:300]
    assert client.calls == 1


def test_first_call_succeeds_costs_exactly_one_call():
    tool, client = _build([FULL], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "Match 1" in out
    assert client.calls == 1


def test_genuinely_empty_exhausts_attempts_and_says_so():
    tool, client = _build([EMPTY], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "NO VISUAL MATCHES" in out
    assert client.calls == 4
    assert "4 times" in out


def test_message_no_longer_tells_the_agent_not_to_retry():
    tool, _ = _build([EMPTY], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "will return the same empty result" not in out
    assert "NOT a transient" not in out


def test_single_attempt_message_omits_the_retry_sentence():
    tool, _ = _build([EMPTY], max_attempts=1)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "NO VISUAL MATCHES" in out
    assert "was issued" not in out.lower()


def test_http_error_is_not_retried_by_the_empty_loop():
    err = SerperRequestError("boom", status_code=400, transient=False, attempts=1)
    tool, client = _build([err], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "REVERSE IMAGE SEARCH FAILED" in out
    assert client.calls == 1


def test_retry_fires_when_items_exist_but_none_parse():
    tool, client = _build([UNPARSEABLE, FULL], max_attempts=4)
    out = _text(tool.execute(image_base64="data:image/jpeg;base64,AAAA"))
    assert "Match 1" in out
    assert client.calls == 2


def test_env_var_controls_attempt_count(monkeypatch):
    monkeypatch.setenv("SEARCH_LENS_EMPTY_MAX_ATTEMPTS", "2")
    importlib.reload(ist)
    assert ist.LENS_EMPTY_MAX_ATTEMPTS == 2
    monkeypatch.setenv("SEARCH_LENS_EMPTY_MAX_ATTEMPTS", "0")
    importlib.reload(ist)
    assert ist.LENS_EMPTY_MAX_ATTEMPTS == 1
    monkeypatch.delenv("SEARCH_LENS_EMPTY_MAX_ATTEMPTS")
    importlib.reload(ist)


def test_default_is_four_attempts(monkeypatch):
    monkeypatch.delenv("SEARCH_LENS_EMPTY_MAX_ATTEMPTS", raising=False)
    importlib.reload(ist)
    assert ist.LENS_EMPTY_MAX_ATTEMPTS == 4
