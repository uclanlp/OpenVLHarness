import os

from tools.serpapi_image_search import tool as serpapi_tool_module
from tools.serpapi_image_search.tool import (
    DEFAULT_NUM_RESULTS,
    SERPAPI_ENDPOINT,
    TOOL_NAME,
    SerpApi_Image_Search_Tool,
)


class _DummyResponse:
    def __init__(self, status_code, payload, reason="OK", text=""):
        self.status_code = status_code
        self._payload = payload
        self.reason = reason
        self.text = text

    @property
    def ok(self):
        return 200 <= int(self.status_code) < 300

    def json(self):
        return self._payload


def test_invalid_image_url():
    tool = SerpApi_Image_Search_Tool()
    try:
        tool.execute(image_url="not_a_valid_url")
        raise AssertionError("Expected ValueError for invalid image_url")
    except ValueError as exc:
        assert "http(s)" in str(exc)


def test_response_parse_limit_and_dedupe():
    fake_payload = {
        "image_results": [
            {
                "position": 1,
                "title": "Match 1",
                "original": "https://images.example.com/1.jpg",
                "link": "https://source.example.com/1",
                "snippet": "snippet 1",
                "width": 100,
                "height": 100,
                "thumbnail": "https://thumbs.example.com/1.jpg",
            },
            {
                "position": 2,
                "title": "Match 2",
                "original": "https://images.example.com/2.jpg",
                "link": "https://source.example.com/2",
                "snippet": "snippet 2",
                "width": 200,
                "height": 200,
                "thumbnail": "https://thumbs.example.com/2.jpg",
            },
            {
                "position": 3,
                "title": "Duplicate Match 1",
                "original": "https://images.example.com/1.jpg",
                "link": "https://source.example.com/1-dup",
                "snippet": "duplicate snippet",
                "width": 101,
                "height": 101,
                "thumbnail": "https://thumbs.example.com/1-dup.jpg",
            },
        ],
        "inline_images": [
            {
                "position": 4,
                "title": "Match 4",
                "thumbnail": "https://images.example.com/4.jpg",
                "link": "https://source.example.com/4",
            },
            {
                "position": 5,
                "title": "Match 5",
                "thumbnail": "https://images.example.com/5.jpg",
                "link": "https://source.example.com/5",
            },
            {
                "position": 6,
                "title": "Match 6",
                "thumbnail": "https://images.example.com/6.jpg",
                "link": "https://source.example.com/6",
            },
        ],
    }
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _DummyResponse(200, fake_payload)

    original_get = serpapi_tool_module.requests.get
    original_api_key_env = os.environ.get("SERPAPI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = "test-api-key"
    try:
        serpapi_tool_module.requests.get = fake_get
        tool = SerpApi_Image_Search_Tool()
        results = tool.execute(image_url="https://example.com/input.jpg")
    finally:
        serpapi_tool_module.requests.get = original_get
        if original_api_key_env is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = original_api_key_env

    assert captured["url"] == SERPAPI_ENDPOINT
    assert captured["params"]["image_url"] == "https://example.com/input.jpg"
    assert captured["timeout"] == 30
    assert len(results) == DEFAULT_NUM_RESULTS
    assert len({item["image_url"] for item in results}) == len(results)
    assert all("source_url" in item for item in results)


if __name__ == "__main__":
    test_invalid_image_url()
    print(f"PASS: {TOOL_NAME} invalid image_url validation")
    test_response_parse_limit_and_dedupe()
    print(f"PASS: {TOOL_NAME} response parsing and default top-{DEFAULT_NUM_RESULTS}")

