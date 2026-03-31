import os
import base64
from io import BytesIO

from PIL import Image

from tools import image_upload_utils
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


def _encode_test_image_base64(image_format, include_data_url=False, content_type=None):
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if not include_data_url:
        return encoded
    content_type = content_type or Image.MIME[image_format]
    return f"data:{content_type};base64,{encoded}"


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


def test_plain_base64_png_preserves_inferred_mime_and_extension():
    captured = {}

    def fake_upload_image_bytes(image_bytes, content_type, extension, provider=None):
        captured["content_type"] = content_type
        captured["extension"] = extension
        captured["provider"] = provider
        captured["size"] = len(image_bytes)
        return "https://example.com/uploaded.png"

    original_upload = image_upload_utils.upload_image_bytes
    try:
        image_upload_utils.upload_image_bytes = fake_upload_image_bytes
        tool = SerpApi_Image_Search_Tool()
        resolved = tool._resolve_image_url(
            image_base64=_encode_test_image_base64("PNG")
        )
    finally:
        image_upload_utils.upload_image_bytes = original_upload

    assert resolved == "https://example.com/uploaded.png"
    assert captured["content_type"] == "image/png"
    assert captured["extension"] == ".png"
    assert captured["size"] > 0


if __name__ == "__main__":
    test_invalid_image_url()
    print(f"PASS: {TOOL_NAME} invalid image_url validation")
    test_response_parse_limit_and_dedupe()
    print(f"PASS: {TOOL_NAME} response parsing and default top-{DEFAULT_NUM_RESULTS}")
    test_plain_base64_png_preserves_inferred_mime_and_extension()
    print(f"PASS: {TOOL_NAME} plain base64 PNG preserves inferred MIME and extension")
