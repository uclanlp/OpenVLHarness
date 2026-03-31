import os
import base64
from io import BytesIO

from PIL import Image

from tools import image_upload_utils
from tools.serpapi_google_lens import tool as lens_tool_module
from tools.serpapi_google_lens.tool import (
    DEFAULT_NUM_RESULTS,
    SERPAPI_ENDPOINT,
    SERPAPI_ENGINE,
    TOOL_NAME,
    SerpApi_Google_Lens_Tool,
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
    Image.new("RGB", (2, 2), (0, 255, 0)).save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if not include_data_url:
        return encoded
    content_type = content_type or Image.MIME[image_format]
    return f"data:{content_type};base64,{encoded}"


def test_invalid_image_url():
    tool = SerpApi_Google_Lens_Tool()
    try:
        tool.execute(image_url="not_a_valid_url")
        raise AssertionError("Expected ValueError for invalid image_url")
    except ValueError as exc:
        assert "http(s)" in str(exc)


def test_no_input_raises():
    tool = SerpApi_Google_Lens_Tool()
    try:
        tool.execute()
        raise AssertionError("Expected ValueError for no input")
    except ValueError as exc:
        assert "one of" in str(exc).lower()


def test_response_parse_limit_and_dedupe():
    fake_payload = {
        "visual_matches": [
            {
                "position": 1,
                "title": "Eiffel Tower - Wikipedia",
                "source": "Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                "thumbnail": "https://thumbs.example.com/1.jpg",
            },
            {
                "position": 2,
                "title": "Eiffel Tower tickets",
                "source": "toureiffel.paris",
                "link": "https://www.toureiffel.paris/en",
                "thumbnail": "https://thumbs.example.com/2.jpg",
            },
            {
                "position": 3,
                "title": "Eiffel Tower - Wikipedia",
                "source": "Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                "thumbnail": "https://thumbs.example.com/3.jpg",
            },
            {
                "position": 4,
                "title": "Visit the Eiffel Tower",
                "source": "TripAdvisor",
                "link": "https://www.tripadvisor.com/eiffel",
                "thumbnail": "https://thumbs.example.com/4.jpg",
            },
            {
                "position": 5,
                "title": "Eiffel Tower history",
                "source": "History.com",
                "link": "https://www.history.com/eiffel",
                "thumbnail": "https://thumbs.example.com/5.jpg",
            },
            {
                "position": 6,
                "title": "Eiffel Tower at night",
                "source": "Unsplash",
                "link": "https://unsplash.com/eiffel",
                "thumbnail": "https://thumbs.example.com/6.jpg",
            },
            {
                "position": 7,
                "title": "Eiffel Tower facts",
                "source": "NatGeo",
                "link": "https://natgeo.com/eiffel",
                "thumbnail": "https://thumbs.example.com/7.jpg",
            },
            {
                "position": 8,
                "title": "Eiffel Tower construction",
                "source": "Engineering",
                "link": "https://engineering.com/eiffel",
                "thumbnail": "https://thumbs.example.com/8.jpg",
            },
            {
                "position": 9,
                "title": "Eiffel Tower photos",
                "source": "Getty",
                "link": "https://getty.com/eiffel",
                "thumbnail": "https://thumbs.example.com/9.jpg",
            },
            {
                "position": 10,
                "title": "Eiffel Tower guide",
                "source": "LonelyPlanet",
                "link": "https://lonelyplanet.com/eiffel",
                "thumbnail": "https://thumbs.example.com/10.jpg",
            },
            {
                "position": 11,
                "title": "Extra result (should be cut off)",
                "source": "Extra",
                "link": "https://extra.com/eiffel",
                "thumbnail": "https://thumbs.example.com/11.jpg",
            },
        ],
    }
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = dict(params)
        captured["timeout"] = timeout
        return _DummyResponse(200, fake_payload)

    original_get = lens_tool_module.requests.get
    original_api_key_env = os.environ.get("SERPAPI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = "test-api-key"
    try:
        lens_tool_module.requests.get = fake_get
        tool = SerpApi_Google_Lens_Tool()
        results = tool.execute(image_url="https://example.com/landmark.jpg")
    finally:
        lens_tool_module.requests.get = original_get
        if original_api_key_env is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = original_api_key_env

    assert captured["url"] == SERPAPI_ENDPOINT
    assert captured["params"]["engine"] == SERPAPI_ENGINE
    assert captured["params"]["url"] == "https://example.com/landmark.jpg"
    assert captured["timeout"] == 60
    assert len(results) == DEFAULT_NUM_RESULTS
    # Check deduplication: the duplicate Wikipedia link should be removed
    links = [item["link"] for item in results]
    assert len(set(links)) == len(links), "Results should be deduplicated by link"
    # Check output fields
    for item in results:
        assert "title" in item
        assert "link" in item
        assert "source" in item
        assert "thumbnail" in item
        assert "position" in item


def test_empty_visual_matches():
    fake_payload = {"visual_matches": []}
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        return _DummyResponse(200, fake_payload)

    original_get = lens_tool_module.requests.get
    original_api_key_env = os.environ.get("SERPAPI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = "test-api-key"
    try:
        lens_tool_module.requests.get = fake_get
        tool = SerpApi_Google_Lens_Tool()
        results = tool.execute(image_url="https://example.com/obscure.jpg")
    finally:
        lens_tool_module.requests.get = original_get
        if original_api_key_env is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = original_api_key_env

    assert results == []


def test_api_key_from_env():
    original_api_key_env = os.environ.get("SERPAPI_API_KEY")
    os.environ["SERPAPI_API_KEY"] = "env-test-key-123"
    try:
        tool = SerpApi_Google_Lens_Tool()
        tool._api_key = None  # reset cache
        key = tool._load_api_key()
        assert key == "env-test-key-123"
    finally:
        if original_api_key_env is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = original_api_key_env


def test_data_url_gif_preserves_extension():
    captured = {}

    def fake_upload_image_bytes(image_bytes, content_type, extension, provider=None):
        captured["content_type"] = content_type
        captured["extension"] = extension
        captured["provider"] = provider
        captured["size"] = len(image_bytes)
        return "https://example.com/uploaded.gif"

    original_upload = image_upload_utils.upload_image_bytes
    try:
        image_upload_utils.upload_image_bytes = fake_upload_image_bytes
        tool = SerpApi_Google_Lens_Tool()
        resolved = tool._resolve_image_url(
            image_base64=_encode_test_image_base64("GIF", include_data_url=True)
        )
    finally:
        image_upload_utils.upload_image_bytes = original_upload

    assert resolved == "https://example.com/uploaded.gif"
    assert captured["content_type"] == "image/gif"
    assert captured["extension"] == ".gif"
    assert captured["size"] > 0


if __name__ == "__main__":
    test_invalid_image_url()
    print(f"PASS: {TOOL_NAME} invalid image_url validation")
    test_no_input_raises()
    print(f"PASS: {TOOL_NAME} no input raises ValueError")
    test_response_parse_limit_and_dedupe()
    print(f"PASS: {TOOL_NAME} response parsing, dedup, and top-{DEFAULT_NUM_RESULTS} limit")
    test_empty_visual_matches()
    print(f"PASS: {TOOL_NAME} empty visual_matches returns []")
    test_api_key_from_env()
    print(f"PASS: {TOOL_NAME} API key loading from env")
    test_data_url_gif_preserves_extension()
    print(f"PASS: {TOOL_NAME} data URL GIF preserves MIME and extension")
    print(f"\nAll {TOOL_NAME} tests passed.")
