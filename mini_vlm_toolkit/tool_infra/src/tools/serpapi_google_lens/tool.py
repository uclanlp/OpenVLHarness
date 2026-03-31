import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail
from tools.image_utils import (
    decode_base64_image_data,
    infer_image_content_type_and_extension,
)

TOOL_NAME = "SerpApi_Google_Lens_Tool"
KEY_FILENAME = "serpapi_api_key"
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), KEY_FILENAME)
# Also check the sibling serpapi_image_search directory for a shared key.
_SIBLING_KEY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "serpapi_image_search", KEY_FILENAME
)
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_ENGINE = "google_lens"
DEFAULT_NUM_RESULTS = 10
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_url": {
            "type": "string",
            "description": "Public HTTP(S) image URL to search with Google Lens",
        },
        "image_path": {
            "type": "string",
            "description": "Local image path (will be uploaded to cloud storage to obtain a public URL)",
        },
        "image_base64": {
            "type": "string",
            "description": "Base64 image data (will be uploaded to cloud storage to obtain a public URL)",
        },
    },
    "oneOf": [
        {"required": ["image_url"]},
        {"required": ["image_path"]},
        {"required": ["image_base64"]},
    ],
    "additionalProperties": False,
}


class SerpApi_Google_Lens_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Identify objects, landmarks, artwork, text, and more in an image "
                "using Google Lens via SerpApi. Returns visual matches with titles, "
                "sources, and links. Use this tool when you need to understand what "
                "is depicted in an image — it can recognize specific places, products, "
                "artworks, plants, animals, etc. Accepts a public image URL, a local "
                "image path, or base64 image data."
            ),
            tool_version="1.0.0",
            input_types={
                "image_url": "str - public HTTP(S) URL of the image to search",
                "image_path": "str - local image path (auto-uploaded to cloud storage)",
                "image_base64": "str - base64-encoded image (auto-uploaded to cloud storage)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["google_lens"],
            output_type=(
                "list[dict] - visual matches with title, source, link, thumbnail, and position"
            ),
            demo_commands=[
                {
                    "command": (
                        "SerpApi_Google_Lens_Tool("
                        'image_url="https://example.com/landmark.jpg"'
                        ")"
                    ),
                    "description": "Identify a landmark or object in an image from a public URL.",
                },
                {
                    "command": (
                        "SerpApi_Google_Lens_Tool("
                        'image_path="/path/to/local/image.jpg"'
                        ")"
                    ),
                    "description": "Identify content in a local image (auto-uploaded to cloud storage).",
                },
            ],
            user_metadata={
                "api_key_file": KEY_FILENAME,
                "api_key_env": "SERPAPI_API_KEY",
                "endpoint": SERPAPI_ENDPOINT,
                "engine": SERPAPI_ENGINE,
                "default_num_results": DEFAULT_NUM_RESULTS,
                "notes": "Shares the same SerpApi API key as SerpApi_Image_Search_Tool.",
            },
        )
        self._api_key = None

    def _load_api_key(self) -> str:
        if self._api_key:
            return self._api_key

        env_key = os.getenv("SERPAPI_API_KEY", "").strip()
        if env_key:
            self._api_key = env_key
            return env_key

        # Check local key file first, then sibling directory
        for path in [KEY_FILE_PATH, _SIBLING_KEY_PATH]:
            resolved = os.path.normpath(path)
            if os.path.exists(resolved):
                with open(resolved, "r") as handle:
                    api_key = handle.read().strip()
                if api_key:
                    self._api_key = api_key
                    return api_key

        raise FileNotFoundError(
            f"Missing SerpApi API key. Set SERPAPI_API_KEY env var, or create "
            f"{KEY_FILE_PATH} with the key on a single line."
        )

    def sanity_check(self):
        try:
            self._load_api_key()
        except Exception as exc:
            return [str(exc)]
        return []

    @staticmethod
    def _validate_image_url(image_url: str) -> str:
        raw_url = str(image_url or "").strip()
        if not raw_url:
            raise ValueError("image_url is required")
        parsed = urlparse(raw_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("image_url must be a valid public http(s) URL")
        return raw_url

    def _resolve_image_url(
        self,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> str:
        """Resolve input to a public URL. Upload to cloud storage if needed."""
        if image_url:
            return self._validate_image_url(image_url)

        from tools.image_upload_utils import upload_image_bytes, upload_image_file

        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"image_path not found: {image_path}")
            return upload_image_file(image_path)

        if image_base64:
            image_bytes, content_type = decode_base64_image_data(image_base64)
            if content_type:
                ext_map = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                    "image/tiff": ".tiff",
                }
                ext = ext_map.get(content_type.lower())
                if not ext:
                    content_type, ext = infer_image_content_type_and_extension(
                        image_bytes
                    )
            else:
                content_type, ext = infer_image_content_type_and_extension(image_bytes)
            return upload_image_bytes(image_bytes, content_type, ext)

        raise ValueError(
            "Provide exactly one of: image_url, image_path, or image_base64"
        )

    @staticmethod
    def _extract_item(item_data: Dict[str, object]) -> Optional[Dict[str, object]]:
        title = item_data.get("title")
        link = item_data.get("link")
        if not title and not link:
            return None

        return {
            "title": title or "",
            "link": link or "",
            "source": item_data.get("source", ""),
            "thumbnail": item_data.get("thumbnail", ""),
            "position": item_data.get("position", ""),
        }

    def execute(
        self,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
    ):
        resolved_url = self._resolve_image_url(image_url, image_path, image_base64)
        api_key = self._load_api_key()

        params = {
            "engine": SERPAPI_ENGINE,
            "url": resolved_url,
            "api_key": api_key,
        }
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=60)
        raise_for_status_with_detail(response, prefix="SerpApi Google Lens failed")

        payload = response.json()
        raw_items: List[Dict[str, object]] = payload.get("visual_matches", [])

        results = []
        seen = set()
        for item_data in raw_items:
            if not isinstance(item_data, dict):
                continue
            match = self._extract_item(item_data)
            if match is None:
                continue
            dedup_key = match["link"] or match["title"]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append(match)
            if len(results) >= DEFAULT_NUM_RESULTS:
                break

        return results
