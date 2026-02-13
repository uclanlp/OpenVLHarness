import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

from tools.base import BaseTool
from tools.http_utils import raise_for_status_with_detail

TOOL_NAME = "SerpApi_Image_Search_Tool"
KEY_FILENAME = "serpapi_api_key"
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), KEY_FILENAME)
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_ENGINE = "google_reverse_image"
DEFAULT_NUM_RESULTS = 5
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_url": {
            "type": "string",
            "description": "Public HTTP(S) image URL for reverse image search",
        },
    },
    "required": ["image_url"],
    "additionalProperties": False,
}


class SerpApi_Image_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Search similar images with SerpApi Google Reverse Image Search "
                "from a single public image URL."
            ),
            tool_version="1.0.0",
            input_types={
                "image_url": "str - public HTTP(S) URL of the image to search",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["image_search"],
            output_type=(
                "list[dict] - top 5 similar image matches with image/source URLs and metadata"
            ),
            demo_commands=[
                {
                    "command": (
                        "SerpApi_Image_Search_Tool("
                        'image_url="https://example.com/dog.jpg"'
                        ")"
                    ),
                    "description": "Find up to 5 reverse-image matches.",
                }
            ],
            user_metadata={
                "api_key_file": KEY_FILENAME,
                "api_key_env": "SERPAPI_API_KEY",
                "endpoint": SERPAPI_ENDPOINT,
                "engine": SERPAPI_ENGINE,
                "default_num_results": DEFAULT_NUM_RESULTS,
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

        if not os.path.exists(KEY_FILE_PATH):
            raise FileNotFoundError(
                f"Missing API key file at {KEY_FILE_PATH}. "
                "Create the file and place the key on a single line."
            )

        with open(KEY_FILE_PATH, "r") as handle:
            api_key = handle.read().strip()

        if not api_key:
            raise ValueError(
                f"Empty API key file at {KEY_FILE_PATH}. "
                "Place the key on a single line."
            )

        self._api_key = api_key
        return api_key

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

    @staticmethod
    def _extract_item(item_data: Dict[str, object]) -> Optional[Dict[str, object]]:
        image_url = item_data.get("original") or item_data.get("thumbnail")
        if not image_url:
            return None

        return {
            "image_url": image_url,
            "title": item_data.get("title", ""),
            "source_url": item_data.get("link", ""),
            "snippet": item_data.get("snippet", ""),
            "position": item_data.get("position", ""),
            "source": item_data.get("source", ""),
            "width": item_data.get("width", ""),
            "height": item_data.get("height", ""),
            "thumbnail": item_data.get("thumbnail", ""),
        }

    def execute(self, image_url: str):
        validated_image_url = self._validate_image_url(image_url)
        api_key = self._load_api_key()

        params = {
            "engine": SERPAPI_ENGINE,
            "image_url": validated_image_url,
            "api_key": api_key,
        }
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        raise_for_status_with_detail(response, prefix="SerpApi image search failed")

        payload = response.json()
        raw_items: List[Dict[str, object]] = payload.get("image_results", [])
        raw_items += payload.get("inline_images", [])

        results = []
        seen = set()
        for item_data in raw_items:
            if not isinstance(item_data, dict):
                continue
            match = self._extract_item(item_data)
            if match is None:
                continue
            match_url = match["image_url"]
            if match_url in seen:
                continue
            seen.add(match_url)
            results.append(match)
            if len(results) >= DEFAULT_NUM_RESULTS:
                break

        return results

