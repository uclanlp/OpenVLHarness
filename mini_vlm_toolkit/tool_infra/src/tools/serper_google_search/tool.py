import os
from typing import Dict, List, Optional

import requests

from tools.base import BaseTool

TOOL_NAME = "Serper_Google_Search_Tool"
KEY_FILENAME = "serper_api_key"
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), KEY_FILENAME)
SERPER_ENDPOINT = "https://google.serper.dev/search"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
    },
    "required": ["query"],
    "additionalProperties": False,
}

DEFAULT_FILTERED_DOMAINS = [
    ".pdf",
    "youtube.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
]


class Serper_Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Search the web with Serper Google Search and return a filtered list "
                "of candidate URLs. Use this tool when you need to discover relevant "
                "pages but do not have a target URL yet. Do not use this tool for "
                "direct question answering: it returns links only (not page content), "
                "and filters out common social/video/PDF domains."
            ),
            tool_version="1.1.0",
            input_types={
                "query": "str - search query",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["web_search"],
            output_type="list[str] - list of URLs",
            demo_commands=[
                {
                    "command": "Serper_Google_Search_Tool(query=\"OpenAI\")",
                    "description": "Fetch top URLs using Serper.",
                }
            ],
            user_metadata={
                "api_key_file": KEY_FILENAME,
            },
        )
        self._api_key = None

    def _load_api_key(self):
        if self._api_key:
            return self._api_key
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
    def _filter_urls(urls: List[str]) -> List[str]:
        filtered = []
        for url in urls:
            if not any(domain in url.lower() for domain in DEFAULT_FILTERED_DOMAINS):
                filtered.append(url)
        return filtered

    def _build_payload(
        self, query: str, gl: Optional[str], hl: Optional[str]
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {"q": query}
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl
        return payload

    def execute(
        self,
        query: str,
        num_results: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ):
        if not query:
            raise ValueError("query is required")
        if num_results is None or num_results <= 0:
            raise ValueError("num_results must be a positive integer")

        api_key = self._load_api_key()
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = self._build_payload(query, gl, hl)
        response = requests.post(
            SERPER_ENDPOINT, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        urls = []
        for item in data.get("organic", [])[:num_results]:
            url = item.get("link") or item.get("url") or ""
            if url:
                urls.append(url)

        return self._filter_urls(urls)
