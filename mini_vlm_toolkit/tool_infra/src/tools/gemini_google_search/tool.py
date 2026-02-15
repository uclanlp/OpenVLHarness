import os
import re
from typing import List

import requests
from google import genai
from google.genai import types

from tools.base import BaseTool

TOOL_NAME = "Gemini_Google_Search_Tool"
KEY_FILENAME = "google_api_key"
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), KEY_FILENAME)
DEFAULT_MODEL = "gemini-2.5-flash"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Web question or search request"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class Gemini_Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Answer a web question using Gemini Google Search with optional "
                "inline citations. Use this tool when you want a concise "
                "web-grounded answer directly. Do not use it when you need only a "
                "list of URLs for follow-up crawling; use Serper_Google_Search_Tool "
                "for URL discovery."
            ),
            tool_version="1.1.0",
            input_types={
                "query": "str - web question or search request",
            },
            input_schema=INPUT_SCHEMA,
            output_type="str",
            demo_commands=[
                {
                    "command": (
                        "Gemini_Google_Search_Tool(query=\"What is the capital of France?\")"
                    ),
                    "description": "Search for a general fact with citations.",
                },
                {
                    "command": (
                        "Gemini_Google_Search_Tool(query=\"Who won Euro 2024?\")"
                    ),
                    "description": "Search for a recent event with citations.",
                },
            ],
            user_metadata={
                "api_key_file": KEY_FILENAME,
                "model_env": "GEMINI_SEARCH_MODEL",
            },
        )
        self.max_retries = 5
        self.search_model = os.getenv("GEMINI_SEARCH_MODEL", DEFAULT_MODEL)
        self._api_key = None
        self._client = None

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

    def _get_client(self):
        if self._client is None:
            api_key = self._load_api_key()
            self._client = genai.Client(api_key=api_key)
        return self._client

    def sanity_check(self):
        try:
            self._load_api_key()
        except Exception as exc:
            return [str(exc)]
        return []

    @staticmethod
    def _get_real_url(url):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        try:
            response = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
            return response.url
        except Exception:
            return url

    @staticmethod
    def _extract_urls(text: str) -> List[str]:
        pattern = re.compile(r"\[\d+\]\((https?://[^\s)]+)\)")
        return pattern.findall(text or "")

    def _reformat_response(self, response_text: str) -> str:
        urls = self._extract_urls(response_text)
        for url in urls:
            direct_url = self._get_real_url(url)
            response_text = response_text.replace(url, direct_url)
        return response_text

    @staticmethod
    def _add_citations(response):
        text = response.text
        supports = response.candidates[0].grounding_metadata.grounding_supports
        chunks = response.candidates[0].grounding_metadata.grounding_chunks

        sorted_supports = sorted(
            supports, key=lambda s: s.segment.end_index, reverse=True
        )

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return text

    def _execute_search(self, query: str, add_citations_flag: bool) -> str:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        client = self._get_client()
        response = None

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.search_model,
                    contents=query,
                    config=config,
                )
                break
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Gemini Google search failed after {self.max_retries} attempts: {exc}"
                    ) from exc

        if response is None:
            raise RuntimeError("Gemini Google search returned no response.")

        response_text = response.text or ""
        if add_citations_flag:
            try:
                response_text = self._add_citations(response)
            except Exception:
                pass

        try:
            response_text = self._reformat_response(response_text)
        except Exception:
            pass

        return response_text

    def execute(self, query: str, add_citations: bool = True):
        if not query:
            raise ValueError("query is required")
        return self._execute_search(query, add_citations)
