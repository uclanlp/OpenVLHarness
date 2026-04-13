"""Shared infrastructure for Serper-based tools.

Provides:
- ``SerperClient`` — thin wrapper for Serper API calls (auth, retry, error handling).
- ``get_serper_api_key`` — load key from env / file.
- ``format_search_results`` — unified result formatter for all Serper text-search endpoints.
- ``strip_think_tags`` — remove ``<think>...</think>`` blocks from LLM output.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM output cleaning
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output (e.g. Qwen3 reasoning traces)."""
    return _THINK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Serper API endpoints
# ---------------------------------------------------------------------------
SERPER_SEARCH_BASE = "https://google.serper.dev"
SERPER_SCRAPE_BASE = "https://scrape.serper.dev"

ALL_SEARCH_TYPES = [
    "search", "images", "news", "scholar",
    "maps", "places", "shopping", "patents",
]

# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------
_SERPER_KEY_ENV = "SERPER_API_KEY"
# Fallback: look for a key file next to any of the Serper tool directories.
_KEY_FILE_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "serper_text_search", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_image_search", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_webpage_visit", "serper_api_key"),
    os.path.join(os.path.dirname(__file__), "serper_google_search", "serper_api_key"),
]

_cached_api_key: Optional[str] = None


def get_serper_api_key() -> str:
    """Load Serper API key from env var or local key file."""
    global _cached_api_key
    if _cached_api_key:
        return _cached_api_key

    env_key = os.getenv(_SERPER_KEY_ENV, "").strip()
    if env_key:
        _cached_api_key = env_key
        return env_key

    for path in _KEY_FILE_CANDIDATES:
        resolved = os.path.normpath(path)
        if os.path.exists(resolved):
            with open(resolved, "r") as f:
                key = f.read().strip()
            if key:
                _cached_api_key = key
                return key

    raise FileNotFoundError(
        f"Missing Serper API key. Set {_SERPER_KEY_ENV} env var, "
        f"or create a serper_api_key file in one of the Serper tool directories."
    )


# ---------------------------------------------------------------------------
# SerperClient
# ---------------------------------------------------------------------------
class SerperClient:
    """Thin HTTP client for Serper API calls."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self._api_key = api_key
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        if not self._api_key:
            self._api_key = get_serper_api_key()
        return self._api_key

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def search(
        self,
        query: str,
        search_type: str = "search",
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call a Serper text-search endpoint.

        Args:
            query: Search query.
            search_type: One of ALL_SEARCH_TYPES.
            num: Number of results.
            gl: Country code (e.g. "us").
            hl: Language code (e.g. "en").

        Returns:
            Raw JSON response dict from Serper.
        """
        if search_type not in ALL_SEARCH_TYPES:
            raise ValueError(f"Invalid search_type: {search_type!r}. Choose from {ALL_SEARCH_TYPES}")

        url = f"{SERPER_SEARCH_BASE}/{search_type}"
        payload: Dict[str, Any] = {"q": query, "num": num}
        if gl:
            payload["gl"] = gl
        if hl:
            payload["hl"] = hl

        response = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def lens(self, image_url: str) -> Dict[str, Any]:
        """Call Serper Google Lens (reverse image search).

        Args:
            image_url: Public URL of the image.

        Returns:
            Raw JSON response dict.
        """
        url = f"{SERPER_SEARCH_BASE}/lens"
        payload = {"url": image_url}
        response = requests.post(url, headers=self._headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def scrape(self, page_url: str) -> Dict[str, Any]:
        """Call Serper webpage scrape endpoint.

        Args:
            page_url: URL to scrape.

        Returns:
            Dict with keys 'text', 'metadata', optionally 'jsonld', 'credits'.
        """
        payload = {"url": page_url}
        response = requests.post(
            SERPER_SCRAPE_BASE, headers=self._headers, json=payload, timeout=60,
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Result formatters
# ---------------------------------------------------------------------------
def format_search_results(search_type: str, data: Dict[str, Any], max_results: int = 10) -> str:
    """Format raw Serper response into agent-readable text."""
    formatter = _FORMATTERS.get(search_type, _format_generic)
    return formatter(data, max_results)


def _format_organic(data: Dict[str, Any], max_results: int) -> str:
    """Format /search, /scholar, /patents responses (organic key)."""
    lines = []

    # Knowledge graph (if present)
    kg = data.get("knowledgeGraph")
    if kg:
        lines.append(f"[Knowledge Graph] {kg.get('title', '')}: {kg.get('description', '')}")
        for k, v in (kg.get("attributes") or {}).items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Answer box (if present)
    ab = data.get("answerBox")
    if ab:
        lines.append(f"[Answer Box] {ab.get('title', '')}")
        lines.append(ab.get("snippet", ""))
        lines.append("")

    items = data.get("organic", [])[:max_results]
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        extra_parts = []
        if item.get("year"):
            extra_parts.append(f"Year: {item['year']}")
        if item.get("citedBy"):
            cited = item["citedBy"]
            if isinstance(cited, dict):
                extra_parts.append(f"Cited by: {cited.get('total', '')}")
        extra = " | ".join(extra_parts)
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {link}")
        if snippet:
            lines.append(f"   {snippet}")
        if extra:
            lines.append(f"   {extra}")
    return "\n".join(lines)


def _format_images(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("images", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        img_url = item.get("imageUrl", "")
        source = item.get("source", "")
        w = item.get("imageWidth", "")
        h = item.get("imageHeight", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   Image: {img_url}")
        if source:
            lines.append(f"   Source: {source}")
        if w and h:
            lines.append(f"   Size: {w}x{h}")
    return "\n".join(lines)


def _format_news(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("news", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        link = item.get("link", "")
        source = item.get("source", "")
        date = item.get("date", "")
        snippet = item.get("snippet", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {link}")
        meta = " | ".join(filter(None, [source, date]))
        if meta:
            lines.append(f"   {meta}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _format_maps(data: Dict[str, Any], max_results: int) -> str:
    """Format /maps responses (places array with rich location data)."""
    items = data.get("places", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        address = item.get("address", "")
        rating = item.get("rating", "")
        count = item.get("ratingCount", "")
        place_type = item.get("type", "")
        phone = item.get("phoneNumber", "")
        website = item.get("website", "")
        lines.append(f"{i}. {title}")
        if place_type:
            lines.append(f"   Type: {place_type}")
        if address:
            lines.append(f"   Address: {address}")
        if rating:
            lines.append(f"   Rating: {rating}" + (f" ({count} reviews)" if count else ""))
        if phone:
            lines.append(f"   Phone: {phone}")
        if website:
            lines.append(f"   Website: {website}")
    return "\n".join(lines)


def _format_places(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("places", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        address = item.get("address", "")
        rating = item.get("rating", "")
        count = item.get("ratingCount", "")
        lines.append(f"{i}. {title}")
        if address:
            lines.append(f"   Address: {address}")
        if rating:
            lines.append(f"   Rating: {rating}" + (f" ({count} reviews)" if count else ""))
    return "\n".join(lines)


def _format_shopping(data: Dict[str, Any], max_results: int) -> str:
    items = data.get("shopping", [])[:max_results]
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        price = item.get("price", "")
        link = item.get("link", "")
        source = item.get("source", "")
        lines.append(f"{i}. {title}")
        if price:
            lines.append(f"   Price: {price}")
        if link:
            lines.append(f"   URL: {link}")
        if source:
            lines.append(f"   Source: {source}")
    return "\n".join(lines)


def _format_generic(data: Dict[str, Any], max_results: int) -> str:
    """Fallback formatter: try organic, then first list-like value."""
    if "organic" in data:
        return _format_organic(data, max_results)
    for key, val in data.items():
        if isinstance(val, list) and val:
            lines = []
            for i, item in enumerate(val[:max_results], 1):
                if isinstance(item, dict):
                    title = item.get("title", "")
                    link = item.get("link", item.get("url", ""))
                    lines.append(f"{i}. {title}")
                    if link:
                        lines.append(f"   URL: {link}")
                else:
                    lines.append(f"{i}. {item}")
            return "\n".join(lines)
    return str(data)


_FORMATTERS = {
    "search": _format_organic,
    "scholar": _format_organic,
    "patents": _format_organic,
    "images": _format_images,
    "news": _format_news,
    "maps": _format_maps,
    "places": _format_places,
    "shopping": _format_shopping,
}
