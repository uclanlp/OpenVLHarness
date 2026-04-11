"""Shared infrastructure for Serper-based tools.

Provides:
- ``SerperClient`` — thin wrapper for Serper API calls (auth, retry, error handling).
- ``get_serper_api_key`` — load key from env / file.
- ``format_search_results`` — unified result formatter for all Serper text-search endpoints.
- Query classifiers — ``SentenceTransformerClassifier``, ``LLMClassifier``, ``create_classifier``.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
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


# ---------------------------------------------------------------------------
# Query classifiers
# ---------------------------------------------------------------------------
_CLASSIFIER_METHOD_ENV = "SEARCH_CLASSIFIER_METHOD"
_CLASSIFIER_MODEL_ENV = "SEARCH_CLASSIFIER_MODEL"
_CLASSIFIER_BASE_ENV = "SEARCH_CLASSIFIER_BASE"
_CLASSIFIER_KEY_ENV = "SEARCH_CLASSIFIER_KEY"
_CLASSIFIER_THRESHOLD_ENV = "SEARCH_CLASSIFIER_THRESHOLD"
_ENABLED_TYPES_ENV = "SEARCH_ENABLED_TYPES"

# Default sentence transformer model
DEFAULT_ST_MODEL = "BAAI/bge-m3"
DEFAULT_CLASSIFIER_THRESHOLD = 0.45

# Single source of truth: what each Serper endpoint handles.
# Used by both SentenceTransformerClassifier (as embedding reference) and
# LLMClassifier (in the routing prompt).  "search" is the catch-all default.
SEARCH_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "search": "general web search for factual or informational queries",
    "images": "find images matching a text description",
    "news": "breaking news, current events, recent happenings",
    "scholar": "academic papers, journal articles, citations, research publications",
    "maps": "find locations on a map, get directions, geographic lookup",
    "places": "discover local businesses, restaurants, shops, hotels nearby",
    "shopping": "product listings, prices, purchase comparisons",
    "patents": "patent filings, inventions, intellectual property search",
}


class BaseClassifier(ABC):
    """Interface for query classifiers."""

    @abstractmethod
    def classify(self, query: str, enabled_types: List[str]) -> str:
        """Classify a query into one of the enabled search types.

        Returns the best matching search type string, or 'search' as fallback.
        """
        ...


class SentenceTransformerClassifier(BaseClassifier):
    """Specialist-detection classifier using sentence-transformer cosine similarity.

    "search" is the default — it is never a candidate for embedding matching.
    Only specialist categories (scholar, news, images, maps, …) compete.
    A specialist must exceed ``threshold`` to win; otherwise the query falls
    back to "search".
    """

    def __init__(self, model_name: Optional[str] = None, threshold: Optional[float] = None):
        self.model_name = model_name or os.getenv(_CLASSIFIER_MODEL_ENV, DEFAULT_ST_MODEL)
        self.threshold = threshold if threshold is not None else float(
            os.getenv(_CLASSIFIER_THRESHOLD_ENV, str(DEFAULT_CLASSIFIER_THRESHOLD))
        )
        self._model = None
        self._specialist_embeddings: Optional[Dict[str, Any]] = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerClassifier. "
                "Install it with: pip install sentence-transformers"
            )
        logger.info("Loading sentence transformer model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-compute embedding for each specialist category from its description."""
        self._specialist_embeddings = {}
        for category, description in SEARCH_TYPE_DESCRIPTIONS.items():
            if category == "search":
                continue  # search is the default, not a specialist
            emb = self._model.encode([description], normalize_embeddings=True)[0]
            self._specialist_embeddings[category] = emb

    def classify(self, query: str, enabled_types: List[str]) -> str:
        import numpy as np

        self._load_model()

        query_emb = self._model.encode([query], normalize_embeddings=True)[0]
        best_type = "search"
        best_score = -1.0

        for cat in enabled_types:
            # "search" is the default fallback — never competes via embedding
            if cat == "search":
                continue
            if cat not in self._specialist_embeddings:
                continue
            score = float(np.dot(query_emb, self._specialist_embeddings[cat]))
            if score > best_score:
                best_score = score
                best_type = cat

        if best_score < self.threshold:
            logger.debug(
                "No specialist above threshold (best=%.3f < %.3f), using 'search'",
                best_score, self.threshold,
            )
            return "search"

        logger.debug("Classified query as specialist '%s' (score=%.3f)", best_type, best_score)
        return best_type


class LLMClassifier(BaseClassifier):
    """Classify queries using an LLM (OpenAI-compatible API)."""

    @staticmethod
    def _build_prompt(query: str, enabled_types: List[str]) -> str:
        """Build the routing prompt dynamically from SEARCH_TYPE_DESCRIPTIONS."""
        type_lines = []
        for t in enabled_types:
            desc = SEARCH_TYPE_DESCRIPTIONS.get(t, t)
            if t == "search":
                type_lines.append(f'- "search" — {desc} (DEFAULT — use when no specialist clearly fits)')
            else:
                type_lines.append(f'- "{t}" — {desc}')
        types_block = "\n".join(type_lines)

        return (
            "You route search queries to the correct Serper API endpoint.\n\n"
            f"Available endpoints:\n{types_block}\n\n"
            "IMPORTANT: \"search\" is the default. Most queries belong to \"search\".\n"
            "Only choose a specialist endpoint when the query clearly and "
            "explicitly targets that vertical. When in doubt, choose \"search\".\n\n"
            f"Query: {query}\n\n"
            "Reply with ONLY the endpoint name (one word, no explanation).\n"
            "Endpoint:"
        )

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv(_CLASSIFIER_BASE_ENV, "")
        self.model = model or os.getenv(_CLASSIFIER_MODEL_ENV, "gpt-4o-mini")
        self.api_key = api_key or os.getenv(_CLASSIFIER_KEY_ENV, "")

    def classify(self, query: str, enabled_types: List[str]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for LLMClassifier. "
                "Install it with: pip install openai"
            )

        prompt = self._build_prompt(query, enabled_types)

        client_kwargs: Dict[str, Any] = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        else:
            client_kwargs["api_key"] = "empty"

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip().lower()
        # Strip quotes/punctuation for robust matching
        cleaned = re.sub(r"[^a-z]", "", raw)
        if cleaned in enabled_types:
            logger.debug("LLM classified query as '%s' (raw: %s)", cleaned, raw)
            return cleaned

        # Fallback: always safe to use search
        logger.warning("LLM returned unparseable '%s', falling back to 'search'", raw)
        return "search"


def create_classifier(method: Optional[str] = None) -> BaseClassifier:
    """Factory: create a query classifier based on config.

    Args:
        method: One of 'sentence_transformer', 'llm'. If None, reads
                from env var SEARCH_CLASSIFIER_METHOD (default: sentence_transformer).

    Returns:
        A BaseClassifier instance.
    """
    method = (method or os.getenv(_CLASSIFIER_METHOD_ENV, "sentence_transformer")).strip().lower()

    if method in ("sentence_transformer", "st"):
        return SentenceTransformerClassifier()
    elif method in ("llm", "openai"):
        return LLMClassifier()
    else:
        logger.warning("Unknown classifier method '%s', falling back to sentence_transformer", method)
        return SentenceTransformerClassifier()


def get_enabled_search_types() -> List[str]:
    """Read the enabled search sub-types from env, default all."""
    raw = os.getenv(_ENABLED_TYPES_ENV, "")
    if not raw.strip():
        return list(ALL_SEARCH_TYPES)
    types = [t.strip().lower() for t in raw.split(",") if t.strip()]
    valid = [t for t in types if t in ALL_SEARCH_TYPES]
    if not valid:
        return list(ALL_SEARCH_TYPES)
    # Always ensure "search" is available as fallback
    if "search" not in valid:
        valid.insert(0, "search")
    return valid
