"""Text Search Tool — Serper-based web search with optional LLM summarization.

Supports all Serper text-search endpoints (search, images, news, scholar,
maps, places, shopping, patents). The agent selects the endpoint via the
``search_type`` parameter (default ``"search"``).

When ``SEARCH_LLM_SUMMARIZE=true``, search results are passed through an
LLM that generates a coherent summary with inline citations. When
``false`` (recommended), raw formatted snippets are returned directly.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool
from tools.serper_common import (
    ALL_SEARCH_TYPES,
    SerperClient,
    format_search_results,
    get_serper_api_key,
    strip_think_tags,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "Text_Search_Tool"

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The search query string. Can be a natural-language question, "
                "a keyword phrase, a product name, an academic paper title, etc. "
                "Choose the appropriate search_type for your query, "
                "or leave it as 'search' for general web results."
            ),
        },
        "search_type": {
            "type": "string",
            "enum": [
                "search",
                "images",
                "news",
                "scholar",
                "maps",
                "places",
                "shopping",
                "patents",
            ],
            "default": "search",
            "description": (
                "Which search endpoint to use. Defaults to 'search' (general web). Options:\n"
                "  'search'   — general web search (titles, snippets, URLs) — use for most queries\n"
                "  'news'     — recent news articles, current events, time-sensitive information\n"
                "  'scholar'  — academic papers, citations, research publications\n"
                "  'images'   — find images matching a text description\n"
                "  'maps'     — map locations with addresses, ratings, and contact info\n"
                "  'places'   — local businesses, restaurants, map results\n"
                "  'shopping' — product listings with prices\n"
                "  'patents'  — patent filings and inventions"
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

# --- LLM summarization prompt ---
# The LLM receives raw search results and generates a coherent summary
# with inline citation markers [1], [2], ... and a Citations block at the end.
SEARCH_SUMMARIZE_PROMPT = """You are a research assistant. Summarize the following search results into a coherent, informative response to the user's query. Your summary must be grounded in the search results — do NOT add information that is not present in the results.

## User Query
{query}

## Search Results (Type: {search_type})
{raw_results}

## Instructions
1. Write a concise but comprehensive summary that directly addresses the user's query.
2. Use inline citation markers like [1], [2], etc. to reference specific search results.
3. After the summary, include a "Citations:" section listing each referenced source.
4. If the search results do not contain enough information, say so honestly.
5. Do NOT hallucinate or invent information not present in the search results.
6. Never alter factual details (dates, names, numbers) from the search results — preserve them exactly as they appear, even if they seem to contradict the query.
7. Include ALL distinct facts from the search results. Do not filter out information based on perceived relevance to the query.

### Type-Specific Rules
- **search / scholar / patents / news**: Citations should list page title + page URL.
- **images**: You MUST preserve the direct image URLs (not just source page URLs) in the Citations section, because downstream tools (e.g., Image_Search_Tool) may need these image URLs as input for reverse image search. Use the format: [N] <title> | Image: <image_url> | Source: <source_page_url>
- **maps**: Include place name, address, rating, and phone number in the summary. No URL citation needed.
- **places**: Include name, address, rating in the summary. No URL citation needed.
- **shopping**: Include product name, price, and source URL in citations.

## Output Format
<summary text with inline citations [1], [2], ...>

Citations:
[1] <title> - <url>
[2] <title> - <url>
...

(For images type, use: [1] <title> | Image: <image_url> | Source: <page_url>)
"""


class Text_Search_Tool(BaseTool):
    """Smart text search with automatic query classification, routing, and LLM summarization.

    Routes queries to the most appropriate Serper endpoint based on query intent.
    Optionally passes results through an LLM to generate a coherent summary with
    inline citations for easy multi-round agent search.
    """

    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Search the web with intelligent query routing and optional LLM "
                "summarization. Given a text query, this tool automatically classifies "
                "the query intent and routes it to the best search endpoint — general "
                "web search, image search (find images by text), news, academic papers "
                "(scholar), maps, local places, shopping, or patents. You can also "
                "specify the search_type to choose a specialist endpoint.\n\n"
                "Returns a coherent summary of search results with inline citations "
                "[1], [2], ... followed by a Citations block listing source titles and "
                "URLs. This format supports multi-round search: read the summary for "
                "quick understanding, then visit specific cited URLs for deeper info.\n\n"
                "Use this tool when you need to discover information, find URLs, or "
                "gather search results. For reading actual webpage content, use "
                "Webpage_Visit_Tool after obtaining URLs from the citations."
            ),
            tool_version="2.0.0",
            input_types={
                "query": "str - search query (natural language or keywords)",
                "search_type": (
                    "str - 'auto' (default) or one of: search, images, news, "
                    "scholar, maps, places, shopping, patents"
                ),
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["smart_search", "text_search"],
            output_type=(
                "str - LLM-generated summary with inline citations [1], [2], ... "
                "followed by a Citations block:\n"
                "  <coherent summary with [1], [2] markers>\n"
                "  Citations:\n"
                "  [1] Title - URL\n"
                "  [2] Title - URL\n"
                "If LLM summarization is disabled, returns raw formatted results."
            ),
            demo_commands=[
                {
                    "command": 'Text_Search_Tool(query="transformer attention paper 2017")',
                    "description": (
                        "Auto-classifies as 'scholar' and returns LLM summary of "
                        "academic papers with citation links."
                    ),
                },
                {
                    "command": 'Text_Search_Tool(query="latest AI breakthroughs", search_type="news")',
                    "description": "Explicitly search news and get summarized results.",
                },
                {
                    "command": 'Text_Search_Tool(query="best pizza near Times Square", search_type="places")',
                    "description": "Search local restaurants — returns summary with ratings and addresses.",
                },
            ],
            user_metadata={
                "api_key_env": "SERPER_API_KEY",
                "llm_summarize_env": "SEARCH_LLM_SUMMARIZE",
                "llm_model_env": "SEARCH_LLM_MODEL",
                "supported_search_types": ALL_SEARCH_TYPES,
            },
        )

        # LLM engine for summarization
        model = (
            os.getenv("SEARCH_LLM_MODEL")
            or os.getenv("LLM_MODEL")
            or DEFAULT_LLM_MODEL
        )
        base_url = (
            os.getenv("SEARCH_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        )
        api_key = (
            os.getenv("SEARCH_LLM_API_KEY")
            or os.getenv("LLM_API_KEY")
            or DEFAULT_LLM_API_KEY
        )
        _enable_thinking = os.getenv("SEARCH_LLM_ENABLE_THINKING", "false").strip().lower() in ("true", "1", "yes")
        self.llm_engine = LLMEngine(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=_enable_thinking,
        )

        self._llm_summarize = os.getenv("SEARCH_LLM_SUMMARIZE", "true").strip().lower() in ("true", "1", "yes")
        self._llm_max_tokens = int(os.getenv("SEARCH_LLM_MAX_TOKENS", "4096"))
        self._num_results = int(os.getenv("SEARCH_NUM_RESULTS", "10"))
        self._client: Optional[SerperClient] = None

    def _get_client(self) -> SerperClient:
        if self._client is None:
            self._client = SerperClient()
        return self._client

    def sanity_check(self):
        try:
            get_serper_api_key()
        except Exception as exc:
            return [str(exc)]
        return []

    # ------------------------------------------------------------------
    # LLM summarization
    # ------------------------------------------------------------------

    def _summarize_with_llm(
        self, query: str, search_type: str, raw_results: str,
    ) -> str:
        """Pass raw search results through LLM to produce summary + citations."""
        prompt = SEARCH_SUMMARIZE_PROMPT.format(
            query=query,
            search_type=search_type,
            raw_results=raw_results,
        )
        try:
            result = self.llm_engine.generate(
                prompt, max_tokens=self._llm_max_tokens, temperature=0.0, top_p=1.0,
            )
            return strip_think_tags(result)
        except Exception as exc:
            logger.warning("LLM summarization failed (%s), returning raw results", exc)
            return raw_results

    # ------------------------------------------------------------------
    # Gemini Google Search integration
    # ------------------------------------------------------------------

    @staticmethod
    def _try_gemini_search(query: str) -> Optional[str]:
        """Attempt Gemini Google Search if enabled. Returns response text or None."""
        if os.getenv("SEARCH_USE_GEMINI", "false").strip().lower() not in ("true", "1", "yes"):
            return None
        try:
            import google.genai as genai
            from google.genai import types

            api_key_file = os.path.join(
                os.path.dirname(__file__), "..", "gemini_google_search", "google_api_key",
            )
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not api_key and os.path.exists(api_key_file):
                with open(api_key_file) as f:
                    api_key = f.read().strip()
            if not api_key:
                return None

            client = genai.Client(api_key=api_key)
            model = os.getenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")
            max_tokens = int(os.getenv("GEMINI_SEARCH_MAX_TOKENS", "8192"))
            tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[tool],
                max_output_tokens=max_tokens,
                system_instruction=(
                    "You are a research assistant. Provide a thorough, well-structured "
                    "answer grounded in search results. Use inline citation markers "
                    "[1], [2], etc. At the end, list all cited sources with titles and URLs "
                    "in a 'Citations:' section. Do NOT hallucinate."
                ),
            )
            response = client.models.generate_content(
                model=model, contents=query, config=config,
            )
            return response.text
        except Exception as exc:
            logger.warning("Gemini search failed (%s), falling back to Serper", exc)
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        search_type: str = "search",
        num_results: Optional[int] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
    ) -> str:
        """Execute a text search with automatic or explicit endpoint routing.

        Args:
            query: Search query string.
            search_type: One of ALL_SEARCH_TYPES (default 'search').
            num_results: Maximum number of results to return (default 10).
            gl: Country code for localized results (e.g. 'us', 'uk', 'cn').
            hl: Language code (e.g. 'en', 'zh', 'fr').

        Returns:
            LLM-generated summary with citations, or raw formatted results
            if LLM summarization is disabled.
        """
        if not query or not isinstance(query, str):
            raise ValueError("query is required and must be a non-empty string")
        if num_results is None:
            num_results = self._num_results

        # --- Optional Gemini path ---
        gemini_result = self._try_gemini_search(query)
        if gemini_result:
            return f"[Search Results — via Gemini Google Search]\n{gemini_result}"

        # --- Resolve search type ---
        resolved_type = search_type.strip().lower() if search_type else "search"

        if resolved_type not in ALL_SEARCH_TYPES:
            logger.warning("Invalid search_type '%s', falling back to 'search'", resolved_type)
            resolved_type = "search"

        logger.info("Text search: type=%s, query=%s", resolved_type, query[:80])

        # --- Call Serper ---
        client = self._get_client()
        data = client.search(
            query=query,
            search_type=resolved_type,
            num=num_results,
            gl=gl,
            hl=hl,
        )

        raw_formatted = format_search_results(resolved_type, data, max_results=num_results)
        if not raw_formatted:
            return f"[Search Results — Type: {resolved_type}]\n(No results found)"

        # --- LLM summarization ---
        if self._llm_summarize:
            summary = self._summarize_with_llm(query, resolved_type, raw_formatted)
            return f"[Search Results — Type: {resolved_type}]\n{summary}"

        return f"[Search Results — Type: {resolved_type}]\n{raw_formatted}"
