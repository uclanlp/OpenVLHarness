"""Text Search Tool — Serper-based web search with optional LLM summarization.

Supports all Serper text-search endpoints (search, images, news, scholar,
maps, places, shopping, patents). The agent selects the endpoint via the
``search_type`` parameter (default ``"search"``).

When ``SEARCH_LLM_SUMMARIZE=true`` (opt-in), search results are passed
through an LLM that generates a coherent summary with inline citations.
When ``false`` (default), raw formatted snippets are returned directly.
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

# --- LLM summarization prompt (rerank + top-K filter + summarize) ---
# Split into system (static role + rules + format) and user (per-call data).
# Semantics: the LLM acts as a curator. It reranks raw results by relevance,
# strictly filters out off-topic / low-quality / near-duplicate items, and
# keeps AT MOST K of the survivors. If filtering wipes out everything (0
# survivors), the LLM emits an explicit "no relevant results" response with
# concrete next-step hints (query rewrites + tool switches) so the agent
# can recover without guessing.

SEARCH_SUMMARIZE_SYSTEM_PROMPT = """You are a research assistant. Your job is to curate web search results for a downstream agent. You do three things, in order:

1. **Rerank** the raw search results by how directly they help answer the user's query. Order from most useful to least useful.
2. **Filter to top-K**: keep AT MOST K results, where K is given in the user message. Drop results that are off-topic, low-quality, near-duplicates of a higher-ranked result, or do not contribute new information. If fewer than K results are useful, keep only the useful ones — do NOT pad to reach K.
3. **Summarize** using ONLY the kept results, with inline citation markers [1], [2], ... numbered in your final ranking order (NOT the original Serper order).

# Grounding rules
- Ground every claim in the kept results. Never invent information.
- Preserve factual details (dates, names, numbers, URLs) exactly as they appear in the results, even if they seem to contradict the query.
- If after filtering you have nothing useful, say so honestly. Do not fabricate.

# When filtering produces zero useful results
If after reranking and filtering NOT a single result is useful enough to keep (none directly address the query, all are off-topic / low-quality / dead-ends), do NOT fabricate a summary. Instead, emit the structured fallback below so the calling agent can pick a different next action.

You must include BOTH a query-rewrite suggestion AND a tool-switch suggestion. Each suggestion must be specific and actionable, not a near-rephrasing of the current attempt.

Format:

NO RELEVANT RESULTS

Why each raw result was rejected:
- [1] <title> — <reason it was rejected, e.g. off-topic, wrong subject, language/region mismatch, near-duplicate of [N]>
- [2] <title> — <reason>
...

Suggested next actions for the agent:

  Query rewrites (1-3 concrete alternatives, each meaningfully different):
  - <alternative query 1> (rationale: <one short reason it should work better>)
  - <alternative query 2> (rationale: ...)

  Tool switches (1-2 concrete alternatives, when applicable):
  - <tool name>(<key args>) — <one short reason this tool would help>
    Available search_type values for Text_Search_Tool: search, images, news, scholar, maps, places, shopping, patents.
    Pick whichever best matches the query intent — e.g.:
      - search   → general web pages (default)
      - images   → looking for visual references / pictures
      - news     → current events, recent happenings
      - scholar  → academic papers, citations
      - maps     → location lookup, coordinates, directions
      - places   → local businesses, restaurants, hotels
      - shopping → product listings, prices, comparisons
      - patents  → patent filings, IP / inventions
    e.g. "Text_Search_Tool(query=..., search_type=scholar)" if the topic is academic and the current type=search returned consumer pages
    e.g. "Text_Search_Tool(query=..., search_type=news)" if the topic is current events
    e.g. "Image_Search_Tool(image_url=...)" — pass EXACTLY ONE of ``image_url`` (public HTTP(S) URL), ``image_path`` (absolute local path), or ``image_base64`` (encoded bytes); schema is oneOf, never combine
    e.g. "Webpage_Visit_Tool(urls=[<url1>, <url2>, ...], goal=<what to extract>)" — ``urls`` is a list (1 to MAX_URLS, default 5); ``goal`` is required and guides what the LLM extracts from each page

# Output format (when at least one result is useful)
<concise summary using ONLY the kept results, with inline [N] citations>

Citations:
[1] <title> - <url>     (← most relevant)
[2] <title> - <url>
...                     (at most K entries)

# Type-specific citation rules
- search / scholar / patents / news: Citations list page title and page URL (default format above).
- images: Preserve the trailing ` <img>` marker after each image URL EXACTLY as it appears in the raw results — it is a framework signal consumed by downstream tools (Image_Search_Tool, ImageStateMemory). Dropping it breaks image-URL routing. Use the citation format: [N] <title> | Image: <image_url> <img> | Source: <page_url>
- maps: Include place name, address, rating, and phone number directly in the summary. No URL citation block needed.
- places: Include name, address, and rating directly in the summary. No URL citation block needed.
- shopping: Include product name and price in the summary; cite source URL.
"""

SEARCH_SUMMARIZE_USER_TEMPLATE = """Query: {query}

Search Results (Type: {search_type}):
{raw_results}

Keep AT MOST {keep_k} (i.e. K = {keep_k}) results after reranking and filtering. Summarize only those."""

# Backup of the prior single-prompt version, kept for rollback.
SEARCH_SUMMARIZE_PROMPT_BK = """You are a research assistant. Summarize the following search results into a coherent, informative response to the user's query. Your summary must be grounded in the search results — do NOT add information that is not present in the results.

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
- **images**: You MUST preserve the direct image URLs (not just source page URLs) in the Citations section, because downstream tools (e.g., Image_Search_Tool) may need these image URLs as input for reverse image search. Use the format EXACTLY (preserving the trailing ` <img>` marker after each image URL — it is a framework signal): [N] <title> | Image: <image_url> <img> | Source: <source_page_url>
- **maps**: Include place name, address, rating, and phone number in the summary. No URL citation needed.
- **places**: Include name, address, rating in the summary. No URL citation needed.
- **shopping**: Include product name, price, and source URL in citations.

## Output Format
<summary text with inline citations [1], [2], ...>

Citations:
[1] <title> - <url>
[2] <title> - <url>
...

(For images type, use EXACTLY (with the trailing ` <img>` framework marker preserved): [1] <title> | Image: <image_url> <img> | Source: <page_url>)
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

        # Default off: ImageStateMemory's URL routing relies on the
        # tool-emitted ``<img>`` sentinel propagating verbatim, and an
        # LLM summarizer between raw_results and the agent risks
        # dropping the marker. Set SEARCH_LLM_SUMMARIZE=true explicitly
        # to opt back in.
        self._llm_summarize = os.getenv("SEARCH_LLM_SUMMARIZE", "false").strip().lower() in ("true", "1", "yes")
        self._llm_max_tokens = int(os.getenv("SEARCH_LLM_MAX_TOKENS", "4096"))
        self._num_results = int(os.getenv("SEARCH_NUM_RESULTS", "10"))
        # Top-K kept by the LLM curator after reranking (separate from
        # ``num_results``: Serper still fetches ``num_results`` raw items;
        # ``keep_k`` only caps the LLM's final output. Defaults to 3 —
        # tight enough to force real ranking, loose enough to leave room
        # for the agent's downstream multi-round search.
        self._keep_k = int(os.getenv("SEARCH_KEEP_K", "3"))
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
        self, query: str, search_type: str, raw_results: str, *, keep_k: int,
    ) -> str:
        """Pass raw search results through LLM to produce summary + citations.

        Uses a system/user split:
        * ``SEARCH_SUMMARIZE_SYSTEM_PROMPT`` — static curator role + rerank /
          filter / summarize rules + output format (cached server-side).
        * ``SEARCH_SUMMARIZE_USER_TEMPLATE`` — per-call data: query,
          search_type, raw results, top-K cap.
        """
        user_prompt = SEARCH_SUMMARIZE_USER_TEMPLATE.format(
            query=query,
            search_type=search_type,
            raw_results=raw_results,
            keep_k=keep_k,
        )
        try:
            result = self.llm_engine.generate(
                user_prompt,
                system_prompt=SEARCH_SUMMARIZE_SYSTEM_PROMPT,
                max_tokens=self._llm_max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            stripped = strip_think_tags(result or "")
            # Empty / whitespace-only LLM output is not a useful summary —
            # fall back to raw_results so the agent still gets the
            # underlying citations + ``<img>`` sentinels intact.
            if not stripped.strip():
                logger.warning(
                    "LLM summarization returned empty content, returning raw results"
                )
                return raw_results
            return stripped
        except Exception as exc:
            logger.warning("LLM summarization failed (%s), returning raw results", exc)
            return raw_results

    # ------------------------------------------------------------------
    # Empty-result fallback (Serper returned 0 raw items)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_no_raw_results(query: str, search_type: str) -> str:
        """Return a structured ``NO RELEVANT RESULTS`` message for the agent.

        Mirrors the LLM-side fallback (when filtering wipes out all results)
        so the agent sees ONE consistent shape regardless of whether the
        emptiness came from Serper or from the LLM curator. No LLM call —
        suggestions are static / generic since we have no useful raw data
        to ground a query rewrite on.
        """
        # Keep ``ALL_SEARCH_TYPES`` quoted in-line so a future renaming /
        # reorder is a one-line edit and stays in sync with the Text_Search
        # tool's actual capabilities.
        type_list = ", ".join(ALL_SEARCH_TYPES)
        return (
            f"[Search Results — Type: {search_type}]\n"
            "NO RELEVANT RESULTS\n\n"
            f"Reason: Serper returned 0 raw results for query={query!r} (search_type={search_type}).\n\n"
            "Suggested next actions for the agent:\n\n"
            "  Query rewrites:\n"
            "  - Try a broader / more general phrasing of the same intent\n"
            "  - Try keywords / entity names instead of a full natural-language question\n"
            "  - Try a different language (e.g. switch en ↔ zh) if the entity has a localized name\n"
            "  - Drop highly specific qualifiers (year, version, region) that may have over-constrained the match\n\n"
            "  Tool switches:\n"
            f"  - Text_Search_Tool(query=..., search_type=<one of: {type_list}>)\n"
            "      Try a different search_type whose vertical fits the intent better — e.g. ``news`` for current "
            "events, ``scholar`` for academic, ``maps`` / ``places`` for locations, ``shopping`` for products, "
            "``patents`` for inventions, ``images`` for visual references.\n"
            "  - Image_Search_Tool(image_url=...)\n"
            "      Reverse image search (Google Lens) when a relevant image is available. "
            "Pass EXACTLY ONE of ``image_url`` (public HTTP(S) URL), ``image_path`` (absolute local path), "
            "or ``image_base64`` (encoded bytes); the schema is oneOf — never combine them.\n"
            "  - Webpage_Visit_Tool(urls=[<url1>, ..., <urlN>], goal=<what to extract from the page>)\n"
            "      If specific URL(s) are already known, fetch and extract content directly. "
            "Both arguments are required: ``urls`` is a list of 1 to 5 HTTP(S) links "
            "(WEBPAGE_VISIT_MAX_URLS env var raises the cap), and ``goal`` is the extraction "
            "intent that guides the per-page LLM extractor.\n"
        )

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
            return self._format_no_raw_results(query, resolved_type)

        # --- LLM summarization ---
        if self._llm_summarize:
            summary = self._summarize_with_llm(
                query, resolved_type, raw_formatted, keep_k=self._keep_k,
            )
            return f"[Search Results — Type: {resolved_type}]\n{summary}"

        return f"[Search Results — Type: {resolved_type}]\n{raw_formatted}"
