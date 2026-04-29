"""Webpage Visit Tool — scrape webpages via Serper and extract content with LLM.

Two-step flow:
1. Fetch page content via Serper Scrape API (fallback: Jina Reader API).
2. Extract evidence and summary using an LLM with structured extraction prompt.

This is the primary Webpage_Visit_Tool implementation backed by Serper's
web scraping API. The legacy Jina/Zhipu-based implementation is available
as ``Webpage_Visit_Tool_BK`` in the ``webpage_visit`` directory.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool
from tools.serper_common import SerperClient, get_serper_api_key, strip_think_tags

logger = logging.getLogger(__name__)

# Extraction prompt: the LLM returns a JSON object with keys: rational, evidence, summary.
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Requirements**
- Return a valid JSON object only (no code fences, Markdown, comments, or additional text).
- The JSON must contain exactly the keys "rational", "evidence", and "summary".
- Each key must map to a string value. Use an empty string if no content is available.
- Do not include any extra keys or explanatory sentences outside the JSON object.

Example:
{{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

TOOL_NAME = "Webpage_Visit_Tool"

JINA_READER_URL = "https://r.jina.ai"
JINA_KEY_FILENAME = "jina_api_key"
# Check both our own directory and the legacy webpage_visit directory.
_JINA_KEY_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), JINA_KEY_FILENAME),
    os.path.join(os.path.dirname(__file__), "..", "webpage_visit", JINA_KEY_FILENAME),
]

MAX_URLS = int(os.getenv("WEBPAGE_VISIT_MAX_URLS", "5"))
MAX_CONTENT_CHARS = int(os.getenv("WEBPAGE_VISIT_MAX_CONTENT_CHARS", "120000"))

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_URLS,
            "description": (
                "List of webpage URLs to visit and extract content from. "
                "Accepts URLs up to the configured maximum. "
                "Each URL should be a valid HTTP(S) link. "
                "If a URL lacks a scheme, 'https://' is automatically prepended."
            ),
        },
        "goal": {
            "type": "string",
            "description": (
                "The goal or query that guides content extraction and summarization. "
                "The LLM will focus on extracting information relevant to this goal "
                "from the webpage content. Be specific — a focused goal yields "
                "better evidence and summaries."
            ),
        },
    },
    "required": ["urls", "goal"],
    "additionalProperties": False,
}


class Webpage_Visit_Tool(BaseTool):
    """Visit webpages via Serper Scrape and extract goal-relevant content with LLM.

    Primary backend: Serper Scrape API (scrape.serper.dev).
    Fallback: Jina Reader API (r.jina.ai).
    Content extraction: LLM-based structured extraction returning
    evidence + summary in JSON format.
    """

    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Visit one or more webpages, extract their text content using "
                "Serper's web scraping API, and summarize the content relevant to "
                "a given goal using an LLM. Returns structured evidence and a "
                "summary for each URL.\n\n"
                "Use this tool when you need to read and understand actual webpage "
                "content for a specific purpose — e.g., after obtaining URLs from "
                "a search tool. Do NOT use this tool for web search or URL "
                "discovery; use Text_Search_Tool or Image_Search_Tool first to "
                "find relevant URLs, then pass those URLs here.\n\n"
                "The tool processes each URL independently: fetches the page content "
                "(with automatic fallback to Jina Reader if Serper fails), truncates "
                "overly long content, and runs LLM extraction to produce structured "
                "output with 'evidence' (relevant quotes/paraphrases) and 'summary' "
                "(concise connection to the goal)."
            ),
            tool_version="1.0.0",
            input_types={
                "urls": "list[str] - webpage URL(s) to visit (1 to configured max)",
                "goal": "str - goal or query guiding content extraction",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["page_visit", "scrape_visit"],
            output_type=(
                "str - concatenated results for each URL, separated by '======='. "
                "Each result contains:\n"
                "  'Evidence in page:' — relevant quotes or paraphrases from the page\n"
                "  'Summary:' — concise summary connecting the evidence to the goal\n"
                "If a URL cannot be fetched, the result contains an error message."
            ),
            demo_commands=[
                {
                    "command": (
                        'Webpage_Visit_Tool(urls=["https://en.wikipedia.org/'
                        'wiki/Python_(programming_language)"], '
                        'goal="What are the key features of Python?")'
                    ),
                    "description": "Visit a Wikipedia page and extract content about Python features.",
                },
                {
                    "command": (
                        'Webpage_Visit_Tool(urls=["https://example.com/page1", '
                        '"https://example.com/page2"], '
                        'goal="Compare the two products.")'
                    ),
                    "description": "Visit multiple pages and extract comparative information.",
                },
            ],
            user_metadata={
                "api_key_env": "SERPER_API_KEY",
                "scrape_endpoint": "https://scrape.serper.dev/",
                "fallback": "Jina Reader API (https://r.jina.ai)",
                "llm_model_env": "WEBPAGE_VISIT_LLM_MODEL",
                "max_urls": MAX_URLS,
                "max_content_chars": MAX_CONTENT_CHARS,
                "notes": (
                    "Serper Scrape costs 2 credits per request (vs 1 for search). "
                    "Falls back to Jina Reader if Serper fails and JINA_API_KEY is set."
                ),
            },
        )

        # LLM engine for content extraction/summarization.
        model = (
            os.getenv("WEBPAGE_VISIT_LLM_MODEL")
            or os.getenv("LLM_MODEL")
            or DEFAULT_LLM_MODEL
        )
        base_url = (
            os.getenv("WEBPAGE_VISIT_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or DEFAULT_LLM_BASE_URL
        )
        api_key = (
            os.getenv("WEBPAGE_VISIT_LLM_API_KEY")
            or os.getenv("LLM_API_KEY")
            or DEFAULT_LLM_API_KEY
        )
        _enable_thinking = os.getenv("WEBPAGE_VISIT_LLM_ENABLE_THINKING", "false").strip().lower() in ("true", "1", "yes")
        self.llm_engine = LLMEngine(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=_enable_thinking,
        )

        self._client: Optional[SerperClient] = None
        self._jina_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Client & key management
    # ------------------------------------------------------------------

    def _get_client(self) -> SerperClient:
        if self._client is None:
            self._client = SerperClient()
        return self._client

    def _load_jina_key(self) -> Optional[str]:
        if self._jina_api_key is not None:
            return self._jina_api_key
        env_key = os.getenv("JINA_API_KEY", "").strip()
        if env_key:
            self._jina_api_key = env_key
            return env_key
        for path in _JINA_KEY_CANDIDATES:
            resolved = os.path.normpath(path)
            if os.path.exists(resolved):
                with open(resolved, "r") as f:
                    key = f.read().strip()
                if key:
                    self._jina_api_key = key
                    return key
        return None

    def sanity_check(self) -> List[str]:
        """Verify that Serper API key (or Jina fallback key) is available."""
        try:
            get_serper_api_key()
        except Exception as exc:
            jina_key = self._load_jina_key()
            if not jina_key:
                return [
                    f"No Serper or Jina API key found. {exc} "
                    "Or set JINA_API_KEY for fallback."
                ]
        return []

    # ------------------------------------------------------------------
    # Page fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Add https:// scheme if missing."""
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _fetch_page_serper(self, url: str) -> Dict[str, Any]:
        """Fetch page content via Serper Scrape API (scrape.serper.dev)."""
        client = self._get_client()
        data = client.scrape(url)
        text = data.get("text", "")
        metadata = data.get("metadata", {})
        if not text:
            raise RuntimeError("Serper Scrape returned empty text")
        return {
            "content": text,
            "description": metadata.get("description", "") or metadata.get("og:description", ""),
        }

    def _fetch_page_jina(self, url: str) -> Dict[str, Any]:
        """Fetch page content via Jina Reader API (fallback)."""
        jina_key = self._load_jina_key()
        if not jina_key:
            raise RuntimeError("Jina API key not available for fallback")
        headers = {"Authorization": f"Bearer {jina_key}"}
        response = requests.post(JINA_READER_URL, headers=headers, data={"url": url}, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Jina Reader API returned HTTP {response.status_code}")
        return {"content": response.text or "", "description": ""}

    def _fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch with Serper primary, Jina fallback."""
        serper_error = None
        try:
            return self._fetch_page_serper(url)
        except Exception as exc:
            serper_error = str(exc)
            logger.warning("Serper scrape failed for %s: %s", url, serper_error)

        jina_error = None
        try:
            return self._fetch_page_jina(url)
        except Exception as exc:
            jina_error = str(exc)
            logger.warning("Jina fallback also failed for %s: %s", url, jina_error)

        parts = []
        if serper_error:
            parts.append(f"Serper: {serper_error}")
        if jina_error:
            parts.append(f"Jina: {jina_error}")
        raise RuntimeError(f"Failed to fetch {url}. " + "; ".join(parts))

    # ------------------------------------------------------------------
    # Content processing
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_content(content: str) -> str:
        """Truncate content to MAX_CONTENT_CHARS if needed."""
        if len(content) <= MAX_CONTENT_CHARS:
            return content
        return content[:MAX_CONTENT_CHARS] + "\n[Content truncated...]"

    def _extract_content(
        self,
        content: str,
        goal: str,
        description: str,
        engine: Optional[LLMEngine] = None,
    ) -> Dict[str, str]:
        """Extract evidence and summary from page content using the LLM."""
        engine = engine or self.llm_engine
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal or "N/A")

        try:
            extract_max_tokens = int(os.getenv("WEBPAGE_VISIT_LLM_MAX_TOKENS", "16384"))
            raw_output = engine.generate(prompt, max_tokens=extract_max_tokens, temperature=0.0, top_p=1.0)
        except Exception:
            return {
                "evidence": content,
                "summary": description or "Extraction service unavailable.",
            }

        return self._parse_extraction_response(raw_output, content, description)

    @staticmethod
    def _parse_extraction_response(
        raw_output: str, content: str, description: str,
    ) -> Dict[str, str]:
        """Parse the LLM extraction JSON response."""
        candidate = strip_think_tags(raw_output)
        if candidate.startswith("`"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return {
                    "evidence": result.get("evidence") or content,
                    "summary": result.get("summary") or description or "",
                }
        except json.JSONDecodeError:
            pass
        return {"evidence": candidate, "summary": candidate}

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_success(url: str, goal: str, evidence: str, summary: str) -> str:
        output = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        output += "Evidence in page: \n" + evidence + "\n\n"
        output += "Summary: \n" + (summary or "No summary generated") + "\n\n"
        return output

    @staticmethod
    def _format_failure(url: str, goal: str, reason: str) -> str:
        output = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        output += "Evidence in page: \n" + reason + "\n\n"
        output += "Summary: \nUnable to retrieve webpage content. Please check the link or try again later.\n\n"
        return output

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _get_llm_engine(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> LLMEngine:
        """Return the LLM engine to use for extraction (with optional overrides)."""
        if not llm_model and not llm_base_url and not llm_api_key:
            return self.llm_engine
        return LLMEngine(
            model=llm_model or self.llm_engine.model,
            base_url=llm_base_url or self.llm_engine.base_url,
            api_key=llm_api_key or self.llm_engine.api_key,
            temperature=0.0,
            top_p=1.0,
        )

    def execute(
        self,
        urls: List[str],
        goal: str,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> str:
        """Visit webpages and extract content relevant to the goal.

        Args:
            urls: List of webpage URLs to visit.
            goal: Goal or query guiding content extraction.
            llm_model: Optional LLM model override for extraction.
            llm_base_url: Optional LLM base URL override.
            llm_api_key: Optional LLM API key override.

        Returns:
            Formatted string with evidence and summary for each URL,
            separated by '=======' dividers.
        """
        if not urls or not isinstance(urls, list):
            raise ValueError("urls must be a non-empty list of strings")
        if len(urls) > MAX_URLS:
            raise ValueError(f"Too many URLs ({len(urls)}). Maximum is {MAX_URLS}.")
        if not goal or not isinstance(goal, str):
            raise ValueError("goal is required and must be a non-empty string")

        engine = self._get_llm_engine(llm_model, llm_base_url, llm_api_key)

        results = []
        for raw_url in urls:
            url = self._normalize_url(raw_url)

            try:
                page_data = self._fetch_page(url)
            except Exception as exc:
                results.append(self._format_failure(url, goal, str(exc)))
                continue

            content = page_data.get("content") or ""
            description = page_data.get("description") or ""

            if not content:
                fallback = description or "Webpage content is empty"
                results.append(self._format_failure(url, goal, fallback))
                continue

            content = self._truncate_content(content)
            extraction = self._extract_content(content, goal, description, engine=engine)
            evidence = extraction.get("evidence") or content
            summary = extraction.get("summary") or description or ""

            results.append(self._format_success(url, goal, evidence, summary))

        return "\n\n=======\n\n".join(results)
