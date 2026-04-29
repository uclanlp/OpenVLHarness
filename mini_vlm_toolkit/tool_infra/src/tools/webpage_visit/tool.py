import json
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

TOOL_NAME = "Webpage_Visit_Tool_BK"

ZHIPU_KEY_FILENAME = "zhipu_api_key"
JINA_KEY_FILENAME = "jina_api_key"
ZHIPU_KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), ZHIPU_KEY_FILENAME)
JINA_KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), JINA_KEY_FILENAME)

# Default Reader API endpoints (matching Vision-DeepResearch reference)
ZHIPU_READER_URL = "https://search-svip.bigmodel.cn/api/paas/v4/reader"
JINA_READER_URL = "https://r.jina.ai"

MAX_URLS = 5
MAX_CONTENT_CHARS = 120000

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": (
                "List of webpage URLs to visit and extract content from (max 5)."
            ),
        },
        "goal": {
            "type": "string",
            "description": (
                "The goal or query for content extraction and summarization. "
                "The tool will extract and summarize content relevant to this goal."
            ),
        },
    },
    "required": ["urls", "goal"],
    "additionalProperties": False,
}

# Extraction prompt matching Vision-DeepResearch's EXTRACTOR_PROMPT.
# The LLM returns a JSON object with keys: rational, evidence, summary.
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


class Webpage_Visit_Tool_BK(BaseTool):
    """[Legacy] Visit webpage(s) via Jina/Zhipu Reader APIs and extract content.

    This is the legacy implementation using Jina/Zhipu Reader backends.
    The primary Webpage_Visit_Tool now uses Serper Scrape API.
    """

    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Visit one or more webpages, convert them to clean markdown "
                "via Reader APIs (Zhipu primary, Jina fallback), and extract "
                "content relevant to a given goal using an LLM. Returns "
                "evidence and a summary for each URL. Use this tool when you "
                "need to read and understand webpage content for a specific "
                "purpose. Do not use this tool for web search or URL "
                "discovery: use a search tool first to find relevant URLs."
            ),
            tool_version="1.0.0",
            input_types={
                "urls": "list[str] - URL(s) to visit (max 5)",
                "goal": "str - goal or query for content extraction",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["visit_webpage", "web_visit"],
            output_type="str - per-URL evidence and summary",
            demo_commands=[
                {
                    "command": (
                        'Webpage_Visit_Tool(urls=["https://en.wikipedia.org/'
                        'wiki/Python_(programming_language)"], '
                        'goal="What is Python programming language?")'
                    ),
                    "description": "Visit a Wikipedia page and extract content.",
                },
                {
                    "command": (
                        'Webpage_Visit_Tool(urls=["https://example.com/page1",'
                        ' "https://example.com/page2"], '
                        'goal="Compare the two products.")'
                    ),
                    "description": "Visit multiple pages and extract content.",
                },
            ],
            user_metadata={
                "api_key_files": [ZHIPU_KEY_FILENAME, JINA_KEY_FILENAME],
            },
        )

        # LLM engine for content extraction/summarization.
        # Tool-specific env vars take priority over the shared LLM_* vars,
        # so that changing the visit-tool model does not affect other tools
        # (e.g. Python_Code_Generator_Tool which exec()s generated code).
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
        self.llm_engine = LLMEngine(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            top_p=1.0,
        )

        # Lazy-loaded API keys
        self._zhipu_api_key: Optional[str] = None
        self._jina_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # API key management (lazy file-based loading)
    # ------------------------------------------------------------------

    def _load_zhipu_key(self) -> Optional[str]:
        """Load the Zhipu API key from file. Returns None if unavailable."""
        if self._zhipu_api_key is not None:
            return self._zhipu_api_key
        if not os.path.exists(ZHIPU_KEY_FILE_PATH):
            return None
        with open(ZHIPU_KEY_FILE_PATH, "r") as handle:
            key = handle.read().strip()
        if not key:
            return None
        self._zhipu_api_key = key
        return key

    def _load_jina_key(self) -> Optional[str]:
        """Load the Jina API key from file. Returns None if unavailable."""
        if self._jina_api_key is not None:
            return self._jina_api_key
        if not os.path.exists(JINA_KEY_FILE_PATH):
            return None
        with open(JINA_KEY_FILE_PATH, "r") as handle:
            key = handle.read().strip()
        if not key:
            return None
        self._jina_api_key = key
        return key

    def sanity_check(self) -> List[str]:
        """Verify that at least one Reader API key is available."""
        zhipu_key = self._load_zhipu_key()
        jina_key = self._load_jina_key()
        if not zhipu_key and not jina_key:
            return [
                f"No Reader API key found. Create at least one of: "
                f"{ZHIPU_KEY_FILE_PATH} or {JINA_KEY_FILE_PATH} "
                f"with the API key on a single line."
            ]
        return []

    # ------------------------------------------------------------------
    # Reader API calls
    # ------------------------------------------------------------------

    def _normalize_url(self, url: str) -> str:
        """Add https:// scheme if missing."""
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _fetch_page_zhipu(self, url: str) -> Dict[str, Any]:
        """Fetch page content via Zhipu Reader API.

        Returns a dict with keys 'content' and 'description'.
        Raises RuntimeError on failure.
        """
        zhipu_key = self._load_zhipu_key()
        if not zhipu_key:
            raise RuntimeError("Zhipu API key not available")

        headers = {
            "Content-Type": "application/json",
            "Authorization": zhipu_key,
            "X-Return-Format": "markdown",
            "X-No-Cache": "false",
            "X-Timeout": "60",
            "X-Retain-Images": "false",
            "X-With-Images-Summary": "false",
            "X-With-Links-Summary": "false",
        }
        body = {"url": url}

        response = requests.post(
            ZHIPU_READER_URL,
            headers=headers,
            data=json.dumps(body, ensure_ascii=False),
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Zhipu Reader API returned HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Zhipu Reader API returned non-JSON payload") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Zhipu Reader API payload structure is invalid")

        if payload.get("code") != 200:
            raise RuntimeError(
                f"Zhipu Reader API returned error code: {payload.get('code')}"
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Zhipu Reader API data field missing or malformed")

        result = {
            "content": data.get("content") or "",
            "description": data.get("description") or "",
        }
        if not result["content"] and not result["description"]:
            raise RuntimeError(
                "Zhipu Reader API returned empty content and description"
            )
        return result

    def _fetch_page_jina(self, url: str) -> Dict[str, Any]:
        """Fetch page content via Jina Reader API (fallback).

        Returns a dict with keys 'content' and 'description'.
        Raises RuntimeError on failure.
        """
        jina_key = self._load_jina_key()
        if not jina_key:
            raise RuntimeError("Jina API key not available")

        headers = {"Authorization": f"Bearer {jina_key}"}
        body = {"url": url}

        response = requests.post(
            JINA_READER_URL,
            headers=headers,
            data=body,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Jina Reader API returned HTTP {response.status_code}"
            )

        return {
            "content": response.text or "",
            "description": "",
        }

    def _fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch page content with Zhipu primary, Jina fallback.

        Returns a dict with keys 'content' and 'description'.
        Raises RuntimeError if both APIs fail.
        """
        zhipu_key = self._load_zhipu_key()
        jina_key = self._load_jina_key()

        zhipu_error = None
        if zhipu_key:
            try:
                return self._fetch_page_zhipu(url)
            except Exception as exc:
                zhipu_error = str(exc)

        jina_error = None
        if jina_key:
            try:
                return self._fetch_page_jina(url)
            except Exception as exc:
                jina_error = str(exc)

        # Both failed or neither key available
        parts = []
        if zhipu_error:
            parts.append(f"Zhipu: {zhipu_error}")
        if jina_error:
            parts.append(f"Jina: {jina_error}")
        if not parts:
            parts.append("No Reader API key configured")
        raise RuntimeError(
            f"Failed to fetch {url}. " + "; ".join(parts)
        )

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
        engine: Optional["LLMEngine"] = None,
    ) -> Dict[str, str]:
        """Extract evidence and summary from page content using the LLM.

        Args:
            engine: LLMEngine to use. Defaults to self.llm_engine.

        Returns a dict with 'evidence' and 'summary' keys.
        Falls back to raw content if extraction fails.
        """
        engine = engine or self.llm_engine
        prompt = EXTRACTOR_PROMPT.format(
            webpage_content=content,
            goal=goal or "N/A",
        )

        try:
            raw_output = engine.generate(
                prompt, max_tokens=16384, temperature=0.0, top_p=1.0
            )
        except Exception:
            # LLM engine failed; fall back to raw content
            return {
                "evidence": content,
                "summary": description or "Extraction service unavailable.",
            }

        # Parse the JSON response from the LLM
        return self._parse_extraction_response(raw_output, content, description)

    @staticmethod
    def _parse_extraction_response(
        raw_output: str, content: str, description: str
    ) -> Dict[str, str]:
        """Parse the LLM extraction response (JSON with rational/evidence/summary).

        Falls back to treating raw output as plain text if JSON parsing fails.
        """
        candidate = raw_output.strip()
        # Strip code fence markers if present
        if candidate.startswith("`"):
            candidate = candidate.strip("`")
            # Handle ```json ... ``` patterns
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

        # Fallback: treat the raw LLM output as the summary
        return {
            "evidence": candidate,
            "summary": candidate,
        }

    # ------------------------------------------------------------------
    # Output formatting (matching Vision-DeepResearch format)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_success(
        url: str, goal: str, evidence: str, summary: str
    ) -> str:
        """Format a successful result for one URL."""
        output = (
            f"The useful information in {url} for user goal "
            f"{goal or 'N/A'} as follows: \n\n"
        )
        output += "Evidence in page: \n" + evidence + "\n\n"
        output += "Summary: \n" + (summary or "No summary generated") + "\n\n"
        return output

    @staticmethod
    def _format_failure(url: str, goal: str, reason: str) -> str:
        """Format a failure result for one URL."""
        output = (
            f"The useful information in {url} for user goal "
            f"{goal or 'N/A'} as follows: \n\n"
        )
        output += "Evidence in page: \n" + reason + "\n\n"
        output += (
            "Summary: \n"
            "Unable to retrieve webpage content. "
            "Please check the link or try again later.\n\n"
        )
        return output

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _get_llm_engine(
        self,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ) -> "LLMEngine":
        """Return the LLM engine to use for extraction.

        If any override is provided, create a one-off engine with the given
        parameters (falling back to the defaults for unset fields).
        Otherwise return the engine created at init time.
        """
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
            urls: List of URLs to visit (1-5).
            goal: Goal or query for content extraction.
            llm_model: Optional LLM model override for content extraction.
            llm_base_url: Optional LLM base URL override.
            llm_api_key: Optional LLM API key override.

        Returns:
            Formatted string with evidence and summary for each URL.
        """
        if not urls or not isinstance(urls, list):
            raise ValueError("urls must be a non-empty list of strings")
        if len(urls) > MAX_URLS:
            raise ValueError(f"Too many URLs ({len(urls)}). Maximum is {MAX_URLS}.")
        if not goal or not isinstance(goal, str):
            raise ValueError("goal is required and must be a non-empty string")

        # Resolve LLM engine (use override if provided)
        engine = self._get_llm_engine(llm_model, llm_base_url, llm_api_key)

        results = []
        for raw_url in urls:
            url = self._normalize_url(raw_url)

            # Fetch page content via Reader API
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

            # Truncate long content
            content = self._truncate_content(content)

            # Extract evidence and summary using the LLM
            extraction = self._extract_content(content, goal, description, engine=engine)
            evidence = extraction.get("evidence") or content
            summary = extraction.get("summary") or description or ""

            results.append(self._format_success(url, goal, evidence, summary))

        return "\n\n=======\n\n".join(results)
