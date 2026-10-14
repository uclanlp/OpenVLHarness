"""Scrape selected webpages and extract evidence for a user-specified goal.

Webpage_Visit_Tool fetches content through Serper, checks common anti-bot
responses, and uses the auxiliary model to extract per-page evidence and
summaries.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openvlharness.services import degradation
from openvlharness.services.models import AuxiliaryModel
from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.serper_common import SerperClient, strip_think_tags

logger = logging.getLogger(__name__)

# System prompt: defines the model's role and the strict output contract that
# downstream parsing depends on (single JSON object, three keys).
EXTRACTOR_SYSTEM_PROMPT = """You are a precise webpage content extractor. Your job is to read raw webpage text together with a user-supplied goal, then return a structured extraction that captures every detail materially relevant to that goal — no more, no less.

You always:
- Stay strictly grounded in the webpage content; do not invent facts that are not present.
- Output a single valid JSON object matching the schema requested by the user message — no Markdown, no code fences, no commentary outside the JSON.
- Preserve the source's exact wording for precision-sensitive details (numbers, names, dates, quotations).
- Skip content that has no bearing on the user's goal."""


# User-message extraction prompt: the LLM returns a JSON object with keys: rationale, evidence, summary.
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction for Evidence**: Identify and extract **all information materially relevant to the goal — do not omit any important detail**. Preserve the source's exact wording where precision matters (numbers, names, dates, quoted phrases). Stay focused on goal-relevant content; do not verbatim-dump entire pages.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritising clarity and judging the contribution of the information to the goal.

**Final Output Requirements**
- Return a valid JSON object only (no code fences, Markdown, comments, or additional text).
- The JSON must contain exactly the keys "rationale", "evidence", and "summary".
- Each key must map to a string value. Use an empty string if no content is available.
- Do not include any extra keys or explanatory sentences outside the JSON object.

Example:
{{"rationale": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

TOOL_NAME = "Webpage_Visit_Tool"

MAX_URLS = 5
MAX_CONTENT_CHARS = 120000

# Retry config for ``_fetch_page``. Transient errors (5xx / 429 / network
# timeout) are retried with exponential backoff; permanent errors (4xx)
# fail fast. Default ceiling: 1s + 2s = 3s of backoff sleep across 3
# attempts, so worst-case latency on a fully-failing URL is roughly
# ~5-9s including request RTTs.
SERPER_MAX_RETRIES = 3
SERPER_BACKOFF_BASE_SEC = 1.0
SERPER_BACKOFF_MAX_SEC = 4.0
_HTTP_CODE_RE = re.compile(r"^\s*(\d{3})\s+")
_TRANSIENT_HTTP_CODES = frozenset({"500", "502", "503", "504", "429"})

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_URLS,
            "description": (
                f"List of webpage URLs to visit and extract content from. "
                f"Up to {MAX_URLS} URLs per call. "
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
    """Extract evidence and summaries from up to five webpages using an auxiliary model.

    Pages are fetched and processed independently. Structured page records accompany
    the rendered text so callers can inspect per-page evidence and errors.
    """

    require_llm_engine = True

    def __init__(self, llm_engine):
        """Initialize metadata and attach the configured auxiliary model.

        Args:
            llm_engine: AuxiliaryModel-compatible generator for evidence extraction.
                The paper configuration uses Qwen multimodal summarization.
        """
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
                "The tool processes each URL independently: fetches the page content, "
                "truncates overly long content, and runs LLM extraction to produce "
                "structured output with 'evidence' (relevant quotes/paraphrases) "
                "and 'summary' (concise connection to the goal). Transient network "
                "failures are auto-retried; pages that block scraping (Cloudflare, "
                "captcha, etc.) or that respond with permanent errors return a "
                "structured failure message with suggestions for recovery."
            ),
            tool_version="1.0.0",
            input_types={
                "urls": f"list[str] - webpage URL(s) to visit (1 to {MAX_URLS} per call)",
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
                "llm_model_source": "services.models",
                "max_urls": MAX_URLS,
                "max_content_chars": MAX_CONTENT_CHARS,
                "max_retries": SERPER_MAX_RETRIES,
                "notes": (
                    "Serper Scrape costs 2 credits per request (vs 1 for search). "
                    f"Transient failures (5xx / 429 / timeout) are retried up to "
                    f"{SERPER_MAX_RETRIES} times with exponential backoff."
                ),
            },
        )

        self.llm_engine = llm_engine

        self._client: Optional[SerperClient] = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> SerperClient:
        """Return the lazily created Serper HTTP client."""
        if self._client is None:
            self._client = SerperClient()
        return self._client

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

    # --- Anti-bot challenge detection ---
    # Cloudflare / hCaptcha / DataDome / Turnstile servers usually return
    # HTTP 200 with a small challenge body, so ``response.raise_for_status``
    # doesn't catch them — but the body is essentially useless to the LLM
    # extractor (it's JavaScript + "verify you're human" text). Detect
    # these heuristically and treat the page as failed so the agent gets
    # a structured "blocked, try a different source" message instead of
    # a gibberish summary.
    _ANTIBOT_TITLE_MARKERS = (
        "just a moment",          # Cloudflare
        "attention required",     # Cloudflare / Imperva
        "please verify",
        "are you human",
        "verifying you are human", # Cloudflare
        "checking your browser",   # Cloudflare interstitial
        "access denied",
        "request unsuccessful",    # DataDome
        "captcha",
    )
    _ANTIBOT_BODY_MARKERS = (
        "cf-browser-verification",
        "cf-challenge",
        "challenge-platform",
        "captcha-bypass",
        "h-captcha",
        "g-recaptcha",
        "turnstile",
        "datadome",
        "px-captcha",
        "perimeterx",
    )
    _ANTIBOT_BODY_MAX_LEN = 4000

    @classmethod
    def _looks_like_anti_bot(cls, content: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Return a short reason string if ``content`` smells like a bot
        challenge page (200 OK but useless), else None.

        Heuristic: short body (< 4000 chars, since real article pages are
        typically much longer) + at least one anti-bot fingerprint in
        either the title or the body. Both signals required so we don't
        false-positive on legitimate short pages that happen to mention
        e.g. "captcha".
        """
        if not isinstance(content, str):
            return None
        body = content
        body_lower = body[:cls._ANTIBOT_BODY_MAX_LEN].lower()
        title = ""
        if isinstance(metadata, dict):
            title = str(metadata.get("title") or metadata.get("og:title") or "").lower()
        title_hit = next((m for m in cls._ANTIBOT_TITLE_MARKERS if m in title), None)
        body_hit = next((m for m in cls._ANTIBOT_BODY_MARKERS if m in body_lower), None)
        # Only treat as anti-bot if the body is short AND we have at
        # least one strong fingerprint. Long bodies that mention these
        # keywords are usually legitimate articles ABOUT bot detection.
        if len(body) < cls._ANTIBOT_BODY_MAX_LEN and (title_hit or body_hit):
            indicator = title_hit or body_hit
            return (
                f"anti-bot challenge page detected (200 OK but body looks like a "
                f"verification interstitial, indicator: {indicator!r})"
            )
        return None

    def _fetch_page_serper(self, url: str) -> Dict[str, Any]:
        """Fetch page content via Serper Scrape API (scrape.serper.dev).

        Detects anti-bot 200-OK responses (Cloudflare challenge / captcha
        interstitial pages) and raises ``RuntimeError`` with a clear
        explanation so the calling layer can format a structured failure
        for the agent. This is in addition to the explicit HTTP error
        path in ``_fetch_page`` which already handles 4xx / 5xx.
        """
        client = self._get_client()
        data = client.scrape(url)
        text = data.get("text", "")
        metadata = data.get("metadata", {}) or {}
        if not text:
            raise RuntimeError("Serper Scrape returned empty text")
        anti_bot = self._looks_like_anti_bot(text, metadata)
        if anti_bot:
            raise RuntimeError(anti_bot)
        return {
            "content": text,
            "description": metadata.get("description", "") or metadata.get("og:description", ""),
        }

    def _fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch via Serper Scrape with transient-error retry.

        Retries up to ``SERPER_MAX_RETRIES`` times on transient failures
        (5xx / 429 / network timeout) with exponential backoff. Permanent
        failures (4xx other than 429) fail fast on the first attempt.

        On final failure, raises ``RuntimeError`` with an agent-readable
        message that includes HTTP code, transient/permanent
        classification, attempt count, and a suggestion for the agent's
        next move.
        """
        last_error: str = ""
        last_code: Optional[str] = None
        last_is_transient: bool = True
        attempts: int = 0

        for attempt in range(SERPER_MAX_RETRIES):
            attempts = attempt + 1
            try:
                return self._fetch_page_serper(url)
            except Exception as exc:
                last_error = str(exc)
                m = _HTTP_CODE_RE.match(last_error)
                last_code = m.group(1) if m else None
                if last_code is not None:
                    last_is_transient = last_code in _TRANSIENT_HTTP_CODES
                else:
                    # No HTTP code → likely network-layer error
                    # (Timeout / ConnectionError); treat as transient.
                    last_is_transient = any(
                        k in last_error for k in ("Timeout", "Connection")
                    )
                logger.warning(
                    "Serper scrape failed for %s (attempt %d/%d, transient=%s): %s",
                    url, attempts, SERPER_MAX_RETRIES, last_is_transient, last_error,
                )
                if not last_is_transient:
                    break  # 4xx → no point retrying
            if attempt < SERPER_MAX_RETRIES - 1 and last_is_transient:
                time.sleep(
                    min(SERPER_BACKOFF_BASE_SEC * (2 ** attempt),
                        SERPER_BACKOFF_MAX_SEC)
                )

        # Build agent-readable error message.
        msg_parts = [f"Failed to fetch {url}"]
        if last_code:
            msg_parts.append(
                f"(HTTP {last_code}, {'transient' if last_is_transient else 'permanent'})"
            )
        if attempts > 1:
            msg_parts.append(f"after {attempts} attempts")
        msg_parts.append(f"Detail: {last_error}.")

        if last_is_transient:
            msg_parts.append(
                "Suggestion: source temporarily unavailable; try again later "
                "or pick a different URL."
            )
        elif last_code == "404":
            msg_parts.append("Suggestion: URL not found; try a different result.")
        elif last_code in ("401", "403"):
            msg_parts.append("Suggestion: access blocked; try a different source.")
        else:
            msg_parts.append("Suggestion: URL is unreachable; try a different result.")

        raise RuntimeError(" ".join(msg_parts))

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
        engine: Optional[AuxiliaryModel] = None,
    ) -> Dict[str, str]:
        """Extract evidence and summary from page content using the LLM."""
        engine = engine or self.llm_engine
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal or "N/A")

        try:
            extract_max_tokens = 16384
            raw_output = engine.generate(
                prompt,
                system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                max_tokens=extract_max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:
            # EXTRACT_DEGRADED is a stable log token for watchdogs; keep it literal.
            logger.warning(
                "EXTRACT_DEGRADED tool=%s reason=%s -- extraction call failed, "
                "returning raw page content (%s)", TOOL_NAME, type(exc).__name__, exc,
            )
            degradation.record("extract", tool=TOOL_NAME, reason=type(exc).__name__)
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
        except json.JSONDecodeError:
            logger.warning(
                "EXTRACT_DEGRADED tool=%s reason=unparseable_json -- returning the raw extraction reply",
                TOOL_NAME,
            )
            degradation.record("extract", tool=TOOL_NAME, reason="unparseable_json")
            return {"evidence": candidate, "summary": candidate}

        evidence = result.get("evidence") if isinstance(result, dict) else None
        summary = result.get("summary") if isinstance(result, dict) else None
        if isinstance(evidence, str) and evidence.strip():
            if not (isinstance(summary, str) and summary.strip()):
                summary = description or ""
            return {"evidence": evidence, "summary": summary}

        # Parseable reply without usable string fields: keep the page content
        # so the agent still gets evidence, and count the fallback.
        logger.warning(
            "EXTRACT_DEGRADED tool=%s reason=invalid_schema -- returning raw page content "
            "(top-level: %s)", TOOL_NAME, type(result).__name__,
        )
        degradation.record("extract", tool=TOOL_NAME, reason="invalid_schema")
        return {"evidence": content, "summary": description or ""}

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_success(url: str, goal: str, evidence: str, summary: str) -> str:
        """Render one URL's goal, extracted evidence and summary as a text block."""
        output = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        output += "Evidence in page: \n" + evidence + "\n\n"
        output += "Summary: \n" + (summary or "No summary generated") + "\n\n"
        return output

    @staticmethod
    def _format_failure(url: str, goal: str, reason: str) -> str:
        """Render a single-URL fetch failure as an agent-readable block.

        Includes cross-tool suggestions so the agent has a recovery path
        instead of just "Unable to retrieve". Self-inclusion preserved:
        the agent may also re-call Webpage_Visit_Tool on a DIFFERENT URL
        from the same prior search result set.
        """
        reason_lower = reason.lower()
        # Classify the failure to give a tighter recovery hint
        is_anti_bot = "anti-bot" in reason_lower or "challenge" in reason_lower
        is_blocked = is_anti_bot or "403" in reason or "401" in reason or "blocked" in reason_lower
        is_not_found = "404" in reason
        is_empty = (
            "empty text" in reason_lower
            or "essentially-empty" in reason_lower
            or "empty body" in reason_lower
            or "no content" in reason_lower
        )
        is_transient = (
            "transient" in reason_lower
            or "timeout" in reason_lower
            or " 429" in reason
            or " 500" in reason
            or " 502" in reason
            or " 503" in reason
            or " 504" in reason
        )

        if is_anti_bot:
            tight_hint = (
                "The page returned a Cloudflare / captcha / anti-bot interstitial instead of "
                "real content. Re-fetching the same URL will NOT help — the source actively "
                "blocks scrapers. Pick a different source for the same fact."
            )
        elif is_blocked:
            tight_hint = (
                "Access is blocked (HTTP 401 / 403 or equivalent). Re-fetching the same URL "
                "will NOT help — the source requires auth or actively denies the scraper. "
                "Pick a different source."
            )
        elif is_not_found:
            tight_hint = (
                "URL not found (HTTP 404). The page has been moved or deleted. Pick a "
                "different URL from the search results, or re-search to find a fresher source."
            )
        elif is_empty:
            tight_hint = (
                "The page returned HTTP 200 (success) but with essentially no extractable text "
                "content. This typically means: (a) the page is JavaScript-only and the scraper "
                "saw an empty shell rather than the rendered DOM, (b) the source silently "
                "blocked the scraper by returning a blank body instead of an explicit challenge "
                "page, or (c) the URL points to a non-HTML resource (PDF / binary / image-only "
                "page) that the scraper can't extract text from. Re-fetching the same URL will "
                "return the same empty body; pick a different source."
            )
        elif is_transient:
            tight_hint = (
                "Transient failure (5xx / 429 / timeout) after 3 retries. The source MAY be "
                "back online in a minute; retrying this URL once more after a short pause is "
                "reasonable. If it fails again, switch to a different source."
            )
        else:
            tight_hint = (
                "Cause unclear (see Reason line). Re-fetching the same URL is unlikely to "
                "help; switch source or tool."
            )

        output = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        output += "Evidence in page: \n" + reason + "\n\n"
        output += (
            "Summary: \n"
            "FETCH FAILED.\n\n"
            f"Reason: {reason}\n"
            f"Diagnosis: {tight_hint}\n\n"
            "Suggested next actions for the agent:\n\n"
            "  - Tool switches (the agent may also re-call this same Webpage_Visit_Tool on a "
            "DIFFERENT URL — nothing is excluded just because the current URL failed):\n"
            "    - Webpage_Visit_Tool(urls=[<a different webpage_url_N>], goal=...)\n"
            "        Try a different URL from the same search results. If the prior search had "
            "multiple results and only this one fetched, the answer may be on another one of them.\n"
            "    - Text_Search_Tool(query=<rephrased>, search_type=<search|news|scholar|...>)\n"
            "        Re-search to surface fresher / non-blocked sources for the same fact. "
            "If the failing URL was the only result, the search query may have been too narrow.\n"
            "    - Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>)\n"
            "        If your question is fundamentally about identifying an image, switch to "
            "reverse image search — bypasses the failing webpage entirely.\n\n"
        )
        return output

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        urls: List[str],
        goal: str,
    ) -> str:
        """Fetch webpages and extract evidence relevant to a shared goal.

        Args:
            urls: Nonempty list of at most five webpage URLs; missing schemes become HTTPS.
            goal: Nonempty description of the information to extract.

        Returns:
            A ServiceResult whose value is rendered page evidence separated by dividers
            and whose metadata.pages contains URL, text, evidence, summary and error
            fields. Fetch failures are represented per page while other pages continue.

        Raises:
            ValueError: URL collection, goal or individual URL is invalid.

        The auxiliary model is supplied at construction. Task arguments cannot replace
        its model, endpoint or credentials.
        """
        if not urls or not isinstance(urls, list):
            raise ToolArgumentError("urls must be a non-empty list of strings")
        if len(urls) > MAX_URLS:
            raise ToolArgumentError(f"Too many URLs ({len(urls)}). Maximum is {MAX_URLS}.")
        if not goal or not isinstance(goal, str):
            raise ToolArgumentError("goal is required and must be a non-empty string")

        engine = self.llm_engine

        results = []
        pages = []
        for raw_url in urls:
            url = self._normalize_url(raw_url)

            try:
                page_data = self._fetch_page(url)
            except Exception as exc:
                rendered = self._format_failure(url, goal, str(exc))
                results.append(rendered)
                pages.append({"url": url, "text": rendered, "evidence": None, "summary": None, "error": str(exc)})
                continue

            content = page_data.get("content") or ""
            description = page_data.get("description") or ""

            if not content:
                fallback = description or "Webpage content is empty"
                rendered = self._format_failure(url, goal, fallback)
                results.append(rendered)
                pages.append({"url": url, "text": rendered, "evidence": None, "summary": None, "error": fallback})
                continue

            content = self._truncate_content(content)
            extraction = self._extract_content(content, goal, description, engine=engine)
            evidence = extraction.get("evidence") or content
            summary = extraction.get("summary") or description or ""

            rendered = self._format_success(url, goal, evidence, summary)
            results.append(rendered)
            pages.append({"url": url, "text": rendered, "evidence": evidence, "summary": summary, "error": None})

        from openvlharness.contracts import ServiceResult
        return ServiceResult("\n\n=======\n\n".join(results), {"pages": pages})
