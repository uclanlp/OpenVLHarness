"""Tool-specific output post-processing — default placeholder."""

PROCESSOR_MODE = "default_placeholder"


def process_tool_output(payload, prepared, tool_metadata=None):
    """Return None to delegate to the orchestrator's default tool-return
    message construction. Tools needing a custom message build a dict
    ``{"user_message": ..., "processor_info": ...}``.
    """
    return None


from ._base import BaseCapability

PUBLIC_SCHEMA = {'type': 'function',
 'function': {'name': 'Webpage_Visit_Tool',
              'description': 'Visit one or more webpages, extract their text content using '
                             "Serper's web scraping API, and summarize the content relevant to a "
                             'given goal using an LLM. Returns structured evidence and a summary '
                             'for each URL.\n'
                             '\n'
                             'Use this tool when you need to read and understand actual webpage '
                             'content for a specific purpose — e.g., after obtaining URLs from a '
                             'search tool. Do NOT use this tool for web search or URL discovery; '
                             'use Text_Search_Tool or Image_Search_Tool first to find relevant '
                             'URLs, then pass those URLs here.\n'
                             '\n'
                             'The tool processes each URL independently: fetches the page content, '
                             'truncates overly long content, and runs LLM extraction to produce '
                             "structured output with 'evidence' (relevant quotes/paraphrases) and "
                             "'summary' (concise connection to the goal). Transient network "
                             'failures are auto-retried; pages that block scraping (Cloudflare, '
                             'captcha, etc.) or that respond with permanent errors return a '
                             'structured failure message with suggestions for recovery.',
              'parameters': {'type': 'object',
                             'properties': {'urls': {'type': 'array',
                                                     'items': {'type': 'string'},
                                                     'minItems': 1,
                                                     'maxItems': 5,
                                                     'description': 'List of webpage URLs to visit '
                                                                    'and extract content from. Up '
                                                                    'to 5 URLs per call. Each URL '
                                                                    'should be a valid HTTP(S) '
                                                                    'link. If a URL lacks a '
                                                                    "scheme, 'https://' is "
                                                                    'automatically prepended.'},
                                            'goal': {'type': 'string',
                                                     'description': 'The goal or query that guides '
                                                                    'content extraction and '
                                                                    'summarization. The LLM will '
                                                                    'focus on extracting '
                                                                    'information relevant to this '
                                                                    'goal from the webpage '
                                                                    'content. Be specific — a '
                                                                    'focused goal yields better '
                                                                    'evidence and summaries.'}},
                             'required': ['urls', 'goal'],
                             'additionalProperties': False}}}

from typing import TypedDict


class PageResult(TypedDict):
    url: str
    text: str
    evidence: str | None
    summary: str | None
    error: str | None


class Output(TypedDict):
    pages: list[PageResult]


class Capability(BaseCapability):
    name = 'Webpage_Visit_Tool'
    backends = {'serper_webpage_visit': 'Webpage_Visit_Tool'}
    schema = PUBLIC_SCHEMA
    process_output = staticmethod(process_tool_output)

    resolve_urls = True

    def validate_result(self, payload):
        result = payload.get("result")
        pages = result.get("pages") if isinstance(result, dict) else payload.get("meta", {}).get("pages")
        if pages is None:
            return
        if not isinstance(pages, list) or any(
                not isinstance(page, dict) or not isinstance(page.get("url"), str)
                or not isinstance(page.get("text", ""), str)
                or any(page.get(field) is not None and not isinstance(page[field], str)
                       for field in ("evidence", "summary", "error")) for page in pages):
            raise ValueError("Malformed webpage result: pages must contain a URL, text and optional evidence/summary/error strings")

    def normalize_result(self, call, result, prepared, binding_refs) -> Output:
        # Structured page records are supplied by the service before text formatting.
        pages = result.get("pages") if isinstance(result, dict) else prepared.get("service_meta", {}).get("pages")
        if pages is None:
            return {"pages": []}
        return {"pages": [{"url": p["url"], "text": p.get("text", ""),
            "evidence": p.get("evidence"), "summary": p.get("summary"), "error": p.get("error")}
            for p in pages]}
