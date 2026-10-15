"""Find visual matches and source pages for an image through Serper Google Lens.

Image_Search_Tool uploads inline image bytes to configured cloud storage,
queries Lens, and uses the auxiliary model to select and summarize matches.
The paper configuration supplies Qwen with query and candidate image pixels.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

from openvlharness.services import degradation
from openvlharness.services.tools.base import BaseTool
from openvlharness.services.tools.image_upload_utils import (
    ImageHostUnreachableError,
    content_hash_from_url,
    forget_verified_base_url,
    public_url_problem,
)
from openvlharness.services.tools.image_utils import (
    decode_base64_image_data,
    infer_image_content_type_and_extension,
)
from openvlharness.services.tools.serper_common import (
    SerperClient,
    SerperRequestError,
    fetch_images_parallel,
    strip_think_tags,
)

logger = logging.getLogger(__name__)

# Serper /lens sometimes answers HTTP 200 with an empty ``organic`` list for an
# image it can match on the next call. SerperClient retries only transient HTTP
# codes, so an empty 200 is retried here instead.
LENS_EMPTY_MAX_ATTEMPTS = max(
    1, int(float(os.getenv("SEARCH_LENS_EMPTY_MAX_ATTEMPTS", "4")))
)
LENS_EMPTY_BACKOFF_BASE_SEC = max(
    0.0, float(os.getenv("SEARCH_LENS_EMPTY_BACKOFF_BASE_SEC", "0.5"))
)
LENS_EMPTY_BACKOFF_MAX_SEC = max(
    0.0, float(os.getenv("SEARCH_LENS_EMPTY_BACKOFF_MAX_SEC", "4.0"))
)


# --- LLM summarize prompt (rerank + index-based top-K + prose summary) ---
# The summarizer receives raw reverse-image-search
# (Google Lens) results with raw URLs, reranks them, and outputs:
#   (1) ``kept_indices`` — a list of 1-based original ranks. The
#       downstream renderer slices the top-K raw entries verbatim from
#       ``raw_results`` using these indices, so URL bytes / image URLs /
#       ``<img>`` sentinel are byte-identical to what Serper returned.
#   (2) ``summary`` — a 3-5 sentence prose paragraph distilling what the
#       FULL set of matches (not just the kept) collectively says about
#       the query image's likely subject. Provides the agent's only
#       window into the rank-K+1..N results.
# The LLM NEVER outputs URLs or ref tokens — URL / image-ref integrity
# is guaranteed by construction.
#
# The text-only and multimodal system prompts share their entire tail
# (suggested_next_actions guidance + output rules + no-rank-leak rule +
# examples) — extracted here to keep the two prompts in lockstep.
_IMAGE_SEARCH_SUMMARIZE_TAIL_RULES = """When ``kept_indices`` is non-empty, ``suggested_next_actions`` is OPTIONAL. Emit it only if the kept matches still leave the visual question under-answered. Otherwise leave it as an empty list.

Available next actions for ``suggested_next_actions`` (the agent may also re-call THIS tool — Image_Search_Tool itself — with a different image_id; nothing is excluded just because it is the same tool that produced this result):
- Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>) — same tool, useful when (a) the agent should first call Zoom_In_Tool / Visual_Grounding_Tool to crop or isolate the subject, THEN reverse-search the new sub-image (pass the resulting ``tool_generated_image_N``), or (b) the agent has another available image_id that is a better visual cue for the same question.
- Text_Search_Tool(query=<text query>, search_type=<search|scholar|patents|news|images|maps|places|shopping>) — switch to text search; useful when the question is fundamentally about a textual fact that the image alone doesn't resolve (entity name, date, attribute, comparison).
- Webpage_Visit_Tool(urls=[webpage_url_N], goal=<concrete goal>) — deep-read a specific URL the agent has already received in a prior tool result (URL must be a runtime ref, not a raw https:// string).

When writing entries in ``suggested_next_actions``, keep ``N`` as a literal placeholder when you refer to ``webpage_url_N`` / ``tool_generated_image_N`` / ``input_image_N`` — never substitute a specific number, since you do not see the agent's reference registry.

When emitting ``suggested_next_actions`` (whether ``kept_indices`` is empty or not), pick whichever subset of the three options above genuinely helps; do not artificially exclude this tool from its own suggestion list. For the empty-kept fallback, always include AT LEAST one concrete next action so the agent can recover without guessing.

Output rules:
- Output ONLY the JSON object. No markdown fences, no prefix, no suffix.
- ``kept_indices`` values MUST be valid 1-based ranks present in the raw input's numbered visual-match list. Out-of-range values are silently dropped.
- ``summary`` is plain prose. Do NOT include ``[N]`` citation markers, do NOT include raw URLs (``https://...``), do NOT include ``webpage_url_N`` or ``tool_generated_image_N`` reference tokens. Preserve names / numbers / dates verbatim from the source results.
- ``summary`` MUST NOT reference candidates by their rank number (no "Candidate 5", "rank 3", "match 2", "candidates 4-7", etc.). The agent only sees the top-K survivors after your rerank — any rank-number reference is meaningless to it. **DO still synthesize information from ALL raw candidates including the non-kept ones** — that is the summary's primary purpose. Just attribute facts by *entity / source / publisher* rather than rank position. Examples — ✗ "Candidate 5 (BBC Radio 3) shows Dialogues des Carmelites"  →  ✓ "A BBC Radio 3 broadcast page identifies the scene as Dialogues des Carmelites"; ✗ "candidates 3-7 confirm Eiffel Tower"  →  ✓ "Multiple sources (Wikipedia, a Reddit thread, a Pinterest pin) corroborate the Eiffel Tower identification". Information value > attribution style; if a low-rank match has a useful disambiguator, include the disambiguator even if you can only name the source as "one source" / "a forum post".
- NEVER fabricate facts. If the raw matches don't address some aspect of the visual identity, omit it from ``summary`` rather than inventing.
"""


IMAGE_SEARCH_SUMMARIZE_SYSTEM_PROMPT = """You are a visual-match summarizer. Given raw reverse image search (Google Lens) results and (implicitly) the query image, do exactly three things:

1. RERANK the numbered matches by how likely each one identifies the same subject as the query image, or otherwise provides useful context for the agent.
2. KEEP at most K most useful matches (K is given in the user message). Emit their original 1-based ranks (the leading ``1.``, ``2.``, ... numbers) as ``kept_indices``. If fewer than K are useful, keep only those — do NOT pad to reach K. If NONE help, emit ``kept_indices: []``.
3. SUMMARIZE what the FULL set of raw matches (not just the kept ones) collectively indicates about the query image's likely subject in a concise 3-5 sentence prose paragraph in the ``summary`` field. Capture specific entity names, dates, attributes verbatim. The summary is the agent's only window into the rank-{K+1}+ matches that did NOT survive the top-K cut, so prioritize information the kept matches alone would NOT surface.

Output STRICT JSON, one object, no prose around it. Schema:

{
  "kept_indices": [<1-based int>, <1-based int>, ...],
  "summary": "<3-5 sentence prose distilling what ALL raw matches indicate about the query image>",
  "no_results_justification": null,
  "suggested_next_actions": []
}

If NO match is useful (all unrelated subjects / low-confidence), emit:

{
  "kept_indices": [],
  "summary": null,
  "no_results_justification": "Brief per-raw-match rejection rationale.",
  "suggested_next_actions": [
    "<concrete next action — see allowed options below>",
    ...
  ]
}

""" + _IMAGE_SEARCH_SUMMARIZE_TAIL_RULES

IMAGE_SEARCH_SUMMARIZE_USER_PROMPT = """Visual Search Results:
{raw_results}

Rerank the numbered matches above by visual-identity relevance and pick AT MOST {keep_k} (i.e. K = {keep_k}) of them. Emit their 1-based original ranks as ``kept_indices`` (most relevant first). Then write a 3-5 sentence ``summary`` distilling what ALL raw matches above (including rank-{keep_k}+1 onwards) collectively indicate about the query image's likely subject — preserve entity names / attributes / dates verbatim. Do NOT include URLs, [N] citation markers, or reference tokens in the summary. Output the strict JSON object specified in the system message."""


# --- Multimodal summarize prompt (rerank uses BOTH text and image bytes) ---
# Used when SEARCH_MULTIMODAL_SUMMARIZE=true. The user prompt is built at
# runtime as an interleaved list of text + image blocks: the query image
# first, then each candidate's text metadata followed by its fetched
# thumbnail. Candidates whose thumbnail fetch failed are pre-filtered
# before this prompt is invoked, so every candidate the summarizer sees
# has an actual image attached.
IMAGE_SEARCH_MULTIMODAL_SYSTEM_PROMPT = """You are a visual-match summarizer. You see the QUERY image (the image the user wants to identify) followed by NUMBERED candidate matches; each candidate has its text metadata AND its own image. Do exactly three things:

1. RERANK the numbered matches by how likely each one identifies the same subject as the query image. Combine THREE signals (in order of importance):
   (a) Visual similarity between the candidate image and the QUERY image — same subject / scene / object instance, not just same category. A candidate whose image looks like a different subject than the query should be down-ranked even if its title looks promising.
   (b) Image-text alignment within the candidate itself — does the candidate's image actually match its own title / source? A title claiming "Eiffel Tower" with an image of a different tower is misleading and should be down-ranked.
   (c) Text relevance — title / source domain reliability.
2. KEEP at most K most useful matches (K is given in the user message). Emit their original 1-based ranks (the leading ``1.``, ``2.``, ... numbers) as ``kept_indices``. If fewer than K are useful, keep only those — do NOT pad to reach K. If NONE help, emit ``kept_indices: []``.
3. SUMMARIZE what the FULL set of raw matches (not just the kept ones) collectively indicates about the query image's likely subject in a concise 3-5 sentence prose paragraph in the ``summary`` field. Capture specific entity names, dates, attributes verbatim. The summary is the agent's only window into the rank-{K+1}+ matches that did NOT survive the top-K cut, so prioritize information the kept matches alone would NOT surface.

Output STRICT JSON, one object, no prose around it. Schema:

{
  "kept_indices": [<1-based int>, <1-based int>, ...],
  "summary": "<3-5 sentence prose distilling what ALL raw matches indicate about the query image>",
  "no_results_justification": null,
  "suggested_next_actions": []
}

If NO match is useful (all show unrelated subjects / low-confidence / image-text mismatch), emit:

{
  "kept_indices": [],
  "summary": null,
  "no_results_justification": "Brief per-raw-match rejection rationale (mention which signal — query-image mismatch / title-image mismatch / unrelated text — drove the rejection).",
  "suggested_next_actions": [
    "<concrete next action — see allowed options below>",
    ...
  ]
}

""" + _IMAGE_SEARCH_SUMMARIZE_TAIL_RULES


IMAGE_SEARCH_MULTIMODAL_USER_PROMPT_HEADER = (
    "QUERY image (the one to identify) is shown first below, "
    "followed by {n} candidate visual matches.\n"
    "Each candidate has its text metadata followed by its image."
)

IMAGE_SEARCH_MULTIMODAL_USER_PROMPT_FOOTER = (
    "Rerank the {n} candidates above by visual-identity relevance to the QUERY image. "
    "Use BOTH the candidate's image AND its text. "
    "Pick AT MOST {keep_k} (K = {keep_k}). Emit their 1-based original ranks as ``kept_indices`` "
    "(most relevant first). Then write the 3-5 sentence ``summary`` per the schema. "
    "Output the strict JSON object specified in the system message — no prose around it."
)


# ----------------------------------------------------------------------
# Summarize-pass helpers (mirrors serper_text_search/tool.py)
# ----------------------------------------------------------------------

def _parse_summarize_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of summarizer LLM JSON: strict → strip fences → balanced ``{...}``."""
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl >= 0:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end < 0:
        return None
    try:
        obj = json.loads(s[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_visual_match_entries(raw_text: str) -> Dict[int, str]:
    """Parse image_search raw_text into ``{1-based rank: verbatim chunk}``.

    Image_search raw output is a flat numbered list of visual matches
    (no Knowledge Graph / Answer Box / Top Stories blocks). Each entry
    starts with ``N. `` and continues until the next ``N. `` or end of
    text. The leading ``[Image Search — Visual Matches]`` header is
    added by ``Image_Search_Tool.execute()`` AFTER summarization, so it
    is not present in the ``raw_text`` we receive here.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return {}
    entries: Dict[int, str] = {}
    pattern = re.compile(r"(?ms)^(\d+)\.\s+.*?(?=^\d+\.\s|\Z)")
    for match in pattern.finditer(raw_text):
        rank = int(match.group(1))
        chunk = match.group(0).rstrip()
        if chunk:
            entries[rank] = chunk
    return entries


def _scrub_summary_text(summary: str) -> str:
    """Defensive sanitation on the LLM-emitted ``summary`` field.

    Mirrors ``serper_text_search/tool.py::_scrub_summary_text``: strip
    URLs / ``[N]`` markers / ``webpage_url_N`` / ``tool_generated_image_N``
    tokens so a poorly-behaved LLM cannot leak URLs into the summary
    region (which the memory module's URL rewriter would otherwise
    register against the URL registry with no anchoring raw entry).
    """
    if not isinstance(summary, str):
        return ""
    s = summary
    s = re.sub(r"\[webpage_url_\d+\]", "", s)
    s = re.sub(r"\[tool_generated_image_\d+\]", "", s)
    s = re.sub(r"\bwebpage_url_\d+\b", "", s)
    s = re.sub(r"\btool_generated_image_\d+\b", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


# ----------------------------------------------------------------------
# Multimodal-summarize helpers (kept-rank → fetched image data URL)
# ----------------------------------------------------------------------

# Placeholder token inserted in place of ``Image: <http_url> <img>`` for kept
# ranks whose thumbnail was successfully fetched. The custom
# ``process_tool_output`` substitutes each placeholder with the framework-assigned
# ``[tool_generated_image_N]`` ref (matching URL rewriter bracket convention) so
# the agent has an explicit per-entry handle to call follow-up tools against the
# specific image bytes that were just rendered.
MULTIMODAL_IMAGE_REF_PLACEHOLDER = "__MULTIMODAL_IMG_REF_PLACEHOLDER__"


def _replace_image_url_line_with_placeholder(entry_text: str) -> str:
    """Replace image URL lines with placeholders while preserving indentation.

    The capability registers returned image bytes and substitutes their memory
    IDs into these placeholders in selected-rank order.
    """
    lines = entry_text.splitlines()
    out: List[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("Image: "):
            indent = ln[: len(ln) - len(stripped)]
            out.append(f"{indent}Image: {MULTIMODAL_IMAGE_REF_PLACEHOLDER}")
        else:
            out.append(ln)
    return "\n".join(out)


def _select_kept_data_urls(
    kept_indices: List[int],
    image_data_urls_by_rank: Dict[int, Optional[str]],
) -> List[str]:
    """Slice the data URLs for kept ranks in kept-rank order. Skip ranks
    whose fetch failed (value is None / missing). Order is preserved so
    the agent can read "rank-1 of kept = first image attached" etc.
    """
    out: List[str] = []
    for rank in kept_indices:
        url = image_data_urls_by_rank.get(rank)
        if isinstance(url, str) and url.startswith("data:image/"):
            out.append(url)
    return out


def _render_summarized_results(
    parsed: Dict[str, Any], *, raw_text: str, keep_k: int,
    image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None,
) -> str:
    """Render selected source entries and a cleaned summary from model JSON.

    Args:
        parsed: Summarizer output with kept_indices and prose fields.
        raw_text: Numbered source entries to select by one-based rank.
        keep_k: Maximum number of selected entries to render.
        image_data_urls_by_rank: Optional fetched images; available selected
            images use placeholders that the capability replaces with IDs.

    Returns:
        Selected entries with source links preserved, a summary with URL and
        reference tokens removed, and optional recovery suggestions. Empty
        selection produces an explicit no-relevant-results explanation.
    """
    kept_raw = parsed.get("kept_indices") or []
    if not isinstance(kept_raw, list):
        kept_raw = []
    kept_indices: List[int] = []
    seen: set = set()
    for v in kept_raw:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i <= 0 or i in seen:
            continue
        seen.add(i)
        kept_indices.append(i)
    kept_indices = kept_indices[:keep_k]

    suggested = parsed.get("suggested_next_actions") or []
    if not isinstance(suggested, list):
        suggested = []

    # Branch 1: empty
    if not kept_indices:
        justification = str(parsed.get("no_results_justification") or "").strip()
        lines = ["NO RELEVANT MATCHES"]
        if justification:
            lines.append("")
            lines.append("Why each raw match was rejected:")
            lines.append(justification)
        if suggested:
            lines.append("")
            lines.append("Suggested next actions:")
            for s in suggested:
                lines.append(f"- {s}")
        return "\n".join(lines)

    # Branch 2: non-empty — slice verbatim by index
    entries = _extract_visual_match_entries(raw_text)
    body_lines: List[str] = []
    new_rank = 0
    for orig_rank in kept_indices:
        entry_text = entries.get(orig_rank)
        if not entry_text:
            continue
        new_rank += 1
        renumbered = re.sub(r"^\d+\.", f"{new_rank}.", entry_text, count=1)
        # Multimodal: when this kept rank has a fetched image, strip its
        # ``Image: <url> <img>`` line — the image bytes are attached to
        # the agent message separately, and keeping the http URL line
        # would lazy-register a second ``tool_generated_image_N`` ref
        # pointing to the same picture.
        if image_data_urls_by_rank is not None:
            data_url = image_data_urls_by_rank.get(orig_rank)
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                renumbered = _replace_image_url_line_with_placeholder(renumbered)
        body_lines.append(renumbered)

    if not body_lines:
        lines = ["NO RELEVANT MATCHES (all kept_indices were out of range)"]
        if suggested:
            lines.append("")
            lines.append("Suggested next actions:")
            for s in suggested:
                lines.append(f"- {s}")
        return "\n".join(lines)

    out_lines: List[str] = list(body_lines)

    summary_text = _scrub_summary_text(parsed.get("summary") or "")
    if summary_text:
        out_lines.append("")
        out_lines.append("[Summary]")
        out_lines.append(summary_text)

    if suggested:
        out_lines.append("")
        out_lines.append("Suggested next actions:")
        for s in suggested:
            out_lines.append(f"- {s}")

    return "\n".join(out_lines)

TOOL_NAME = "Image_Search_Tool"
DEFAULT_NUM_RESULTS = 10

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {
            "type": "string",
            "description": (
                "Reference ID of the image to search, drawn from the Environment "
                "Update's 'Currently available image IDs' list. Typically "
                "``input_image_N`` (user-provided) or ``tool_generated_image_N`` "
                "(produced by a prior tool call). The framework resolves the ID "
                "to image bytes before dispatch — do NOT pass raw URLs, file "
                "paths, or base64 data here."
            ),
        },
    },
    "required": ["image_id"],
    "additionalProperties": False,
}


class Image_Search_Tool(BaseTool):
    """Upload query images, retrieve Lens matches and summarize visual evidence.

    Uses the configured auxiliary model with fetched query/candidate pixels,
    retaining up to three selected matches from ten retrieved results.
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
                "Identify objects, landmarks, artwork, text, products, plants, "
                "animals, and more in an image using Google Lens reverse image "
                "search. Returns a list of visual matches with titles, source URLs, "
                "links, and thumbnails.\n\n"
                "Use this tool when you need to understand what is depicted in an "
                "image — it can recognize specific places, products, artworks, "
                "logos, plants, animals, celebrities, etc. Pass the target image "
                "via its ``image_id`` (``input_image_N`` or "
                "``tool_generated_image_N``) — the framework resolves the ID to "
                "image bytes before dispatch.\n\n"
                "This tool performs REVERSE image search (image → information). "
                "To search for images by text description, use Text_Search_Tool "
                "with search_type='images' instead.\n\n"
                "Search ONE image per call. To reverse-search several images, "
                "invoke this tool once per image — results from successive calls "
                "remain visible in your context, so you can correlate them."
            ),
            tool_version="1.0.0",
            input_types={
                "image_id": "str - image reference ID (input_image_N or tool_generated_image_N)",
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["reverse_image_search", "visual_search"],
            output_type=(
                "list[dict] - list of visual matches, each with keys:\n"
                "  'title' (str): description or name of the matched content\n"
                "  'link' (str): URL to the source page\n"
                "  'source' (str): domain name of the source\n"
                "  'thumbnail' (str): URL to a thumbnail image\n"
                "  'position' (int): rank position in results"
            ),
            demo_commands=[
                {
                    "command": (
                        'Image_Search_Tool(image_id="input_image_1")'
                    ),
                    "description": (
                        "Reverse-search the user-provided input image. "
                        "Returns matching subjects, places, products, etc."
                    ),
                },
                {
                    "command": (
                        'Image_Search_Tool(image_id="tool_generated_image_3")'
                    ),
                    "description": (
                        "Reverse-search a sub-image produced by a prior tool "
                        "call (e.g., Zoom_In or Visual_Grounding crop)."
                    ),
                },
            ],
            user_metadata={
                "api_key_env": "SERPER_API_KEY",
                "endpoint": "https://google.serper.dev/lens",
                "default_num_results": DEFAULT_NUM_RESULTS,
                "notes": (
                    "Uses Serper Google Lens API. The framework resolves "
                    "``image_id`` to image bytes before dispatch; the tool "
                    "then auto-uploads the bytes to cloud storage to obtain a "
                    "public URL for Serper Lens via image_upload_utils."
                ),
            },
        )
        self.llm_engine = llm_engine
        # Official paper algorithmic constants; model connection is injected.
        self._llm_summarize = True
        self._llm_max_tokens = 4096
        self._num_results = 10
        self._keep_k = 3
        self._multimodal_summarize = True

        self._client: Optional[SerperClient] = None

    def _get_client(self) -> SerperClient:
        """Return the lazily created Serper HTTP client."""
        if self._client is None:
            self._client = SerperClient()
        return self._client

    # ------------------------------------------------------------------
    # Structured error / empty-result rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _image_recovery_options() -> str:
        """High-level recovery hints for hard-error paths (Serper Lens HTTP
        failure / 0 visual matches / input resolution failure).

        Kept deliberately abstract — no tool names, no arg signatures — so
        the menu does not drift when tool surface area changes. The agent
        has the live tool registry and can resolve each hint to a concrete
        call. Hard-error paths are also rare, so a high-fidelity menu is
        not worth the maintenance cost.

        The normal summarize path does NOT use this — there the
        summarizer LLM emits ``suggested_next_actions`` in its JSON, with
        access to the actual tool registry semantics.
        """
        return (
            "  - Retry a visual lookup with a different input image, or a tighter crop of "
            "the salient subject.\n"
            "  - Pivot to text-based search using descriptive keywords for the visual "
            "content.\n"
            "  - Directly retrieve a specific source URL if any prior tool output "
            "referenced one.\n"
            "  - For text-heavy images (slides, signage, posters), extract the text first "
            "and search via text rather than visually.\n"
        )

    @classmethod
    def _format_lens_error(
        cls,
        image_ref: str,
        error: "SerperRequestError",
    ) -> str:
        """Render a Serper Lens failure as a structured agent-readable message.

        Shape mirrors Text_Search_Tool._format_serper_error so the agent
        sees consistent recovery guidance across search tools. Includes:

        * Why it failed (transient vs permanent + HTTP status + attempts)
        * A targeted retry hint (or no-retry hint, by failure mode)
        * The shared recovery-options menu — see ``_image_recovery_options``
        """
        if error.account_level:
            return (
                "[Image Search Error]\n"
                "REVERSE IMAGE SEARCH UNAVAILABLE\n\n"
                f"Reason: the search provider rejected the request because of our search "
                f"account (HTTP {error.status_code}: credits exhausted, key invalid or payment "
                "required), NOT because of the image. Every reverse image search, text search "
                "and webpage visit will fail the same way until an operator fixes the account.\n"
                f"Image: {image_ref}\n\n"
                "What to do: do NOT retry, crop, zoom or try another image_id or search tool; "
                "none of that can help. Continue without web search: answer from the image and "
                "your own knowledge, and state which facts you could not verify.\n"
            )
        if error.transient:
            cause = (
                f"Serper Lens (reverse image search) temporarily unavailable "
                f"(HTTP {error.status_code}, transient) after {error.attempts} attempt(s). "
                f"The Serper backend is briefly down or rate-limiting; this is NOT a "
                f"problem with the image."
            )
            retry_hint = (
                "  - Retry the SAME image_id shortly (after ~30-60s). Serper outages\n"
                "      typically resolve within a minute.\n\n"
            )
        elif error.status_code in (401, 403):
            cause = (
                f"Serper authentication / authorization rejected (HTTP {error.status_code}, "
                f"permanent). The API key is missing, invalid, or quota-exhausted."
            )
            retry_hint = (
                "  - Retrying with the same image_id will NOT help — this is a permanent\n"
                "      auth failure that needs operator intervention. Try a different\n"
                "      recovery option below.\n\n"
            )
        elif error.status_code == 400:
            cause = (
                f"Serper Lens rejected the image (HTTP 400, permanent). The image URL "
                f"is unreachable, the format is unsupported, or the image is too small "
                f"/ too large for Google Lens. Detail: {error}"
            )
            retry_hint = (
                "  - Retrying the same image_id will NOT help. Choose a different recovery\n"
                "      option below — most commonly a different image_id (a higher-resolution\n"
                "      version of the same content, or a Zoom_In / Visual_Grounding crop).\n\n"
            )
        else:
            cause = (
                f"Serper Lens request failed (HTTP {error.status_code or '?'}, permanent) "
                f"after {error.attempts} attempt(s). Detail: {error}"
            )
            retry_hint = (
                "  - Retrying with the EXACT same image_id is unlikely to help. Pick a\n"
                "      different recovery option below.\n\n"
            )
        return (
            f"[Image Search Error]\n"
            f"REVERSE IMAGE SEARCH FAILED\n\n"
            f"Reason: {cause}\n"
            f"Image: {image_ref}\n\n"
            "Suggested next actions for the agent (each bullet below is an INDEPENDENT\n"
            "alternative — pick whichever fits, no need to chain them):\n\n"
            f"{retry_hint}"
            f"{cls._image_recovery_options()}"
        )

    @classmethod
    def _format_no_matches(cls, image_ref: str, attempts: int = 1) -> str:
        """Render an empty Lens result with explanations and image-search recovery options.

        ``attempts`` is how many times the identical Lens request was already
        issued (see LENS_EMPTY_MAX_ATTEMPTS). The message must not claim that
        retrying is pointless; an empty reply is often transient.
        """
        if attempts > 1:
            retried = (
                f" The identical request was issued {attempts} times "
                f"(Serper intermittently drops result sets), and every attempt came "
                f"back empty, so a transient drop is now unlikely to be the cause."
            )
        else:
            retried = ""
        return (
            "[Image Search — Visual Matches]\n"
            "NO VISUAL MATCHES\n\n"
            f"Reason: Serper Lens returned 0 reverse-image matches for {image_ref}. "
            "This usually means the image content is generic / non-distinctive (a blank "
            "wall, a common object photographed without context), the image resolution is "
            "too low for Google Lens to anchor, or the subject is too obscure for the "
            f"Lens index.{retried}\n\n"
            "Suggested next actions for the agent (each bullet below is an INDEPENDENT\n"
            "alternative — pick whichever fits, no need to chain them):\n\n"
            f"{cls._image_recovery_options()}"
        )

    @staticmethod
    def _published_image_problem(url: str) -> Optional[str]:
        """Fetch the published query image the way Serper does; None when it is served."""
        content_hash = content_hash_from_url(url)
        if content_hash is None:
            return None
        problem = public_url_problem(url, content_hash)
        if problem is None:
            return None
        degradation.record("image_host", tool=TOOL_NAME, reason="unreachable")
        forget_verified_base_url(url)
        logger.error(
            "IMAGE_HOST_UNREACHABLE tool=%s url=%s -- %s. Serper Lens returned no matches "
            "because it cannot fetch the query image.", TOOL_NAME, url, problem,
        )
        return problem

    @classmethod
    def _format_image_unreachable(cls, image_ref: str, detail: str) -> str:
        """Structured message when the published query image is not reachable online."""
        return (
            "[Image Search Error]\n"
            "QUERY IMAGE NOT REACHABLE\n\n"
            "Reason: the reverse image search service could not fetch the query image "
            f"from where it was published ({detail}). This is a problem with the image "
            f"hosting on our side, NOT with the image itself.\n"
            f"Image: {image_ref}\n\n"
            "Suggested next actions for the agent (each bullet below is an INDEPENDENT\n"
            "alternative — pick whichever fits, no need to chain them):\n\n"
            "  - Retrying the same image_id will NOT help until the hosting recovers.\n"
            "      Prefer Text_Search with a precise description of the image for now.\n\n"
            f"{cls._image_recovery_options()}"
        )

    @classmethod
    def _format_input_error(cls, reason: str, detail: str = "") -> str:
        """Structured message for input-side failures (upload service down,
        local file missing, invalid image bytes, etc.).

        These are NOT Serper failures — the call never reached Serper. We
        still emit a recovery-guided message instead of letting a raw
        exception propagate to the agent.
        """
        detail_line = f"\nDetail: {detail}" if detail else ""
        return (
            "[Image Search Error]\n"
            "INPUT FAILED\n\n"
            f"Reason: {reason}{detail_line}\n\n"
            "Suggested next actions for the agent (each bullet below is an INDEPENDENT\n"
            "alternative — pick whichever fits, no need to chain them):\n\n"
            f"{cls._image_recovery_options()}"
        )

    def _resolve_image_url(self, image_base64: str) -> str:
        """Decode the resolved inline image and upload it to configured cloud storage.

        Returns a signed HTTP(S) URL suitable for Serper Lens. The capability supplies
        image_base64 after resolving the model's image ID; this helper does not resolve
        IDs itself. Missing data raises ValueError; upload failures propagate.
        """
        from openvlharness.services.tools.image_upload_utils import upload_image_bytes

        if not image_base64:
            raise ValueError("image_id resolved to empty image_base64")

        image_bytes, content_type = decode_base64_image_data(image_base64)
        if content_type:
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/bmp": ".bmp",
                "image/tiff": ".tiff",
            }
            ext = ext_map.get(content_type.lower())
            if not ext:
                content_type, ext = infer_image_content_type_and_extension(image_bytes)
        else:
            content_type, ext = infer_image_content_type_and_extension(image_bytes)
        return upload_image_bytes(image_bytes, content_type, ext)

    @staticmethod
    def _extract_item(item_data: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Extract a visual match from a raw result item."""
        # Serper /lens returns ``imageUrl`` (source) + ``thumbnailUrl``
        # (gstatic CDN preview); the field name ``thumbnail`` does not
        # exist. Iterate per-candidate so that a data: URL in imageUrl
        # does not block falling through to a usable http thumbnailUrl.
        title = item_data.get("title")
        link = item_data.get("link")
        if not title and not link:
            return None
        thumbnail = ""
        for key in ("imageUrl", "thumbnailUrl"):
            cand = str(item_data.get(key) or "").strip()
            if cand.lower().startswith(("http://", "https://")):
                thumbnail = cand
                break
        return {
            "title": title or "",
            "link": link or "",
            "source": item_data.get("source", ""),
            "thumbnail": thumbnail,
            "position": item_data.get("position", ""),
        }

    def _format_raw_results(self, results: List[Dict[str, object]]) -> str:
        """Render numbered visual matches with source links and optional thumbnail URLs.

        Compact source labels preserve URL paths as semantic context. Image links carry
        an <img> marker so memory rewriting recognizes them without a filename suffix.
        """
        MAX_PATH_LEN = 80
        lines = []
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. {item.get('title', '')}")
            link = item.get("link")
            thumbnail = str(item.get("thumbnail") or "").strip()
            existing_source = str(item.get("source") or "").strip()
            if link:
                try:
                    parsed = urlparse(str(link))
                    host = (parsed.netloc or "").lower().removeprefix("www.")
                    path = parsed.path or ""
                except Exception:
                    host = ""
                    path = ""
                compact = (host + path).rstrip("/")
                if compact and len(compact) > MAX_PATH_LEN:
                    tail_seg = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
                    compact = (
                        f"{host}/.../{tail_seg}"
                        if tail_seg and (len(host) + 4 + len(tail_seg)) <= MAX_PATH_LEN + 10
                        else compact[: MAX_PATH_LEN - 3] + "..."
                    )
                display = compact or existing_source
                if display:
                    lines.append(f"   Source: {display}   {link}")
                else:
                    lines.append(f"   URL: {link}")
            elif existing_source:
                lines.append(f"   Source: {existing_source}")
            if thumbnail:
                lines.append(f"   Image: {thumbnail} <img>")
        return "\n".join(lines)

    def _build_multimodal_user_prompt(
        self,
        raw_text: str,
        *,
        query_image_data_url: Optional[str],
        image_data_urls_by_rank: Dict[int, Optional[str]],
    ) -> List[Dict[str, Any]]:
        """Build the interleaved text+image content list for the multimodal
        summarizer call. Layout:

            [text: "QUERY image (the one to identify)..."]
            [image: <query image>]                           (if available)
            [text: "Candidate 1 metadata:"]
            [image: <candidate 1's thumbnail>]               (if fetched)
            [text: "Candidate 2 metadata:\n[image unavailable]"]
            ...
            [text: footer with K and instructions]
        """
        entries = _extract_visual_match_entries(raw_text)
        n = len(entries)
        parts: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": IMAGE_SEARCH_MULTIMODAL_USER_PROMPT_HEADER.format(n=n),
            },
        ]
        if isinstance(query_image_data_url, str) and query_image_data_url.startswith("data:image/"):
            parts.append({
                "type": "image_url",
                "image_url": {"url": query_image_data_url},
            })
        else:
            parts.append({
                "type": "text",
                "text": "[QUERY image unavailable — fetch failed; rerank using candidate text + candidate images only.]",
            })

        for rank in sorted(entries.keys()):
            data_url = image_data_urls_by_rank.get(rank)
            entry_text = entries[rank]
            parts.append({"type": "text", "text": entry_text})
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                # Defensive: pre-filter in ``execute`` removes fetch-failed
                # candidates before this builder runs, so this branch is
                # not expected to fire. Kept as a safety net.
                parts.append({"type": "text", "text": "[image unavailable]"})

        parts.append({
            "type": "text",
            "text": IMAGE_SEARCH_MULTIMODAL_USER_PROMPT_FOOTER.format(
                n=n, keep_k=self._keep_k,
            ),
        })
        return parts

    def _summarize_with_llm(
        self,
        raw_text: str,
        *,
        query_image_data_url: Optional[str] = None,
        image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None,
    ) -> Tuple[str, List[str]]:
        """Rank raw visual matches using the auxiliary model and render selected evidence.

        Available query/candidate images are interleaved with text for multimodal
        ranking. The model chooses kept_indices; the renderer selects source entries
        from raw_text and strips reference tokens from the prose summary.

        Returns:
            A (rendered_text, kept_data_urls) tuple in selected-rank order. Model,
            parsing or empty-output failures return (raw_text, []).
        """
        multimodal = image_data_urls_by_rank is not None
        system_prompt = (
            IMAGE_SEARCH_MULTIMODAL_SYSTEM_PROMPT if multimodal
            else IMAGE_SEARCH_SUMMARIZE_SYSTEM_PROMPT
        )
        if multimodal:
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": self._build_multimodal_user_prompt(
                        raw_text,
                        query_image_data_url=query_image_data_url,
                        image_data_urls_by_rank=image_data_urls_by_rank or {},
                    ),
                },
            ]
            generate_arg: Any = messages
        else:
            generate_arg = IMAGE_SEARCH_SUMMARIZE_USER_PROMPT.format(
                raw_results=raw_text, keep_k=self._keep_k,
            )

        try:
            result = self.llm_engine.generate(
                generate_arg,
                system_prompt=system_prompt,
                max_tokens=self._llm_max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception as exc:
            # SUMMARIZE_DEGRADED is a stable log token for watchdogs; keep it literal.
            logger.warning(
                "SUMMARIZE_DEGRADED tool=%s reason=%s -- LLM summarize call failed, "
                "returning raw results (%s)", TOOL_NAME, type(exc).__name__, exc,
            )
            degradation.record("summarize", tool=TOOL_NAME, reason=type(exc).__name__)
            return raw_text, []

        stripped = strip_think_tags(result or "")
        if not stripped.strip():
            logger.warning(
                "SUMMARIZE_DEGRADED tool=%s reason=empty_output -- returning raw results", TOOL_NAME,
            )
            degradation.record("summarize", tool=TOOL_NAME, reason="empty_output")
            return raw_text, []

        parsed = _parse_summarize_json(stripped)
        if parsed is None:
            logger.warning(
                "SUMMARIZE_DEGRADED tool=%s reason=unparseable_json -- returning raw results. "
                "Head: %r", TOOL_NAME, stripped[:200],
            )
            degradation.record("summarize", tool=TOOL_NAME, reason="unparseable_json")
            return raw_text, []

        # Schema deviation guard (matches serper_text_search). Two
        # modes that would otherwise be silently swallowed:
        #   (a) "kept_indices" key missing
        #   (b) "kept_indices" present but null / wrong type
        # Empty list [] is preserved (legitimate "no relevant matches").
        kept_value = parsed.get("kept_indices")
        if not isinstance(kept_value, list):
            logger.warning(
                "SUMMARIZE_DEGRADED tool=%s reason=kept_indices_invalid -- LLM summarize JSON: "
                "'kept_indices' missing or not a list "
                "(top-level keys: %s); returning raw results to avoid silent data loss",
                TOOL_NAME, list(parsed.keys()),
            )
            degradation.record("summarize", tool=TOOL_NAME, reason="kept_indices_invalid")
            return raw_text, []

        rendered = _render_summarized_results(
            parsed,
            raw_text=raw_text,
            keep_k=self._keep_k,
            image_data_urls_by_rank=image_data_urls_by_rank,
        )

        kept_data_urls: List[str] = []
        if image_data_urls_by_rank is not None:
            # Re-apply the renderer's normalization on kept_indices (positive,
            # deduped, cap at keep_k, preserve order) so kept_data_urls order
            # matches the order entries appear in ``rendered``.
            seen: set = set()
            normalized: List[int] = []
            for v in kept_value:
                try:
                    i = int(v)
                except (TypeError, ValueError):
                    continue
                if i <= 0 or i in seen:
                    continue
                seen.add(i)
                normalized.append(i)
            normalized = normalized[: self._keep_k]
            kept_data_urls = _select_kept_data_urls(
                normalized, image_data_urls_by_rank,
            )

        return rendered, kept_data_urls

    def execute(
        self,
        image_base64: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Upload an image, search Lens and summarize the selected visual matches.

        Args:
            image_base64: Inline image bytes or an image data URL. The capability
                resolves the model's image_id into this field before dispatch.

        Returns:
            Rendered evidence text, or a mapping with text and kept_image_data_urls
            when selected candidate pixels are available. Upload/search failures and
            empty matches produce explanatory text with recovery suggestions.

        Raises:
            ValueError: Image data is missing or invalid.
            FileNotFoundError: No Serper credential is configured.

        Uploads use configured GCS or S3 credentials. The signed query URL and candidate
        thumbnails are downloaded for multimodal summarization. Unavailable candidate
        images are filtered; if all fail, summarization uses text metadata.
        """
        # Catch input-side failures (S3 / cloud upload down, base64 corrupt)
        # and emit a structured agent-readable message. ValueError on a
        # missing / corrupt base64 payload is let through — those are
        # framework-level errors (memory module failed to resolve image_id),
        # not recoverable agent-arg errors.
        image_ref_for_error = "<image_base64>" if image_base64 else "<unspecified>"
        try:
            resolved_url = self._resolve_image_url(image_base64)
        except ValueError:
            raise
        except ImageHostUnreachableError as exc:
            return self._format_image_unreachable(image_ref_for_error, str(exc))
        except Exception as exc:
            logger.warning(
                "Image_Search_Tool: input resolution failed (%s): %s",
                type(exc).__name__, exc,
            )
            return self._format_input_error(
                reason="Image upload / resolution service failed before the search could run.",
                detail=f"{type(exc).__name__}: {exc}",
            )

        # --- Call Serper Lens (with retry inside SerperClient; structured
        # error on final failure) ---
        # The retry condition is ``not results`` (not ``not raw_items``) so it
        # fires on exactly the states that render NO VISUAL MATCHES. Transport
        # errors are left to SerperClient; this loop must not double-retry them.
        client = self._get_client()
        results: List[Dict[str, object]] = []
        lens_attempts = 0

        for attempt in range(1, LENS_EMPTY_MAX_ATTEMPTS + 1):
            lens_attempts = attempt
            try:
                payload = client.lens(resolved_url)
            except SerperRequestError as exc:
                logger.warning(
                    "Image_Search_Tool: Serper Lens failed (transient=%s, status=%s, attempts=%d): %s",
                    exc.transient, exc.status_code, exc.attempts, exc,
                )
                return self._format_lens_error(image_ref_for_error, exc)

            raw_items: List[Dict[str, object]] = payload.get("organic", []) or []

            results = []
            seen = set()
            for item_data in raw_items:
                if not isinstance(item_data, dict):
                    continue
                match = self._extract_item(item_data)
                if match is None:
                    continue
                dedup_key = match["link"] or match["title"]
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                results.append(match)
                if len(results) >= self._num_results:
                    break

            if results:
                if attempt > 1:
                    logger.info(
                        "Image_Search_Tool: Serper Lens returned %d matches on attempt %d/%d "
                        "after %d empty response(s) for the same URL.",
                        len(results), attempt, LENS_EMPTY_MAX_ATTEMPTS, attempt - 1,
                    )
                break

            if attempt < LENS_EMPTY_MAX_ATTEMPTS:
                backoff = min(
                    LENS_EMPTY_BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
                    LENS_EMPTY_BACKOFF_MAX_SEC,
                )
                logger.warning(
                    "Image_Search_Tool: Serper Lens returned 0 matches (HTTP 200) on "
                    "attempt %d/%d; retrying the same URL in %.1fs.",
                    attempt, LENS_EMPTY_MAX_ATTEMPTS, backoff,
                )
                time.sleep(backoff)

        if not results:
            # Serper answers an empty 200 both for an image without matches and for
            # an image it could not fetch, so check the public URL before reporting.
            problem = self._published_image_problem(resolved_url)
            if problem is not None:
                return self._format_image_unreachable(image_ref_for_error, problem)
            # Render an explicit no-matches result with recovery suggestions.
            # Includes cross-tool suggestions so the agent has a recovery path.
            return self._format_no_matches(image_ref_for_error, attempts=lens_attempts)

        raw_text = self._format_raw_results(results)

        if self._llm_summarize:
            # Multimodal path: pre-fetch the QUERY image AND the candidate
            # thumbnails so the summarizer LLM (Qwen3-VL) can rerank using
            # both modalities, and so kept-rank bytes can be injected into
            # the agent's user message as real pixels (not lazy http refs).
            #
            # Fetch-failed candidates are pre-filtered BEFORE the rerank,
            # so the summarizer only judges candidates whose image bytes
            # are actually present. If ALL candidate fetches fail, we
            # fall back to a text-only summarize over the metadata, with
            # a banner telling the agent no image bytes were retrieved.
            query_data_url: Optional[str] = None
            image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None
            all_fetches_failed = False
            if self._multimodal_summarize:
                candidate_urls: List[str] = []
                for item in results:
                    thumb = str(item.get("thumbnail") or "").strip()
                    if thumb and thumb.lower().startswith(("http://", "https://")):
                        candidate_urls.append(thumb)
                fetched = fetch_images_parallel(
                    [resolved_url] + candidate_urls,
                )
                query_data_url = fetched.get(resolved_url)
                # Pre-filter: keep only candidates whose thumbnail fetched
                # successfully. The summarizer's whole job is visual
                # similarity, so a candidate with no image bytes is not
                # judgeable — including it forces a fake text-only fallback
                # that pollutes the rerank.
                fetched_pairs: List[Tuple[Dict[str, object], str]] = []
                for item in results:
                    thumb = str(item.get("thumbnail") or "").strip()
                    data_url = fetched.get(thumb) if thumb else None
                    if isinstance(data_url, str) and data_url.startswith("data:image/"):
                        fetched_pairs.append((item, data_url))

                if not fetched_pairs:
                    # All-fail path: no candidate has bytes. Fall back to
                    # text-only summarize over metadata; strip ``Image: <url> <img>``
                    # lines from raw_text so the framework's URL rewriter does
                    # not lazy-register tool_generated_image_N refs pointing
                    # at URLs we already know are fetch-failure-prone.
                    all_fetches_failed = True
                    results_no_images = [
                        {**item, "thumbnail": ""} for item in results
                    ]
                    raw_text = self._format_raw_results(results_no_images)
                    image_data_urls_by_rank = None
                else:
                    # Partial-fail or full-success: rebuild raw_text and the
                    # rank-to-data-url map from the filtered candidates only.
                    # Ranks are renumbered to 1..N_fetched so the summarizer's
                    # ``kept_indices`` land back on the right entries.
                    filtered_results = [item for item, _ in fetched_pairs]
                    raw_text = self._format_raw_results(filtered_results)
                    image_data_urls_by_rank = {
                        rank: data_url
                        for rank, (_, data_url) in enumerate(fetched_pairs, 1)
                    }

            summary, kept_data_urls = self._summarize_with_llm(
                raw_text,
                query_image_data_url=query_data_url,
                image_data_urls_by_rank=image_data_urls_by_rank,
            )
            if all_fetches_failed:
                banner = (
                    "Image fetch failed for ALL candidates. The candidate metadata "
                    "below (titles and source URLs) is available but no image bytes "
                    "were retrieved.\n"
                )
                text_payload = f"[Image Search — Visual Matches]\n{banner}\n{summary}"
            else:
                text_payload = f"[Image Search — Visual Matches]\n{summary}"

            # When any kept-rank image was successfully fetched, return a
            # dict result. The framework's _extract_images_and_scrub_result
            # walks the dict, registers each ``data:image/...`` string as
            # ``tool_generated_image_N``, and the default process_tool_output
            # builder attaches the bytes to the deferred role=user message.
            # In text-only mode (no fetched bytes) we keep the string
            # contract so default text rendering applies unchanged.
            if kept_data_urls:
                return {
                    "text": text_payload,
                    "kept_image_data_urls": kept_data_urls,
                }
            return text_payload

        return f"[Image Search — Visual Matches]\n{raw_text}"
