"""Search Serper endpoints and summarize evidence relevant to a text query.

Text_Search_Tool supports web, image, news, scholar, maps, places, shopping and
patent searches. The paper configuration uses Qwen multimodal summarization
for image results and preserves selected source entries and links.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from openvlharness.services.tools.base import BaseTool, ToolArgumentError
from openvlharness.services.tools.serper_common import (
    ALL_SEARCH_TYPES,
    SerperClient,
    SerperRequestError,
    fetch_images_parallel,
    format_search_results,
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

# --- LLM summarization prompt (rerank + index-based top-K + prose summary) ---
# Split into system (static role + rules + schema) and user (per-call data).
# The summarizer receives raw search results with raw
# URLs (https://...), reranks them, and outputs:
#   (1) ``kept_indices`` — a list of 1-based original ranks (NOT URLs, NOT
#       refs, NOT key_facts). The downstream renderer slices the top-K raw
#       entries verbatim from ``raw_results`` using these indices, so the
#       agent sees the ORIGINAL Serper snippet text + URL byte-for-byte.
#   (2) ``summary`` — a 3-5 sentence prose paragraph that distills the
#       query-relevant facts from ALL raw results (not just the kept). This
#       is the agent's only window into the rank-4..N results that were
#       culled, so it should prioritize facts the kept results would NOT
#       surface (e.g. dates/numbers/entities present in rank 5 but absent
#       from rank 1-3).
# The LLM NEVER outputs URLs or ref tokens — URL integrity is guaranteed
# by construction (no LLM hand touches the URLs). If filtering wipes out
# every result (kept_indices=[]), the LLM emits a structured "no relevant
# results" response with query rewrites + next-action hints.

TEXT_SEARCH_SUMMARIZE_SYSTEM_PROMPT = """You are a research summarizer. Given raw web search results and a user query, do exactly three things:

1. RERANK the numbered organic results by how directly each helps answer the query.
2. KEEP at most K most useful results (K is given in the user message). Emit their original 1-based ranks (the leading ``1.``, ``2.``, ... numbers) as ``kept_indices``. If fewer than K are useful, keep only those — do NOT pad to reach K. If NONE of the results help, emit ``kept_indices: []``.
3. SUMMARIZE the query-relevant facts from the FULL set of raw results (not just the kept ones) into a concise 3-5 sentence prose paragraph in the ``summary`` field. Include specific facts verbatim — numbers, dates, named entities, prices, addresses. The summary is the agent's only window into the raw results that did NOT survive the top-K cut, so do not waste it on generic restatement of the kept results; prioritize information that the kept results alone would NOT surface.

What is eligible for ``kept_indices``:
- ONLY the numbered organic results (``1.``, ``2.``, ... — the main list at the bottom of the raw input). Use their 1-based original ranks verbatim.
- ``[Knowledge Graph]``, ``[Answer Box]``, ``[Top Stories]`` blocks are NOT eligible for ``kept_indices`` (they have no organic rank). If they contain query-relevant facts, capture those facts in ``summary`` (entity name + description + key attributes for KG, snippet for Answer Box, headline + outlet + date for each Top Story). Do NOT silently drop these structured top-of-raw blocks — they often carry the most directly relevant content for entity / factual / current-events queries.

Output STRICT JSON, one object, no prose around it. Schema:

{
  "kept_indices": [<1-based int>, <1-based int>, ...],
  "summary": "<3-5 sentence prose distillation of the query-relevant facts across ALL raw results>",
  "no_results_justification": null,
  "suggested_queries": [],
  "suggested_next_actions": []
}

If NO result is useful enough to keep (all off-topic / low-quality / dead-ends), emit:

{
  "kept_indices": [],
  "summary": null,
  "no_results_justification": "Brief per-raw-result rejection rationale (e.g. 'result 1 was off-topic, result 2 was a near-duplicate, ...').",
  "suggested_queries": [
    "<alternative query 1> (rationale: <one short reason>)",
    "<alternative query 2> (rationale: ...)"
  ],
  "suggested_next_actions": [
    "<Tool_Name>(<key args>) — <one short reason>",
    ...
  ]
}

When ``kept_indices`` is non-empty, ``suggested_queries`` and ``suggested_next_actions`` are OPTIONAL. Emit them only if the kept results still leave the query under-answered (e.g. the answer is hinted at but not fully resolved). Otherwise leave them as empty lists.

Available next actions (the agent may also re-call THIS tool — Text_Search_Tool itself — with a different search_type or query; nothing is excluded just because it is the same tool that produced this result):
- Text_Search_Tool(query=<rewritten query>, search_type=<search|scholar|patents|news|images|maps|places|shopping>) — same tool, useful when the issue is wrong sub-mode (e.g. an academic paper question that should go to ``scholar`` instead of ``search``, or a news / current-events query that should go to ``news``, or a product / price lookup that should go to ``shopping``, or a venue / location lookup that should go to ``maps``/``places``).
- Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>) — reverse image search; useful when the query is fundamentally about identifying the subject of an image rather than text retrieval.
- Webpage_Visit_Tool(urls=[webpage_url_N], goal=<concrete goal>) — deep-read a specific URL the agent has already received in a prior tool result (URL must be a runtime ref, not a raw https:// string).

When writing entries in ``suggested_next_actions``, keep ``N`` as a literal placeholder when you refer to ``webpage_url_N`` / ``tool_generated_image_N`` / ``input_image_N`` — never substitute a specific number, since you do not see the agent's reference registry.

When emitting ``suggested_next_actions`` (whether ``kept_indices`` is empty or not), pick whichever subset of the three options above genuinely helps; do not artificially exclude this tool from its own suggestion list. For the empty-kept fallback, always include AT LEAST one concrete next action so the agent can recover without guessing.

Output rules:
- Output ONLY the JSON object. No markdown fences, no prefix, no suffix.
- ``kept_indices`` values MUST be valid 1-based ranks present in the raw input's numbered organic list. Out-of-range values are silently dropped; do not waste a slot on a hallucinated rank.
- ``summary`` is plain prose. Do NOT include ``[N]`` citation markers, do NOT include raw URLs (``https://...``), do NOT include ``webpage_url_N`` or ``tool_generated_image_N`` reference tokens. Preserve numbers / names / dates verbatim from the source results.
- ``summary`` MUST NOT reference results by their rank number (no "Result 7", "rank 3", "results 4-9", "the second result", etc.). The agent only sees the top-K survivors after your rerank — any rank-number reference is meaningless to it. **DO still synthesize information from ALL raw results including the non-kept ones** — that is the summary's primary purpose. Just attribute facts by *source / publisher / domain / title fragment* rather than rank position. Examples — ✗ "Result 7 from the Stanford lecture notes describes latent vectors"  →  ✓ "Stanford lecture notes describe latent vectors as foundational to 3D generation"; ✗ "results 3-9 mention the Forbidden City"  →  ✓ "Multiple sources (a Reddit thread, an arXiv paper, a Wikipedia entry) corroborate the Forbidden City reference". Information value > attribution style; if a low-rank result has a useful disambiguator, include it even if you can only name the source as "one source" / "a forum post" / "a 2025 paper".
- NEVER fabricate facts. If the raw results don't address some aspect of the query, omit it from ``summary`` rather than inventing.
"""

TEXT_SEARCH_SUMMARIZE_USER_PROMPT = """Query: {query}

Search Results (Type: {search_type}):
{raw_results}

Rerank the numbered organic results above by query relevance and pick AT MOST {keep_k} (i.e. K = {keep_k}) of them. Emit their 1-based original ranks as ``kept_indices`` (most relevant first). Then write a 3-5 sentence ``summary`` distilling the query-relevant facts from ALL raw results above — including Knowledge Graph / Answer Box / Top Stories if present, and including rank-{keep_k}+1 onwards results that did not make the top-K cut. Preserve numbers / dates / names verbatim; do NOT include URLs, [N] citation markers, or reference tokens. Output the strict JSON object specified in the system message."""


# --- Multimodal summarize prompt (search_type=images only) ---
# Used when ``search_type=images`` AND ``SEARCH_MULTIMODAL_SUMMARIZE=true``.
# The user prompt is built at runtime as interleaved text+image blocks so
# the summarizer (Qwen3-VL) can rerank using both the candidate text and
# its fetched thumbnail.
TEXT_SEARCH_IMAGES_MULTIMODAL_SYSTEM_PROMPT = """You are a research summarizer specialized in image search. The user query is a TEXT query that searches for images. You see NUMBERED candidate images; each candidate has its text metadata AND its own thumbnail image. Do exactly three things:

1. RERANK the numbered candidates by how well each one answers the text query. Combine THREE signals (in order of importance):
   (a) Image-query relevance — does the candidate image actually depict what the text query asks for? A candidate whose image clearly shows a different subject should be down-ranked even if its title looks promising.
   (b) Image-text alignment within the candidate itself — does the candidate's image match its own title / source? A title claiming "Eiffel Tower" with an image of a different tower is misleading and should be down-ranked.
   (c) Text relevance — title / source domain reliability / size if visible.
2. KEEP at most K most useful candidates (K is given in the user message). Emit their original 1-based ranks (the leading ``1.``, ``2.``, ... numbers) as ``kept_indices``. If fewer than K are useful, keep only those — do NOT pad to reach K. If NONE are useful, emit ``kept_indices: []``.
3. SUMMARIZE what the FULL set of raw candidates (not just the kept ones) collectively shows about the query — entities depicted, source coverage breadth, visual diversity, any caveat patterns (off-topic clusters, etc.) — in a concise 3-5 sentence prose paragraph in the ``summary`` field. Capture specific entity names, dates, attributes verbatim where they appear in titles / sources.

Output STRICT JSON, one object, no prose around it. Schema:

{
  "kept_indices": [<1-based int>, <1-based int>, ...],
  "summary": "<3-5 sentence prose>",
  "no_results_justification": null,
  "suggested_queries": [],
  "suggested_next_actions": []
}

If NO candidate is useful enough to keep (all off-query / image-text mismatch), emit:

{
  "kept_indices": [],
  "summary": null,
  "no_results_justification": "Brief per-raw-candidate rejection rationale (mention which signal — image-query mismatch / title-image mismatch / unrelated text — drove the rejection).",
  "suggested_queries": [
    "<alternative query 1> (rationale: <one short reason>)"
  ],
  "suggested_next_actions": [
    "<Tool_Name>(<key args>) — <one short reason>"
  ]
}

When ``kept_indices`` is non-empty, ``suggested_queries`` and ``suggested_next_actions`` are OPTIONAL.

Available next actions:
- Text_Search_Tool(query=<rewritten query>, search_type=<search|scholar|patents|news|images|maps|places|shopping>) — same tool, different sub-mode or rewritten query.
- Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>) — reverse image search, useful when the agent already has a reference image and wants to identify its subject.
- Webpage_Visit_Tool(urls=[webpage_url_N], goal=<concrete goal>) — deep-read a specific URL the agent has already received in a prior tool result.

When writing entries in ``suggested_next_actions``, keep ``N`` as a literal placeholder when you refer to ``webpage_url_N`` / ``tool_generated_image_N`` / ``input_image_N`` — never substitute a specific number, since you do not see the agent's reference registry.

Output rules:
- Output ONLY the JSON object. No markdown fences, no prefix, no suffix.
- ``kept_indices`` values MUST be valid 1-based ranks present in the raw input's numbered list. Out-of-range values are silently dropped.
- ``summary`` is plain prose. Do NOT include ``[N]`` citation markers, do NOT include raw URLs, do NOT include ``webpage_url_N`` or ``tool_generated_image_N`` reference tokens. Preserve names / numbers / dates verbatim.
- ``summary`` MUST NOT reference candidates by their rank number (no "Candidate 5", "rank 3", "match 2", "candidates 4-7", etc.). The agent only sees the top-K survivors after your rerank — any rank-number reference is meaningless to it. **DO still synthesize information from ALL candidates including the non-kept ones** — that is the summary's primary purpose. Just attribute facts by *entity / source / publisher* rather than rank position. Examples — ✗ "Candidate 5 (BBC Radio 3) shows Dialogues des Carmelites"  →  ✓ "A BBC Radio 3 broadcast page identifies the scene as Dialogues des Carmelites"; ✗ "candidates 3-7 confirm Eiffel Tower"  →  ✓ "Multiple sources (Wikipedia, a Reddit thread, a Pinterest pin) corroborate the Eiffel Tower identification". Information value > attribution style; if a low-rank match has a useful disambiguator, include the disambiguator even if you can only name the source as "one source" / "a forum post".
- NEVER fabricate facts.
"""


TEXT_SEARCH_IMAGES_MULTIMODAL_HEADER = (
    "Text query: {query}\n\n"
    "{n} candidate images below — each candidate has its text metadata "
    "followed by its thumbnail."
)

TEXT_SEARCH_IMAGES_MULTIMODAL_FOOTER = (
    "Rerank the {n} candidates above by image-query relevance to: ``{query}``. "
    "Use BOTH the candidate's thumbnail AND its text. "
    "Pick AT MOST {keep_k} (K = {keep_k}). Emit their 1-based original ranks as "
    "``kept_indices`` (most relevant first). Then write the 3-5 sentence ``summary`` "
    "per the schema. Output the strict JSON object specified in the system message — "
    "no prose around it."
)


# ----------------------------------------------------------------------
# Summarize-pass helpers: JSON parsing + deterministic rendering
# ----------------------------------------------------------------------

def _parse_summarize_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the summarizer LLM's JSON output.

    Tries strict json.loads first. If the model wrapped the JSON in
    markdown fences or added a leading natural-language preface, fall
    back to extracting the largest balanced ``{...}`` substring.

    Returns the parsed dict, or ``None`` if no valid JSON can be
    recovered.
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Strip markdown fences.
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
    # Locate the first ``{`` and the matching closing ``}``.
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
    candidate = s[start:end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_organic_entries(raw_text: str) -> Dict[int, str]:
    """Parse ``raw_text`` produced by ``format_search_results``, return
    ``{1-based rank: verbatim entry text}`` for organic numbered entries.

    Knowledge Graph / Answer Box / Top Stories blocks are intentionally
    excluded from the dict because they have no organic rank and their
    facts are meant to be captured in the LLM's ``summary`` field instead.

    The parser strips any leading special blocks (``[Top Stories]`` /
    ``[Answer Box]`` / ``[Knowledge Graph]``) by finding the last such
    block's blank-line terminator and continuing past it. The remaining
    text is split by ``^\\d+\\.\\s`` patterns into per-entry chunks.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return {}

    text = raw_text
    for marker in ("[Top Stories]", "[Answer Box]", "[Knowledge Graph]"):
        pos = text.find(marker)
        if pos < 0:
            continue
        end = text.find("\n\n", pos)
        if end < 0:
            # Marker present but no blank-line terminator — malformed input;
            # give up rather than risk swallowing legitimate organic entries.
            return {}
        text = text[end + 2:]

    entries: Dict[int, str] = {}
    pattern = re.compile(r"(?ms)^(\d+)\.\s+.*?(?=^\d+\.\s|\Z)")
    for match in pattern.finditer(text):
        rank = int(match.group(1))
        chunk = match.group(0).rstrip()
        if chunk:
            entries[rank] = chunk
    return entries


def _scrub_summary_text(summary: str) -> str:
    """Defensive sanitation on the LLM-emitted ``summary`` field.

    The prompt forbids URLs / ``[N]`` markers / ``webpage_url_N`` /
    ``tool_generated_image_N`` tokens inside ``summary``. We strip them
    anyway so a poorly-behaved LLM does not leak a raw URL (which would
    get registered by the memory module's URL rewriter and pollute the
    URL registry with an entry not anchored to any top-K raw block) or
    leak a phantom ref token (which would survive the rewriter but
    point to no registered URL, so any visit-call by the agent would
    fail).
    """
    if not isinstance(summary, str):
        return ""
    s = summary
    # Strip [N] citation markers and ref-token shapes (whether bracketed
    # or bare). Order matters: bracketed forms first so the inner bare
    # form regex does not run on already-stripped substrings.
    s = re.sub(r"\[webpage_url_\d+\]", "", s)
    s = re.sub(r"\[tool_generated_image_\d+\]", "", s)
    s = re.sub(r"\bwebpage_url_\d+\b", "", s)
    s = re.sub(r"\btool_generated_image_\d+\b", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    # Strip raw URLs (any scheme).
    s = re.sub(r"https?://\S+", "", s)
    # Collapse internal whitespace runs but preserve sentence boundaries.
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


# ----------------------------------------------------------------------
# Multimodal-summarize helpers (search_type=images branch)
# ----------------------------------------------------------------------

# Placeholder token inserted in place of ``Image: <http_url> <img>`` for kept
# ranks whose thumbnail was successfully fetched. The custom
# ``process_tool_output`` substitutes each placeholder with the framework-assigned
# ``[tool_generated_image_N]`` ref in kept-rank order.
MULTIMODAL_IMAGE_REF_PLACEHOLDER = "__MULTIMODAL_IMG_REF_PLACEHOLDER__"


def _replace_image_url_line_with_placeholder(entry_text: str) -> str:
    """Replace ``   Image: <url> <img>`` in a raw entry with
    ``   Image: <PLACEHOLDER>`` (indentation preserved). The custom
    ``process_tool_output`` substitutes the placeholder with the assigned
    ``[tool_generated_image_N]`` ref.
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
    whose fetch failed (value is ``None`` / missing).
    """
    out: List[str] = []
    for rank in kept_indices:
        url = image_data_urls_by_rank.get(rank)
        if isinstance(url, str) and url.startswith("data:image/"):
            out.append(url)
    return out


def _render_summarized_results(
    parsed: Dict[str, Any],
    *,
    search_type: str,
    raw_text: str,
    keep_k: int,
    image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None,
) -> str:
    """Render selected source entries and a cleaned summary from model JSON.

    Args:
        parsed: Summarizer output with kept_indices and prose fields.
        search_type: Search endpoint label used when parsing source entries.
        raw_text: Numbered source entries to select by one-based rank.
        keep_k: Maximum number of selected entries to render.
        image_data_urls_by_rank: Optional fetched images; available selected
            images use placeholders that the capability replaces with IDs.

    Returns:
        Selected entries with source links preserved, a summary with URL and
        reference tokens removed, and optional recovery suggestions. Empty
        selection produces an explicit no-relevant-results explanation.
    """
    # NOTE: do NOT prepend "[Search Results — Type: X]" — caller adds it.

    kept_raw = parsed.get("kept_indices") or []
    if not isinstance(kept_raw, list):
        kept_raw = []
    # Defensive: positive ints, dedupe, cap at keep_k, preserve order.
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

    suggested_queries = parsed.get("suggested_queries") or []
    suggested_next_actions = parsed.get("suggested_next_actions") or []
    if not isinstance(suggested_queries, list):
        suggested_queries = []
    if not isinstance(suggested_next_actions, list):
        suggested_next_actions = []

    # Branch 1: empty (NO RELEVANT RESULTS fallback)
    if not kept_indices:
        justification = str(parsed.get("no_results_justification") or "").strip()
        lines = ["NO RELEVANT RESULTS"]
        if justification:
            lines.append("")
            lines.append("Why each raw result was rejected:")
            lines.append(justification)
        if suggested_queries:
            lines.append("")
            lines.append("Suggested query rewrites:")
            for q in suggested_queries:
                lines.append(f"- {q}")
        if suggested_next_actions:
            lines.append("")
            lines.append("Suggested next actions:")
            for t in suggested_next_actions:
                lines.append(f"- {t}")
        return "\n".join(lines)

    # Branch 2: non-empty — slice verbatim raw entries by index, renumber
    entries = _extract_organic_entries(raw_text)
    body_lines: List[str] = []
    new_rank = 0
    for orig_rank in kept_indices:
        entry_text = entries.get(orig_rank)
        if not entry_text:
            continue  # Index out of range → silently drop
        new_rank += 1
        # Renumber the leading "<orig_rank>." to the new contiguous index.
        renumbered = re.sub(r"^\d+\.", f"{new_rank}.", entry_text, count=1)
        # Multimodal: strip ``Image: <url> <img>`` for kept ranks whose
        # thumbnail was successfully fetched; the bytes are attached to
        # the agent message separately, and keeping the http URL line
        # would lazy-register a second ``tool_generated_image_N`` ref
        # pointing to the same picture.
        if image_data_urls_by_rank is not None:
            data_url = image_data_urls_by_rank.get(orig_rank)
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                renumbered = _replace_image_url_line_with_placeholder(renumbered)
        body_lines.append(renumbered)

    if not body_lines:
        # All kept_indices were invalid — degrade gracefully to the
        # empty-kept fallback so the agent gets recovery hints rather
        # than an empty top-K block.
        lines = ["NO RELEVANT RESULTS (all kept_indices were out of range)"]
        if suggested_queries:
            lines.append("")
            lines.append("Suggested query rewrites:")
            for q in suggested_queries:
                lines.append(f"- {q}")
        if suggested_next_actions:
            lines.append("")
            lines.append("Suggested next actions:")
            for t in suggested_next_actions:
                lines.append(f"- {t}")
        return "\n".join(lines)

    out_lines: List[str] = list(body_lines)

    summary_text = _scrub_summary_text(parsed.get("summary") or "")
    if summary_text:
        out_lines.append("")
        out_lines.append("[Summary]")
        out_lines.append(summary_text)

    if suggested_queries:
        out_lines.append("")
        out_lines.append("Suggested query rewrites:")
        for q in suggested_queries:
            out_lines.append(f"- {q}")
    if suggested_next_actions:
        out_lines.append("")
        out_lines.append("Suggested next actions:")
        for t in suggested_next_actions:
            out_lines.append(f"- {t}")

    return "\n".join(out_lines)


class Text_Search_Tool(BaseTool):
    """Search a selected Serper endpoint and summarize relevant evidence.

    Requests ten results, retains up to three ranked entries and uses the injected
    auxiliary model. Image search supplies fetched candidate pixels to the
    summarizer; other search types use text evidence.
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
            tool_description='Search the caller-selected endpoint using search_type: general web search (the default), images, news, scholar, maps, places, shopping, or patents. Summarizes the retrieved evidence and returns retained, reranked results with source titles, snippets or metadata, and available links. Image search can include retained images and a multimodal summary. Use the summary and retained sources to plan follow-up searches; use Webpage_Visit_Tool to read a source page in detail.',
            tool_version="2.0.0",
            input_types={
                "query": "str - search query (natural language or keywords)",
                "search_type": (
                    "str - 'search' (default) or one of: images, news, scholar, "
                    "maps, places, shopping, patents"
                ),
            },
            input_schema=INPUT_SCHEMA,
            tool_aliases=["smart_search", "text_search"],
            output_type=("Summary plus retained source entries with available links; image search "
                         "can return text and retained image data. Summarization failures "
                         "return formatted source results with an explicit fallback notice."),
            demo_commands=[
                {
                    "command": 'Text_Search_Tool(query="transformer attention paper 2017", search_type="scholar")',
                    "description": (
                        "Search scholarly sources and return a summary with retained source links."
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
                "llm_summarize": True,
                "llm_model_source": "services.models",
                "supported_search_types": ALL_SEARCH_TYPES,
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
    # LLM summarization
    # ------------------------------------------------------------------

    def _build_images_multimodal_user_prompt(
        self,
        query: str,
        raw_text: str,
        *,
        image_data_urls_by_rank: Dict[int, Optional[str]],
    ) -> List[Dict[str, Any]]:
        """Build the interleaved text+image content list for the
        search_type=images multimodal summarizer call.
        """
        entries = _extract_organic_entries(raw_text)
        n = len(entries)
        parts: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": TEXT_SEARCH_IMAGES_MULTIMODAL_HEADER.format(query=query, n=n),
            },
        ]
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
            "text": TEXT_SEARCH_IMAGES_MULTIMODAL_FOOTER.format(
                query=query, n=n, keep_k=self._keep_k,
            ),
        })
        return parts

    def _summarize_with_llm(
        self,
        query: str,
        search_type: str,
        raw_results: str,
        *,
        keep_k: int,
        image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None,
    ) -> Tuple[str, List[str]]:
        """Rank raw entries and render selected evidence with a prose summary.

        Uses one auxiliary-model call to produce kept_indices and summary. Entry text
        is selected from raw_results by rank, preserving source links. For image
        searches, available pixels are interleaved with candidate text.

        Returns:
            A (rendered_text, kept_data_urls) tuple in selected-rank order. Model or
            JSON parsing failures use deterministic result-selection fallbacks.
        """
        multimodal = image_data_urls_by_rank is not None
        if multimodal:
            system_prompt = TEXT_SEARCH_IMAGES_MULTIMODAL_SYSTEM_PROMPT
            generate_arg: Any = [
                {
                    "role": "user",
                    "content": self._build_images_multimodal_user_prompt(
                        query,
                        raw_results,
                        image_data_urls_by_rank=image_data_urls_by_rank or {},
                    ),
                }
            ]
        else:
            system_prompt = TEXT_SEARCH_SUMMARIZE_SYSTEM_PROMPT
            generate_arg = TEXT_SEARCH_SUMMARIZE_USER_PROMPT.format(
                query=query,
                search_type=search_type,
                raw_results=raw_results,
                keep_k=keep_k,
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
            logger.warning("LLM summarize call failed (%s), returning raw results", exc)
            return raw_results, []

        stripped = strip_think_tags(result or "")
        if not stripped.strip():
            logger.warning("LLM summarize returned empty content, returning raw results")
            return raw_results, []

        parsed = _parse_summarize_json(stripped)
        if parsed is None:
            logger.warning(
                "LLM summarize output is not parseable JSON, returning raw results. "
                "Head: %r", stripped[:200],
            )
            return raw_results, []

        # Schema validation: the required result-selection field
        # is ``kept_indices``. Two deviation modes that would otherwise
        # silently emit "NO RELEVANT RESULTS" (losing real Serper data),
        # distinguished from a *legitimate* empty kept_indices (LLM
        # explicitly said no results found):
        #   (a) "kept_indices" key missing entirely
        #   (b) "kept_indices" present but null / wrong type
        # An empty list ([]) is preserved — that's the legitimate
        # "LLM judged no results relevant" case, which
        # ``_render_summarized_results`` handles via the structured
        # NO RELEVANT RESULTS block.
        kept_value = parsed.get("kept_indices")
        if not isinstance(kept_value, list):
            logger.warning(
                "LLM summarize JSON: 'kept_indices' is missing or not a list "
                "(top-level keys: %s); returning raw results to avoid silent data loss",
                list(parsed.keys()),
            )
            return raw_results, []

        rendered = _render_summarized_results(
            parsed,
            search_type=search_type,
            raw_text=raw_results,
            keep_k=keep_k,
            image_data_urls_by_rank=image_data_urls_by_rank,
        )

        kept_data_urls: List[str] = []
        if image_data_urls_by_rank is not None:
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
            normalized = normalized[:keep_k]
            kept_data_urls = _select_kept_data_urls(
                normalized, image_data_urls_by_rank,
            )

        return rendered, kept_data_urls

    # ------------------------------------------------------------------
    # Structured error formatting (Serper request failed)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_serper_error(
        query: str,
        search_type: str,
        error: "SerperRequestError",
    ) -> str:
        """Render a ``SerperRequestError`` into an agent-readable message.

        The output is intentionally similar in shape to ``_format_no_raw_results``
        so the agent sees consistent structure across all "search failed"
        scenarios. Includes:

        * Why it failed (transient vs permanent + HTTP status + attempt count)
        * Query rewrites (always useful)
        * Cross-tool suggestions including Text_Search itself (per the
          self-inclusion rule — re-calling with a different search_type
          is a valid recovery path even when the same query just failed)
        """
        type_list = ", ".join(ALL_SEARCH_TYPES)
        if error.transient:
            cause = (
                f"Serper temporarily unavailable (HTTP {error.status_code}, transient) "
                f"after {error.attempts} attempt(s). The Serper backend is briefly down "
                f"or rate-limiting; this is NOT a problem with the query."
            )
            retry_hint = (
                "  - Retry the SAME search shortly (after ~30-60s). Serper outages typically "
                "resolve within a minute; the same query should work again.\n"
            )
        elif error.status_code in (401, 403):
            cause = (
                f"Serper authentication / authorization rejected (HTTP {error.status_code}, "
                f"permanent). The API key is missing, invalid, or quota-exhausted."
            )
            retry_hint = (
                "  - Retrying with the same arguments will NOT help — this is a permanent "
                "auth failure that needs operator intervention. Try a different tool below.\n"
            )
        else:
            cause = (
                f"Serper request failed (HTTP {error.status_code or '?'}, permanent) "
                f"after {error.attempts} attempt(s). Detail: {error}"
            )
            retry_hint = (
                "  - Retrying with the EXACT same arguments is unlikely to help — try a "
                "different search_type, a reworded query, or a different tool below.\n"
            )
        return (
            f"[Search Error — Type: {search_type}]\n"
            f"SEARCH FAILED\n\n"
            f"Reason: {cause}\n"
            f"Query: {query!r}\n\n"
            "Suggested next actions for the agent:\n\n"
            f"{retry_hint}"
            "  - Query rewrites (try one of these instead, then re-call Text_Search_Tool):\n"
            "    - Use broader / more general phrasing of the same intent\n"
            "    - Use keywords / entity names instead of a full natural-language question\n"
            "    - Switch language (en ↔ zh, or the entity's native locale)\n"
            "    - Drop highly specific qualifiers (year, region, version) that may have over-constrained the match\n\n"
            "  - Tool switches (the agent may also re-call this same Text_Search_Tool with a different search_type "
            "— nothing is excluded just because it produced this error):\n"
            f"    - Text_Search_Tool(query=..., search_type=<one of: {type_list}>)\n"
            "        Try a different sub-mode. ``news`` for current events, ``scholar`` for academic, "
            "``maps`` / ``places`` for locations, ``shopping`` for products, ``patents`` for inventions, "
            "``images`` for visual references.\n"
            "    - Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>)\n"
            "        If your question has a visual component (identify an object / building / artwork in an image), "
            "switch to reverse image search.\n"
            "    - Webpage_Visit_Tool(urls=[webpage_url_N], goal=<concrete goal>)\n"
            "        If you already have a relevant URL ref from prior tool output, deep-read that page directly "
            "— bypasses the failing search endpoint.\n"
        )

    # ------------------------------------------------------------------
    # Empty-result fallback (Serper returned 0 raw items)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_no_raw_results(query: str, search_type: str) -> str:
        """Return a structured ``NO RELEVANT RESULTS`` message for the agent.

        Mirrors the LLM-side fallback (when filtering wipes out all results)
        so the agent sees ONE consistent shape regardless of whether the
        emptiness came from Serper or from the LLM summarizer. No LLM call —
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
            "  - Image_Search_Tool(image_id=<input_image_N | tool_generated_image_N>)\n"
            "      Reverse image search (Google Lens) when a relevant image is available. "
            "Pass the image via its ``image_id`` from the Environment Update's "
            "'Currently available image IDs' list (``input_image_N`` or "
            "``tool_generated_image_N``).\n"
            "  - Webpage_Visit_Tool(urls=[<url1>, ..., <urlN>], goal=<what to extract from the page>)\n"
            "      If specific URL(s) are already known, fetch and extract content directly. "
            "Both arguments are required: ``urls`` is a list of 1 to 5 HTTP(S) links "
            "and ``goal`` is the extraction "
            "intent that guides the per-page LLM extractor.\n"
        )

    # ------------------------------------------------------------------
    # Serper search execution
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        search_type: str = "search",
    ) -> Union[str, Dict[str, Any]]:
        """Search a selected endpoint and return summarized evidence.

        Args:
            query: Search query string.
            search_type: One of ALL_SEARCH_TYPES; defaults to search.

        Returns:
            Formatted text with selected result entries and a summary. Image search
            returns a mapping with text and kept_image_data_urls when selected images
            were fetched successfully. Exhausted search errors and empty results
            return explanatory text with recovery suggestions.

        Raises:
            ValueError: search_type is unsupported.
            FileNotFoundError: No Serper credential is configured.

        Result count, retained rank count and summarization policy are service settings,
        not task arguments. Failed image fetches fall back to available text evidence.
        """
        if not query or not isinstance(query, str):
            raise ToolArgumentError("query is required and must be a non-empty string")
        num_results, gl, hl = self._num_results, None, None

        # --- Resolve search type ---
        resolved_type = search_type.strip().lower() if search_type else "search"

        if resolved_type not in ALL_SEARCH_TYPES:
            logger.warning("Invalid search_type '%s', falling back to 'search'", resolved_type)
            resolved_type = "search"

        logger.info("Text search: type=%s, query=%s", resolved_type, query[:80])

        # --- Call Serper (with retry inside SerperClient; we catch the
        # final SerperRequestError and emit a structured agent-readable
        # error so the call never bubbles out as a raw HTTPError) ---
        client = self._get_client()
        try:
            data = client.search(
                query=query,
                search_type=resolved_type,
                num=num_results,
                gl=gl,
                hl=hl,
            )
        except SerperRequestError as exc:
            logger.warning(
                "Text_Search_Tool: Serper search failed (transient=%s, status=%s, attempts=%d): %s",
                exc.transient, exc.status_code, exc.attempts, exc,
            )
            return self._format_serper_error(query, resolved_type, exc)

        raw_formatted = format_search_results(resolved_type, data, max_results=num_results)
        if not raw_formatted:
            return self._format_no_raw_results(query, resolved_type)

        # --- LLM summarization ---
        if self._llm_summarize:
            image_data_urls_by_rank: Optional[Dict[int, Optional[str]]] = None
            all_fetches_failed = False
            # Only the search_type=images branch goes multimodal — other
            # types return webpages, not images, and pre-fetching per-result
            # OG-image previews would add HTTP cost with low rerank value.
            #
            # Fetch-failed candidates are pre-filtered BEFORE the rerank,
            # so the summarizer only judges candidates whose image bytes
            # are actually present. If ALL candidate fetches fail, we
            # fall back to a text-only summarize over the metadata, with
            # a banner telling the agent no image bytes were retrieved.
            if self._multimodal_summarize and resolved_type == "images":
                items = data.get("images", []) if isinstance(data, dict) else []
                items_limited = items[:num_results]
                candidate_urls: List[str] = []
                for item in items_limited:
                    if not isinstance(item, dict):
                        candidate_urls.append("")
                        continue
                    img_url = ""
                    for key in ("imageUrl", "thumbnailUrl"):
                        cand = str(item.get(key) or "").strip()
                        if cand.lower().startswith(("http://", "https://")):
                            img_url = cand
                            break
                    candidate_urls.append(img_url)
                fetched = fetch_images_parallel(
                    [u for u in candidate_urls if u],
                )
                # Pre-filter: keep only items whose thumbnail fetched
                # successfully. The summarizer's whole job is visual
                # judgment, so a candidate with no image bytes is not
                # judgeable — including it forces a fake text-only fallback
                # that pollutes the rerank.
                fetched_pairs: List[Tuple[Dict[str, Any], str]] = []
                for item, c_url in zip(items_limited, candidate_urls):
                    if not isinstance(item, dict):
                        continue
                    data_url = fetched.get(c_url) if c_url else None
                    if isinstance(data_url, str) and data_url.startswith("data:image/"):
                        fetched_pairs.append((item, data_url))

                if not fetched_pairs:
                    # All-fail path: rebuild raw_formatted from the original
                    # items but with image URLs stripped, so the framework's
                    # URL rewriter does not lazy-register tool_generated_image_N
                    # refs pointing at URLs we already know are fetch-failure-prone.
                    all_fetches_failed = True
                    items_no_imgs: List[Any] = []
                    for item in items_limited:
                        if isinstance(item, dict):
                            cleaned = dict(item)
                            cleaned.pop("imageUrl", None)
                            cleaned.pop("thumbnailUrl", None)
                            items_no_imgs.append(cleaned)
                        else:
                            items_no_imgs.append(item)
                    filtered_data = dict(data) if isinstance(data, dict) else {}
                    filtered_data["images"] = items_no_imgs
                    raw_formatted = format_search_results(
                        resolved_type, filtered_data, max_results=num_results,
                    )
                    image_data_urls_by_rank = None
                else:
                    # Partial-fail or full-success: rebuild raw_formatted and
                    # the rank-to-data-url map from the filtered items only.
                    # Ranks are renumbered to 1..N_fetched so the summarizer's
                    # ``kept_indices`` land back on the right entries.
                    filtered_items = [item for item, _ in fetched_pairs]
                    filtered_data = dict(data) if isinstance(data, dict) else {}
                    filtered_data["images"] = filtered_items
                    raw_formatted = format_search_results(
                        resolved_type, filtered_data, max_results=num_results,
                    )
                    image_data_urls_by_rank = {
                        rank: data_url
                        for rank, (_, data_url) in enumerate(fetched_pairs, 1)
                    }

            summary, kept_data_urls = self._summarize_with_llm(
                query,
                resolved_type,
                raw_formatted,
                keep_k=self._keep_k,
                image_data_urls_by_rank=image_data_urls_by_rank,
            )
            if all_fetches_failed:
                banner = (
                    "Image fetch failed for ALL candidates. The candidate metadata "
                    "below (titles and source URLs) is available but no image bytes "
                    "were retrieved.\n"
                )
                text_payload = (
                    f"[Search Results — Type: {resolved_type}]\n{banner}\n{summary}"
                )
            else:
                text_payload = f"[Search Results — Type: {resolved_type}]\n{summary}"
            if kept_data_urls:
                return {
                    "text": text_payload,
                    "kept_image_data_urls": kept_data_urls,
                }
            return text_payload

        return f"[Search Results — Type: {resolved_type}]\n{raw_formatted}"
