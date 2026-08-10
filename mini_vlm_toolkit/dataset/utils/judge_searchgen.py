"""Judge_SearchGen — verbatim port of the SearchGen-Bench evaluation protocol.

SearchGen-Bench ("Search Beyond What Can Be Taught: Evolving the Knowledge
Boundary in Agentic Visual Generation", arXiv 2607.05382; GitHub
HaozheH3/SearchGen; HF JasperHaozhe/SearchGen-Bench) is a 751-prompt
knowledge-intensive image-generation benchmark. Each sample carries a
verification checklist, a weighted task-specific rubric, 0-6 visual
reference images and 0-4 textual knowledge gap slots. A single multimodal
LLM judge call per sample scores 10 components on a 0-3 scale
(half-points): checklist, adaptive_rubric, prompt_faithfulness,
image_quality, text_rendering, ai_naturalness, composition_and_aesthetics,
physical_plausibility, visual_reference, text_reference. The paper reports
component means x 100/3 on a 0-100 scale.

Ported byte-exact from ``SearchGen/evaluation/searchgen_eval/``:
  - ``judge_common.py``  -> JUDGE_SYSTEM_PROMPT, XML tag constants,
    ``build_eval_prompt_text`` and all its helpers, the full XML/JSON
    parser + repair chain, ``parsed_judge_labeled_scores_03``.
  - ``evaluator.py``     -> ``build_request`` visual-context construction
    (here: :func:`build_visual_context_from_slots`).
  - ``scoring.py``       -> ``COMPONENTS`` + the scores->component mapping
    (here: :func:`component_means_from_scores`).
  - ``image_utils.py``   -> judge-side image downscale (<=1.5MP, <=1568
    max edge, JPEG q95). Upstream inlines the result as a base64 data
    URL; we write the identically-resized JPEG to a cache file and pass
    the PATH through the framework's flat role-message schema
    (``api/base.py::preprocess_message_with_role``) — the API wrapper
    base64-encodes it itself. Pixel content is identical; only the
    transport codec differs. Datasets should pass ``img_detail='high'``
    to ``build_judge``.

Upstream is Apache-2.0 (Copyright 2026 Jayce-Ping, Haozhe Wang).

The only deliberate difference vs upstream: the judge MODEL is whatever
``--judge-config`` provides (upstream uses any OpenAI-compatible endpoint;
no model is hardcoded in the protocol).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# System prompt — judge_common.py::JUDGE_SYSTEM_PROMPT, byte-exact.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are an image quality evaluator. Be rigorous and evidence-based. "
    "Output strict XML only, using the root element <evaluation> and the required tag names.\n\n"
    "Follow these steps in order:\n\n"
    "Step 1 — Task understanding\n"
    "- user_prompt: identify required subjects, attributes, actions, scene, style, quality, and in-image text.\n"
    "- verification_checklist: map each item to concrete visual evidence that satisfies or violates it.\n"
    "- evaluation_rubric: clarify what each dimension judges.\n"
    "- visual_reference_context: identify reference-dependent criteria.\n"
    "- textual_knowledge_gap_context: critical/important textual gaps from prompt analysis (metadata only); "
    "what factual or textual elaboration the target should reflect.\n\n"
    "Step 2 — Image roles\n"
    "- 'Reference image N' = comparison evidence only. Never score reference image quality.\n"
    "- 'Image to assess' = the only image scored.\n\n"
    "Step 3 — Evidence extraction (observable facts only, no inference)\n"
    "  a) Subjects & identity — entities, distinguishing traits, count.\n"
    "  b) Attributes — clothing, colors, materials, textures, proportions.\n"
    "  c) Actions & poses — gestures, interactions, body language.\n"
    "  d) Scene & environment — setting, background, weather, time of day.\n"
    "  e) Composition — framing, spatial arrangement, depth, balance.\n"
    "  f) Color & lighting — palette, light direction, shadows, consistency.\n"
    "  g) Style — artistic medium, realism level, stylistic cues.\n"
    "  h) In-image text — content, legibility, spelling, placement.\n"
    "  i) Physical plausibility — anatomy (hands, faces, limbs), object physics, spatial consistency.\n"
    "  j) Artifacts — blur, aliasing, seams, duplications, impossible geometry.\n"
    "  k) AI tells — unnatural smoothness, plastic textures, warped backgrounds, uncanny features.\n\n"
    "Step 4 — Reference comparison\n"
    "- Compare target against references for reference-dependent criteria.\n"
    "- Note which identity traits are preserved, altered, or missing.\n"
    "- Map each reference to the checklist items and rubric dimensions it informs.\n\n"
    "Step 5 — Scoring scale (0–3, half-points allowed)\n"
    "  0 = Not met / contradicted.  1 = Weak, major issues.\n"
    "  2 = Mostly met, minor issues.  3 = Fully met, correct and coherent.\n\n"
    "Step 5b — Auditable scores (each scored row's <reason>)\n"
    "Use three lines in order: (1) Anchor: <0…3 in 0.5 steps> — why, with Step 3 cite. "
    "(2) Adjustments: bullets \"Δ=±0.5|±1 …\" each with evidence, or \"none\". "
    "(3) Computed final: X — X must equal Anchor+ΣΔ (clamp to [0,3], 0.5 steps) and MUST match <score>. "
    "reference_alignment: 2–4 sentences only, no Anchor template.\n\n"
    "Step 6 — Score checklist and rubric\n"
    "Score each checklist item, each task-specific rubric dimension, and ALL generic dimensions below.\n"
    "Every scored row's <reason> follows Step 5b.\n\n"
    "  prompt_faithfulness — Are all requested subjects, attributes, actions, scene, and style present and accurate?\n\n"
    "  image_quality — Visual clarity/sharpness? Artifacts or defects? "
    "Style/lighting/perspective coherence? Physical plausibility and anatomy correctness?\n\n"
    "  text_rendering — If no in-image text is required and none is present, leave the score element empty in XML; "
    "otherwise give a score in 0–3. Does rendered text match required content? Readable? Spelled? Placed well?\n\n"
    "  ai_naturalness — Does the image look AI-generated? Check: organic texture realism vs AI smoothness? "
    "Fine-detail plausibility vs uncanny uniformity? Environment grounded vs dreamlike/generic? "
    "Would a careful viewer readily identify this as AI-generated?\n\n"
    "  composition_and_aesthetics — Well-framed and balanced? Clear depth and spatial arrangement? "
    "Color harmony and consistent lighting? Visually appealing overall?\n\n"
    "  physical_plausibility — Physical rules adherence: anatomy (fingers, limbs, proportions, pose)? "
    "Object physics (gravity, support, balance)? Spatial consistency (perspective, scale, occlusion)? "
    "Material properties (reflections, transparency, texture)? Lighting/shadow consistency? "
    "Leave score empty if not applicable (abstract/stylized content).\n\n"
    "Step 7 — Visual reference fidelity\n"
    "- Set visual_reference applicable=true only if visual references are provided in the evaluation context.\n"
    "- When applicable, compare the target image against each reference for identity preservation, attribute "
    "consistency, and style fidelity. Check: are key facial features / distinguishing traits maintained? "
    "Are colors, textures, and proportions consistent? Is the overall style preserved?\n"
    "- Score preservation of critical identity, attributes, and style from references onto the target.\n"
    "- When applicable, visual_reference <reason> follows Step 5b.\n\n"
    "Step 7b — Textual knowledge / textual elaboration fidelity\n"
    "- Set text_reference applicable=true only if textual knowledge gaps are listed in the evaluation context.\n"
    "- Using gap metadata (entity, reasoning, suggested lookup intent) and visible target evidence only, score how "
    "correctly the image renders what those gaps require (facts, labels, standards, wording, numbers, symbols, etc.).\n"
    "- Do not use live web search.\n"
    "- When applicable, text_reference <reason> follows Step 5b.\n\n"
    "Step 8 — Consistency checks\n"
    "- Computed final = <score> for every scored row; reasons must match scores.\n"
    "- Reasons and scores must not contradict. Major failures → score < 2. Minor-only issues → score > 1.\n"
    "- When a rubric score applies, use values in {0, 0.5, 1, 1.5, 2, 2.5, 3}. When a dimension does not apply, "
    "leave that dimension's <score> empty (no characters between the tags).\n"
    "- Include a dimension element for every required rubric row; all numeric scores are for the target image only.\n\n"
    "Output the target image evaluation as a single XML block. "
    "No preface, no code fences, no text after the root element. "
    "Escape <, >, & in text (e.g. &amp;)."
)

# ---------------------------------------------------------------------------
# XML tag constants — judge_common.py, byte-exact.
# ---------------------------------------------------------------------------
JUDGE_XML_ROOT_TAG = "evaluation"
JUDGE_XML_REFERENCE_ALIGNMENT = "reference_alignment"
JUDGE_XML_ALIGNMENT_ITEM = "item"
JUDGE_XML_REF_ATTR = "ref"
JUDGE_XML_CHECKLIST_LINKS = "checklist_links"
JUDGE_XML_RUBRIC_LINKS = "rubric_links"
JUDGE_XML_CHECKLIST = "checklist"
JUDGE_XML_CHECKLIST_ITEM = "item"
JUDGE_XML_CRITERION = "criterion"
JUDGE_XML_RUBRIC = "rubric"
JUDGE_XML_DIMENSION = "dimension"
JUDGE_XML_VISUAL_REFERENCE = "visual_reference"
JUDGE_XML_TEXT_REFERENCE = "text_reference"
JUDGE_XML_APPLICABLE = "applicable"
JUDGE_XML_REASON = "reason"
JUDGE_XML_SCORE = "score"
JUDGE_XML_NAME_ATTR = "name"
JUDGE_XML_LINK = "link"
JUDGE_XML_I_ATTR = "i"

LEGACY_JUDGE_XML_ROOT = "judge_output"
LEGACY_JUDGE_XML_REF_LABEL = "ref_label"
LEGACY_JUDGE_XML_LINKED_CL = "linked_checklist"
LEGACY_JUDGE_XML_LINKED_RUB = "linked_rubric"
LEGACY_JUDGE_XML_REF_REASON = "ref_reason"
LEGACY_JUDGE_XML_CL_ROW = "cl_row"
LEGACY_JUDGE_XML_LINE_TEXT = "line_text"
LEGACY_JUDGE_XML_RUBRIC_LINE = "rubric_line"
LEGACY_JUDGE_XML_VISUAL_REF = "visual_ref"
LEGACY_JUDGE_XML_APPLIES = "applies"

JUDGE_OUTPUT_XML_SHAPE_FENCED = f"""```xml
<{JUDGE_XML_ROOT_TAG}>
  <{JUDGE_XML_REFERENCE_ALIGNMENT}>
    <{JUDGE_XML_ALIGNMENT_ITEM} {JUDGE_XML_REF_ATTR}="Reference image 1">
      <{JUDGE_XML_CHECKLIST_LINKS}>…</{JUDGE_XML_CHECKLIST_LINKS}>
      <{JUDGE_XML_RUBRIC_LINKS}>…</{JUDGE_XML_RUBRIC_LINKS}>
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    </{JUDGE_XML_ALIGNMENT_ITEM}>
  </{JUDGE_XML_REFERENCE_ALIGNMENT}>
  <{JUDGE_XML_CHECKLIST}>
    <{JUDGE_XML_CHECKLIST_ITEM} {JUDGE_XML_I_ATTR}="0">
      <{JUDGE_XML_CRITERION}>…</{JUDGE_XML_CRITERION}>
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
      <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
    </{JUDGE_XML_CHECKLIST_ITEM}>
  </{JUDGE_XML_CHECKLIST}>
  <{JUDGE_XML_RUBRIC}>
    <{JUDGE_XML_DIMENSION} {JUDGE_XML_NAME_ATTR}="…">
      <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
      <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
    </{JUDGE_XML_DIMENSION}>
  </{JUDGE_XML_RUBRIC}>
  <{JUDGE_XML_VISUAL_REFERENCE}>
    <{JUDGE_XML_APPLICABLE}>…</{JUDGE_XML_APPLICABLE}>
    <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
  </{JUDGE_XML_VISUAL_REFERENCE}>
  <{JUDGE_XML_TEXT_REFERENCE}>
    <{JUDGE_XML_APPLICABLE}>…</{JUDGE_XML_APPLICABLE}>
    <{JUDGE_XML_REASON}>…</{JUDGE_XML_REASON}>
    <{JUDGE_XML_SCORE}>…</{JUDGE_XML_SCORE}>
  </{JUDGE_XML_TEXT_REFERENCE}>
</{JUDGE_XML_ROOT_TAG}>
```"""


# ---------------------------------------------------------------------------
# Prompt-text builders — judge_common.py, byte-exact.
# ---------------------------------------------------------------------------

def normalize_rubric(rubric: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rubric, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in rubric.items():
        if not isinstance(val, dict):
            continue
        weight = val.get("weight", 1.0)
        if isinstance(weight, (int, float)):
            w = float(weight)
        else:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 1.0
        out[str(key)] = {
            "weight": w,
            "description": str(val.get("description", "")),
        }
    return out


def get_reference_count_fields_default(used_count: int) -> Dict[str, int]:
    n = max(0, int(used_count))
    return {
        "expected_reference_images_count": n,
        "selected_reference_images_count": n,
        "used_reference_images_count": n,
    }


def build_reference_image_metadata_map(visual_context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    url_to_meta: Dict[str, Dict[str, Any]] = {}
    candidates = visual_context.get("critical_candidates")
    if not isinstance(candidates, list):
        return url_to_meta

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        matched = candidate.get("matched_search_results")
        if not isinstance(matched, list):
            continue
        for match in matched:
            if not isinstance(match, dict):
                continue
            image_url = match.get("image_url")
            if not isinstance(image_url, str):
                continue
            image_url = image_url.strip()
            if not image_url or image_url in url_to_meta:
                continue
            url_to_meta[image_url] = {
                "candidate_index": candidate.get("candidate_index"),
                "entity": candidate.get("entity"),
                "severity": candidate.get("severity"),
                "candidate_reasoning": candidate.get("reasoning"),
                "query": match.get("query"),
                "selected_image_title": match.get("selected_image_title"),
                "source_file": match.get("source_file"),
                "reference_local_path": match.get("local_path"),
            }
    return url_to_meta


def get_reference_count_fields_from_row(row: Dict[str, Any]) -> Dict[str, int]:
    details = row.get("augmented_generation_details")
    if not isinstance(details, dict):
        return get_reference_count_fields_default(0)

    def to_non_negative_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value if value >= 0 else 0
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return 0
        return ivalue if ivalue >= 0 else 0

    return {
        "expected_reference_images_count": to_non_negative_int(
            details.get("expected_reference_images_count", 0)
        ),
        "selected_reference_images_count": to_non_negative_int(
            details.get("selected_reference_images_count", 0)
        ),
        "used_reference_images_count": to_non_negative_int(details.get("used_reference_images_count", 0)),
    }


def _slot_reference_focus(
    visual_context: Dict[str, Any],
    slot_index: int,
    url_to_meta: Dict[str, Dict[str, Any]],
    ref_slot_url: str,
) -> Tuple[str, str, str]:
    if isinstance(ref_slot_url, str) and ref_slot_url.startswith("flowfactory://"):
        cands = visual_context.get("critical_candidates")
        if isinstance(cands, list) and 0 <= slot_index < len(cands):
            c = cands[slot_index]
            if isinstance(c, dict):
                entity = str(c.get("entity") or "unknown").strip()
                severity = str(c.get("severity") or "unknown").strip()
                reasoning = str(c.get("reasoning") or "").strip()
                return entity, severity, reasoning or "No reasoning provided."
        return "unknown", "unknown", "No reasoning provided."

    meta = url_to_meta.get(ref_slot_url, {})
    entity = str(meta.get("entity") or "").strip()
    severity = str(meta.get("severity") or "").strip()
    reasoning = str(meta.get("candidate_reasoning") or "").strip()

    # Fallback: if URL didn't match the map, use critical_candidates by slot index
    if not entity:
        cands = visual_context.get("critical_candidates")
        if isinstance(cands, list) and 0 <= slot_index < len(cands):
            c = cands[slot_index]
            if isinstance(c, dict):
                entity = str(c.get("entity") or "unknown").strip()
                severity = str(c.get("severity") or "unknown").strip()
                reasoning = str(c.get("reasoning") or "").strip()

    return entity or "unknown", severity or "unknown", reasoning or "No reasoning provided."


def interleaved_ref_context_block(
    *,
    slot_index: int,
    visual_context: Dict[str, Any],
    url_to_meta: Dict[str, Dict[str, Any]],
    ref_slot_url: str,
) -> str:
    entity, severity, reasoning = _slot_reference_focus(
        visual_context, slot_index, url_to_meta, ref_slot_url
    )
    if len(reasoning) > 220:
        reasoning = reasoning[:217].rstrip() + "..."
    return (
        f"Reference image {slot_index + 1} context:\n"
        f"- entity: {entity}\n"
        f"- severity: {severity}\n"
        f"- why this reference matters: {reasoning or 'No reasoning provided.'}"
    )


def build_eval_prompt_text(
    *,
    row: Dict[str, Any],
    verification_checklist: Sequence[Any],
    evaluation_rubric: Dict[str, Any],
    visual_context: Dict[str, Any],
    variant: str,
    reference_slot_urls: Sequence[str],
) -> str:
    """
    Build the judge user-text block (instructions + evaluation context).

    ``row["user_prompt"]`` must be the original user-facing task. The checklist, rubric,
    and prompt-faithfulness score are all defined against that text rather than the
    generator's refined conditioning prompt.
    """
    checklist = list(verification_checklist)
    rubric = normalize_rubric(evaluation_rubric)
    reference_counts = get_reference_count_fields_from_row(row)
    if reference_counts["used_reference_images_count"] <= 0 and len(reference_slot_urls) > 0:
        reference_counts = get_reference_count_fields_default(len(reference_slot_urls))

    user_prompt = str(row.get("user_prompt", "") or "").strip()
    url_to_meta = build_reference_image_metadata_map(visual_context=visual_context)

    image_layout_lines: List[str] = []
    for i, ref_url in enumerate(reference_slot_urls):
        entity, severity, reasoning = _slot_reference_focus(visual_context, i, url_to_meta, ref_url)
        image_layout_lines.append(f"Reference image {i + 1} (entity={entity}): <image>")
    image_layout_lines.append("Image to assess: <image>")
    image_layout_text = "\n".join(image_layout_lines)

    checklist_lines = [f"{j}. {str(item)}" for j, item in enumerate(checklist, start=1)]
    checklist_text = "\n".join(checklist_lines) if checklist_lines else "None provided."

    rubric_lines: List[str] = []
    for dimension, dim_meta in rubric.items():
        desc = str(dim_meta.get("description", "") or "").strip()
        weight = dim_meta.get("weight")
        weight_text = f", weight={weight}" if weight is not None else ""
        rubric_lines.append(f"- {dimension}{weight_text}: {desc or 'No description provided.'}")
    rubric_text = "\n".join(rubric_lines) if rubric_lines else "None provided."
    rubric_text_with_required_generic = (
        f"{rubric_text}\n"
        "- prompt_faithfulness: Required generic — subjects, attributes, actions, scene, and style vs user prompt.\n"
        "- image_quality: Required generic — clarity, artifacts, style/lighting/perspective coherence, plausibility.\n"
        "- text_rendering: Required generic — text accuracy, readability, spelling, font, placement (leave score empty in output if no text required).\n"
        "- ai_naturalness: Required generic — organic realism vs AI smoothness, detail plausibility, grounded environment.\n"
        "- composition_and_aesthetics: Required generic — framing, balance, depth, color harmony, overall appeal."
    )
    rubric_text_with_required_generic += (
        "\n- physical_plausibility: Required generic — physical plausibility of depicted content. "
        "Evaluate ONLY aspects that should follow physical rules given the user's request. "
        "Ground every claim with specific visual evidence. "
        "Check: (a) Human/creature anatomy — finger count, hand structure, facial symmetry, "
        "limb proportions, joint articulation, body-part connectivity, pose biomechanical "
        "soundness (grip on held objects, weight-bearing stance, natural posture)? "
        "(b) Object physics — gravity, support/balance, weight distribution; do objects float "
        "without support or intersect impossibly? "
        "(c) Spatial consistency — perspective correctness, vanishing points, scale between "
        "entities (character height vs environment, foreground vs background proportions), "
        "occlusion ordering, depth coherence? "
        "(d) Material properties — do metals reflect, glass transmit, fabric drape, liquids "
        "flow correctly? Are surface textures physically plausible under depicted lighting? "
        "(e) Lighting/shadow physics — shadow direction consistent with light source(s), "
        "reflection angles correct, no impossible self-shadowing? "
        "If the request is for abstract, stylized, or intentionally surreal content where "
        "physical rules do not apply, leave score empty in output."
    )

    critical_focus_lines: List[str] = []
    critical_candidates = visual_context.get("critical_candidates")
    if isinstance(critical_candidates, list):
        for candidate in critical_candidates:
            if not isinstance(candidate, dict):
                continue
            entity = str(candidate.get("entity") or "unknown").strip()
            severity = str(candidate.get("severity") or "unknown").strip()
            reasoning = str(candidate.get("reasoning") or "").strip()
            if len(reasoning) > 220:
                reasoning = reasoning[:217].rstrip() + "..."
            critical_focus_lines.append(
                f"- entity='{entity}', severity='{severity}', reason='{reasoning or 'No reasoning provided.'}'"
            )
    critical_focus_text = "\n".join(critical_focus_lines) if critical_focus_lines else "None."

    text_knowledge_lines: List[str] = []
    tk_slots = visual_context.get("eval_text_knowledge_slots")
    if isinstance(tk_slots, list):
        for slot in tk_slots:
            if not isinstance(slot, dict):
                continue
            entity = str(slot.get("entity") or "unknown").strip()
            severity = str(slot.get("severity") or "unknown").strip()
            reasoning = str(slot.get("reasoning") or "").strip()
            sug = str(slot.get("suggested_search") or "").strip()
            if len(reasoning) > 220:
                reasoning = reasoning[:217].rstrip() + "..."
            if len(sug) > 160:
                sug = sug[:157].rstrip() + "..."
            text_knowledge_lines.append(
                f"- entity='{entity}', severity='{severity}', "
                f"suggested_lookup_intent='{sug or 'N/A'}', reason='{reasoning or 'No reasoning provided.'}'"
            )
    text_knowledge_focus_text = "\n".join(text_knowledge_lines) if text_knowledge_lines else "None."
    has_text_knowledge = bool(visual_context.get("has_text_knowledge_gaps_for_eval"))

    # --- Build conditional sections ---
    has_visual_refs = bool(visual_context.get("has_critical_references")) and len(reference_slot_urls) > 0

    visual_ref_section = ""
    if has_visual_refs:
        visual_ref_section = (
            "\nVisual-reference setup:\n"
            f"- attached_reference_images: {len(reference_slot_urls)}\n"
            f"- used_reference_images_count (from generation): {reference_counts.get('used_reference_images_count')}\n"
            "- critical reference focus:\n"
            f"{critical_focus_text}\n"
            "- Guideline: Compare the target image against each reference for identity preservation (facial features, "
            "distinguishing traits), attribute consistency (colors, textures, proportions), and style fidelity. "
            "Penalize missing or altered identity traits; reward faithful reproduction.\n"
        )

    textual_knowledge_section = ""
    if has_text_knowledge and text_knowledge_lines:
        textual_knowledge_section = (
            "\nTextual knowledge gap (textual reference):\n"
            f"- gap_count: {len(text_knowledge_lines)}\n"
            "- gaps to assess (metadata only; no retrieved search text):\n"
            f"{text_knowledge_focus_text}\n"
            "- Guideline: Using the gap metadata (entity, reasoning, suggested lookup intent) and visible evidence "
            "in the target image only, assess whether the image correctly renders the factual/textual content "
            "these gaps describe (labels, numbers, symbols, standards, proper nouns, etc.).\n"
        )

    # --- Build XML output format block ---
    xml_format_block = (
        "XML output blocks:\n"
        f"1) reference_alignment — one {JUDGE_XML_ALIGNMENT_ITEM} per reference slot "
        f"(attribute {JUDGE_XML_REF_ATTR}=\"Reference image k\" matching the image layout): "
        f"{JUDGE_XML_CHECKLIST_LINKS} and {JUDGE_XML_RUBRIC_LINKS} (plain text or multiple {JUDGE_XML_LINK} children "
        "for checklist phrases / rubric dimension names), then reason (how the reference should influence judgment "
        "of the target only).\n"
        f"2) checklist — one {JUDGE_XML_CHECKLIST_ITEM} per verification line (optional {JUDGE_XML_I_ATTR}="
        f"0-based index): {JUDGE_XML_CRITERION} (the checklist line being scored), reason (Step 5b), "
        "score (always present and numeric).\n"
        f"3) rubric — one {JUDGE_XML_DIMENSION} per task-specific and generic row (attribute {JUDGE_XML_NAME_ATTR}="
        "exact dimension name): reason (Step 5b), then score; if N/A, empty <score> and brief reason.\n"
        f"4) visual_reference — {JUDGE_XML_APPLICABLE} true/false; when true: Step 5b reason + score; when false: short reason, score=0.\n"
        f"5) text_reference — {JUDGE_XML_APPLICABLE} true/false; when true: Step 5b reason + score; when false: short reason, score=0.\n\n"
        f"Row ordering: if all {JUDGE_XML_CHECKLIST_ITEM} rows carry a non-empty {JUDGE_XML_I_ATTR} attribute, "
        "ascending index order; otherwise document order.\n\n"
        f"{JUDGE_OUTPUT_XML_SHAPE_FENCED}\n"
    )

    # --- Assemble user message (query-specific only) ---
    parts: List[str] = []

    parts.append(
        "Evaluation context:\n\n"
        "Task prompt:\n"
        f"{user_prompt or 'N/A'}\n\n"
        "Verification checklist:\n"
        f"{checklist_text}\n\n"
        "Evaluation rubric:\n"
        f"{rubric_text_with_required_generic}"
    )

    if visual_ref_section:
        parts.append(visual_ref_section)

    if textual_knowledge_section:
        parts.append(textual_knowledge_section)

    parts.append(f"\n\n{xml_format_block}")

    parts.append(
        f"\nImage layout (slot order):\n{image_layout_text}\n\n"
        f"Return strict XML only (single <{JUDGE_XML_ROOT_TAG}> document; no legacy <{LEGACY_JUDGE_XML_ROOT}>).\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing — judge_common.py, byte-exact (full repair chain).
# ---------------------------------------------------------------------------

def _strip_assistant_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```xml"):
        cleaned = cleaned[6:]
    elif cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _repair_whitespace_in_xml_close_tags(text: str) -> str:
    r"""
    Normalize ``</ rubric_links>``-style closings to ``</rubric_links>``.

    Invalid XML only; well-formed judge output is unchanged.
    """
    return re.sub(r"</\s+([A-Za-z][\w:-]*)\s*>", r"</\1>", text)


def _extract_balanced_root_fragment(text: str, local_tag: str) -> Optional[Tuple[int, int]]:
    """
    Return ``(start, end)`` slice indices of the first balanced ``<local_tag>...</local_tag>`` subtree.

    Matching is case-insensitive for both open and close. Returns ``None`` if the root is unclosed
    (truncated output).
    """
    tag_re_open = re.compile(rf"<{re.escape(local_tag)}\b[^>]*>", re.IGNORECASE)
    m = tag_re_open.search(text)
    if not m:
        return None
    start = m.start()
    i = m.end()
    depth = 1
    tag_re_open2 = re.compile(rf"<{re.escape(local_tag)}\b[^>]*>", re.IGNORECASE)
    tag_re_close = re.compile(rf"</{re.escape(local_tag)}\s*>", re.IGNORECASE)
    while i < len(text) and depth > 0:
        next_open = tag_re_open2.search(text, i)
        next_close = tag_re_close.search(text, i)
        if next_close is None:
            return None
        next_open_start = next_open.start() if next_open is not None else len(text) + 1
        close_start = next_close.start()
        if next_open is not None and next_open_start < close_start:
            depth += 1
            i = next_open.end()
        else:
            depth -= 1
            if depth == 0:
                return (start, next_close.end())
            i = next_close.end()
    return None


def _extract_first_judge_root_xml_fragment(cleaned: str) -> Optional[str]:
    """Extract the first ``evaluation`` XML document from the response."""
    span = _extract_balanced_root_fragment(cleaned, JUDGE_XML_ROOT_TAG)
    if span is None:
        return None
    start, end = span
    return cleaned[start:end]


def _preprocess_judge_xml_text_for_parse(text: str) -> str:
    """
    Strip code fences and apply :func:`_repair_common_judge_xml_typos` before root extraction.

    Keeps behavior predictable for well-formed ``<evaluation>`` output; does not apply ad-hoc
    repairs for malformed model-specific corner cases.
    """
    cleaned = _strip_assistant_code_fences(text)
    cleaned = _repair_whitespace_in_xml_close_tags(cleaned)
    cleaned = _repair_common_judge_xml_typos(cleaned)
    return cleaned


def _repair_malformed_score_closing_tags(text: str) -> str:
    """
    Fix frontier XML where ``</score>`` is truncated to ``</5`` before a real ``</score>``, or to ``</5>``.

    Observed shapes: ``<score>2</5</score>``, ``<score>2</5></score>``, and ``<score>1</5>`` only.
    """
    out = text
    out = re.sub(r"(<score>[^<]*)</5(?=</score>)", r"\1", out)
    out = re.sub(r"(<score>[^<]*)</5>\s*</score>", r"\1</score>", out)
    out = re.sub(r"(<score>[^<]*)</5>", r"\1</score>", out)
    return out


def _repair_broken_evaluation_criterion_tags(text: str) -> str:
    """
    Fix ``<criterion>`` rows where ``</criterion>`` is missing before ``<reason>``, or mangled as ``?</␠…>``.

    Frontier sometimes drops ``</criterion>`` so the line becomes ``...?</`` + spaces + ``<reason>``
    (a truncated ``</...>`` before ``<reason>``), or only ``?`` + spaces + ``<reason>``.
    Another failure mode is ``...</?`` + spaces + ``>`` without a proper closing tag name.
    """
    out = text
    out = re.sub(
        r"(<criterion>[^<]*\?)</\s*<reason>",
        r"\1</criterion>\n      <reason>",
        out,
    )
    out = re.sub(
        r"(<criterion>[^<]*\?)\s+<reason>",
        r"\1</criterion>\n      <reason>",
        out,
    )
    out = re.sub(r"\?</\s+>", r"?</criterion>", out)
    return out


def _repair_common_judge_xml_typos(text: str) -> str:
    """
    Fix tag spellings that otherwise make ElementTree reject otherwise well-formed judge XML.

    Frontier models occasionally emit ``large_text`` instead of ``criterion`` (``<evaluation>`` schema)
    or legacy ``line_text`` (``<judge_output>`` schema).
    """
    out = text
    out = _repair_malformed_score_closing_tags(out)
    out = _repair_broken_evaluation_criterion_tags(out)
    if "large_text" not in out:
        return out
    # Prefer criterion when the document uses the default <evaluation> root (new template).
    if re.search(r"<evaluation\b", out, flags=re.IGNORECASE):
        return out.replace("<large_text>", "<criterion>").replace("</large_text>", "</criterion>")
    return out.replace("<large_text>", "<line_text>").replace("</large_text>", "</line_text>")


def _repair_bare_xml_ampersands_in_fragment(xml_fragment: str) -> str:
    """
    Escape ``&`` that are not the start of a well-formed XML entity or numeric character reference.

    Models often emit natural-language ``A & B`` inside element text or attributes; a bare ``&``
    is invalid XML and makes :func:`xml.etree.ElementTree.fromstring` raise :exc:`ParseError`.
    ⛔ Only XML's five PREDEFINED named entities are legal here: amp, lt, gt, quot,
    apos. ``xml.etree.ElementTree`` does not know HTML entities, so treating every
    ``&name;`` as valid (the previous behaviour) left ``&copy;`` untouched and
    ``fromstring()`` still raised ParseError. Numeric references are preserved.
    """
    return re.sub(
        r"&(?!(amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);)",
        "&amp;",
        xml_fragment,
    )


def _normalise_root_tag_case(xml_fragment: str) -> str:
    """Rewrite the fragment's outermost open/close tag to the canonical root casing.

    XML tag names are case-sensitive, so ``<Evaluation>...</evaluation>`` is simply
    not well-formed and :func:`xml.etree.ElementTree.fromstring` rejects it. The
    judge does emit mismatched casing occasionally, and matching the closing tag
    case-insensitively during extraction is not enough on its own -- the parse still
    fails. Normalising both tags here is what makes that recovery path real.

    Only the OUTERMOST pair is touched; inner elements are left exactly as-is so a
    genuinely malformed body still fails loudly instead of being papered over.
    """
    root = JUDGE_XML_ROOT_TAG
    # opening tag: <Evaluation ...>  ->  <evaluation ...>
    xml_fragment = re.sub(
        rf"\A(\s*<)\s*{root}(\b[^>]*>)",
        lambda m: m.group(1) + root + m.group(2),
        xml_fragment,
        count=1,
        flags=re.IGNORECASE,
    )
    # closing tag: </Evaluation>  ->  </evaluation>
    xml_fragment = re.sub(
        rf"(</)\s*{root}(\s*>\s*)\Z",
        lambda m: m.group(1) + root + m.group(2),
        xml_fragment,
        count=1,
        flags=re.IGNORECASE,
    )
    return xml_fragment


def _element_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_applicable_bool(raw: str) -> bool:
    t = str(raw or "").strip().lower()
    if t in ("true", "1", "yes"):
        return True
    if t in ("false", "0", "no", ""):
        return False
    return False


def _linked_link_texts(container: Optional[ET.Element], *, line_tag: str) -> List[str]:
    if container is None:
        return []
    return [_element_text(ch) for ch in list(container) if ch.tag == line_tag]


def _links_container_phrases(container: Optional[ET.Element], *, link_tag: str = JUDGE_XML_LINK) -> List[str]:
    """
    Collect checklist/rubric link phrases from ``<checklist_links>`` / ``<rubric_links>``.

    Accepts either multiple ``<link>`` children (legacy style) or plain text / newline-separated
    phrases inside the container element.
    """
    if container is None:
        return []
    link_texts = _linked_link_texts(container, line_tag=link_tag)
    if link_texts:
        return [t for t in (x.strip() for x in link_texts) if t]
    raw = _element_text(container)
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return lines
    return [raw.strip()]


def _checklist_item_criterion_text(ent: ET.Element) -> str:
    crit_el = ent.find(JUDGE_XML_CRITERION)
    if crit_el is None:
        crit_el = ent.find(LEGACY_JUDGE_XML_LINE_TEXT)
    return _element_text(crit_el)


def _visual_ref_block_from_evaluation_root(root: ET.Element) -> Optional[ET.Element]:
    vre = root.find(JUDGE_XML_VISUAL_REFERENCE)
    if vre is not None:
        return vre
    return root.find(LEGACY_JUDGE_XML_VISUAL_REF)


def _visual_ref_applicable_element(vre_root: ET.Element) -> Optional[ET.Element]:
    for tag in (JUDGE_XML_APPLICABLE, LEGACY_JUDGE_XML_APPLIES):
        el = vre_root.find(tag)
        if el is not None:
            return el
    return None


def _xml_attr_i_sort_key(el: ET.Element) -> int:
    raw = el.get(JUDGE_XML_I_ATTR)
    if raw is None or str(raw).strip() == "":
        return 0
    try:
        return int(str(raw).strip())
    except ValueError:
        return 0


def _xml_row_children_ordered(parent: Optional[ET.Element], row_tag: str) -> List[ET.Element]:
    if parent is None:
        return []
    rows = [c for c in list(parent) if c.tag == row_tag]
    if not rows:
        return []
    if all(
        c.get(JUDGE_XML_I_ATTR) is not None and str(c.get(JUDGE_XML_I_ATTR) or "").strip() != ""
        for c in rows
    ):
        return sorted(rows, key=_xml_attr_i_sort_key)
    return rows


def _root_local_tag(root: ET.Element) -> str:
    if "}" in root.tag:
        return str(root.tag).split("}", 1)[-1]
    return str(root.tag)


def _parse_evaluation_format_to_dict(root: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse ``<evaluation>`` (reference_alignment / checklist / rubric / visual_reference / text_reference)."""
    out: Dict[str, Any] = {
        "reference_criterion_alignment": [],
        "checklist_scores": [],
        "rubric_scores": {},
    }

    ral = root.find(JUDGE_XML_REFERENCE_ALIGNMENT)
    if ral is not None:
        for ent in _xml_row_children_ordered(ral, JUDGE_XML_ALIGNMENT_ITEM):
            ref_attr = ent.get(JUDGE_XML_REF_ATTR)
            ref_label = str(ref_attr).strip() if ref_attr else ""
            if not ref_label:
                ref_label_el = ent.find(LEGACY_JUDGE_XML_REF_LABEL)
                ref_label = _element_text(ref_label_el)
            cl_el = ent.find(JUDGE_XML_CHECKLIST_LINKS)
            rb_el = ent.find(JUDGE_XML_RUBRIC_LINKS)
            lcheck_legacy = ent.find(LEGACY_JUDGE_XML_LINKED_CL)
            lrub_legacy = ent.find(LEGACY_JUDGE_XML_LINKED_RUB)
            if cl_el is None and lcheck_legacy is not None:
                cl_el = lcheck_legacy
            if rb_el is None and lrub_legacy is not None:
                rb_el = lrub_legacy
            w_el = ent.find(JUDGE_XML_REASON)
            if w_el is None:
                w_el = ent.find(LEGACY_JUDGE_XML_REF_REASON)
            linked_cl = _links_container_phrases(cl_el)
            if not linked_cl and lcheck_legacy is not None:
                linked_cl = _linked_link_texts(lcheck_legacy, line_tag=JUDGE_XML_LINK)
            linked_rb = _links_container_phrases(rb_el)
            if not linked_rb and lrub_legacy is not None:
                linked_rb = _linked_link_texts(lrub_legacy, line_tag=JUDGE_XML_LINK)
            out["reference_criterion_alignment"].append(
                {
                    "reference_image": ref_label,
                    "linked_checklist_items": linked_cl,
                    "linked_rubric_dimensions": linked_rb,
                    "reason": _element_text(w_el),
                }
            )

    cs = root.find(JUDGE_XML_CHECKLIST)
    if cs is not None:
        for ent in _xml_row_children_ordered(cs, JUDGE_XML_CHECKLIST_ITEM):
            w_el = ent.find(JUDGE_XML_REASON)
            s_el = ent.find(JUDGE_XML_SCORE)
            out["checklist_scores"].append(
                {
                    "item": _checklist_item_criterion_text(ent),
                    "score": _element_text(s_el),
                    "reason": _element_text(w_el),
                }
            )
        if not out["checklist_scores"]:
            for ent in _xml_row_children_ordered(cs, LEGACY_JUDGE_XML_CL_ROW):
                text_el = ent.find(LEGACY_JUDGE_XML_LINE_TEXT)
                w_el = ent.find(JUDGE_XML_REASON)
                s_el = ent.find(JUDGE_XML_SCORE)
                out["checklist_scores"].append(
                    {
                        "item": _element_text(text_el),
                        "score": _element_text(s_el),
                        "reason": _element_text(w_el),
                    }
                )

    rub = root.find(JUDGE_XML_RUBRIC)
    if rub is not None:
        for ch in rub:
            if ch.tag not in (JUDGE_XML_DIMENSION, LEGACY_JUDGE_XML_RUBRIC_LINE):
                continue
            dim = ch.get(JUDGE_XML_NAME_ATTR)
            if not isinstance(dim, str) or not dim.strip():
                continue
            w_el = ch.find(JUDGE_XML_REASON)
            s_el = ch.find(JUDGE_XML_SCORE)
            out["rubric_scores"][dim.strip()] = {
                "score": _element_text(s_el),
                "reason": _element_text(w_el),
            }

    vre_root = _visual_ref_block_from_evaluation_root(root)
    if vre_root is not None:
        app_el = _visual_ref_applicable_element(vre_root)
        w_el = vre_root.find(JUDGE_XML_REASON)
        s_el = vre_root.find(JUDGE_XML_SCORE)
        out["visual_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    tre_root = root.find(JUDGE_XML_TEXT_REFERENCE)
    if tre_root is not None:
        app_el = _visual_ref_applicable_element(tre_root)
        w_el = tre_root.find(JUDGE_XML_REASON)
        s_el = tre_root.find(JUDGE_XML_SCORE)
        out["text_reference_evaluation"] = {
            "applicable": _parse_applicable_bool(_element_text(app_el)),
            "score": _element_text(s_el),
            "reason": _element_text(w_el),
        }

    if not out["rubric_scores"] and not out["checklist_scores"]:
        return None
    return out


def parse_judge_xml_to_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse judge XML into the common result shape (``reference_criterion_alignment``,
    ``checklist_scores``, ``rubric_scores``, ``visual_reference_evaluation``, ``text_reference_evaluation``) for
    :func:`parsed_judge_labeled_scores_03`.

    The root element is extracted with case-insensitive matching on the closing tag so a
    well-formed document such as ``<Evaluation>...</evaluation>`` still parses (the previous
    single-regex path required identical casing on the close tag).
    """
    cleaned = _preprocess_judge_xml_text_for_parse(text)
    fragment = _extract_first_judge_root_xml_fragment(cleaned)
    if not fragment:
        return None
    fragment = _repair_bare_xml_ampersands_in_fragment(fragment)
    # ⛔ Extracting the root case-insensitively is not enough on its own: XML tag
    # names ARE case-sensitive, so `<Evaluation>...</evaluation>` still raises
    # ParseError and this documented recovery path could never succeed (it just
    # burned a retry). Normalise the root open/close tags to the canonical casing
    # before parsing.
    fragment = _normalise_root_tag_case(fragment)
    try:
        root = ET.fromstring(fragment)
    except ET.ParseError:
        return None

    if _root_local_tag(root).casefold() != JUDGE_XML_ROOT_TAG.casefold():
        return None
    return _parse_evaluation_format_to_dict(root)


def try_parse_judge_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort parse of a judge message: try XML first, then JSON, including fenced
    blocks and brace-bounded JSON.
    """
    xml_parsed = parse_judge_xml_to_dict(text)
    if xml_parsed is not None:
        return xml_parsed
    return try_extract_json_obj(text)


def try_extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    candidate = text[start:end]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _score_to_float_03(x: Any, *, ctx: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"{ctx}: expected numeric score, got bool")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)", x.strip())
        if not m:
            raise ValueError(f"{ctx}: could not parse score from string {x!r}")
        v = float(m.group(1))
    else:
        raise TypeError(f"{ctx}: expected int/float/str score, got {type(x).__name__}: {x!r}")
    if v < 0.0 or v > 3.0:
        raise ValueError(f"{ctx}: score must be in [0, 3], got {v}")
    return v


def parsed_judge_labeled_scores_03(parsed: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Extract ordered (key, score) pairs from judge JSON or XML-parsed dicts.

    Keys are stable across samples that share the same rubric/checklist layout:
    ``checklist:{i}``, ``rubric:{dim}``, and optionally ``visual_reference_evaluation`` / ``text_reference_evaluation``.
    Scores are on the model's native ``[0, 3]`` scale. A rubric dimension is **skipped** when ``score`` is
    missing, empty, or legacy ``N/A``/``NA``; checklist scores must be present and non-empty; when
    ``visual_reference_evaluation.applicable`` or ``text_reference_evaluation.applicable`` is true, an empty score is skipped.
    """
    out: List[Tuple[str, float]] = []

    checklist = parsed.get("checklist_scores")
    if isinstance(checklist, list):
        for idx, item in enumerate(checklist):
            if not isinstance(item, dict):
                raise TypeError(
                    f"checklist_scores[{idx}] expected dict, got {type(item).__name__}: {item!r}"
                )
            if "score" not in item:
                raise KeyError(f"checklist_scores[{idx}] missing 'score' key: {item!r}")
            raw_chk = item["score"]
            if raw_chk is None or (isinstance(raw_chk, str) and raw_chk.strip() == ""):
                raise ValueError(
                    f"checklist_scores[{idx}].score must be a numeric value in [0, 3] (empty score not allowed for checklist); "
                    f"got {raw_chk!r}"
                )
            v = _score_to_float_03(raw_chk, ctx=f"checklist_scores[{idx}].score")
            out.append((f"checklist:{idx}", v))

    rubric_scores = parsed.get("rubric_scores")
    if rubric_scores is None:
        rubric_scores = {}
    if not isinstance(rubric_scores, dict):
        raise TypeError(
            f"rubric_scores expected dict, got {type(rubric_scores).__name__}: {rubric_scores!r}"
        )
    for dim, payload in rubric_scores.items():
        if not isinstance(payload, dict):
            raise TypeError(
                f"rubric_scores[{dim!r}] expected dict, got {type(payload).__name__}: {payload!r}"
            )
        if "score" not in payload:
            continue
        raw_score = payload["score"]
        if raw_score is None or (isinstance(raw_score, str) and raw_score.strip() == ""):
            continue
        if isinstance(raw_score, str) and raw_score.strip().upper() in {"N/A", "NA"}:
            continue
        v = _score_to_float_03(raw_score, ctx=f"rubric_scores[{dim}].score")
        out.append((f"rubric:{dim}", v))

    vre = parsed.get("visual_reference_evaluation")
    if isinstance(vre, dict) and vre.get("applicable") is True:
        if "score" in vre:
            raw_v = vre["score"]
            if raw_v is not None and not (isinstance(raw_v, str) and raw_v.strip() == ""):
                v = _score_to_float_03(raw_v, ctx="visual_reference_evaluation.score")
                out.append(("visual_reference_evaluation", v))

    tre = parsed.get("text_reference_evaluation")
    if isinstance(tre, dict) and tre.get("applicable") is True:
        if "score" in tre:
            raw_t = tre["score"]
            if raw_t is not None and not (isinstance(raw_t, str) and raw_t.strip() == ""):
                tsc = _score_to_float_03(raw_t, ctx="text_reference_evaluation.score")
                out.append(("text_reference_evaluation", tsc))

    if not out:
        raise ValueError(
            "parsed judge JSON produced no scores (empty checklist_scores and rubric_scores); "
            f"keys present: {list(parsed.keys())}"
        )
    return out


# ---------------------------------------------------------------------------
# Component aggregation — scoring.py::COMPONENTS + the exact scores->component
# mapping from scoring.py::aggregate (applied per-sample; the dataset class
# takes cross-sample means with the upstream default "skip" policy).
# ---------------------------------------------------------------------------
COMPONENTS = ["checklist", "adaptive_rubric", "prompt_faithfulness", "image_quality", "text_rendering",
              "ai_naturalness", "composition_and_aesthetics", "physical_plausibility", "visual_reference", "text_reference"]

_GENERIC_RUBRIC_DIMS = {
    "prompt_faithfulness", "image_quality", "text_rendering", "ai_naturalness",
    "composition_and_aesthetics", "physical_plausibility",
}


def component_means_from_scores(scores: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Per-sample component means (upstream scoring.py mapping, verbatim logic).

    Returns {component: mean or None-if-absent} on the native 0-3 scale.
    """
    mapping = {
        "checklist": lambda k: k.startswith("checklist:"),
        "adaptive_rubric": lambda k: k.startswith("rubric:") and k.split(":", 1)[1] not in _GENERIC_RUBRIC_DIMS,
        "visual_reference": lambda k: k == "visual_reference_evaluation",
        "text_reference": lambda k: k == "text_reference_evaluation",
    }
    out: Dict[str, Optional[float]] = {}
    for component in COMPONENTS:
        predicate = mapping.get(component, lambda k, c=component: k == f"rubric:{c}")
        matched = [float(v) for k, v in scores.items() if predicate(k)]
        out[component] = (sum(matched) / len(matched)) if matched else None
    return out


# ---------------------------------------------------------------------------
# Adapters for the mini-vlm-toolkit judge protocol (NOT upstream code):
# visual-context construction (mirrors evaluator.py::build_request) and
# flat role-message building. Image transport: identically-resized JPEG
# files (image_utils.py downscale, q95) passed as paths.
# ---------------------------------------------------------------------------

def prepare_judge_image(src_path: str, cache_dir: str,
                        max_total_pixels: int = 1_500_000, max_edge: int = 1568) -> str:
    """Downscale ``src_path`` exactly as upstream ``local_image_to_data_url``
    (<=1.5MP, <=1568 max edge, LANCZOS, RGB JPEG q95) and cache the result.

    Returns the cached JPEG path (stable name derived from the source path,
    so repeat judge runs reuse it).
    """
    import hashlib

    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(osp.abspath(src_path).encode('utf-8')).hexdigest()[:16]
    stem = osp.splitext(osp.basename(src_path))[0]
    dst = osp.join(cache_dir, f'{stem}.{key}.jpg')
    if osp.isfile(dst):
        return dst

    with Image.open(src_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_edge / max(width, height), (max_total_pixels / (width * height)) ** 0.5)
        if scale < 1.0:
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        # ⛔ Unique temp name per call: reference images are SHARED between samples,
        # so parallel graders (num-proc > 1) hit the same dst. With a fixed
        # '<dst>.tmp' one worker replaces the file another is still writing,
        # producing intermittent save failures or a truncated cache entry.
        fd, tmp = tempfile.mkstemp(prefix=osp.basename(dst) + '.', suffix='.tmp',
                                   dir=osp.dirname(dst) or '.')
        os.close(fd)
        try:
            image.save(tmp, format='JPEG', quality=95)
            os.replace(tmp, dst)          # atomic within the same directory
        except BaseException:
            if osp.exists(tmp):
                os.unlink(tmp)
            raise
    return dst


def build_visual_context_from_slots(reference_slots: List[Dict[str, Any]],
                                    ref_paths: List[str],
                                    text_knowledge_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror upstream ``evaluator.py::build_request`` context construction.

    ``reference_slots`` are the eval_metadata ``visual_reference_slots`` and
    ``ref_paths`` the (resized) local image files, one per slot, used here as
    the slot identifiers (upstream uses base64 data URLs for the same role).
    """
    candidates = []
    for index, (slot, path) in enumerate(zip(reference_slots, ref_paths)):
        candidates.append({
            "candidate_index": index, "entity": slot.get("entity"), "severity": slot.get("severity"),
            "reasoning": slot.get("reasoning"), "suggested_search": None, "search_type": None,
            "matched_search_results": [{
                "query": slot.get("query"), "image_url": path,
                "local_path": path, "selected_image_title": slot.get("title"), "source_file": None,
            }],
        })
    return {
        "has_critical_references": bool(ref_paths),
        "critical_candidate_indices": list(range(len(ref_paths))),
        "critical_candidates": candidates,
        "eval_text_knowledge_slots": text_knowledge_slots or [],
        "has_text_knowledge_gaps_for_eval": bool(text_knowledge_slots),
    }


def build_judge_searchgen_messages(
    *,
    sample_id: str,
    user_prompt: str,
    verification_checklist: List[Any],
    evaluation_rubric: Dict[str, Any],
    reference_slots: List[Dict[str, Any]],
    ref_image_paths: List[str],
    text_knowledge_slots: List[Dict[str, Any]],
    assess_image_path: str,
) -> list:
    """Return the flat role-message list for one judge call.

    Layout matches upstream ``build_judge_interleaved_user_content``: full
    text block, then per reference (context text + image), then
    ``Image to assess:`` + target image. Images are pre-resized paths.
    """
    visual_context = build_visual_context_from_slots(
        reference_slots, ref_image_paths, text_knowledge_slots)
    judge_row = {
        "user_prompt": user_prompt, "sample_id": sample_id,
        "augmented_generation_details": {
            "used_reference_images_count": len(ref_image_paths),
            "selected_reference_images_count": len(ref_image_paths),
            "expected_reference_images_count": len(ref_image_paths),
        },
    }
    prompt = build_eval_prompt_text(
        row=judge_row, verification_checklist=verification_checklist,
        evaluation_rubric=evaluation_rubric, visual_context=visual_context,
        variant="augmented", reference_slot_urls=ref_image_paths,
    )
    url_to_meta = build_reference_image_metadata_map(visual_context=visual_context)

    items: list = [
        {"role": "system", "type": "text", "value": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "type": "text", "value": prompt},
    ]
    for i, ref_path in enumerate(ref_image_paths):
        ctx_text = interleaved_ref_context_block(
            slot_index=i, visual_context=visual_context,
            url_to_meta=url_to_meta, ref_slot_url=str(ref_path),
        )
        items.append({"role": "user", "type": "text", "value": ctx_text})
        items.append({"role": "user", "type": "image", "value": ref_path})
    items.append({"role": "user", "type": "text", "value": "Image to assess:"})
    items.append({"role": "user", "type": "image", "value": assess_image_path})
    return items
