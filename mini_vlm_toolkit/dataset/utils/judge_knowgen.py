"""Judge_KnowGen — verbatim port of the Gen-Searcher KnowGen-Bench paper grader.

KnowGen-Bench is a knowledge-grounded text-to-image benchmark introduced in
Gen-Searcher (arXiv 2603.28767). The grader compares a generated image
against a ground-truth reference image (both accessed via file path) using
GPT-4.1 (default; overridable) as a multimodal judge. Four dimensions per
sample, each in {0, 0.5, 1}:

  - ``faithfulness``       — overall prompt adherence (presence & structure)
  - ``visual_correctness`` — GT visual-feature agreement (strict)
  - ``text_accuracy``      — required readable text (with NA flag)
  - ``aesthetics``         — composition & polish vs GT

Plus:
  - ``text_accuracy_na``  — boolean; true when the prompt required no readable text
  - ``rationale``         — 5-10 sentence justification

We port the SYSTEM_PROMPT and the per-sample user message construction
byte-exact from ``Gen-Searcher/KnowGen_Eval/gpt_eval_knowgen.py``. The judge
is invoked per-sample via the standard mini-vlm-toolkit judge protocol.

Aggregate scoring: our framework accepts the per-dim scores and treats
``correct`` as `1` if the mean of the four normalised dims >= 0.5, else 0.
The raw four scores are surfaced in ``verdict`` for downstream analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# System prompt — Gen-Searcher release (gpt_eval_knowgen.py SYSTEM_PROMPT),
# byte-exact. ACTIVE.
# ---------------------------------------------------------------------------
JUDGE_KNOWGEN_SYSTEM_PROMPT = r"""You are a strict and professional expert evaluator for AI-generated image grounded with world knowledge (MODEL EVALUATION).

You will receive:
1) A task prompt (what the image must show).
2) Image 1: the generated image (model output to be evaluated).
3) Image 2: the ground-truth reference image (a strong reference implementation).

All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

Critical clarification (VERY IMPORTANT):
- This is NOT a pixel-level similarity task.
- Image 2 (GT) is a REFERENCE for intended identity, key grounded details, and stable visual attributes.
  Image 1 may use a different camera angle/layout as long as it still satisfies the prompt.
- Focus on whether prompt-required, externally-checkable (search-grounded) details are correctly AND verifiably realized in Image 1.
- Do NOT assume correctness if a key detail is not clearly visible/readable. If unverifiable, score lower.

Output format (MUST follow exactly):
Output ONLY one valid JSON object with EXACTLY these keys:
{
  "rationale": string,
  "faithfulness": number,
  "visual_correctness": number,
  "text_accuracy": number,
  "aesthetics": number,
  "text_accuracy_na": boolean
}
SCORING SCALE (VERY IMPORTANT):
- Each score MUST be exactly one of: 0, 0.5, 1
- 1 (Exemplary) is rare and requires perfect success for that dimension.
- 0.5 (Conditional) means mostly correct but not perfect.
- 0 (Rejected) means failed on important requirements.

- "rationale" must be 5-10 short sentences, evidence-based, referring only to what is visible.
- "text_accuracy_na" should be true if the prompt does not require any readable text, otherwise it should be false.

Implicit required step (ENFORCED via rationale):
- In the rationale, you MUST explicitly list the extracted prompt hard constraints (2-5, or more if needed) BEFORE scoring.
  If you cannot identify the constraints, you must still list what you believe are the hard constraints.

Evaluation procedure (follow silently, but the rationale MUST reflect it):
1) Extract the prompt's TOP hard constraints (2-5, or more if needed): required subjects/identities, setting/props,
   relations/counts, required style, and any externally-checkable requirements (readable text/landmark/logo/badge/version/year/etc.).
2) Score Image 1 against the constraints. Use Image 2 only as a reference for stable identity/visual attributes and grounded evidence.
3) If a key requirement is not verifiable (too small/blurred/occluded/warped), do NOT assume it is correct; score lower.
4) Assessment of the primary subjects' visual identity correctness and consistency is mandatory in every case.

Boundary between visual_correctness vs text_accuracy:
- Visual-only grounded cues (subject visual features, logo SHAPE, badge EMBLEM geometry, landmark facade/massing, outfit/weapon silhouette, object geometry)
  belong to visual_correctness.
- Any grounded cue that must be READ as text (spelling, year numbers, titles, institution names, badge text) belongs to text_accuracy.

==========================
STRICT 3-LEVEL RUBRICS
(Each dimension uses ONLY {0, 0.5, 1})
==========================

1) faithfulness (overall prompt adherence: presence & structure only; not GT-identity correctness):
- This score does NOT require matching GT's exact identity or fine-grained visual features; it focuses on whether Image 1 includes the prompt-requested elements and scene structure (who/what is present, what is happening, where it happens, and the required style/format).

(Exemplary) Score = 1 ONLY IF:
- Image 1 clearly includes everything the prompt asks for in terms of visible content and structure:
  all required subjects/entities are present, the required setting and key props appear,
  required actions/relations/counts are shown, and the required style/format is followed.
- Any required in-scene evidence elements requested by the prompt (e.g., a plaque/sign, a map, a report paper, a badge) are present as elements.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 includes almost all prompt-requested content and structure, with only minor omissions or minor staging differences
  that do not change what the scene is supposed to depict (e.g., small placement differences, slight simplification of a secondary prop).

(Rejected) Score = 0 IF:
- One or more prompt-requested essential elements are not shown at all, or the scene structure clearly does not match the prompt's request
  (e.g., missing a required subject/entity, missing the required setting, missing the required key prop/evidence element,
  missing the requested action/relationship/count, or not following the requested style/format).

2) visual_correctness (GT visual-feature agreement is the core; extremely strict):
(Exemplary) Score = 1 ONLY IF:
- The prompt-required primary subjects/objects in Image 1 match the GT reference (Image 2) in visual characteristics
  with NO substantive changes.
- This means: the same face/hairstyle silhouette, the same armor/clothing design and key colors/patterns,
  the same distinctive props/object geometry, the same emblem/logo/landmark facade/massing cues when applicable, etc.
- Any meaningful difference in these stable visual features disqualifies a score of 1.

(Conditional) Score = 0.5 ONLY IF:
- Image 1 can still be considered the same overall visual instance as the GT, and the differences are limited to relatively minor variations, allowing some changes to the visual features (face, hairstyle, armor design, key colors/patterns, key prop shapes), while the overall identity and major visual features remain recognizable and broadly consistent.
- IMPORTANT: "same role archetype" (generic knight/princess/warrior) alone does NOT qualify for 0.5.

(Rejected) Score = 0 IF:
- Any substantive mismatch vs GT in stable visual features of the required subjects/objects
  (different face/hair/armor design/color scheme/emblem/prop geometry/landmark cues),
  even if the overall scene still looks plausible or stylistically similar.

3) text_accuracy (required readable text; ALL relevant text must be correct AND very clearly readable; NO partial credit for wrong text):
Rule:
- If the prompt does NOT require any readable text: you MUST output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON. In your rationale state that the prompt did not require readable text.
- If the prompt DOES require readable text: output "text_accuracy_na": false and score "text_accuracy" (0, 0.5, or 1) per the criteria below.

(Exemplary) Score = 1 ONLY IF:
- ALL required text AND any prompt-involved text elements are:
  (a) present,
  (b) very clearly readable (crisp, unambiguous),
  (c) correct and consistent with the prompt's requirements.
(Conditional) Score = 0.5 ONLY IF:
- Much of the required/prompt-involved text is readable and generally correct, and although parts may contain inaccuracies, omissions, or deviations, the overall meaning remains clear and is not seriously inconsistent with the prompt requirements.
(Rejected) Score = 0 IF:
- Any required/prompt-involved text is missing, unclear, not very readable, gibberish, placeholder, OR incorrect.
- Even if perfectly readable, if content is not correct, text_accuracy MUST be 0.

4) aesthetics:
(Exemplary) Score = 1 ONLY IF:
- Masterpiece-level composition and polish, AND Image 1 is NOT worse than GT in overall aesthetic quality.
(Conditional) Score = 0.5 ONLY IF:
- Very beautiful and polished, but slightly worse than GT (ONLY slightly) OR slightly less refined than top-tier.
(Rejected) Score = 0 IF:
- Anything clearly worse than GT in a noticeable way, OR merely average/OK-looking, OR cluttered/awkward framing,
  OR visible artifacts/noise that harm the overall appeal.

Rationale requirements (MANDATORY):
- Start with: "Constraints:" and list the extracted constraints (2-5, or more if needed).
- State whether the prompt required readable text; if not required, output "text_accuracy_na": true and "text_accuracy": 0.5 in the JSON and say so in the rationale.
- Mention 2-5 key comparisons (or more if needed) to GT focused on stable identity/visual traits (NOT demanding identical layout).
- Keep within 10 sentences.

Output JSON only. No markdown. No extra text."""


# ---------------------------------------------------------------------------
# Message building
#
# Image transport: we pass file PATHS using the framework's flat role-message
# schema (``api/base.py::preprocess_message_with_role`` — each item is
# ``{role, type, value}``); the API wrapper base64-encodes them itself.
# Upstream sent JPEG-q100 data-URLs (max side 4096); content is equivalent,
# only the transport codec differs. Datasets should pass
# ``img_detail='high'`` to ``build_judge`` so the judge sees full detail.
# ---------------------------------------------------------------------------
_SCORE_KEYS = ("faithfulness", "visual_correctness", "text_accuracy", "aesthetics")


def build_judge_knowgen_user_message(
    sample_id: str,
    prompt: str,
    generated_image_path: str,
    gt_image_path: str,
) -> list:
    """Build the user-message items (framework flat role schema).

    Text byte-exact to Gen-Searcher's ``_build_user_message``.
    """
    import os.path as osp

    text = (
        f"Sample id: {sample_id}\n\n"
        f"Task prompt (image requirement):\n{prompt}\n\n"
        "Image 1: the **generated image** (to be evaluated).\n"
        "Image 2: the **ground-truth reference image**.\n\n"
        "Output a single JSON object with the three scores and a short rationale."
    )
    items: list = [{"role": "user", "type": "text", "value": text}]
    for p in [generated_image_path, gt_image_path]:
        if p and osp.isfile(p):
            items.append({"role": "user", "type": "image", "value": p})
        else:
            items.append({
                "role": "user", "type": "text",
                "value": f"\n[WARN] failed to load image: {p}, err=file not found\n",
            })
    return items


# ---------------------------------------------------------------------------
# Parsing — tolerant JSON extractor (matches upstream _parse_llm_json intent)
# ---------------------------------------------------------------------------

def _clip01(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def parse_judge_knowgen_judgement(content: str) -> Dict[str, Any]:
    """Parse the JSON response from the KnowGen judge model.

    Returns dict with:
      - ``verdict``: parsed dict from LLM (or {} on failure)
      - ``correct``: 1 if mean-of-four >= 0.5 else 0
      - ``mean_score``: mean of the four dims in [0, 1]
      - ``faithfulness``, ``visual_correctness``, ``text_accuracy``, ``aesthetics``
      - ``text_accuracy_na``: bool
      - ``rationale``: str
      - ``reasoning``: 1-line summary derived from rationale + scores
    """
    text = (content or "").strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Attempt to locate a JSON object.
    obj: Optional[dict] = None
    try:
        obj = json.loads(text)
    except Exception:
        # Fallback 1: greedy match of the first {...} block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            snippet = m.group(0)
            # Tolerate trailing commas before } or ]
            snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                obj = json.loads(snippet)
            except Exception:
                obj = None

    # Fallback 2: JSON is malformed (e.g. an unescaped newline/quote inside the
    # rationale string — a common LLM mistake). Rather than throwing the whole
    # verdict away and scoring 0 (which silently DEFLATES the benchmark), extract
    # the four dimension scores + rationale directly with regex. Only used when
    # JSON parsing failed; a normal well-formed response never reaches here.
    if not isinstance(obj, dict):
        salvaged: dict[str, Any] = {}
        for k in _SCORE_KEYS:
            mm = re.search(rf'"{k}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
            if mm:
                salvaged[k] = float(mm.group(1))
        na = re.search(r'"text_accuracy_na"\s*:\s*(true|false)', text, re.IGNORECASE)
        if na:
            salvaged["text_accuracy_na"] = na.group(1).lower() == "true"
        rat = re.search(r'"rationale"\s*:\s*"(.*?)"\s*(?:,\s*"|\})', text, re.DOTALL)
        if rat:
            salvaged["rationale"] = rat.group(1).replace('\\n', '\n').replace('\\"', '"')
        # Accept the salvage only if we recovered at least the four scores, so we
        # never fabricate a passing verdict from noise.
        if all(k in salvaged for k in _SCORE_KEYS):
            obj = salvaged

    if not isinstance(obj, dict):
        return dict(
            verdict={},
            correct=0,
            mean_score=0.0,
            faithfulness=0.0,
            visual_correctness=0.0,
            text_accuracy=0.0,
            aesthetics=0.0,
            text_accuracy_na=False,
            rationale="",
            reasoning="judge failed to parse",
        )

    scores = {k: _clip01(obj.get(k, 0.0)) for k in _SCORE_KEYS}
    text_accuracy_na = bool(obj.get("text_accuracy_na", False))
    rationale = str(obj.get("rationale", ""))
    mean_score = sum(scores.values()) / len(scores)
    correct = 1 if mean_score >= 0.5 else 0

    reasoning = (
        f"F={scores['faithfulness']:.1f} "
        f"V={scores['visual_correctness']:.1f} "
        f"T={scores['text_accuracy']:.1f}"
        f"{' (NA)' if text_accuracy_na else ''} "
        f"A={scores['aesthetics']:.1f} "
        f"| mean={mean_score:.2f}"
    )

    return dict(
        verdict=obj,
        correct=correct,
        mean_score=mean_score,
        faithfulness=scores["faithfulness"],
        visual_correctness=scores["visual_correctness"],
        text_accuracy=scores["text_accuracy"],
        aesthetics=scores["aesthetics"],
        text_accuracy_na=text_accuracy_na,
        # Store the FULL rationale (5-10 evidence-based sentences). No truncation:
        # this is analysis data and must be preserved intact.
        rationale=rationale,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Convenience: build the ChatML messages the judge model consumes.
# ---------------------------------------------------------------------------

def build_judge_knowgen_messages(
    sample_id: str,
    prompt: str,
    generated_image_path: str,
    gt_image_path: str,
) -> list:
    """Return the flat role-message list for a single judge call."""
    return [
        {"role": "system", "type": "text", "value": JUDGE_KNOWGEN_SYSTEM_PROMPT},
        *build_judge_knowgen_user_message(
            sample_id=sample_id,
            prompt=prompt,
            generated_image_path=generated_image_path,
            gt_image_path=gt_image_path,
        ),
    ]
