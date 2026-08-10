"""Judge_T2IFactualBench — port of the T2I-FactualBench multi-round VQA grader.

T2I-FactualBench (arXiv 2404.19467, "Benchmarking the Factuality of
Text-to-Image Models with Knowledge-Intensive Concepts") uses a 3-layer
evaluation pipeline structured around three prompt families:

  1. concept factuality  → 0-4 score from 4 sub-criteria
                           (Shape / Color / Texture / Feature Details);
                           applied to every sample regardless of tier
  2. task completeness   → binary 0/1;
                           applied to SKCI (action / attribute /
                           background variation) and MKCC (interaction /
                           size / differentiating) tiers
  3. integration factuality → binary 0/1;
                              applied to MKCC only (interaction /
                              differentiating tasks)

Original prompts live in
``T2I-FactualBench/eval_multi_round/prompt_process.py`` — this file
ports the essential prompt strings + parsers, and the tier-aware
metric formulas from the paper.

Metric aggregation (normalized to [0, 1] per our interpretation for
cross-tier comparability):
  - SKCM: Concept_Score / 4
  - SKCI: (Concept_Score / 4 + Task_Score) / 2
  - MKCC: (Concept_Score / 4 + Task_Score + Integration_Score) / 3

⚠️ Documented deviation — integration factuality (MKCC only):
the integration sub-score here is a binary 0/1 rubric of our own design
(positioning / scene consistency / interaction plausibility), NOT a verbatim
port of upstream's integration prompt. Only the concept judge is a 0-4
``Total Rating`` normalized by 4. Consequently MKCC numbers produced by this
grader are NOT directly comparable to the paper's MKCC column; use them for
relative comparison between our own runs. Reconciling this against
``eval_multi_round/prompt_process.py`` is tracked as follow-up work.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Feature-details table — per T2I-FactualBench/eval_multi_round/prompt_process.py
# Category → what to focus on for "Feature Details" sub-score.
# ---------------------------------------------------------------------------
_FEATURE_DETAILS_ANIMAL_LIKE = (
    "Focus on facial details, limbs, and skin texture to ensure they "
    "align with the natural or expected representation of the {concept}."
)
_FEATURE_DETAILS_FUNCTIONAL = (
    "Focus on functional components, material details, and decorative "
    "elements to ensure they align with the natural or expected "
    "representation of the {concept}."
)
_FEATURE_DETAILS_ARCHITECTURE = (
    "Examine the {concept}'s architectural details, unique decorations, "
    "and structural features, ensuring key elements and iconic details "
    "are precisely represented with correct proportions, symmetry, "
    "spatial layout and accurately reflect the original design."
)
_FEATURE_DETAILS_EVENT = (
    "Evaluate the {concept}'s portrayal by examining the historical "
    "accuracy and representation of key figures, attire, and iconic "
    "scenes within the context of the event. Ensure that the visual "
    "narrative accurately depicts significant moments and the overall "
    "atmosphere, maintaining fidelity to documented historical accounts "
    "and cultural settings."
)

_FEATURE_DETAILS_MAP: Dict[str, str] = {}
for k in ("animal", "pet", "person", "plant"):
    _FEATURE_DETAILS_MAP[k] = _FEATURE_DETAILS_ANIMAL_LIKE
for k in (
    "hat", "bag", "car", "clothes", "food", "music_instrument",
    "electronic", "sport_equipment", "other", "music", "sport", "artifact",
):
    _FEATURE_DETAILS_MAP[k] = _FEATURE_DETAILS_FUNCTIONAL
for k in ("celestial", "landmark", "natural_landform", "location"):
    _FEATURE_DETAILS_MAP[k] = _FEATURE_DETAILS_ARCHITECTURE
_FEATURE_DETAILS_MAP["event"] = _FEATURE_DETAILS_EVENT


def _feature_details_for(type_: str, concept: str) -> str:
    template = _FEATURE_DETAILS_MAP.get(type_, _FEATURE_DETAILS_FUNCTIONAL)
    return template.format(concept=concept)


# ---------------------------------------------------------------------------
# Prompt: concept factuality (all tiers)
# ---------------------------------------------------------------------------

def concept_prompt(concept: str, type_: str, sentence: str) -> str:
    """Build the concept-factuality judge prompt.

    Byte-adjacent to ``concept_prompt_process`` in prompt_process.py:4-58.
    Two images will be sent alongside: <image> = generated image,
    <image> = concept reference image.
    """
    feature_details = _feature_details_for(type_, concept)
    return (
        f"You are provided with a concept image and a generated image. "
        f"First, learn the concept of {concept} from the concept image. "
        f"Then evaluate the {concept} in the generated image based on "
        f"your own knowledge of {concept} and the concept of {concept} "
        f"you learned from the concept image.\n\n"
        f"Concept Evaluation:\n"
        f"Evaluate whether the {concept} in the generated image matches "
        f"the conceptual characteristics of the {concept} based on:\n"
        f"1. Shape Accuracy: Focus on the overall outline and structure "
        f"of the generated {concept}. Assess whether the overall "
        f"silhouette, pose, and proportions align with the common shapes "
        f"associated with the concept.\n"
        f"2. Color Accuracy: Assess whether the generated {concept}'s "
        f"color scheme and lighting conditions align with the natural or "
        f"expected hues, saturation, and brightness characteristic of "
        f"the concept.\n"
        f"3. Texture Representation: Evaluate the realism and clarity of "
        f"the {concept}'s textures, ensuring authentic representation in "
        f"key areas, free from blurriness, pixelation, or artificial "
        f"effects, to uphold realistic integrity.\n"
        f"4. Feature Details: {feature_details}\n\n"
        f"Notice that: You should only consider the concept of "
        f"{concept} in the generated image.\n"
        f"Evaluation Criteria:\n"
        f"Evaluate based on the criteria above. Score from 0 to 4 based "
        f"on how many criteria are fully met:\n"
        f"- 0: None met\n"
        f"- 1: One met\n"
        f"- 2: Two met\n"
        f"- 3: Three met\n"
        f"- 4: All met\n"
        f"Explain your rating by providing a very brief core reason. "
        f"Keep your explanation concise and to the point.\n"
        f"Input:\n"
        f"The generated image <image>. Caption of the generated image "
        f"is {sentence}.\n"
        f"The concept image <image>. Caption of the concept image is "
        f"{concept}.\n\n"
        f"Output Format(Notice that output must be without any bold "
        f"formatting):\n"
        f"Total Rating: Your Rating\n"
        f"Shape Accuracy: 0/1 Reason:\n"
        f"Color Accuracy: 0/1 Reason:\n"
        f"Texture Representation: 0/1 Reason:\n"
        f"Feature Details: 0/1 Reason:\n"
    )


# ---------------------------------------------------------------------------
# Prompt: task completeness (SKCI: action / attribute / background variation)
# ---------------------------------------------------------------------------

def task_prompt_skci(
    task_type: str,
    concept: str,
    type_: str,
    addition: str,
) -> str:
    """Build the SKCI task-completeness prompt.

    task_type ∈ {"action", "attribute", "background variation", "variation"}.
    """
    if task_type == "action":
        return (
            f"You are provided with a generated image, its corresponding "
            f"caption, a knowledge concept {type_} {concept}, and an "
            f"action: {addition}. Your task is:\n"
            f"1. Confirm Concept Presence: Check whether {concept} is "
            f"present in the image. If it is not present, the Task Score "
            f"should be 0.\n"
            f"2. Confirm Action: If {concept} is present, determine "
            f"whether the {type_} {concept} in the generated image is "
            f"either dynamically performing the action of {addition} or "
            f"is being dynamically acted upon. If either of these "
            f"conditions is true, the Task Score should be 1; if neither "
            f"is true, the Task Score should be 0.\n"
            f"Evaluation Order: Please first confirm the presence of "
            f"the concept, then verify the action of knowledge concept.\n\n"
            f"Output Format (Note: the output must be without any bold "
            f"formatting):\n"
            f"Task Score:\n"
            f"Reason:\n"
        )
    if task_type == "attribute":
        return (
            f"You will be provided with a generated image, a knowledge "
            f"concept {type_} {concept}, and an attribute related to "
            f"concept: {addition}. Your task is:\n"
            f"1. Confirm Concept Presence: Check whether {concept} is "
            f"present in the image. If it is not present, the Task Score "
            f"should be 0.\n"
            f"2. Confirm Attribute Change: If {concept} is present, "
            f"assess whether the knowledge concept exhibits the "
            f"attribute {addition}. If it does, the Task Score should be "
            f"1; otherwise, the Task Score should be 0.\n"
            f"Evaluation Order: Please first confirm the presence of "
            f"the concept, then verify the knowledge concept attribute.\n"
            f"Output Format (Note: do not use any bold formatting):\n"
            f"Task Score:\n"
            f"Reason:\n"
        )
    # background variation / variation
    if task_type not in ("background variation", "variation"):
        # Do NOT silently fall through: an unrecognised task type would be
        # graded with the wrong rubric and produce a plausible-looking but
        # meaningless benchmark number. Fail loudly instead.
        raise ValueError(
            f'unknown SKCI task type {task_type!r}; expected one of '
            '"action", "attribute", "background variation", "variation"'
        )
    return (
        f"You will be provided with a generated image, a background "
        f"knowledge concept {type_} {concept}, and an attribute related "
        f"to background: {addition}. Your task is:\n"
        f"1. Confirm Concept Presence: Check whether {concept} is "
        f"present in the image. If it is not present, the Task Score "
        f"should be 0.\n"
        f"2. Confirm Background Attribute: If {concept} is present, "
        f"assess whether the related background exhibits the attribute "
        f"{addition}. If it does, the Task Score should be 1; otherwise, "
        f"the Task Score should be 0.\n"
        f"Evaluation Order: Please first confirm the presence of the "
        f"concept, then verify its background attribute.\n\n"
        f"Output Format (Note: do not use any bold formatting):\n"
        f"Task Score:\n"
        f"Reason:\n"
    )


# ---------------------------------------------------------------------------
# Prompt: task completeness (MKCC: interaction / size / differentiating)
# ---------------------------------------------------------------------------

def task_prompt_mkcc(
    task_type: str,
    concepts: List[str],
    types: List[str],
    additions: List[str],
) -> str:
    """Build the MKCC task-completeness prompt.

    task_type ∈ {"interaction", "size", "differentiating"}.
    concepts / types have len 2; additions varies by task.
    """
    c1, c2 = concepts[0], concepts[1]
    t1, t2 = types[0], types[1]

    if task_type == "interaction":
        interaction = additions[0] if additions else "interacting"
        return (
            f"You are provided with a generated image and two knowledge "
            f"concepts: {t1} {c1} and {t2} {c2}. Additionally, an "
            f"interaction action is specified: {interaction}. Your task "
            f"is to:\n"
            f"1. Confirm Presence: check whether both {c1} and {c2} are "
            f"present in the image. If either is missing, the Task Score "
            f"should be 0.\n"
            f"2. Interaction Verification: determine whether they are "
            f"dynamically interacting through the specified interaction "
            f"of {interaction}. If this interaction is occurring, the "
            f"Task Score should be 1; if not, the Task Score should be 0.\n"
            f"Evaluation Order: Please first confirm the presence of "
            f"the concepts, then proceed to verify their interaction.\n\n"
            f"Output Format (Note: do not use any bold formatting):\n"
            f"Task Score:\n"
            f"Reason:\n"
        )
    if task_type == "size":
        # Per prompt_process.py: additions[0] is expected relative size
        # for concept_1 ("large" vs "small" etc.).
        rel = additions[0] if additions else "larger"
        return (
            f"You are provided with a generated image and two knowledge "
            f"concepts: {t1} {c1} and {t2} {c2}. Your task is to:\n"
            f"1. Confirm Presence: check whether both {c1} and {c2} are "
            f"present in the image. If either is missing, the Task Score "
            f"should be 0.\n"
            f"2. Size Restoration: if both are present, assess whether "
            f"their size proportions in the image match their real-world "
            f"counterparts. Estimate the relative volume, height, or "
            f"recognizable size ratio of each concept in the image to "
            f"ensure they reflect the real-world size differences. If "
            f"{c1} is indeed {rel} than {c2} as expected, the Task Score "
            f"should be 1; otherwise, it should be 0.\n\n"
            f"Output Format (Note: do not use any bold formatting):\n"
            f"Task Score:\n"
            f"Reason:\n"
        )
    # differentiating
    if task_type != "differentiating":
        # Same reasoning as task_prompt_skci: a wrong rubric silently produces
        # a meaningless score, so reject unknown task types.
        raise ValueError(
            f'unknown MKCC task type {task_type!r}; expected one of '
            '"interaction", "size", "differentiating"'
        )
    a1 = additions[0] if len(additions) > 0 else ""
    a2 = additions[1] if len(additions) > 1 else ""
    return (
        f"You are provided with a generated image and two knowledge "
        f"concepts: {t1} {c1} (attribute: {a1}) and {t2} {c2} (attribute: "
        f"{a2}). Your task is to:\n"
        f"1. Confirm Presence: check whether both {c1} and {c2} are "
        f"present in the image. If either is missing, the Task Score "
        f"should be 0.\n"
        f"2. Attribute Differentiation: if both are present, verify that "
        f"{c1} clearly exhibits attribute {a1} AND {c2} clearly exhibits "
        f"attribute {a2}. Only if both attribute checks pass, the Task "
        f"Score is 1; otherwise, 0.\n\n"
        f"Output Format (Note: do not use any bold formatting):\n"
        f"Task Score:\n"
        f"Reason:\n"
    )


# ---------------------------------------------------------------------------
# Prompt: integration factuality (MKCC only)
# ---------------------------------------------------------------------------

def integration_prompt(concepts: List[str], types: List[str], sentence: str) -> str:
    c1, c2 = concepts[0], concepts[1]
    t1, t2 = types[0], types[1]
    return (
        f"You are provided with a generated image whose caption is: "
        f"{sentence}\n"
        f"It should contain two knowledge concepts: {t1} {c1} and {t2} "
        f"{c2}. Your task is to evaluate the INTEGRATION FACTUALITY — "
        f"whether the two concepts are composed in a physically and "
        f"semantically coherent way:\n"
        f"1. Relative positioning (are they placed together sensibly?)\n"
        f"2. Scene consistency (do lighting / style / perspective match "
        f"between the two)?\n"
        f"3. Interaction plausibility (does the co-appearance look "
        f"natural, not photoshopped or discontinuous)?\n"
        f"Score:\n"
        f"- 1 if all three criteria are broadly satisfied.\n"
        f"- 0 if any criterion fails (jarring size mismatch, mismatched "
        f"lighting, one concept looks pasted-in, etc.).\n\n"
        f"Output Format (no bold formatting):\n"
        f"Integration Score:\n"
        f"Reason:\n"
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_concept_rating(content: str) -> Dict[str, Any]:
    """Extract Total Rating (0-4) from the concept-judge output."""
    text = content or ""
    m = re.search(r"Total\s*Rating\s*[::]?\s*([0-4])", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        return dict(concept_score=s, raw=text, reasoning=f"Total Rating: {s}")
    # Fallback: bare "Rating: N" or "Score: N"
    m = re.search(r"(?:Rating|Score)\s*[::]?\s*([0-4])", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        return dict(concept_score=s, raw=text, reasoning=f"Rating: {s}")
    return dict(concept_score=0, raw=text[:500], reasoning="unparseable concept rating")


def parse_task_score(content: str) -> Dict[str, Any]:
    """Extract Task Score (0/1) from the task-judge output."""
    text = content or ""
    m = re.search(r"Task\s*Score\s*[::]?\s*([01])", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        return dict(task_score=s, raw=text, reasoning=f"Task Score: {s}")
    return dict(task_score=0, raw=text[:500], reasoning="unparseable task score")


def parse_integration_score(content: str) -> Dict[str, Any]:
    text = content or ""
    m = re.search(r"Integration\s*Score\s*[::]?\s*([01])", text, re.IGNORECASE)
    if m:
        s = int(m.group(1))
        return dict(integration_score=s, raw=text, reasoning=f"Integration Score: {s}")
    return dict(integration_score=0, raw=text[:500], reasoning="unparseable integration score")


# ---------------------------------------------------------------------------
# Metric aggregation (paper formulas, normalized to [0, 1])
# ---------------------------------------------------------------------------

def metric_for_tier(
    tier: str,
    concept_score: int,
    task_score: Optional[int],
    integration_score: Optional[int],
) -> float:
    """Compute the paper metric for one sample.

    Normalized to [0, 1] range for cross-tier comparison:
      - SKCM: Concept / 4
      - SKCI: (Concept / 4 + Task) / 2
      - MKCC: (Concept / 4 + Task + Integration) / 3
    """
    c = max(0, min(4, int(concept_score))) / 4.0
    if tier == "SKCM":
        return c
    t = max(0, min(1, int(task_score or 0)))
    if tier == "SKCI":
        return (c + t) / 2.0
    i = max(0, min(1, int(integration_score or 0)))
    if tier == "MKCC":
        return (c + t + i) / 3.0
    return c


# ---------------------------------------------------------------------------
# Multimodal message builders — framework flat role schema ({role, type, value});
# the API wrapper base64-encodes image paths itself. Pass img_detail='high' to
# build_judge so the judge sees full detail.
# ---------------------------------------------------------------------------

_SYSTEM = "You are an image assessment expert."


def _image_or_warn(path: str) -> Dict[str, Any]:
    import os.path as osp
    if path and osp.isfile(path):
        return {"role": "user", "type": "image", "value": path}
    return {
        "role": "user", "type": "text",
        "value": f"\n[WARN] failed to load image: {path}, err=file not found\n",
    }


def build_concept_messages(
    prompt_text: str,
    generated_image_path: str,
    concept_image_path: str,
) -> List[Dict[str, Any]]:
    """Build (system, user) for the concept judge.

    Order matters per prompt template: <image> generated, then <image> concept.
    """
    return [
        {"role": "system", "type": "text", "value": _SYSTEM},
        {"role": "user", "type": "text", "value": prompt_text},
        _image_or_warn(generated_image_path),
        _image_or_warn(concept_image_path),
    ]


def build_task_messages(
    prompt_text: str,
    generated_image_path: str,
) -> List[Dict[str, Any]]:
    """Task judge sees only the generated image."""
    return [
        {"role": "system", "type": "text", "value": _SYSTEM},
        {"role": "user", "type": "text", "value": prompt_text},
        _image_or_warn(generated_image_path),
    ]


build_integration_messages = build_task_messages  # same shape
