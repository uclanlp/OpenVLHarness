"""Judge_WISE — verbatim port of the WISE benchmark evaluation protocol.

WISE (World-knowledge-Informed Semantic Evaluation) is a knowledge-intensive
text-to-image benchmark from Niu et al. 2025 (arXiv 2503.07265). The judge
takes (PROMPT, EXPLANATION, generated_image) and outputs a binary score
{0, 1} — 1 iff the image correctly realizes the knowledge-based meaning
of the prompt AND is visually usable (no obvious collapse / severe
deformation / garbled main objects / etc.).

Prompt is byte-exact to ``WISE/vllm_eval.py`` (build_evaluation_messages).
The paper judges with Qwen3.5-35B-A3B via vLLM; we default to whatever
model ``judge_rui.yaml`` specifies (usually gpt-4o). The scoring rubric
is identical either way.

Aggregation reference (WISE/calculate_verified.py):
  - Score is binary {0, 1}
  - Categories are prompt_id-range-based:
      CULTURE   : 1-400
      TIME      : 401-520
      SPACE     : 521-640
      BIOLOGY   : 641-760
      PHYSICS   : 761-880
      CHEMISTRY : 881-1000
  - WISE score = mean of binary scores per category, and overall.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# System prompt — byte-exact from WISE/vllm_eval.py
# ---------------------------------------------------------------------------
JUDGE_WISE_SYSTEM_PROMPT = (
    "You are a professional text-to-image quality auditor. "
    "Evaluate the image strictly according to the protocol."
)


# ---------------------------------------------------------------------------
# User prompt template — byte-exact from WISE/vllm_eval.py
# ---------------------------------------------------------------------------
_JUDGE_WISE_USER_TEMPLATE = """Please evaluate this generated image for the WISE benchmark and return ONLY one binary score.

# WISE Text-to-Image Evaluation Protocol

## What WISE Is Evaluating
WISE is a knowledge-intensive text-to-image benchmark. Many prompts do not directly state the final visual answer. Instead, the model must use commonsense, cultural, scientific, spatial, or temporal knowledge to infer what should appear in the image.

Your job is not to judge whether the image is beautiful. Your job is to judge whether the generated image correctly realizes the knowledge-based meaning of the prompt and is visually usable.

## Input Fields

**PROMPT**
The original text-to-image prompt given to the image generation model. It may contain an implicit clue rather than the explicit final answer.

**EXPLANATION**
The reference interpretation used for judging. It explains the intended answer, the required knowledge reasoning chain, and the visual evidence that should appear in a correct image. Treat EXPLANATION as the ground-truth judging guide.

For example:
- If PROMPT says "the round pastry commonly shared during Mid-Autumn Festival family gatherings", EXPLANATION may specify mooncakes. A correct image should show mooncakes, not just any festival food.
- If PROMPT says "a plant kept for many days beside a bright one-sided window", EXPLANATION may specify phototropism. A correct image should show the plant bending toward the light source.
- If PROMPT says "a street in New York when it is midnight in Beijing", EXPLANATION may specify the corresponding local time and expected lighting/activity. A correct image should reflect that inferred local time, not simply show Beijing or generic night.

## How To Judge

Evaluate the image using these checks:
1. Does the image contain the main objects or scene required by the PROMPT?
2. Does it satisfy the intended knowledge-based answer described in the EXPLANATION?
3. Are important relations correct, such as spatial layout, temporal state, physical effect, biological behavior, cultural object, or scientific phenomenon?
4. Is the image visually usable for judging, without obvious collapse, severe deformation, unreadable main objects, or major artifacts?

## Binary Score

**Score: 1**
Give 1 only when both conditions are met:
- The image is semantically correct according to both PROMPT and EXPLANATION.
- The image has no obvious generation failure that prevents reliable judging.

Minor aesthetic weakness, ordinary composition, non-photorealistic style, or lack of artistic beauty should not by itself cause rejection if the semantic target is correct and the image is clear.

**Score: 0**
Give 0 if any of the following applies:
- The image misses the intended answer in EXPLANATION.
- The image only follows surface words in PROMPT but fails the required knowledge inference.
- Key objects, attributes, states, behaviors, or relations are missing or wrong.
- The image contradicts the prompt or explanation.
- The main visual evidence is ambiguous enough that a human judge could not confidently verify correctness.
- The image has obvious visual collapse, severe deformation, garbled main objects, impossible structure, or artifacts that interfere with evaluation.

If there is serious doubt, return 0.

## Output Format

Return exactly one line and nothing else:

Score: 0

or

Score: 1

---

PROMPT: "{prompt_text}"
EXPLANATION: "{explanation_text}"

Return only `Score: 0` or `Score: 1`."""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def build_judge_wise_messages(
    prompt_text: str,
    explanation_text: str,
    generated_image_path: str,
) -> List[Dict[str, Any]]:
    """Return (system, user) message pair for a single WISE judge call.

    Byte-exact structure to ``WISE/vllm_eval.py::build_evaluation_messages``.
    """
    import os.path as osp

    user_text = _JUDGE_WISE_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        explanation_text=explanation_text,
    )
    # Framework flat role schema ({role, type, value}); the API wrapper
    # base64-encodes image paths itself. Pass img_detail='high' to
    # build_judge so the judge sees full detail.
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "type": "text", "value": JUDGE_WISE_SYSTEM_PROMPT},
        {"role": "user", "type": "text", "value": user_text},
    ]
    if generated_image_path and osp.isfile(generated_image_path):
        msgs.append({"role": "user", "type": "image", "value": generated_image_path})
    else:
        msgs.append({
            "role": "user", "type": "text",
            "value": f"\n[WARN] failed to load image: {generated_image_path}, err=file not found\n",
        })
    return msgs


# ---------------------------------------------------------------------------
# Parsing — byte-exact to WISE/vllm_eval.py::extract_scores
# ---------------------------------------------------------------------------

def parse_judge_wise_judgement(content: str) -> Dict[str, Any]:
    """Extract binary score {0, 1} from judge output.

    Byte-exact regex to WISE/vllm_eval.py::extract_scores:
      1. Look for '*Score*: 0|1' with optional markdown asterisks / colon variants
      2. Fallback: single-line bare '0' or '1'
      3. Otherwise: return correct=0 with 'unparseable' reasoning

    Returns:
        dict with:
            correct   : int (0 or 1)
            score     : float (0.0 or 1.0), None if unparseable
            reasoning : str (short summary)
            raw       : str (original judge content, untruncated)
    """
    text = (content or "").strip()

    # Primary regex from vllm_eval.py:65
    match = re.search(r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b", text, re.IGNORECASE)
    if match:
        s = int(match.group(1))
        return dict(
            correct=s,
            score=float(s),
            reasoning=f"Score: {s}",
            raw=text,
        )

    # Fallback: bare single-line 0 or 1 (vllm_eval.py:71-73)
    nums = re.findall(r"(?m)^\s*([01])\s*$", text)
    if len(nums) == 1:
        s = int(nums[0])
        return dict(
            correct=s,
            score=float(s),
            reasoning=f"Score: {s} (bare-line fallback)",
            raw=text,
        )

    return dict(
        correct=0,
        score=None,
        reasoning="judge failed to parse",
        raw=text,
    )


# ---------------------------------------------------------------------------
# Category assignment — matches WISE/calculate_verified.py prompt_id ranges
# ---------------------------------------------------------------------------

def wise_category_from_prompt_id(prompt_id: int) -> str:
    """Return the paper category ('CULTURE' | 'TIME' | 'SPACE' | 'BIOLOGY' |
    'PHYSICS' | 'CHEMISTRY') for a WISE prompt_id.

    Ranges are from WISE/calculate_verified.py:
      CULTURE   : 1-400
      TIME      : 401-520
      SPACE     : 521-640
      BIOLOGY   : 641-760
      PHYSICS   : 761-880
      CHEMISTRY : 881-1000
    """
    p = int(prompt_id)
    if 1 <= p <= 400:
        return "CULTURE"
    if 401 <= p <= 520:
        return "TIME"
    if 521 <= p <= 640:
        return "SPACE"
    if 641 <= p <= 760:
        return "BIOLOGY"
    if 761 <= p <= 880:
        return "PHYSICS"
    if 881 <= p <= 1000:
        return "CHEMISTRY"
    return "UNKNOWN"
