"""Judge_FactIP — byte-exact port of the Unify-Agent FactIP paper grader.

FactIP (arXiv 2603.29620, Unify-Agent) is a knowledge-grounded image
generation benchmark focused on IP identity preservation. 500 prompts
across 12 fine-grained categories (Character / Object / Scene groups).

Judge: GPT-4o (default), 5 integer dimensions in [0, 10]:
  - clarity           (5% weight)
  - content_quality   (10% weight)
  - aesthetics        (10% weight)
  - text_relevance_ip (75% weight — dominant, per paper)
  - overall_score     (holistic; NOT used for aggregation)

Overall score (per calculate.py:26-32):
    O = 0.05·clarity + 0.10·content_quality + 0.10·aesthetics + 0.75·text_relevance_ip

We port the SYSTEM_PROMPT verbatim from Unify-Agent/eval/score_factip.py:50-83.
The judge sees (Prompt, generated_image, GT1, GT2) — 3 images total.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System prompt — byte-exact from Unify-Agent/eval/score_factip.py:50-83
# ---------------------------------------------------------------------------
JUDGE_FACTIP_SYSTEM_PROMPT = """You are an image evaluation assistant. Compare AS (assistant-generated image) against GT1 and GT2 (ground-truth images) given the Prompt.

Your task is to return exactly 6 fields: 5 integer scores (0-10) and 1 rationale string.

Evaluate these 5 dimensions independently:

1. "clarity": image sharpness, absence of blur/artifacts/noise, and richness of visible details.
2. "content_quality": faithfulness to the Prompt, subject completeness, and semantic coherence.
3. "aesthetics": visual appeal, composition, lighting/color harmony, and style consistency.
4. "text_relevance_ip": IP identity consistency. This means whether AS preserves the same character/object/IP identity as GT1/GT2 based on distinctive traits (e.g. face, hairstyle, costume, colors, species/object-defining features). Do NOT require exact matching of pose, background, camera angle, or composition.
5. "overall_score": holistic judgment of the final image quality. This should not be a simple average of the above scores.

Important instructions:
- Use GT1 and GT2 jointly to infer the stable identity and attributes of the IP.
- Do not penalize AS for differences that also vary between GT1 and GT2.
- Score each dimension independently before assigning the overall score.
- All five score fields must be integers from 0 to 10.
- "rationale" must be a single short string with at least two concrete evidence points, separated by semicolons.

Overall score rubric:
- 0: total failure
- 1-3: severe issues
- 4-6: usable but clearly worse than GT
- 7-9: good and comparable to GT
- 10: nearly perfect

You MUST respond with ONLY this exact JSON structure, with all 6 keys present and no extra keys:

{"clarity": 7, "content_quality": 8, "aesthetics": 7, "text_relevance_ip": 8, "overall_score": 7, "rationale": "Evidence 1; Evidence 2"}

Do not use markdown fences.
Do not output any text before or after the JSON.
If uncertain, still output the best-effort JSON with all required keys."""


_SCORE_KEYS = ("clarity", "content_quality", "aesthetics", "text_relevance_ip", "overall_score")

# Aggregation weights (calculate.py:26-32)
_WEIGHTS = {
    "clarity": 0.05,
    "content_quality": 0.10,
    "aesthetics": 0.10,
    "text_relevance_ip": 0.75,
}


def build_judge_factip_messages(
    sample_id: str,
    prompt: str,
    generated_image_path: str,
    gt1_path: str,
    gt2_path: str,
) -> List[Dict[str, Any]]:
    """Build (system, user) messages for the FactIP judge.

    User payload = text (id + prompt + explanation) + 3 images: AS, GT1, GT2.
    Framework flat role schema ({role, type, value}); the API wrapper
    base64-encodes image paths itself. Pass img_detail='high' to build_judge
    so the judge sees full detail.
    """
    import os.path as osp

    text = (
        f"Sample id: {sample_id}\n\n"
        f"Task prompt (image requirement):\n{prompt}\n\n"
        "Image 1 (AS): the **generated image** (to be evaluated).\n"
        "Image 2 (GT1): the **first ground-truth reference image**.\n"
        "Image 3 (GT2): the **second ground-truth reference image**.\n\n"
        "Output ONLY the required JSON object with all 6 keys."
    )
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "type": "text", "value": JUDGE_FACTIP_SYSTEM_PROMPT},
        {"role": "user", "type": "text", "value": text},
    ]
    for label, path in (("AS", generated_image_path), ("GT1", gt1_path), ("GT2", gt2_path)):
        if path and osp.isfile(path):
            msgs.append({"role": "user", "type": "image", "value": path})
        else:
            msgs.append({
                "role": "user", "type": "text",
                "value": f"\n[WARN] failed to load {label} image: {path}, err=file not found\n",
            })
    return msgs


def _clip_int_0_10(x) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        return 0
    return max(0, min(10, v))


def parse_judge_factip_judgement(content: str) -> Dict[str, Any]:
    """Parse the FactIP judge output.

    Returns dict with:
      - ``verdict``: parsed dict from LLM (or {} on failure)
      - ``correct``: 1 if weighted_overall >= 5.0 else 0 (0-10 scale midpoint)
      - ``weighted_overall``: 0.05·C + 0.10·CQ + 0.10·A + 0.75·TRI (0-10 scale)
      - ``clarity`` / ``content_quality`` / ``aesthetics`` / ``text_relevance_ip`` / ``overall_score``: raw 0-10
      - ``rationale``: str
      - ``reasoning``: short 1-line summary
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    obj: Optional[dict] = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            snippet = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            try:
                obj = json.loads(snippet)
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return dict(
            verdict={},
            correct=0,
            weighted_overall=0.0,
            clarity=0,
            content_quality=0,
            aesthetics=0,
            text_relevance_ip=0,
            overall_score=0,
            rationale="",
            reasoning="judge failed to parse",
        )

    scores = {k: _clip_int_0_10(obj.get(k, 0)) for k in _SCORE_KEYS}
    weighted = sum(_WEIGHTS[k] * scores[k] for k in _WEIGHTS)
    rationale = str(obj.get("rationale", ""))

    reasoning = (
        f"C={scores['clarity']} CQ={scores['content_quality']} "
        f"A={scores['aesthetics']} TRI={scores['text_relevance_ip']} "
        f"O={scores['overall_score']} | weighted={weighted:.2f}"
    )

    return dict(
        verdict=obj,
        correct=1 if weighted >= 5.0 else 0,
        weighted_overall=weighted,
        clarity=scores["clarity"],
        content_quality=scores["content_quality"],
        aesthetics=scores["aesthetics"],
        text_relevance_ip=scores["text_relevance_ip"],
        overall_score=scores["overall_score"],
        rationale=rationale[:500],
        reasoning=reasoning,
    )
