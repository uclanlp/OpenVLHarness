"""Judge_RS — Rui's iterative LLM-judge prompt family.

v2 (``v2_20260428``) is ported from the multimodal-search-r1 paper
(arXiv 2506.20670, paper appendix). Adds a reasoning step (XML-tagged
``<judge>`` + ``<reason>``) and an image-aware rule (Rule 9 lets
text-only judging gracefully ignore the image). Compared to v1, the
reasoning step is the dominant fix — v1 emitted bare ``Yes``/``No``
which caused systematic false negatives on semantic-equivalence
cases (case-only diffs / trailing-period / synonyms / category-vs-
instance / paraphrase) on older GPT-4o snapshots. See
``personal/log.md`` 2026-04-27 entry for the full v1 error analysis.

v1 (``v1_20260425``) is byte-exact ported from
``MM-DeepResearch/eval/prompt.py`` (the ``JUDGE_SYSTEM_PROMPT`` /
``JUDGE_USER_PROMPT`` 11-rule pair). Kept here as ``*_BK`` constants
for reproducibility / rollback; not exported as the active prompt.

Used by: ``browsecomp_vl`` when ``judge_kwargs['judge_prompt'] = 'judge_rs'``
(or env ``MVT_JUDGE_PROMPT=judge_rs``). Other datasets (mmsearch /
simplevqa_en / livevqa_v2) not yet hooked up — pending after BCVL
results validate v2.

HLE remains the default judge for everything.
"""

from __future__ import annotations

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# v2 prompt — multimodal-search-r1 paper (XML output, reasoning step,
# image-aware Rule 9). ACTIVE.
# ---------------------------------------------------------------------------
JUDGE_RS_SYSTEM_PROMPT = """You are an AI assistant tasked with evaluating the correctness of model responses based on an image, question, and ground truth answer. Your judgment should follow these principles:
1. Consider the image, question, and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match \"Yes\" or \"No\" to be correct.
9. If the judgment can be made based solely on the text, you may choose to ignore the input image, as some images may be unfamiliar to you and could affect your judgment. Refer to the image only when necessary to minimize misjudgment.
10. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
Your output must be in the following format:
<judge>Yes/No</judge>
<reason>Explanation of why the answer is correct or incorrect.</reason>"""


JUDGE_RS_USER_PROMPT = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Candidate Answers: {candidate_answers}
Model Response: {model_response}
Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question, Ground Truth Answer and Candidate Answers. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.
Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""


# Concatenated single-prompt form for ``model.generate(prompt)`` callers
# that don't expose a system/user split. Order: system, blank line, user.
JUDGE_RS_PROMPT = JUDGE_RS_SYSTEM_PROMPT + "\n\n" + JUDGE_RS_USER_PROMPT


# Version tag — bump when the prompt changes. Persisted into acc.csv /
# llm_judge.json so completed runs remain traceable to the exact judge
# protocol that produced the score.
JUDGE_RS_VERSION = "judge_rs_v2_20260428"


# ---------------------------------------------------------------------------
# v1 prompt — kept as backup for reproducibility / rollback. NOT ACTIVE.
# v1 was byte-exact port of MM-DeepResearch/eval/prompt.py (11 rules,
# bare Yes/No output). Run BCVL with v1 by manually swapping
# JUDGE_RS_*_PROMPT = JUDGE_RS_*_PROMPT_BK in this file.
# ---------------------------------------------------------------------------
JUDGE_RS_SYSTEM_PROMPT_BK = """You are an AI assistant tasked with evaluating the correctness of model responses based on the question, and ground truth answer.
Your judgment should follow these principles:
1. Consider the question, and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match \"Yes\" or \"No\" to be correct.
9. If the model response contains refusal statements, and does not directly answer the question, it must be judged incorrect.
10. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
11. If the model response is empty, it is incorrect.
Your output must be in the following format: Yes or No"""


JUDGE_RS_USER_PROMPT_BK = """Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Candidate Answers: {candidate_answers}
Model Response: {model_response}
Evaluation Instructions
Evaluate whether the Model Response is correct based on the Question, Ground Truth Answer and Candidate Answers.
Follow the predefined judgment rules and provide a clear Yes/No answer without any illustrations.
Output Format Yes or No"""


JUDGE_RS_PROMPT_BK = JUDGE_RS_SYSTEM_PROMPT_BK + "\n\n" + JUDGE_RS_USER_PROMPT_BK
JUDGE_RS_VERSION_BK = "judge_rs_v1_20260425"


# ---------------------------------------------------------------------------
# Parsing — v2 expects XML tags; falls back through bare yes/no for
# non-compliant LLM outputs.
# ---------------------------------------------------------------------------
_YES = ("yes", "y", "true", "correct")
_NO = ("no", "n", "false", "incorrect")

_JUDGE_TAG_RE = re.compile(r"<judge>\s*(yes|no)\b[^<]*</judge>", re.IGNORECASE)
_REASON_TAG_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)


def parse_judge_rs_judgement(response_text: str) -> dict:
    """Parse v2 XML-tagged verdict, with multi-layer fallbacks.

    Expected v2 output:
      <judge>Yes</judge>
      <reason>...</reason>

    Robust against common LLM deviations (uppercase tags, trailing
    punctuation, missing reason tag, raw 'Yes'/'No' without tags).

    Returns a dict with same keys as ``parse_hle_judgement`` for
    drop-in compatibility:
      - correct (1 or 0)
      - extracted_answer (always "" — Judge_RS doesn't extract)
      - reasoning (str)
      - confidence (int; 100 normally, 0 on parse failure)
    """
    text = str(response_text or "").strip()

    # 1) Primary: <judge>Yes/No</judge> tag (v2 spec).
    m = _JUDGE_TAG_RE.search(text)
    if m:
        verdict = m.group(1).strip().lower()
        reason_m = _REASON_TAG_RE.search(text)
        reasoning = reason_m.group(1).strip() if reason_m else ""
        if not reasoning:
            reasoning = f"<judge> only (no <reason>): {text[:200]}"
        return dict(
            correct=1 if verdict == "yes" else 0,
            extracted_answer="",
            reasoning=reasoning[:500],
            confidence=100,
        )

    # 2) Fallback: bare token / first-line match (LLM ignored XML schema).
    low = text.lower()
    first_line = low.splitlines()[0].strip(" .:!?,") if low else ""
    if first_line in _YES:
        return dict(correct=1, extracted_answer="", reasoning=f"bare verdict (no tags): {text[:200]}", confidence=100)
    if first_line in _NO:
        return dict(correct=0, extracted_answer="", reasoning=f"bare verdict (no tags): {text[:200]}", confidence=100)

    # 3) Fallback: first word.
    first_word = re.split(r"\s+", low, maxsplit=1)[0].strip(" .:!?,") if low else ""
    if first_word in _YES:
        return dict(correct=1, extracted_answer="", reasoning=f"first-word verdict (no tags): {text[:200]}", confidence=100)
    if first_word in _NO:
        return dict(correct=0, extracted_answer="", reasoning=f"first-word verdict (no tags): {text[:200]}", confidence=100)

    # 4) Fallback: explicit yes/no token in body, exclusively.
    has_yes = bool(re.search(r"\byes\b", low))
    has_no = bool(re.search(r"\bno\b", low))
    if has_yes and not has_no:
        return dict(correct=1, extracted_answer="", reasoning=f"yes-token in body: {text[:300]}", confidence=100)
    if has_no and not has_yes:
        return dict(correct=0, extracted_answer="", reasoning=f"no-token in body: {text[:300]}", confidence=100)

    # 5) Unparseable — caller can retry; signal failure via reserved sentinel.
    return dict(correct=0, extracted_answer="", reasoning="judge_rs failed to parse", confidence=0)


# ---------------------------------------------------------------------------
# Grading helper (mirrors grade_with_hle_judge signature)
# ---------------------------------------------------------------------------
def _flatten(value: Any) -> list:
    """Flatten one level of list-of-lists; pass through non-lists."""
    if not isinstance(value, list):
        return [value] if value not in (None, "") else []
    out = []
    for x in value:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def grade_with_judge_rs(model, question: str, correct_answer: Any, response: str,
                        retry: int = 3) -> dict:
    """Run the Judge_RS prompt on a single (question, gt, response) tuple.

    Sends ``JUDGE_RS_SYSTEM_PROMPT`` and ``JUDGE_RS_USER_PROMPT`` as
    SEPARATE roles via ``model.generate_inner`` so the OpenAI judge
    receives a real ``system`` role (not a single concatenated user
    message). Skips ``model.generate``'s ``preprocess_message_with_role``
    path so ``self.system_prompt`` on the OpenAIWrapper instance is not
    mutated across calls.

    ``correct_answer`` may be a string, list, or list-of-lists. The first
    variant becomes ``Ground Truth Answer``; remaining variants become
    ``Candidate Answers`` (Rule 10 — match any → correct). When there is
    only one variant, ``Candidate Answers`` is set to ``"(none)"``.

    Returns the parsed judgement dict (see ``parse_judge_rs_judgement``).
    On total failure after ``retry`` attempts: ``correct=0`` with
    ``reasoning='judge unreachable after retries'``.
    """
    flat = _flatten(correct_answer)
    if not flat:
        gt_primary = ""
        candidates = "(none)"
    else:
        gt_primary = str(flat[0])
        rest = [str(x) for x in flat[1:] if str(x).strip()]
        candidates = " | ".join(rest) if rest else "(none)"

    msgs_role_split = [
        {"role": "system", "content": [{"type": "text", "value": JUDGE_RS_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "value": JUDGE_RS_USER_PROMPT.format(
            question=str(question),
            ground_truth_answer=gt_primary,
            candidate_answers=candidates,
            model_response=str(response),
        )}]},
    ]

    last_text: Optional[str] = None
    for _ in range(retry):
        try:
            code, ans, _log = model.generate_inner(msgs_role_split)
        except Exception:
            continue
        if code != 0 or not ans:
            continue
        last_text = ans
        parsed = parse_judge_rs_judgement(ans)
        if parsed.get("reasoning") != "judge_rs failed to parse":
            return parsed

    return dict(
        correct=0,
        extracted_answer="",
        reasoning="judge unreachable after retries"
            if last_text is None
            else f"unparseable: {str(last_text)[:300]}",
        confidence=0,
    )
