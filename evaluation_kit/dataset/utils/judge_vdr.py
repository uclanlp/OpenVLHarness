"""Judge_VDR — contains-based LLM judge from the Vision-DeepResearch paper.

The prompt is the judge prompt from the appendix of the Vision-DeepResearch
(VDR) paper (arXiv 2601.22060), copied verbatim, including the
``[Deep Research Report]`` field name and the one-line
``correct: [yes/no] reasoning: [your explanation]`` output format;
``parse_judge_vdr_judgement`` also accepts the two-line
``correct: ...\\nreasoning: ...`` form.

Different from ``judge_rs`` (MMSearch-R1 paper judge):
- Judge_RS is rule-based (10-rule semantic-equivalence checklist, XML output)
- Judge_VDR is contains-based ("does the report contain the correct answer
  anywhere?" — looser, suitable for long deep-research style outputs)

Used as a per-benchmark opt-in via env ``MMDR_JUDGE_PROMPT=judge_vdr`` or
``judge_kwargs['judge_prompt'] = 'judge_vdr'``. No evaluator uses it by
default; each keeps its own default prompt (the HLE-style judge for
MMSearch, MMSearch-Plus, BrowseComp-VL, VDR-Bench and SimpleVQA).
"""

from __future__ import annotations

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Prompt from the Vision-DeepResearch paper appendix.
# ---------------------------------------------------------------------------
JUDGE_VDR_PROMPT = """You are an impartial judge evaluating whether a deep research report contains the correct answer.
[Question] {question}
[Correct Answer] {correct_answer}
[Deep Research Report] {response}
Task: Determine if the deep research report contains the correct answer anywhere in its content.
Instructions:
1. Read through the entire research report carefully
2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
3. Check if the information in the report is consistent with the correct answer
4. The answer does NOT need to be in a specific format or labelled as "final answer"
5. Provide your reasoning
6. Answer with "yes" if the report contains the correct answer, "no" if it doesn't or contradicts it
Output format: correct: [yes/no] reasoning: [your explanation]"""


# Version tag — bump when the prompt changes. Persisted into acc.csv /
# llm_judge.json so completed runs remain traceable to the exact judge
# protocol that produced the score.
JUDGE_VDR_VERSION = "judge_vdr_v1_20260513"


# ---------------------------------------------------------------------------
# Parsing — v1 expects ``correct: yes/no\nreasoning: ...`` plain text,
# falls through several layers for non-compliant LLM outputs.
# ---------------------------------------------------------------------------
_YES = ("yes", "y", "true", "correct")
_NO = ("no", "n", "false", "incorrect")

# Primary: ``correct: yes`` / ``correct: no`` (case-insensitive, optional
# brackets / colons / dashes around the verdict, optional trailing punctuation).
_CORRECT_LINE_RE = re.compile(
    r"correct\s*[:\-]\s*\[?\s*(yes|no|y|n)\b",
    re.IGNORECASE,
)
_REASONING_LINE_RE = re.compile(
    r"reasoning\s*[:\-]\s*(.+?)(?:\Z|\n\s*correct\s*[:\-])",
    re.IGNORECASE | re.DOTALL,
)


def parse_judge_vdr_judgement(response_text: str) -> dict:
    """Parse v1 plain-text verdict, with multi-layer fallbacks.

    Expected v1 output:
      correct: yes
      reasoning: <explanation>

    Robust against common LLM deviations (variant casing, surrounding
    brackets / dashes, missing reasoning field, raw 'Yes'/'No' without
    keys).

    Returns a dict with the same keys as ``parse_hle_judgement`` /
    ``parse_judge_rs_judgement`` for drop-in compatibility:
      - correct (1 or 0)
      - extracted_answer (always "" — Judge_VDR doesn't extract)
      - reasoning (str)
      - confidence (int; 100 normally, 0 on parse failure)
    """
    text = str(response_text or "").strip()

    # 1) Primary: ``correct: yes`` / ``correct: no`` keyword.
    m = _CORRECT_LINE_RE.search(text)
    if m:
        verdict = m.group(1).strip().lower()
        is_yes = verdict in _YES
        r = _REASONING_LINE_RE.search(text)
        reasoning = r.group(1).strip() if r else ""
        if not reasoning:
            reasoning = f"<correct> only (no <reasoning>): {text[:200]}"
        return dict(
            correct=1 if is_yes else 0,
            extracted_answer="",
            reasoning=reasoning,
            confidence=100,
        )

    # 2) Fallback: bare first-line verdict (LLM ignored keyword schema).
    low = text.lower()
    first_line = low.splitlines()[0].strip(" .:!?,") if low else ""
    if first_line in _YES:
        return dict(correct=1, extracted_answer="",
                    reasoning=f"bare verdict (no keys): {text[:200]}", confidence=100)
    if first_line in _NO:
        return dict(correct=0, extracted_answer="",
                    reasoning=f"bare verdict (no keys): {text[:200]}", confidence=100)

    # 3) Fallback: first word.
    first_word = re.split(r"\s+", low, maxsplit=1)[0].strip(" .:!?,") if low else ""
    if first_word in _YES:
        return dict(correct=1, extracted_answer="",
                    reasoning=f"first-word verdict (no keys): {text[:200]}", confidence=100)
    if first_word in _NO:
        return dict(correct=0, extracted_answer="",
                    reasoning=f"first-word verdict (no keys): {text[:200]}", confidence=100)

    # 4) Fallback: exclusive yes/no token in body.
    has_yes = bool(re.search(r"\byes\b", low))
    has_no = bool(re.search(r"\bno\b", low))
    if has_yes and not has_no:
        return dict(correct=1, extracted_answer="",
                    reasoning=f"yes-token in body: {text[:300]}", confidence=100)
    if has_no and not has_yes:
        return dict(correct=0, extracted_answer="",
                    reasoning=f"no-token in body: {text[:300]}", confidence=100)

    # 5) Unparseable — caller can retry; signal failure via reserved sentinel.
    return dict(correct=0, extracted_answer="",
                reasoning="judge_vdr failed to parse", confidence=0)


# ---------------------------------------------------------------------------
# Grading helper (mirrors grade_with_judge_rs / grade_with_hle_judge signature)
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


def grade_with_judge_vdr(model, question: str, correct_answer: Any, response: str,
                         retry: int = 3) -> dict:
    """Run the Judge_VDR prompt on a single (question, gt, response) tuple.

    Sends a single concatenated user message via ``model.generate_inner`` (VDR
    paper prompt has no system/user split — it's one flat instruction).
    Skips ``model.generate``'s ``preprocess_message_with_role`` path so
    ``self.system_prompt`` on the OpenAIWrapper instance is not mutated
    across calls.

    ``correct_answer`` may be a string, list, or list-of-lists. The VDR
    prompt has no Candidate Answers field (unlike Judge_RS), so when
    multiple variants are given they are joined with `` | `` into the
    ``[Correct Answer]`` field; the judge sees them as one alternative
    answer string.

    Returns the parsed judgement dict (see ``parse_judge_vdr_judgement``).
    On total failure after ``retry`` attempts: ``correct=0`` with
    ``reasoning='judge unreachable after retries'``.
    """
    flat = _flatten(correct_answer)
    if not flat:
        gt_text = ""
    elif len(flat) == 1:
        gt_text = str(flat[0])
    else:
        gt_text = " | ".join(str(x) for x in flat if str(x).strip())

    user_prompt = JUDGE_VDR_PROMPT.format(
        question=str(question),
        correct_answer=gt_text,
        response=str(response),
    )

    msgs = [{"role": "user", "content": [{"type": "text", "value": user_prompt}]}]

    last_text: Optional[str] = None
    for _ in range(retry):
        try:
            code, ans, _log = model.generate_inner(msgs)
        except Exception:
            continue
        if code != 0 or not ans:
            continue
        last_text = ans
        parsed = parse_judge_vdr_judgement(ans)
        if parsed.get("reasoning") != "judge_vdr failed to parse":
            return parsed

    return dict(
        correct=0,
        extracted_answer="",
        reasoning="judge unreachable after retries"
            if last_text is None
            else f"unparseable: {str(last_text)[:300]}",
        confidence=0,
    )
