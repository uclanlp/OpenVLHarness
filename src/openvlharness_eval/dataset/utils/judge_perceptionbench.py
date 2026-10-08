"""PerceptionBench LLM-judge prompt + parsing helpers.

Byte-faithful port of MoonshotAI's PerceptionBench grader
(``eval/judge_prompt.txt`` + ``decode_judge`` in ``eval/eval.py``,
https://github.com/MoonshotAI/PerceptionBench).

PerceptionBench answers are short, free-form and uniquely determined; the
paper grades each response 0/1 with an LLM ``teacher`` that reads the
question, the student (model) answer, and the reference answer, then emits a
``[reason] ... [judge] True/False`` verdict. The reference answer is treated
as absolutely correct; the judge only checks whether the student's final
result matches it.

We keep the prompt and the True/False decoding identical to upstream so our
numbers stay comparable to the public leaderboard. Substitution uses
``str.replace`` (not ``str.format``) exactly like upstream, so any stray
braces in a question/answer never break templating.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# The exact upstream teacher-grading prompt (eval/judge_prompt.txt).
# Placeholders {problem} / {assistant_answer} / {reference_answer} are filled
# via str.replace, matching upstream judge_general_qa().
# ---------------------------------------------------------------------------
PERCEPTIONBENCH_JUDGE_PROMPT = """Please act as a professional teacher and grade the student's answer. Below are the question, the student's answer, and the reference answer. Based on the question and the reference answer, analyze the student's answer and judge whether it correctly answers the question. I will provide several examples to help you understand how to judge.

========== [Output Format] ==========
Return your judgment in the following format. The first field [reason] gives the reason for your judgment; the second field [judge] gives your verdict as a single boolean, i.e., True or False. Make sure your output ends with True or False.
[reason]
<a brief justification of no more than 100 tokens>
[judge]
False

========== [Notice] ==========
0. You only need to compare the consistency between the reference answer and the student's answer, focusing especially on the content after summarizing phrases such as "Final answer:". The reference answer is absolutely correct; judge solely by whether the student's final result matches it, even if the solution process is correct.
1. If the question contains multiple sub-questions, output the student's answer, the reference answer, and the consistency for each sub-question in [reason]; judge correct only when all sub-questions are consistent.
2. For multiple-answer questions, the student's answer must contain all correct answers without any extra ones. Pay special attention when the reference answer contains "or": decide whether all answers must be given based on the specific question.
3. If you believe the student's answer equals the reference answer after simplification, you must provide the complete simplification process in [reason] until the two are equal; otherwise it cannot be judged correct. You may not vaguely claim that "they can be made equal".
4. For numerical answers, integers or values with at most 4 significant figures must match exactly; values with more than 4 significant figures must match within 4 significant figures. In [reason], convert both to standard scientific notation, first compare the order of magnitude, then compare the first 4 significant figures. If the units differ, convert to a common unit first.
5. For English writing questions, if the student does not answer in English, judge it incorrect.
6. For physics, chemistry, and biology questions, if the reference answer contains a technical term, the student's answer must contain that exact term; synonyms are not accepted.
7. When the question asks to explain a term, redundant explanation is not penalized, but missing key points must be judged incorrect.
8. For multiple-choice questions, judge incorrect whenever the student's selected option differs from the reference answer, regardless of the content.

========== [Examples] ==========
========== [Example 1.1] ==========
Question: omitted    Student answer: 34    Reference answer: (1) 12; (2) 34
[reason] The student answer 34 only addresses the second sub-question and leaves the first unanswered.
[judge] False

========== [Example 1.2] ==========
Question: omitted    Student answer: 13; 34    Reference answer: (1) 12; (2) 34
[reason] Sub-question 1: student 13 vs. reference 12, inconsistent. Sub-question 2: student 34 vs. reference 34, consistent. Overall, incorrect.
[judge] False

========== [Example 2.1] ==========
Question: omitted    Student answer: A    Reference answer: ABC
[reason] The student answer A is incomplete; the correct answer is ABC.
[judge] False

========== [Example 2.2] ==========
Question: omitted    Student answer: ABC    Reference answer: AC
[reason] The student answer ABC includes an extra B; the correct answer is AC.
[judge] False

========== [Example 3.1] ==========
Question: omitted    Student answer: y''(0) = 13e^2 - e    Reference answer: y''(0) = 12e^2 - 1
[reason] The student answer differs in form from the reference and cannot be made equal through simplification.
[judge] False

========== [Example 3.2] ==========
Question: omitted    Student answer: y''(0) = 12e^2 - e    Reference answer: y''(0) = (12e - 1)e
[reason] 12e^2 - e = (12e - 1)e, which is equivalent to the reference answer.
[judge] True

========== [Example 4.1] ==========
Question: omitted    Student answer: 349    Reference answer: 342
[reason] Integer answers must match exactly; 349 differs from 342.
[judge] False

========== [Example 4.2] ==========
Question: omitted    Student answer: 0.325    Reference answer: 0.618
[reason] In scientific notation, 0.325 = 3.250e-1 and 0.618 = 6.180e-1; the exponents match but the first significant figure differs.
[judge] False

========== [Example 4.3] ==========
Question: omitted    Student answer: 48.67    Reference answer: 48.675
[reason] 48.67 = 4.867e1 and 48.675 = 4.868e1 (to 4 significant figures); the exponents match but the 4th significant figure differs.
[judge] False

========== [Example 4.4] ==========
Question: omitted    Student answer: 4.85e4    Reference answer: 485
[reason] 4.85e4 vs. 485 = 4.85e2; the two differ in order of magnitude.
[judge] False

========== [Example 8.1] ==========
Question: A. 11  B. 22  C. 33  D. 44    Reference answer: A. 11    Student answer: C. 11
[reason] The student option C differs from the reference option A.
[judge] False

========== [Example 8.2] ==========
Question: A. 11  B. 22  C. 33  D. 44    Reference answer: B    Student answer: C
[reason] The student option C differs from the reference option B.
[judge] False

========== [Notice] ==========
0. You only need to compare the consistency between the reference answer and the student's answer, focusing especially on the content after summarizing phrases such as "Final answer:". The reference answer is absolutely correct; judge solely by whether the student's final result matches it, even if the solution process is correct.
1. If the question contains multiple sub-questions, output the student's answer, the reference answer, and the consistency for each sub-question in [reason]; judge correct only when all sub-questions are consistent.
2. For multiple-answer questions, the student's answer must contain all correct answers without any extra ones. Pay special attention when the reference answer contains "or": decide whether all answers must be given based on the specific question.
3. If you believe the student's answer equals the reference answer after simplification, you must provide the complete simplification process in [reason] until the two are equal; otherwise it cannot be judged correct. You may not vaguely claim that "they can be made equal".
4. For numerical answers, integers or values with at most 4 significant figures must match exactly; values with more than 4 significant figures must match within 4 significant figures. In [reason], convert both to standard scientific notation, first compare the order of magnitude, then compare the first 4 significant figures. If the units differ, convert to a common unit first.
5. For English writing questions, if the student does not answer in English, judge it incorrect.
6. For physics, chemistry, and biology questions, if the reference answer contains a technical term, the student's answer must contain that exact term; synonyms are not accepted.
7. When the question asks to explain a term, redundant explanation is not penalized, but missing key points must be judged incorrect.
8. For multiple-choice questions, judge incorrect whenever the student's selected option differs from the reference answer, regardless of the content.

Now you may begin grading.
========== [Question] ==========
{problem}
========== [Student Answer] ==========
{assistant_answer}
========== [Reference Answer] ==========
{reference_answer}
========== [Your Judgment] ==========
"""

# Sentinel reasoning emitted when the judge output can't be parsed; the caller
# treats this as "retry" (matches parse_hle_judgement's convention).
_UNPARSEABLE = "judge failed to parse"


def normalize_escape(s: str) -> str:
    """Port of upstream ``normalize_escape`` — tidy stray backslash escapes.

    Keeps behaviour identical to eval/eval.py so judge inputs match upstream.
    """
    wl = " 0123456789-"
    s = re.sub(rf"\\+[{wl}]", r"\\\\" + " ", s)
    s = re.sub(rf"\\+(?![{wl}])", r"\\", s)
    return re.sub(r"\\+n", "\n", s)


def parse_perceptionbench_judgement(response_text: str) -> dict:
    """Decode a ``[reason] ... [judge] True/False`` verdict.

    Faithful port of upstream ``decode_judge``. Returns a dict with:
      - correct (1 or 0): verdict; correct iff "true" appears after [judge]
      - reasoning (str):  the [reason] text (or the ``_UNPARSEABLE`` sentinel)

    If either tag is missing, returns ``correct=0`` with ``reasoning`` set to
    the sentinel so the caller can retry.
    """
    resp = str(response_text or "").strip()
    if "[reason]" not in resp or "[judge]" not in resp:
        return dict(correct=0, reasoning=_UNPARSEABLE)
    reason = resp.split("[judge]")[0].split("[reason]")[-1].strip()
    verdict = resp.split("[judge]")[-1].strip()
    return dict(correct=1 if "true" in verdict.lower() else 0, reasoning=reason)


def grade_with_perceptionbench_judge(model, problem: Any, reference: Any, prediction: Any,
                                     retry: int = 3) -> dict:
    """Grade one (question, reference, prediction) tuple with the PerceptionBench judge.

    Mirrors ``grade_with_hle_judge`` in ``hle_judge.py``: fills the exact
    upstream prompt, calls ``model.generate``, and retries on API failure or
    unparseable output. Returns the parsed dict (see
    ``parse_perceptionbench_judgement``); truncates the stored reasoning to keep
    result files small (upstream caps at 200 chars).

    On total failure after ``retry`` attempts, returns ``correct=0`` with an
    explanatory reasoning string.
    """
    prob = normalize_escape(str(problem)).strip()
    ref = normalize_escape(str(reference if reference is not None else "")).strip()
    pred = normalize_escape(str(prediction)).strip()

    prompt = (PERCEPTIONBENCH_JUDGE_PROMPT
              .replace("{problem}", prob)
              .replace("{assistant_answer}", pred)
              .replace("{reference_answer}", ref))

    last_text: Optional[str] = None
    for _ in range(retry):
        ans = model.generate(prompt)
        if "Failed to obtain answer via API" in (ans or ""):
            continue
        last_text = ans
        parsed = parse_perceptionbench_judgement(ans)
        if parsed.get("reasoning") != _UNPARSEABLE:
            parsed["reasoning"] = str(parsed.get("reasoning", ""))[:200]
            return parsed

    return dict(
        correct=0,
        reasoning="judge unreachable after retries"
            if last_text is None
            else f"unparseable: {str(last_text)[:180]}",
    )
