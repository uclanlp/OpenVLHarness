"""RealXBench-style lenient QA judge prompt + parser.

Locally-developed lenient QA judge protocol used by RealXBench. NOT the
HLE protocol — for cross-paper alignment with HLE / WebWatcher /
BrowseComp-VL paper numbers, use ``hle_judge`` instead.

Distinguishing properties of this protocol (vs HLE):
  - Output is EXACTLY one token: ``CORRECT`` or ``INCORRECT`` (no JSON).
  - Leniency rules are listed *explicitly* in the prompt
    (synonyms, number/date formats, verbose answers, capitalisation).
  - The dataset side is responsible for short-answer extraction (e.g.
    via the ``Answer:`` regex) and for joining multiple alternative
    ground truths with ``" or "`` before passing as ``gt_answer``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
REALXBENCH_JUDGE_PROMPT = """You are an evaluator for question-answering systems.

Given a question, the ground-truth answer, and a model's prediction, judge whether the prediction is **semantically correct** (i.e. it conveys the same answer as the ground truth, even if the wording differs).

Be lenient about:
- Synonyms (e.g. "England" vs "United Kingdom")
- Number formats (e.g. "1" vs "One")
- Date formats (e.g. "September 29 to October 4, 2024" vs "2024-09-29 to 2024-10-04")
- Verbose answers that contain the correct answer plus extra explanation
- Capitalization and punctuation

Be strict about:
- Wrong facts (e.g. wrong color, wrong name, wrong number)
- Refusal or "I don't know" responses

Question: {question}
Ground Truth: {gt_answer}
Prediction: {prediction}

Respond with EXACTLY one word: either "CORRECT" or "INCORRECT". No explanation."""


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

def parse_realxbench_verdict(raw_response: str) -> str:
    """Parse the judge's raw response into a verdict label.

    Returns one of ``"correct"`` / ``"incorrect"`` / ``"error"``. The
    canonical RealXBench rule is: ``CORRECT`` substring without
    ``INCORRECT`` substring → correct; ``INCORRECT`` substring →
    incorrect; otherwise ``error``.
    """
    raw_upper = (raw_response or "").strip().upper()
    if "CORRECT" in raw_upper and "INCORRECT" not in raw_upper:
        return "correct"
    if "INCORRECT" in raw_upper:
        return "incorrect"
    return "error"
