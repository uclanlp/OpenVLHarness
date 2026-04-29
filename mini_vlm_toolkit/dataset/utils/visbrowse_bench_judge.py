"""VisBrowse-Bench paper-protocol LLM judge prompt + parser.

Direct port of the upstream VisBrowse-Bench ``eval.py`` judge: a minimal
yes/no semantic-equivalence grader. This is the *default* judge for our
VBB benchmark so reported numbers align with the paper.

Output is a free-form response containing the first ``yes`` or ``no``
token; the parser scans for ``\\b(yes|no)\\b`` (case-insensitive) and
returns ``"yes"`` / ``"no"`` / ``None`` (None → caller should retry).

Multi-variant ground truths are graded *per variant* (any-of) by the
caller — that loop lives in the dataset module since it depends on the
``ground_truth_list`` field shape.
"""

from __future__ import annotations

import re
from typing import Optional


# Upstream eval.py preserves a typo ("groud truth") in the prompt body.
# DO NOT correct it — fixing the typo drifts our scores away from the
# VisBrowse-Bench paper baseline.
VISBROWSE_BENCH_JUDGE_PROMPT = """You are a rigorous question-and-answer quality assessment expert. Given a question, your task is to compare the "groud truth" and the "model answer", determine whether they are semantically equivalent, and output "yes" or "no".
question: {question}
ground truth: {correct_answer}
model answer: {response}
"""


_YES_NO_RE = re.compile(r"\b(yes|no)\b")


def parse_visbrowse_bench_verdict(raw_response: str) -> Optional[str]:
    """Parse the judge's raw response into ``"yes"`` / ``"no"`` / ``None``.

    Upstream protocol: scan for the first ``\\b(yes|no)\\b`` token in the
    response (case-insensitive after strip+lower). Returns ``None`` if
    neither token is found, in which case the caller should retry the
    judge call.
    """
    raw = (raw_response or "").strip().lower()
    m = _YES_NO_RE.search(raw)
    if m:
        return m.group(1)
    return None
