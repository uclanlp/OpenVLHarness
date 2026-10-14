"""HLE-style LLM judge prompt + parsing helpers.

This is the same prompt protocol used by:
  - HLE official ``evaluate_hle_official.py``
  - WebWatcher's eval pipeline (uniformly applied across hle / livevqa /
    mmsearch / simplevqa / browsecomp-vl)
  - BrowseComp-VL paper's ``evaluate_hle_official.py``

Selectable with ``judge_prompt: hle`` on mmsearch, mmsearch_plus, vdrbench,
hle and browsecomp_vl, for comparison with papers that adopt the HLE judge
protocol; these benchmarks default to ``judge_vdr``.

The judge is asked to extract the model's final answer, compare against
ground truth, and emit a JSON object with binary ``correct: yes|no``.
Aggregated accuracy = mean(correct == "yes") × 100.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# The HLE-style judge prompt
# ---------------------------------------------------------------------------
HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.

Respond **strictly in a valid JSON object** matching this schema:
{{
  "extracted_final_answer": string,
  "reasoning": string,
  "correct": "yes" | "no",
  "confidence": integer,
  "strict": true
}}"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
_YES_TOKENS = ("y", "yes", "true", "positive")
_NO_TOKENS = ("n", "no", "false", "negative")


def parse_hle_judgement(response_text: str) -> dict:
    """Parse the JSON returned by an HLE-style judge.

    Returns a dict with keys:
      - correct (1 or 0):                 the binary verdict
      - extracted_answer (str):           judge's extracted final answer
      - reasoning (str):                  judge's reasoning text
      - confidence (int, default 100):    judge's reported confidence

    If the response is malformed or no JSON object is found, falls back
    to scanning for a bare ``yes``/``no`` token; on total failure
    returns ``correct=0``.
    """
    text = str(response_text or "").strip()

    # Strategy 1: locate a JSON object and parse it
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("correct", "")).strip().lower()
            return dict(
                correct=1 if verdict in _YES_TOKENS else 0,
                extracted_answer=str(obj.get("extracted_final_answer", "")),
                reasoning=str(obj.get("reasoning", "")),
                confidence=int(obj.get("confidence", 100))
                    if str(obj.get("confidence", "")).strip().isdigit()
                    else 100,
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: bare yes/no token
    raw = text.lower()
    if "correct: yes" in raw or raw in _YES_TOKENS:
        return dict(correct=1, extracted_answer="", reasoning=text, confidence=100)
    if "correct: no" in raw or raw in _NO_TOKENS:
        return dict(correct=0, extracted_answer="", reasoning=text, confidence=100)

    return dict(correct=0, extracted_answer="", reasoning="judge failed to parse", confidence=0)


def grade_with_hle_judge(model, question: str, correct_answer: Any, response: str,
                         retry: int = 3) -> dict:
    """Run the HLE-style judge on a single (question, gt, response) tuple.

    ``correct_answer`` may be a string, list, or list-of-lists; we flatten
    one level and pick the first variant for the judge call (matching the
    upstream ``evaluate_hle_official.py`` convention of using only the
    first ground-truth variant).

    Returns the parsed judgement dict (see ``parse_hle_judgement``); if
    the model never returns a parseable response after ``retry`` attempts,
    ``correct=0`` with ``reasoning='judge unreachable'``.
    """
    if isinstance(correct_answer, list):
        flat = []
        for x in correct_answer:
            if isinstance(x, list):
                flat.extend(x)
            else:
                flat.append(x)
        ca = str(flat[0]) if flat else ""
    else:
        ca = str(correct_answer)

    prompt = HLE_JUDGE_PROMPT.format(
        question=str(question),
        correct_answer=ca,
        response=str(response),
    )

    last_text: Optional[str] = None
    for _ in range(retry):
        ans = model.generate(prompt)
        if "Failed to obtain answer via API" in (ans or ""):
            continue
        last_text = ans
        parsed = parse_hle_judgement(ans)
        if parsed.get("reasoning") != "judge failed to parse":
            return parsed

    return dict(
        correct=0,
        extracted_answer="",
        reasoning="judge unreachable after retries"
            if last_text is None
            else f"unparseable: {str(last_text)[:300]}",
        confidence=0,
    )
