"""Judge_WorldVQA — byte-exact port of the WorldVQA paper grader.

WorldVQA (arXiv 2602.02537v1, Moonshot AI) — "Measuring Atomic World
Knowledge in Multimodal Large Language Models" — evaluates via an LLM
judge that classifies predictions into ``Correct`` / ``Incorrect`` /
``Unattempted``:

  - Correct   — model answer contains core information of ground truth,
                semantically consistent, granularity equal or finer.
  - Incorrect — contradicts GT, wrong entity/value, or too coarse.
  - Unattempted — model explicitly refuses / says "I don't know" / etc.

Byte-exact prompt from ``WorldVQA/eval/eval.py::JUDGE_WORLDQA_PROMPT_EN``.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# ---------------------------------------------------------------------------
# System prompt — byte-exact from WorldVQA/eval/eval.py:22
# ---------------------------------------------------------------------------
JUDGE_WORLDVQA_PROMPT = """
### Role
You are an expert judge specialized in evaluating the correctness of answers. Your task is to assess whether a model-generated answer is correct based on a given question, the model's response, and the ground truth answer.

### Task: Evaluate Answer Correctness
Please classify the model's response into one of the following three categories. Ignore differences in formatting, punctuation, language (Chinese vs. English), or abbreviations/full names. Focus strictly on the **core semantics** and the **level of detail (granularity)**:

1. **Correct**:
    - The model answer contains the core information of the ground truth.
    - The model answer is semantically consistent with the ground truth and contains no contradictions.
    - **The granularity of the model answer is equal to or finer than the ground truth.**
    - Extra irrelevant information is allowed as long as it does not conflict with the ground truth.

2. **Incorrect**:
    - The model answer provides information that contradicts the ground truth.
    - The model answer provides the wrong specific entity, value, or description.
    - **The granularity of the model answer is coarser than the ground truth**, leading to incomplete or insufficiently specific information.
    - Even if the model expresses uncertainty but follows up with a wrong answer (e.g., "I'm not sure, maybe it's B" when the truth is A), it is considered Incorrect.

3. **Unattempted**:
    - The model explicitly states it does not know the answer (e.g., "I don't know," "I cannot answer this question").
    - The model suggests the user search elsewhere (e.g., "Please search the internet").
    - The model answer contains no information from the ground truth but provides no incorrect or contradictory information.

### Output Format
Please strictly follow this two-line format for your output:
1. **Evaluation**: [A brief explanation of your reasoning]
2. **Label**: [Final classification: "Correct", "Incorrect", or "Unattempted"]

---
### Input

Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}

Evaluation:
"""


def build_judge_worldvqa_prompt(question: str, model_answer: str, ground_truth_answer: str) -> str:
    """Fill the WorldVQA judge template with the current sample."""
    return JUDGE_WORLDVQA_PROMPT.format(
        question=question,
        model_answer=model_answer,
        ground_truth_answer=ground_truth_answer,
    )


def parse_judge_worldvqa_judgement(content: str) -> Dict[str, Any]:
    """Parse the judge output into a structured verdict.

    Returns dict with:
      - ``correct``: 1 (Correct) / 0 (Incorrect or Unattempted)
      - ``category``: 'correct' | 'incorrect' | 'unattempted'
      - ``reasoning``: extracted evaluation string (truncated 500)
      - ``raw``: original content (truncated 500)
    """
    text = (content or "").strip()

    # Try to extract Label
    label = None
    m = re.search(r'label\**\s*[::]?\s*\**\s*"?(correct|incorrect|unattempted)', text, re.IGNORECASE)
    if m:
        label = m.group(1).lower()
    else:
        # Fallback: look for any of the three keywords in the last 200 chars
        tail = text[-200:].lower()
        for candidate in ('correct', 'incorrect', 'unattempted'):
            # Match whole word only
            if re.search(r'\b' + candidate + r'\b', tail):
                # Prefer specific match; incorrect trumps correct if both appear
                label = candidate
                # Don't break — allow later candidates to override
        # If no label found, treat as incorrect
        if label is None:
            label = 'incorrect'

    # Extract evaluation reasoning
    reason = ''
    m = re.search(r'evaluation\s*[::]?\s*([^\n]*)', text, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()

    if label == 'correct':
        correct = 1
    else:
        correct = 0

    return dict(
        correct=correct,
        category=label,
        reasoning=(reason or f'Label: {label}'),
        raw=text,
    )
