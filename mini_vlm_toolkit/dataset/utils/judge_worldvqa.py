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
### Examples

**Example 1 (Incorrect - Granularity Mismatch/Too Coarse)**
Input:
'''
Question: 图片中属于什么类型的田地？
Model Answer: 图片中展示的是梯田。梯田是在山坡地上开垦并修筑的阶梯状农田。
Ground Truth Answer: 龙脊梯田
'''
Evaluation: 标准答案特指“龙脊梯田”，模型只回答了通用的“梯田”。模型答案层级比答案层级更粗略，未能提供标准答案所需的特指信息，属于层级不一致导致的回答错误。
Label: Incorrect

**Example 2 (Correct - Finer Granularity)**
Input:
'''
Question: What weather phenomenon is in the image?
Model Answer: Based on the visual evidence in the image, the weather phenomenon shown is a **severe storm with extremely high winds**, most likely a **tornado** or a very powerful **hurricane/typhoon**.
Ground Truth Answer: High winds
'''
Evaluation: The ground truth is "high winds," and a "tornado" is a more specific and granular type of high wind. The semantics are correct and the detail is finer.
Label: Correct

**Example 3 (Correct)**
Input:
'''
Question: 图中内容是什么品牌的logo？
Model Answer: via浏览器
Ground Truth Answer: via
'''
Evaluation: 模型答案“via浏览器”包含了标准答案“via”，核心语义一致，且“via浏览器”是更具体的描述，层级上是匹配的。
Label: Correct

**Example 4 (Unattempted)**
Input:
'''
Question: Which athlete is in the image?
Model Answer: I cannot answer this question as I do not have relevant sports data.
Ground Truth Answer: Wout Weghorst
'''
Evaluation: The model explicitly states its inability to answer and provides no incorrect information.
Label: Unattempted

**Example 5 (Incorrect)**
Input:
'''
Question: 图片中展示的是什么苹果品种？
Model Answer: 我觉得可能是阿克苏苹果。
Ground Truth Answer: 烟台苹果
'''
Evaluation: 虽然模型用了“可能”等词汇，但它给出的具体答案“阿克苏苹果”与标准答案“烟台苹果”不符，提供了错误信息。
Label: Incorrect

**Example 6 (Unattempted)**
Input:
'''
Question: What is the name of the insect in this image?
Model Answer: This is a photo of an insect. To find the species, consult an entomologist or use recognition software.
Ground Truth Answer: Japanese rhinoceros beetle
'''
Evaluation: The model does not attempt to name the insect and suggests the user search elsewhere, providing no incorrect information.
Label: Unattempted

---
### Current Task
Input:
'''
Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth_answer}
'''

Evaluation:
"""


def build_judge_worldvqa_prompt(question: str, model_answer: str, ground_truth_answer: str) -> str:
    """Fill the WorldVQA judge template with the current sample."""
    return JUDGE_WORLDVQA_PROMPT.format(
        question=question,
        model_answer=model_answer,
        ground_truth_answer=ground_truth_answer,
    )


_LABEL_LINE = re.compile(
    r'^[ \t]*(?:\d+\.[ \t]*)?\**[ \t]*label[ \t]*\**[ \t]*[:：][ \t]*\**[ \t]*["\'“]?'
    r'(correct|incorrect|unattempted)["\'”]?[ \t]*\**[ \t]*\.?[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)


def parse_judge_worldvqa_judgement(content: str) -> Dict[str, Any]:
    """Parse the judge output into a structured verdict.

    Returns dict with:
      - ``correct``: 1 (Correct) / 0 (Incorrect or Unattempted)
      - ``category``: 'correct' | 'incorrect' | 'unattempted' | 'parse_error'
      - ``reasoning``: the judge's evaluation text
      - ``raw``: the full judge reply
    """
    text = (content or "").strip()

    # The verdict is the last line of the form "Label: <verdict>" (optionally numbered, bolded or
    # quoted). Mentions of a label inside the reasoning, or a label line with trailing text, do not
    # count; no such line is a parse failure that the caller retries, never a verdict.
    matches = list(_LABEL_LINE.finditer(text))
    if not matches:
        return dict(correct=0, category='parse_error', reasoning='judge_worldvqa failed to parse', raw=text)
    m = matches[-1]
    label = m.group(1).lower()

    # The prompt ends with "Evaluation:", so the reasoning is whatever precedes the label.
    reason = text[:m.start()]
    em = re.search(r'evaluation\**\s*[:：]?\s*\**', reason, re.IGNORECASE)
    if em:
        reason = reason[em.end():]
    reason = re.sub(r'[\s*]*\d*\.?[\s*]*$', '', reason).strip()

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
