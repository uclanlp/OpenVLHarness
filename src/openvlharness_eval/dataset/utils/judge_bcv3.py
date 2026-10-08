"""Judge_BCV3 — verbatim port of the BrowseComp-V3 (`bcv3`) paper grader.

NAMING ALIAS: when Rui refers to "BCV3 judge" / "BrowseComp-V3 paper judge",
that is THIS judge (``judge_bcv3``). It is a TWO-DIMENSIONAL grader:

  - ``answer_correct`` (boolean) — does the prediction match the
    ground-truth final answer in core meaning?
  - ``process_score`` (float in [0.0, 1.0]) — fraction of ground-truth
    sub-goals whose ``key_info`` is covered by the prediction.

In the BrowseComp-V3 release the grader is implemented in
``examples/eval_rollout_results.py`` (PROMPT_TEMPLATE + RESPONSE_FORMAT
constants). This module ports both verbatim plus the structured-JSON
output schema and the verdict extractor.

Differences from the other judges wired into mini-vlm-toolkit:

  - ``judge_vdr``: contains-based, single yes/no, no sub-goal handling
  - ``judge_rs``: rule-based, single XML correct/incorrect
  - ``hle``: HLE-style yes/no with confidence
  - ``judge_bcv3``: TWO scores (binary + continuous) + structured JSON

How predictions feed the BCV3 judge in our framework:

The original BCV3 evaluator expects predictions structured as
``{final_answer, subgoals: [...]}`` — but our framework just stores a raw
prediction string per sample. To bridge: we pass the raw prediction
string as the ``pred_answer`` AND as a single-element ``pred_subgoals``
list. The judge can still:

  - Compute ``answer_correct`` reliably (compares against
    ``gt_answer``, no sub-goals needed).
  - Compute ``process_score`` by checking whether each GT sub-goal's
    ``key_info`` appears anywhere in the prediction text.

If the model is later prompted to emit explicit ``Sub-goal 1: ...`` style
output, a richer parser can split the prediction into a real list. For
the v1 wiring we treat the prediction as one piece of text.

Used as a per-benchmark opt-in via env ``MMDR_JUDGE_PROMPT=judge_bcv3``
or ``judge_kwargs['judge_prompt'] = 'judge_bcv3'``. Default for
``BrowseCompV3`` is ``judge_vdr`` (consistent with the rest of the
search-bench family); ``judge_bcv3`` is the opt-in to recover the paper
protocol + process_score breakdown.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# v1 prompt — BrowseComp-V3 release (eval_rollout_results.py
# PROMPT_TEMPLATE), byte-exact. ACTIVE.
# ---------------------------------------------------------------------------
JUDGE_BCV3_PROMPT = """You are a professional AI evaluation agent. Your task is to evaluate a model’s performance on multi-step reasoning tasks along two dimensions:

1. Correctness of the final answer.
2. Quality of the reasoning process.

You will be given the following input fields:

- Question: {question}

- Ground Truth (Reference Annotation):
  - Gold final answer: {gt_answer}
  - Ground-truth sub-goals: {formatted_sub_goals}
    Each ground-truth sub-goal includes a key_info field (critical facts or intermediate conclusions) that specifies the condition for that sub-goal to be considered achieved.

- Model Prediction:
  - Predicted final answer: {pred_answer}
  - Predicted sub-goals: {formatted_pred_subgoals}

Your evaluation must be based strictly on this provided information. Do not introduce external knowledge beyond what is necessary for consistency checking.

------------------------------------------------------------

EVALUATION RULES
------------------------------------------------------------

EVALUATION PRIORITY:
1. First, evaluate answer_correct based solely on final answer comparison.
2. Second, calculate process_score based on sub-goal achievement.
3. Third, IF AND ONLY IF answer_correct is true, check for obvious anomalies for flagging (this does NOT change answer_correct).

I. ANSWER CORRECTNESS (Final Answer)

Determine whether the model’s predicted final answer matches the ground-truth final answer:

- CORRECT:
  - The predicted final answer and the ground-truth final answer are equivalent in core meaning.
  - Differences in wording, format, or minor stylistic variations are allowed.

- INCORRECT:
  - The predicted final answer and the ground-truth final answer do not match in core meaning.
  - If multiple possible answers are acceptable according to the ground truth, treat the prediction as CORRECT only if it is semantically compatible with at least one acceptable answer.

IMPORTANT: The answer_correct value is determined SOLELY by comparing the predicted final answer with the ground-truth final answer. Anomaly detection (Section III) does NOT affect this value.

II. PROCESS SCORE (Reasoning Process Quality)

1. Ground-truth anchored evaluation:
   - The list of ground-truth sub-goals defines the authoritative set of sub-goals to be evaluated.
   - For each ground-truth sub-goal i, you must determine whether this sub-goal is achieved by searching over ALL predicted sub-goals.
   - The matching between ground-truth sub-goals and predicted sub-goals is content-based (by key_info), not position-based. The order of predicted sub-goals may differ from the ground truth.
   - The number of elements in subgoal_vector and subgoal_details MUST equal the number of ground-truth sub-goals.
   - If there are more predicted sub-goals than ground-truth sub-goals, these extra predicted sub-goals MUST NOT create additional entries in subgoal_vector or subgoal_details. However, you may still use them for anomaly detection (fabrication, contradictions, faulty logic).
2. Scoring each ground-truth sub-goal:
   For each ground-truth sub-goal G_i with its key_info:
   - ACHIEVED (1 point):
     - There exists at least one predicted sub-goal whose content correctly identifies and recovers the key_info of G_i (semantic equivalence is allowed).
     - The information is reasonable, does not involve fabricated facts, and does not conflict with any key_info in the ground truth.
   - NOT_ACHIEVED (0 points), if any of the following holds:
     - No predicted sub-goal correctly captures the key_info of G_i.
     - All candidate predicted sub-goals that mention related content contain clear fabrication or contradict the ground-truth key_info.
     - The prediction effectively omits this sub-goal (i.e., no predicted sub-goal provides the required key_info).
3. Process score calculation:
   - PROCESS_SCORE = (number of ACHIEVED ground-truth sub-goals) / (total number of ground-truth sub-goals).
   - You must also output a subgoal_vector such as [1,0,1,...], where each entry is 1 if the corresponding ground-truth sub-goal is ACHIEVED and 0 otherwise.
   - The length of subgoal_vector MUST equal the total number of ground-truth sub-goals.
   - subgoal_details MUST contain one entry per ground-truth sub-goal, in ground-truth order (index 1..N), describing whether that ground-truth sub-goal is achieved and why.


III. ANOMALY DETECTION RULE (FOR HUMAN REVIEW)

⚠️ TRIGGER CONDITION: This rule applies ONLY when answer_correct is true (i.e., the predicted final answer matches the ground-truth final answer).

⚠️ CRITICAL CLARIFICATION:
- Anomaly detection is for FLAGGING purposes only.
- anomaly_flag does NOT change the answer_correct value.
- answer_correct reflects only whether the final answer is correct.
- Flagged anomalies indicate cases that may need human review to verify the reasoning quality.

When the final answer is correct, perform a careful check for potential reasoning anomalies. However, you must be CONSERVATIVE and set anomaly_flag to true ONLY when there is CLEAR and UNMISTAKABLE evidence of severe issues.

Flag anomaly_flag as true ONLY in these cases:

1. Explicit Guessing or Unsupported Answering:
   - The prediction explicitly states guessing behavior (phrases like \"I guess,\" \"I’ll try,\" \"randomly choosing\").
   - The final answer has absolutely zero connection to the reasoning provided in sub-goals.
   - There is no plausible logical path from the sub-goals to the answer.

   DO NOT flag if: The reasoning is weak but present, or the connection is unclear but possible.

2. Severe Factual Contradiction with Ground Truth:
   - A predicted sub-goal states a fact that directly, unmistakably contradicts a key_info in the ground-truth sub-goals.
   - Example of flaggable contradiction: Predicted says \"Event occurred in 1990\" but ground-truth key_info clearly states \"Event occurred in 2000.\"
   - This must be a clear factual conflict, not a different interpretation or reasoning approach.

   DO NOT flag if: The prediction uses different reasoning, omits information, or has minor inconsistencies.

3. Logically Impossible Reasoning Chain:
   - The sub-goals contain statements that are internally contradictory in a way that makes the reasoning logically impossible.
   - Example: \"X > Y and Y > Z, therefore X < Z\" (violates transitivity).

   DO NOT flag if: The reasoning has gaps, is incomplete, or uses a suboptimal approach.

⚠️ WHEN IN DOUBT, DO NOT FLAG:
- If you are uncertain whether an anomaly exists, set anomaly_flag to false.
- Missing or incomplete reasoning is already captured by process_score.
- Different reasoning approaches are acceptable.
- The bar for flagging should be HIGH - only flag obvious, severe issues.

IV. CONFIDENCE LEVEL

You must output a confidence level describing how reliable your evaluation is, based on clarity and sufficiency of the information given:

- HIGH:
  - Evidence is clear and sufficient; sub-goal alignment and correctness judgments are straightforward.

- MEDIUM:
  - Some minor ambiguity exists (e.g., slightly unclear wording), but you can still make a reasonably stable judgment.

- LOW:
  - The ground truth and/or prediction are highly ambiguous, incomplete, or under-specified, making the evaluation unstable or speculative.

------------------------------------------------------------

OUTPUT SPECIFICATION
------------------------------------------------------------

You MUST output a single JSON object that conforms to the provided JSON schema. Do not include any additional text outside of this JSON object.

The JSON object must include:

- answer_correct (boolean):
  - true if the predicted final answer matches the ground-truth final answer in core meaning.
  - false if the predicted final answer does not match the ground-truth final answer.
  - NOTE: This value is NOT affected by anomaly_flag. It reflects only the final answer comparison.

- process_score (number in [0.0, 1.0]):
  - The PROCESS_SCORE value defined above.

- subgoal_vector (array of 0/1 integers):
  - Binary vector where each element corresponds to a ground-truth sub-goal, in order.
  - Length = number of ground-truth sub-goals.

- subgoal_details (array):
  - Each element is an object describing the evaluation of one ground-truth sub-goal.
  - One element per ground-truth sub-goal, with strictly aligned indices.

- anomaly_flag (boolean):
  - true if any anomaly (fabrication, faulty-logic success, contradiction of key_info) is detected.
  - false otherwise.

- anomaly_reason (string):
  - If anomaly_flag is true, concisely describe the anomaly.
  - If anomaly_flag is false, set this to \"None\".

- confidence (string):
  - One of \"HIGH\", \"MEDIUM\", or \"LOW\".

- judge_reasoning (string):
  - A concise but explicit explanation of why you reached this evaluation, referencing both process and final answer.

Your response must be valid JSON and satisfy the JSON schema given in the response_format configuration.
"""


# Version tag — bump when the prompt changes. Persisted into acc.csv /
# llm_judge.json so completed runs remain traceable to the exact judge
# protocol that produced the score.
JUDGE_BCV3_VERSION = "judge_bcv3_v1_20260623"


# Response format schema (verbatim from BrowseComp-V3 release). Used when
# the OpenAI SDK supports structured outputs; otherwise the parser falls
# back to extracting JSON from a free-form response.
JUDGE_BCV3_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "multistep_reasoning_evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "answer_correct": {"type": "boolean"},
                "process_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "subgoal_vector": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}},
                "subgoal_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subgoal_index": {"type": "integer", "minimum": 1},
                            "status": {"type": "string", "enum": ["ACHIEVED", "NOT_ACHIEVED"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["subgoal_index", "status", "reason"],
                        "additionalProperties": False,
                    },
                },
                "anomaly_flag": {"type": "boolean"},
                "anomaly_reason": {"type": "string"},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "judge_reasoning": {"type": "string"},
            },
            "required": [
                "answer_correct", "process_score", "subgoal_vector",
                "subgoal_details", "anomaly_flag", "anomaly_reason",
                "confidence", "judge_reasoning",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ---------------------------------------------------------------------------
# Formatting helpers — verbatim from eval_rollout_results.py
# ---------------------------------------------------------------------------
def format_subgoals(gt_subgoals: List[Dict[str, Any]]) -> str:
    """Render GT sub-goals as a bullet list with key_info exposed."""
    lines = []
    for sg in gt_subgoals or []:
        sid = sg.get("sg_id")
        desc = sg.get("description", "")
        key_info = sg.get("key_info", "")
        lines.append(f"- SG{sid}: {desc} | key_info: {key_info}")
    return "\n".join(lines) if lines else "无"  # 无 (none)


def format_pred_subgoals(pred_subgoals: List[str]) -> str:
    """Render predicted sub-goals (or the single raw prediction we pass in)."""
    if not pred_subgoals:
        return "无"
    return "\n".join(f"- {s}" for s in pred_subgoals)


# ---------------------------------------------------------------------------
# Verdict parsing — robust JSON extraction matching the upstream behavior.
# Falls through:
#   1) direct json.loads on the whole string
#   2) last ```json ... ``` fenced block
#   3) first {...} block
# ---------------------------------------------------------------------------
def extract_verdict(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    raw = str(text).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    fenced = list(re.finditer(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE))
    for m in reversed(fenced):
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return {}


def parse_judge_bcv3_judgement(text: Any) -> Dict[str, Any]:
    """Public alias mirroring the ``parse_judge_<judge>_judgement`` convention
    used by ``judge_vdr`` / ``judge_rs`` so this judge plugs into the
    existing per-bench dispatch with the same signature.

    Returns a dict with:
      - ``correct`` (bool): True iff ``answer_correct`` is True in verdict
      - ``process_score`` (float): from verdict, clipped to [0, 1]
      - ``verdict`` (dict): raw parsed verdict (or {} on parse failure)
      - ``confidence`` (str): "HIGH" / "MEDIUM" / "LOW" / "" if missing
      - ``anomaly_flag`` (bool)
      - ``anomaly_reason`` (str)
    """
    v = extract_verdict(text)
    correct = bool(v.get("answer_correct"))
    process = v.get("process_score")
    try:
        process = float(process)
    except (TypeError, ValueError):
        process = 0.0
    process = max(0.0, min(1.0, process))
    return {
        "correct": correct,
        "process_score": process,
        "verdict": v,
        "confidence": v.get("confidence", ""),
        "anomaly_flag": bool(v.get("anomaly_flag")),
        "anomaly_reason": v.get("anomaly_reason", ""),
    }


def build_judge_bcv3_prompt(
    question: str,
    gt_answer: str,
    gt_subgoals: List[Dict[str, Any]],
    pred_answer: str,
    pred_subgoals: List[str],
) -> str:
    """Render the BCV3 judge prompt with the 5 placeholder fields filled."""
    return JUDGE_BCV3_PROMPT.format(
        question=question,
        gt_answer=gt_answer,
        formatted_sub_goals=format_subgoals(gt_subgoals or []),
        pred_answer=pred_answer,
        formatted_pred_subgoals=format_pred_subgoals(pred_subgoals or []),
    )
