"""Geometry3K reward adapter for direct-answer scoring."""

from __future__ import annotations

import re
from typing import Any, Optional


def _extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _fallback_score(solution_str: str, ground_truth: str) -> tuple[float, float, str]:
    pred_boxed = _extract_boxed(solution_str)
    pred = _normalize_text(pred_boxed if pred_boxed else solution_str)
    gt = _normalize_text(ground_truth)
    acc = 1.0 if pred and gt and pred == gt else 0.0
    return acc, acc, "fallback_exact_match"


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict[str, float | str]:
    del data_source, extra_info, kwargs

    try:
        from verl.utils.reward_score.geo3k import compute_score as geo3k_score

        score = float(geo3k_score(solution_str, ground_truth))
        acc = 1.0 if score >= 0.5 else 0.0
        scorer = "verl_geo3k"
    except Exception:
        score, acc, scorer = _fallback_score(solution_str, ground_truth)

    return {"score": float(score), "acc": float(acc), "scorer": scorer}
