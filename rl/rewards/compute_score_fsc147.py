"""Reward function for FSC147 visual counting with SAM3 tool use."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from mini_vlm_toolkit.dataset.utils.fsc147 import extract_integer_from_prediction

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{(\d+)\}")


def _extract_boxed_int(text: str) -> Optional[int]:
    """Return the integer inside \\boxed{N}, or None if no valid boxed expression found."""
    match = _BOXED_PATTERN.search(text or "")
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _extract_int_fallback(text: str) -> Optional[int]:
    """Best-effort integer extraction for accuracy when \\boxed{} is absent."""
    return extract_integer_from_prediction(text)


def _extract_tool_call_stats(solution_str: str) -> tuple[int, int]:
    total = 0
    wellformed = 0
    for block in _TOOL_CALL_PATTERN.findall(solution_str or ""):
        total += 1
        try:
            payload = json.loads(block)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if "name" not in payload or "arguments" not in payload:
            continue
        if not isinstance(payload["name"], str):
            continue
        if not isinstance(payload["arguments"], dict):
            continue
        wellformed += 1
    return total, wellformed


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> dict:
    del data_source, kwargs
    extra_info = extra_info or {}

    gt_int = int(ground_truth)
    boxed_int = _extract_boxed_int(solution_str)
    # For accuracy: prefer boxed value; fall back to best-effort extraction.
    pred_int = boxed_int if boxed_int is not None else _extract_int_fallback(solution_str)

    format_correct = boxed_int is not None
    acc = 1.0 if (pred_int is not None and pred_int == gt_int) else 0.0
    format_bonus = 0.1 if format_correct else 0.0

    tool_calls, wellformed = _extract_tool_call_stats(solution_str)
    tool_bonus = 0.1 if tool_calls > 0 else 0.0
    malformed_penalty = 0.05 if tool_calls > wellformed else 0.0

    score = acc + format_bonus + tool_bonus - malformed_penalty

    wellformed_rate = float(wellformed) / float(tool_calls) if tool_calls > 0 else 0.0

    num_turns = extra_info.get("num_turns")
    avg_tool_turns = float(tool_calls)
    if isinstance(num_turns, (int, float)) and num_turns > 0:
        avg_tool_turns = float(tool_calls) / float(num_turns)

    return {
        "score": float(score),
        "acc": float(acc),
        "format_correct": float(format_correct),
        "format_bonus": float(format_bonus),
        "tool_bonus": float(tool_bonus),
        "malformed_tool_penalty": float(malformed_penalty),
        "wellformed_tool_json_rate": float(wellformed_rate),
        "avg_tool_turns": float(avg_tool_turns),
    }
