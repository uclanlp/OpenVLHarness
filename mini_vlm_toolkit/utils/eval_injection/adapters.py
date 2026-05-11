"""Per-dataset eval-injection adapters.

Each adapter reads a dataset's per-instance eval output (judge xlsx, per_row csv,
gpt_eval.json, or the eval_file itself) and returns a mapping
``task_id -> {"correct": bool|None, "score": float|None, "raw": {...}}``
that the merger writes into ``full_trajectories/<task_id>.json``.

Adapter signature: ``(result_file, dataset_obj, judge_kwargs) -> dict[str, dict]``.
Returning an empty dict is a no-op (warning is logged by the caller).
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any, Dict, Optional

import pandas as pd

from ...smp import load
from ...smp.file import get_intermediate_file_path


_JUDGE_NAME_MAP = {"chatgpt-0125": "openai", "gpt-4-0125": "gpt4"}


def _judge_tag(judge_kwargs: Dict[str, Any]) -> str:
    model = judge_kwargs.get("model", "exact_matching")
    if model is None or model == "exact_matching":
        return "exact_matching"
    name = _JUDGE_NAME_MAP.get(model, model)
    if "/" in name:
        name = name.split("/")[-1]
    return name


def _resolve_result_xlsx(eval_file: str, judge_tag: str) -> Optional[str]:
    """Find the per-instance result xlsx written by MCQ-style evaluators.

    Tries the ``_<judge_tag>_result.xlsx`` pattern first; if not on disk, falls
    back to globbing ``_*_result.xlsx`` and picking the most recently modified
    match. Returns ``None`` if nothing matches.
    """
    candidate = get_intermediate_file_path(eval_file, f"_{judge_tag}_result", "xlsx")
    if os.path.exists(candidate):
        return candidate
    stem = eval_file.rsplit(".", 1)[0]
    matches = sorted(glob.glob(f"{stem}_*_result.xlsx"), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _safe_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no", ""):
            return False
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return bool(f)


def _row_raw(row: pd.Series, cols) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = row[c]
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            out[c] = v
    return out


def _task_id(idx: Any) -> str:
    """Normalize an index value into the same str form used for trajectory filenames.

    Pandas may load an integer ``index`` column as ``float64`` if any NaN snuck
    in, so ``str(idx)`` would be ``'5.0'`` and miss the ``5.json`` file. Cast
    integer-valued floats back to int form first.
    """
    try:
        f = float(idx)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(idx)


# ---------------------------------------------------------------------------
# Adapter 1: MCQ-style result xlsx (image_mcq, freak_mcq, freak_vqa, spatial bench MCQ,
#            RoboSpatial, RefSpatial, SparBench)
# ---------------------------------------------------------------------------

def mcq_result_adapter(
    result_file: str,
    dataset_obj,
    judge_kwargs: Dict[str, Any],
    *,
    correct_col: Optional[str] = "hit",
    score_col: Optional[str] = None,
    score_col_alt: Optional[str] = None,
    raw_cols=(),
) -> Dict[str, Dict[str, Any]]:
    """Read ``_<judge_tag>_result.xlsx`` and lift per-instance correctness.

    ``correct_col`` maps to ``eval.correct`` (cast to bool) and, if no
    ``score_col`` is given, also seeds ``eval.score = float(correct)``. When the
    benchmark mixes a binary and a continuous metric (SparBench), pass
    ``score_col_alt`` to use the continuous column wherever ``correct_col`` is
    NaN.
    """
    judge_tag = _judge_tag(judge_kwargs)
    xlsx_path = _resolve_result_xlsx(result_file, judge_tag)
    if xlsx_path is None:
        return {}

    df = pd.read_excel(xlsx_path)
    if "index" not in df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = row["index"]
        if pd.isna(idx):
            continue
        task_id = _task_id(idx)

        correct: Optional[bool] = None
        score: Optional[float] = None

        if correct_col and correct_col in df.columns:
            raw_correct = row[correct_col]
            if pd.notna(raw_correct):
                correct = _safe_bool(raw_correct)
                if score_col is None:
                    score = float(correct) if correct is not None else None

        if score_col and score_col in df.columns:
            score = _safe_float(row[score_col])
            if correct is None and score is not None:
                # Default threshold for continuous metrics; callers override
                # via correct_col=None when they want correct=null instead.
                correct = score > 0.0

        if (correct is None or score is None) and score_col_alt and score_col_alt in df.columns:
            alt = _safe_float(row[score_col_alt])
            if alt is not None:
                if score is None:
                    score = alt
                if correct is None:
                    correct = alt >= 0.5

        out[task_id] = {
            "correct": correct,
            "score": score,
            "raw": _row_raw(row, raw_cols),
        }
    return out


# ---------------------------------------------------------------------------
# Adapter 2: Counting per_row.csv (FSC147, PixmoCount, CountQA)
# ---------------------------------------------------------------------------

def counting_adapter(
    result_file: str,
    dataset_obj,
    judge_kwargs: Dict[str, Any],
    *,
    use_avg3_score: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Read ``_per_row.csv`` written by ``emit_counting_per_row``.

    ``use_avg3_score=True`` (CountQA) sets ``score`` to ``(exact_match + ra5 +
    ra10) / 3`` to match the dataset's reporting metric. Otherwise ``score``
    falls back to ``exact_match``.
    """
    csv_path = get_intermediate_file_path(result_file, "_per_row", "csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if "index" not in df.columns:
        return {}

    raw_cols = [c for c in ("gt", "pred", "abs_error", "sq_error", "ra5", "ra10", "is_focused")
                if c in df.columns]

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = row["index"]
        if pd.isna(idx):
            continue
        task_id = _task_id(idx)

        exact = _safe_bool(row.get("exact_match")) if "exact_match" in df.columns else None
        if use_avg3_score and {"exact_match", "ra5", "ra10"}.issubset(df.columns):
            try:
                score = (
                    float(row["exact_match"]) + float(row["ra5"]) + float(row["ra10"])
                ) / 3.0
            except (TypeError, ValueError):
                score = float(exact) if exact is not None else None
        else:
            score = float(exact) if exact is not None else None

        out[task_id] = {
            "correct": exact,
            "score": score,
            "raw": _row_raw(row, raw_cols),
        }
    return out


# ---------------------------------------------------------------------------
# Adapter 3: SimpleVQA gpt_eval.json (positional alignment with eval_file)
# ---------------------------------------------------------------------------

_SIMPLEVQA_LABEL_TO_SCORE = {"正确": 1.0, "错误": 0.0, "未尝试": 0.0}


def simplevqa_adapter(
    result_file: str,
    dataset_obj,
    judge_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    json_path = os.path.join(os.path.dirname(result_file), "gpt_eval.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        return {}

    data = load(result_file)
    if "index" not in data.columns:
        return {}
    indices = list(data["index"])
    if len(indices) != len(records):
        # Positional alignment is unsafe; fall back to no-op.
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for idx, rec in zip(indices, records):
        if pd.isna(idx) or not isinstance(rec, dict):
            continue
        judge_res = rec.get("judge_res") or {}
        verdict = judge_res.get("model_response") if isinstance(judge_res, dict) else None
        if isinstance(verdict, dict):
            verdict = verdict.get("conclusion") or verdict.get("model_response")
        verdict = str(verdict).strip() if verdict is not None else None

        score = _SIMPLEVQA_LABEL_TO_SCORE.get(verdict)
        correct = (verdict == "正确") if verdict is not None else None

        out[_task_id(idx)] = {
            "correct": correct,
            "score": score,
            "raw": {"judge_response": verdict, "gt": rec.get("answer")},
        }
    return out


# ---------------------------------------------------------------------------
# Adapter 4: in-place eval_file overwritten with score columns
#            (MMSearch family, RealXBench)
# ---------------------------------------------------------------------------

def inplace_score_adapter(
    result_file: str,
    dataset_obj,
    judge_kwargs: Dict[str, Any],
    *,
    correct_col: Optional[str] = None,
    score_col: Optional[str] = None,
    score_from_correct: bool = False,
    correct_threshold: Optional[float] = None,
    raw_cols=(),
) -> Dict[str, Dict[str, Any]]:
    """Read columns the dataset's ``evaluate()`` wrote back into ``result_file``.

    - ``correct_col`` (e.g. ``is_correct``) drives ``eval.correct``.
    - ``score_from_correct=True`` (paper-protocol judge wins) sets
      ``score = float(correct)``.
    - Otherwise, ``score_col`` (e.g. ``f1_score`` / ``requery_score``) sets the
      scalar score, and ``correct`` is derived via ``correct_threshold``
      (when provided) or left as ``None``.
    """
    if not os.path.exists(result_file):
        return {}
    df = load(result_file)
    if "index" not in df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = row["index"]
        if pd.isna(idx):
            continue
        task_id = _task_id(idx)

        correct: Optional[bool] = None
        score: Optional[float] = None

        if correct_col and correct_col in df.columns:
            correct = _safe_bool(row[correct_col])

        if score_from_correct:
            score = float(correct) if correct is not None else None
        elif score_col and score_col in df.columns:
            score = _safe_float(row[score_col])
            if correct is None and score is not None and correct_threshold is not None:
                correct = score >= correct_threshold

        out[task_id] = {
            "correct": correct,
            "score": score,
            "raw": _row_raw(row, raw_cols),
        }
    return out


# ---------------------------------------------------------------------------
# Adapter 5: VSI-Bench _result.xlsx (no judge tag in filename; mixed MCA/NA)
# ---------------------------------------------------------------------------

def vsibench_adapter(
    result_file: str,
    dataset_obj,
    judge_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    xlsx_path = get_intermediate_file_path(result_file, "_result", "xlsx")
    if not os.path.exists(xlsx_path):
        return {}
    df = pd.read_excel(xlsx_path)
    if "index" not in df.columns or "score" not in df.columns:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = row["index"]
        if pd.isna(idx):
            continue
        score = _safe_float(row["score"])
        # MCA rows score 0/1; NA rows carry MRA in [0,1]. Threshold @ 0.5 covers
        # both cleanly.
        correct = (score >= 0.5) if score is not None else None
        out[_task_id(idx)] = {
            "correct": correct,
            "score": score,
            "raw": _row_raw(row, ("type",)),
        }
    return out
