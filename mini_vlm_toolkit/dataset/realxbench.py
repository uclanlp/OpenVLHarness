"""
RealXBench dataset integration (glowol/RealXBench on HuggingFace).

A VQA benchmark from the DeepEyesV2 paper with 194 open-source samples.
Each sample has an image, a query, ground truth answer(s), and three binary
task-type flags: perception, search, reason.

Evaluation: token-level F1 + optional LLM judge, with breakdown by task type.
"""

import collections
import logging
import os
import re
import string
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403 - provides LMUDataRoot, load, dump, osp
from ..smp.file import get_intermediate_file_path

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "glowol/RealXBench"

ANSWER_FORMAT_INSTRUCTION = (
    "\n\nYou may provide reasoning first, but you must end your response "
    "with exactly one line in this format: Answer: <your concise answer>"
)


# ---------------------------------------------------------------------------
# HuggingFace loading helper
# ---------------------------------------------------------------------------

def _load_hf_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required to use RealXBench.\n"
            "Install via `pip install datasets`."
        ) from exc

    env_cache_dir = os.environ.get("HF_DATASETS_CACHE_DIR")
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        return load_dataset(_HF_DATASET_ID, split="train", cache_dir=env_cache_dir)

    try:
        return load_dataset(_HF_DATASET_ID, split="train")
    except TypeError as e:
        if "must be called with a dataclass type or instance" not in str(e):
            raise
        fallback_cache_dir = osp.join(LMUDataRoot(), "hf_cache", "datasets")
        os.makedirs(fallback_cache_dir, exist_ok=True)
        logger.warning(
            "RealXBench: HF datasets cache seems incompatible; retrying with cache_dir=%s.",
            fallback_cache_dir,
        )
        return load_dataset(_HF_DATASET_ID, split="train", cache_dir=fallback_cache_dir)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_to_pil(img: Any) -> Optional[Any]:
    if img is None:
        return None
    if hasattr(img, "save") and callable(getattr(img, "save")):
        return img
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            from PIL import Image
            import io
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if "path" in img and img["path"] is not None and osp.exists(img["path"]):
            from PIL import Image
            return Image.open(img["path"]).convert("RGB")
    if isinstance(img, bytes):
        from PIL import Image
        import io
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str) and img.strip() and osp.exists(img.strip()):
        from PIL import Image
        return Image.open(img.strip()).convert("RGB")
    return None


def _save_image(img: Any, path: str) -> bool:
    pil = _image_to_pil(img)
    if pil is None:
        return False
    try:
        pil.save(path)
        return True
    except Exception as e:
        logger.warning("RealXBench: failed to save image to %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Answer normalization & F1 (same as MMSearch)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    if text is None:
        text = ""
    text = str(text)

    def remove_articles(s: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _get_tokens(text: str) -> List[str]:
    if not text:
        return []
    return _normalize_answer(text).split()


def _f1_score(prediction: str, gold: str) -> float:
    gold_tokens = _get_tokens(gold)
    pred_tokens = _get_tokens(prediction)
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / float(len(pred_tokens))
    recall = num_same / float(len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def _extract_answer(prediction: str) -> str:
    if not prediction:
        return ""
    match = re.search(r"Answer:\s*(.*)", prediction, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return prediction.strip()


# ---------------------------------------------------------------------------
# Task type label helper
# ---------------------------------------------------------------------------

def _task_type_label(row) -> str:
    """Build a human-readable task type label from the three binary flags."""
    parts = []
    if row.get("perception", 0) == 1:
        parts.append("perception")
    if row.get("search", 0) == 1:
        parts.append("search")
    if row.get("reason", 0) == 1:
        parts.append("reason")
    return "+".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# LLM judge — RealXBench lenient protocol (utils/realxbench_judge.py).
# Note: this is NOT the HLE judge protocol; for cross-paper alignment with
# HLE / WebWatcher / BrowseComp-VL paper numbers, switch to
# utils/hle_judge.py.
# ---------------------------------------------------------------------------

from .utils.realxbench_judge import (
    REALXBENCH_JUDGE_PROMPT,
    parse_realxbench_verdict,
)


def _normalize_alts(alt_answers) -> list:
    if isinstance(alt_answers, str):
        return [a.strip() for a in alt_answers.split(" or ") if a.strip()]
    return alt_answers if isinstance(alt_answers, list) else []


def _judge_one(judge, row) -> Dict[str, Any]:
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)

    # Check all alternative answers
    alt_answers = _normalize_alts(row.get("alternative_gt_answers", []))
    all_answers = [gt] + alt_answers
    gt_combined = " or ".join(all_answers)

    prompt = REALXBENCH_JUDGE_PROMPT.format(
        question=question,
        gt_answer=gt_combined,
        prediction=pred_extracted or pred,
    )

    verdict = "error"
    raw_response = ""
    try:
        raw_response = judge.generate(prompt)
        verdict = parse_realxbench_verdict(raw_response)
    except Exception as e:
        logger.warning("RealXBench judge error: %s", e)

    return {"verdict": verdict, "raw_response": raw_response}


def _judge_one_judge_rs(judge, row) -> Dict[str, Any]:
    """Judge_RS variant. Opt-in via env ``MVT_JUDGE_PROMPT=judge_rs``."""
    from .utils.judge_rs import grade_with_judge_rs

    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)
    alt_answers = _normalize_alts(row.get("alternative_gt_answers", []))
    correct_answer_list = [gt] + alt_answers

    try:
        parsed = grade_with_judge_rs(
            judge, question, correct_answer_list, pred_extracted or pred,
        )
    except Exception as e:
        logger.warning("RealXBench Judge_RS error: %s", e)
        return {"verdict": "error", "raw_response": ""}

    if parsed["correct"] == 1:
        verdict = "correct"
    elif parsed.get("reasoning") in (
            "judge_rs failed to parse", "judge unreachable after retries"):
        verdict = "error"
    else:
        verdict = "incorrect"
    return {"verdict": verdict, "raw_response": str(parsed.get("reasoning", ""))[:200]}


def _run_llm_judge(eval_file: str, data: pd.DataFrame, judge_kwargs: Dict[str, Any]):
    """Switch between RealXBench paper-protocol judge (default) and
    Judge_RS via env ``MVT_JUDGE_PROMPT`` or ``judge_kwargs['judge_prompt']``.
    Cache + score files are suffixed with ``_judge_rs`` when active.
    """
    from .utils import build_judge
    from ..utils.mp_util import track_progress_rich

    judge_kwargs_local = dict(judge_kwargs)
    judge_kwargs_local.pop("enable_llm_judge", None)
    nproc = judge_kwargs_local.pop("nproc", 4)
    # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
    # the single source of truth; no per-evaluator default).

    judge_name = (
        os.environ.get('MVT_JUDGE_PROMPT')
        or judge_kwargs_local.pop('judge_prompt', None)
        or 'realxbench'
    ).strip().lower()
    if judge_name not in ('realxbench', 'judge_rs'):
        raise ValueError(f"Unknown judge_prompt {judge_name!r}; expected 'realxbench' or 'judge_rs'")
    judge_kwargs_local.pop('judge_prompt', None)

    judge = build_judge(**judge_kwargs_local)
    assert judge.working(), "RealXBench LLM judge requires a working OpenAI API"

    if judge_name == 'judge_rs':
        judge_one_fn = _judge_one_judge_rs
        cache_stem = "_realxbench_llm_judge_judge_rs_cache"
    else:
        judge_one_fn = _judge_one
        cache_stem = "_realxbench_llm_judge_cache"

    judge_cache_file = get_intermediate_file_path(eval_file, cache_stem, "pkl")
    cached: Dict[Any, Dict[str, Any]] = {}
    if osp.exists(judge_cache_file):
        cached = load(judge_cache_file)

    tups = []
    indices = []
    for _, row in data.iterrows():
        idx = row["index"]
        if idx in cached:
            continue
        tups.append((judge, row))
        indices.append(idx)

    if indices:
        track_progress_rich(
            judge_one_fn,
            tups,
            nproc=nproc,
            chunksize=nproc,
            keys=indices,
            save=judge_cache_file,
        )
        cached = load(judge_cache_file)

    verdicts = {idx: cached[idx] for idx in data["index"] if idx in cached}
    n_total = len(data)
    n_correct = sum(1 for v in verdicts.values() if v.get("verdict") == "correct")
    n_incorrect = sum(1 for v in verdicts.values() if v.get("verdict") == "incorrect")
    n_error = n_total - n_correct - n_incorrect
    accuracy = n_correct / n_total if n_total else 0.0

    # By task type
    by_task_type: Dict[str, Dict[str, Any]] = {}
    for _, row in data.iterrows():
        idx = row["index"]
        v = verdicts.get(idx, {}).get("verdict")
        is_correct = 1 if v == "correct" else 0

        tt = _task_type_label(row)
        by_task_type.setdefault(tt, {"length": 0, "correct": 0})
        by_task_type[tt]["length"] += 1
        by_task_type[tt]["correct"] += is_correct

    for v in by_task_type.values():
        v["accuracy"] = v["correct"] / v["length"] if v["length"] else 0.0

    judge_summary = {
        "total": {"total_length": n_total, "accuracy": accuracy},
        "correct": n_correct,
        "incorrect": n_incorrect,
        "error": n_error,
        "task_type": by_task_type,
    }

    score_stem = "_realxbench_llm_judge_judge_rs" if judge_name == 'judge_rs' else "_realxbench_llm_judge"
    judge_score_file = get_intermediate_file_path(eval_file, score_stem, "json")
    dump(judge_summary, judge_score_file)
    return judge_summary, verdicts


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RealXBench(ImageBaseDataset):
    """
    RealXBench VQA benchmark (DeepEyesV2).

    194 open-source samples with image + query + answer + task type flags.
    Evaluation: token-level F1 + optional LLM judge, broken down by
    task type (perception / search / reason and their combinations).
    """

    TYPE = "VQA"

    DATASET_URL = {
        "RealXBench": "https://huggingface.co/datasets/glowol/RealXBench",
    }
    DATASET_MD5 = {
        "RealXBench": "",
    }

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["RealXBench"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        if dataset != "RealXBench":
            raise ValueError(f"Unsupported dataset name for RealXBench: {dataset}")

        hf_ds = _load_hf_dataset()
        df = hf_ds.to_pandas()

        img_root = osp.join(lmu_root, "images", "RealXBench")
        os.makedirs(img_root, exist_ok=True)

        def _cache_image(idx, row):
            img = row.get("image", None)
            if img is None:
                return []
            filename = f"{idx}.jpg"
            path = osp.join(img_root, filename)
            if not osp.exists(path):
                if not _save_image(img, path):
                    return []
            return filename

        data = pd.DataFrame()
        data["index"] = [str(i) for i in range(len(df))]
        data["question"] = df["query"].astype(str)

        # Parse answers: primary answer + alternatives split by " or "
        raw_answers = df["answer"].astype(str)
        primary_answers = []
        alt_answers_list = []
        for ans in raw_answers:
            parts = [a.strip() for a in ans.split(" or ") if a.strip()]
            primary_answers.append(parts[0] if parts else ans)
            alt_answers_list.append(parts[1:] if len(parts) > 1 else [])

        data["answer"] = primary_answers
        data["alternative_gt_answers"] = alt_answers_list

        # Task type flags
        data["perception"] = df.get("perception", 0).astype(int)
        data["search"] = df.get("search", 0).astype(int)
        data["reason"] = df.get("reason", 0).astype(int)

        # Cache images
        data["image_path"] = [
            _cache_image(i, df.iloc[i]) for i in range(len(df))
        ]

        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs: List[Dict[str, Any]] = []
        val = line.get("image_path", "")
        if isinstance(val, list):
            img_rel = val[0] if len(val) > 0 else ""
        else:
            img_rel = str(val or "").strip()
        if img_rel:
            img_path = img_rel
            if not osp.isabs(img_path):
                img_path = osp.join(self.img_root, img_rel)
            msgs.append({"type": "image", "value": img_path})

        question = str(line.get("question", "") or "").strip()
        msgs.append({"type": "text", "value": question})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data and "answer" in data, (
            f"RealXBench.evaluate expects columns 'prediction' and 'answer' in {eval_file}"
        )

        # Normalize alternative answers
        def _to_alt_list(val: Any) -> List[str]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return []
            if isinstance(val, list):
                return [str(x) for x in val]
            s = str(val).strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                import json
                import ast
                # Try JSON first, then Python repr (TSV round-trip uses repr)
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(s)
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed]
                    except Exception:
                        continue
                return [s]
            return [s]

        data = data.copy()
        data["alternative_gt_answers"] = data.get("alternative_gt_answers", []).apply(_to_alt_list)

        # Compute F1
        f1_scores: List[float] = []
        for _, row in data.iterrows():
            pred = _extract_answer(str(row["prediction"]))
            gold = str(row["answer"])
            best = _f1_score(pred, gold)
            for alt in row["alternative_gt_answers"]:
                score_alt = _f1_score(pred, alt)
                if score_alt > best:
                    best = score_alt
            f1_scores.append(best)

        data["f1_score"] = f1_scores
        data["is_correct"] = [s >= 0.5 for s in f1_scores]
        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        # Aggregate by task type
        def _agg_by_task_type() -> Dict[str, Dict[str, float]]:
            groups: Dict[str, List[float]] = {}
            for _, row in data.iterrows():
                tt = _task_type_label(row)
                groups.setdefault(tt, []).append(row["f1_score"])
            return {
                name: {
                    "length": len(vals),
                    "average_f1": float(np.mean(vals)) if vals else 0.0,
                }
                for name, vals in groups.items()
            }

        summary: Dict[str, Any] = {
            "total": {
                "total_length": len(f1_scores),
                "average_f1": avg_f1,
            },
            "task_type": _agg_by_task_type(),
        }

        score_file = get_intermediate_file_path(eval_file, "_realxbench_f1", "json")
        dump(summary, score_file)

        # LLM judge always runs alongside F1 (paper protocol). F1 above is
        # retained in `*_realxbench_f1.json`.
        try:
            judge_summary, judge_verdicts = _run_llm_judge(
                eval_file=eval_file,
                data=data,
                judge_kwargs=judge_kwargs,
            )
            summary["llm_judge"] = judge_summary
            data["is_correct"] = [
                judge_verdicts.get(idx, {}).get("verdict") == "correct"
                for idx in data["index"]
            ]
        except Exception as exc:
            logger.warning(f"LLM judge evaluation failed: {exc}")

        try:
            dump(data, eval_file)
        except Exception as exc:
            logger.warning(f"Failed to write evaluated data back to {eval_file}: {exc}")

        return summary
