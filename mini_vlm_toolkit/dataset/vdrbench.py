"""
VDR-Bench dataset integration (Osilly/VDR-Bench on HuggingFace).

A VQA benchmark from the Vision-DeepResearch project with 2000 samples.
Each sample has an image, a question, and a ground-truth answer.
IDs encode difficulty (base / hard / hard1) and category
(Game, Art&Music, Sci&Tech, Sports, Nature, Architecture, Movie,
People, Object, Other).

Evaluation: token-level F1 + optional LLM judge, with breakdown by
difficulty and category.
"""

import collections
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403
from ..smp.file import get_intermediate_file_path

logger = logging.getLogger(__name__)

_HF_DATASET_IDS = {
    "VDRBench": "Osilly/VDR-Bench-testmini",      # 500 samples (default, matches paper)
    "VDRBench_Full": "Osilly/VDR-Bench",           # 2000 samples (full set)
}

ANSWER_FORMAT_INSTRUCTION = (
    "\n\nYou may provide reasoning first, but you must end your response "
    "with exactly one line in this format: Answer: <your concise answer>"
)


# ---------------------------------------------------------------------------
# HuggingFace loading helper
# ---------------------------------------------------------------------------

def _load_hf_dataset(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required to use VDR-Bench.\n"
            "Install via `pip install datasets`."
        ) from exc

    hf_id = _HF_DATASET_IDS.get(dataset_name, _HF_DATASET_IDS["VDRBench"])

    env_cache_dir = os.environ.get("HF_DATASETS_CACHE_DIR")
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        return load_dataset(hf_id, split="train", cache_dir=env_cache_dir)

    try:
        return load_dataset(hf_id, split="train")
    except TypeError as e:
        if "must be called with a dataclass type or instance" not in str(e):
            raise
        fallback = osp.join(LMUDataRoot(), "hf_cache", "datasets")
        os.makedirs(fallback, exist_ok=True)
        return load_dataset(hf_id, split="train", cache_dir=fallback)


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
    return None


def _save_image(img: Any, path: str) -> bool:
    pil = _image_to_pil(img)
    if pil is None:
        return False
    try:
        pil.save(path)
        return True
    except Exception as e:
        logger.warning("VDR-Bench: failed to save image to %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# ID parsing
# ---------------------------------------------------------------------------

def _parse_id(sample_id: str):
    """Parse VDR-Bench ID into (difficulty, category).

    ID format examples: base_Game_464, hard1_Art&Music_175, hard_Sports_632
    """
    parts = sample_id.split("_")
    difficulty = parts[0] if parts else "unknown"
    category = parts[1] if len(parts) >= 3 else "unknown"
    return difficulty, category


# ---------------------------------------------------------------------------
# Answer normalization & F1
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Normalize following VDR's original implementation:
    lowercase → collapse whitespace → strip surrounding punctuation (keep hyphens).
    """
    s = str(text) if text is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # Remove surrounding (not internal) punctuation, but keep hyphens
    s = re.sub(r"^[^\w\-]+|[^\w\-]+$", "", s)
    return s


def _f1_score(prediction: str, gold: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    pred_tokens = _normalize_answer(prediction).split()
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _extract_answer(prediction: str) -> str:
    if not prediction:
        return ""
    match = re.search(r"Answer:\s*(.*)", prediction, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    return prediction.strip()


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """You are an impartial judge evaluating whether a deep research report contains the correct answer.

[Question]
{question}

[Correct Answer]
{gt_answer}

[Deep Research Report]
{prediction}

Task: Determine if the deep research report contains the correct answer anywhere in its content.

Instructions:
1. Read through the entire research report carefully
2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)
3. Check if the information in the report is consistent with the correct answer
4. The answer does NOT need to be in a specific format or labeled as "final answer"
5. Provide your reasoning
6. Answer with "yes" if the report contains the correct answer, "no" if it doesn't or contradicts it

Output format:
correct: [yes/no]
reasoning: [your explanation]"""


def _judge_one(judge, row) -> Dict[str, Any]:
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)

    prompt = _JUDGE_PROMPT.format(
        question=question,
        gt_answer=gt,
        prediction=pred_extracted or pred,
    )

    verdict = "error"
    raw_response = ""
    try:
        raw_response = judge.generate(prompt)
        raw_lower = raw_response.strip().lower()
        # Parse "correct: yes/no" format (VDR original format)
        if "correct: yes" in raw_lower or "correct:yes" in raw_lower:
            verdict = "correct"
        elif "correct: no" in raw_lower or "correct:no" in raw_lower:
            verdict = "incorrect"
        # Fallback: check for standalone CORRECT/INCORRECT
        elif "incorrect" not in raw_lower and "correct" in raw_lower:
            verdict = "correct"
        elif "incorrect" in raw_lower:
            verdict = "incorrect"
    except Exception as e:
        logger.warning("VDR-Bench judge error: %s", e)

    return {"verdict": verdict, "raw_response": raw_response}


def _run_llm_judge(eval_file: str, data: pd.DataFrame, judge_kwargs: Dict[str, Any]):
    from .utils import build_judge
    from ..utils.mp_util import track_progress_rich

    judge_kwargs_local = dict(judge_kwargs)
    judge_kwargs_local.pop("enable_llm_judge", None)
    nproc = judge_kwargs_local.pop("nproc", 4)
    max_tokens = judge_kwargs_local.pop("judge_max_tokens", 128)

    judge = build_judge(max_tokens=max_tokens, **judge_kwargs_local)
    assert judge.working(), "VDR-Bench LLM judge requires a working OpenAI API"

    judge_cache_file = get_intermediate_file_path(eval_file, "_vdrbench_llm_judge_cache", "pkl")
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
            _judge_one, tups, nproc=nproc, chunksize=nproc,
            keys=indices, save=judge_cache_file,
        )
        cached = load(judge_cache_file)

    verdicts = {idx: cached[idx] for idx in data["index"] if idx in cached}
    n_total = len(data)
    n_correct = sum(1 for v in verdicts.values() if v.get("verdict") == "correct")
    n_incorrect = sum(1 for v in verdicts.values() if v.get("verdict") == "incorrect")
    n_error = n_total - n_correct - n_incorrect
    accuracy = n_correct / n_total if n_total else 0.0

    by_difficulty: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, Any]] = {}
    for _, row in data.iterrows():
        idx = row["index"]
        v = verdicts.get(idx, {}).get("verdict")
        is_correct = 1 if v == "correct" else 0

        diff = str(row.get("difficulty", "unknown"))
        by_difficulty.setdefault(diff, {"length": 0, "correct": 0})
        by_difficulty[diff]["length"] += 1
        by_difficulty[diff]["correct"] += is_correct

        cat = str(row.get("category", "unknown"))
        by_category.setdefault(cat, {"length": 0, "correct": 0})
        by_category[cat]["length"] += 1
        by_category[cat]["correct"] += is_correct

    for d in (by_difficulty, by_category):
        for v in d.values():
            v["accuracy"] = v["correct"] / v["length"] if v["length"] else 0.0

    judge_summary = {
        "total": {"total_length": n_total, "accuracy": accuracy},
        "correct": n_correct,
        "incorrect": n_incorrect,
        "error": n_error,
        "difficulty": by_difficulty,
        "category": by_category,
    }

    judge_score_file = get_intermediate_file_path(eval_file, "_vdrbench_llm_judge", "json")
    dump(judge_summary, judge_score_file)
    return judge_summary, verdicts


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VDRBench(ImageBaseDataset):
    """
    VDR-Bench VQA benchmark (Vision-DeepResearch).

    2000 samples with image + query + answer. Evaluation: token-level F1
    + optional LLM judge, broken down by difficulty (base / hard / hard1)
    and category (Game, Art&Music, Sci&Tech, Sports, Nature, Architecture,
    Movie, People, Object, Other).
    """

    TYPE = "VQA"

    DATASET_URL = {
        "VDRBench": "https://huggingface.co/datasets/Osilly/VDR-Bench-testmini",
        "VDRBench_Full": "https://huggingface.co/datasets/Osilly/VDR-Bench",
    }
    DATASET_MD5 = {
        "VDRBench": "",
        "VDRBench_Full": "",
    }

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["VDRBench", "VDRBench_Full"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        hf_ds = _load_hf_dataset(dataset)
        df = hf_ds.to_pandas()

        img_root = osp.join(lmu_root, "images", "VDRBench")
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
        data["index"] = df["id"].astype(str)
        data["question"] = df["question"].astype(str)
        data["answer"] = df["answer"].astype(str)

        # Parse difficulty and category from ID
        difficulties = []
        categories = []
        for sid in df["id"]:
            diff, cat = _parse_id(str(sid))
            difficulties.append(diff)
            categories.append(cat)
        data["difficulty"] = difficulties
        data["category"] = categories

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
        question = question + ANSWER_FORMAT_INSTRUCTION
        msgs.append({"type": "text", "value": question})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data and "answer" in data

        # Compute exact match (VDR's primary non-judge metric) and F1 (auxiliary)
        em_scores: List[float] = []
        f1_scores: List[float] = []
        for _, row in data.iterrows():
            pred = _extract_answer(str(row["prediction"]))
            gold = str(row["answer"])
            pred_norm = _normalize_answer(pred)
            gold_norm = _normalize_answer(gold)
            em_scores.append(1.0 if pred_norm == gold_norm else 0.0)
            f1_scores.append(_f1_score(pred, gold))

        data = data.copy()
        data["em_score"] = em_scores
        data["f1_score"] = f1_scores
        data["is_correct"] = [s >= 0.5 for s in em_scores]

        def _agg_by_key(key: str, scores, score_name):
            if key not in data:
                return {}
            groups = {}
            for i, (_, row) in enumerate(data.iterrows()):
                name = str(row.get(key, "") or "")
                if not name:
                    continue
                groups.setdefault(name, []).append(scores[i])
            return {
                name: {"length": len(vals), f"average_{score_name}": float(np.mean(vals)) if vals else 0.0}
                for name, vals in groups.items()
            }

        summary: Dict[str, Any] = {
            "total": {
                "total_length": len(em_scores),
                "exact_match": float(np.mean(em_scores)) if em_scores else 0.0,
                "average_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            },
            "difficulty": _agg_by_key("difficulty", em_scores, "em"),
            "category": _agg_by_key("category", em_scores, "em"),
        }

        score_file = get_intermediate_file_path(eval_file, "_vdrbench_f1", "json")
        dump(summary, score_file)

        if judge_kwargs.get("enable_llm_judge"):
            try:
                judge_summary, judge_verdicts = _run_llm_judge(
                    eval_file=eval_file, data=data, judge_kwargs=judge_kwargs,
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
