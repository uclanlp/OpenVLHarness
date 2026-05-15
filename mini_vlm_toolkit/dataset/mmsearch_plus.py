"""
MMSearch-Plus dataset integration (Cie1/MMSearch-Plus on HuggingFace).

A harder, provenance-aware VQA benchmark with 311 curated samples.
Unlike original MMSearch, this requires genuine visual reasoning.
Each sample has 1-5 images, a question, multiple valid answers,
and metadata (category, difficulty, subtask, arxiv_id, video_url).

The dataset is encrypted; decryption uses canary "MMSearch-Plus".

Evaluation: token-level F1 + optional LLM judge, with breakdown by
difficulty (easy / difficult), category, and subtask.
"""

import base64
import hashlib
import io
import logging
import os
import re
import string
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *  # noqa: F401,F403
from ..smp.file import get_intermediate_file_path

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "Cie1/MMSearch-Plus"
_CANARY = "MMSearch-Plus"

ANSWER_FORMAT_INSTRUCTION = (
    "\n\nYou may provide reasoning first, but you must end your response "
    "with exactly one line in this format: Answer: <your concise answer>"
)


# ---------------------------------------------------------------------------
# Decryption helpers (from MMSearch-Plus/decrypt_after_load.py)
# ---------------------------------------------------------------------------

def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[:length % len(key)]


def _decrypt_text(ciphertext_b64: str, password: str) -> str:
    if not ciphertext_b64:
        return ciphertext_b64 or ""
    try:
        encrypted = base64.b64decode(ciphertext_b64)
        key = _derive_key(password, len(encrypted))
        decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])
        return decrypted.decode("utf-8")
    except Exception:
        return ciphertext_b64


# ---------------------------------------------------------------------------
# HuggingFace loading + decryption
# ---------------------------------------------------------------------------

def _load_hf_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required to use MMSearch-Plus.\n"
            "Install via `pip install datasets`."
        ) from exc

    env_cache_dir = os.environ.get("HF_DATASETS_CACHE_DIR")
    kwargs = {"split": "train"}
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        kwargs["cache_dir"] = env_cache_dir

    ds = load_dataset(_HF_DATASET_ID, **kwargs)

    # Decrypt text fields
    def _decrypt_sample(sample):
        for field in ("question", "video_url", "arxiv_id"):
            if sample.get(field):
                sample[field] = _decrypt_text(sample[field], _CANARY)
        if sample.get("answer"):
            sample["answer"] = [
                _decrypt_text(a, _CANARY) if a else a for a in sample["answer"]
            ]
        return sample

    ds = ds.map(_decrypt_sample, desc="Decrypting MMSearch-Plus")
    return ds


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
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, bytes):
        from PIL import Image
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
        logger.warning("MMSearch-Plus: failed to save image to %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Answer normalization & F1 (same as MMSearch)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    if text is None:
        text = ""
    text = str(text)

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)

    def remove_punc(s):
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def _f1_score(prediction: str, gold: str) -> float:
    """Set-based token F1 (matches lmms-eval's MMSearch-Plus implementation)."""
    pred_tokens = set(_normalize_answer(prediction).split())
    gold_tokens = set(_normalize_answer(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(gold))


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
# Default judge: HLE-style prompt (extract_final_answer + binary correct
# yes/no JSON). Same prompt used by hle / mmsearch / simplevqa / bcvl /
# vdrbench so cross-benchmark numbers are directly comparable, and
# WebWatcher-aligned for paper comparisons. See utils/hle_judge.py.
#
# The previous homebrew "lenient/strict CORRECT/INCORRECT" prompt is
# retained as ``_LEGACY_JUDGE_PROMPT`` for reference but no longer
# referenced by the active code path.
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement


_LEGACY_JUDGE_PROMPT = """You are an evaluator for question-answering systems.

Given a question, the ground-truth answer(s), and a model's prediction, judge whether the prediction is **semantically correct** (i.e. it conveys the same answer as any of the ground truths, even if the wording differs).

Be lenient about:
- Synonyms and paraphrases
- Number formats (e.g. "1" vs "One")
- Date formats
- Verbose answers that contain the correct answer plus extra explanation
- Capitalization and punctuation

Be strict about:
- Wrong facts (e.g. wrong name, wrong title, wrong number)
- Refusal or "I don't know" responses

Question: {question}
Ground Truth Answer(s): {gt_answer}
Prediction: {prediction}

Respond with EXACTLY one word: either "CORRECT" or "INCORRECT". No explanation."""


def _parse_alts(alt) -> list:
    if isinstance(alt, str):
        import ast
        import json
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(alt)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                continue
        return []
    return alt if isinstance(alt, list) else []


def _judge_one(judge, row) -> Dict[str, Any]:
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)

    # Combine primary + alternatives for judge — HLE prompt convention is
    # to use a single ``correct_answer`` field, so we OR-join the variants.
    alt = _parse_alts(row.get("alternative_gt_answers", []))
    all_answers = [gt] + alt
    gt_combined = " OR ".join(str(a) for a in all_answers if a)

    prompt = HLE_JUDGE_PROMPT.format(
        question=question,
        correct_answer=gt_combined,
        response=pred_extracted or pred,
    )

    verdict = "error"
    raw_response = ""
    try:
        raw_response = judge.generate(prompt)
        parsed = parse_hle_judgement(raw_response)
        if parsed["correct"] == 1:
            verdict = "correct"
        elif parsed["correct"] == 0 and parsed.get("reasoning") not in (
                "judge failed to parse", "judge unreachable after retries"):
            verdict = "incorrect"
    except Exception as e:
        logger.warning("MMSearch-Plus judge error: %s", e)

    return {"verdict": verdict, "raw_response": raw_response}


def _judge_one_judge_rs(judge, row) -> Dict[str, Any]:
    """Judge_RS variant of ``_judge_one``. Uses role-split system+user
    via ``grade_with_judge_rs`` helper. Opt-in via env var
    ``MMDR_JUDGE_PROMPT=judge_rs`` or ``judge_kwargs['judge_prompt']='judge_rs'``.
    """
    from .utils.judge_rs import grade_with_judge_rs

    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)
    alt = _parse_alts(row.get("alternative_gt_answers", []))
    correct_answer_list = [gt] + alt  # primary + candidates

    try:
        parsed = grade_with_judge_rs(
            judge, question, correct_answer_list, pred_extracted or pred,
        )
    except Exception as e:
        logger.warning("MMSearch-Plus Judge_RS error: %s", e)
        return {"verdict": "error", "raw_response": ""}

    if parsed["correct"] == 1:
        verdict = "correct"
    elif parsed.get("reasoning") in (
            "judge_rs failed to parse", "judge unreachable after retries"):
        verdict = "error"
    else:
        verdict = "incorrect"
    return {"verdict": verdict, "raw_response": str(parsed.get("reasoning", ""))[:200]}


def _judge_one_judge_vdr(judge, row) -> Dict[str, Any]:
    """Judge_VDR variant of ``_judge_one``. Uses VDR paper's contains-based
    prompt via ``grade_with_judge_vdr`` helper. Opt-in via env var
    ``MMDR_JUDGE_PROMPT=judge_vdr`` or ``judge_kwargs['judge_prompt']='judge_vdr'``.
    """
    from .utils.judge_vdr import grade_with_judge_vdr

    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")
    pred_extracted = _extract_answer(pred)
    alt = _parse_alts(row.get("alternative_gt_answers", []))
    correct_answer_list = [gt] + alt

    try:
        parsed = grade_with_judge_vdr(
            judge, question, correct_answer_list, pred_extracted or pred,
        )
    except Exception as e:
        logger.warning("MMSearch-Plus Judge_VDR error: %s", e)
        return {"verdict": "error", "raw_response": ""}

    if parsed["correct"] == 1:
        verdict = "correct"
    elif parsed.get("reasoning") in (
            "judge_vdr failed to parse", "judge unreachable after retries"):
        verdict = "error"
    else:
        verdict = "incorrect"
    return {"verdict": verdict, "raw_response": str(parsed.get("reasoning", ""))[:200]}


def _run_llm_judge(eval_file: str, data: pd.DataFrame, judge_kwargs: Dict[str, Any]):
    """Run LLM judge. Switch between HLE (default), Judge_RS, and Judge_VDR via
    env var ``MMDR_JUDGE_PROMPT`` or ``judge_kwargs['judge_prompt']``.
    Cache + score files are suffixed with ``_judge_rs`` / ``_judge_vdr`` when
    those variants are active.
    """
    from .utils import build_judge
    from ..utils.mp_util import track_progress_rich

    judge_kwargs_local = dict(judge_kwargs)
    judge_kwargs_local.pop("enable_llm_judge", None)
    nproc = judge_kwargs_local.pop("nproc", 4)
    # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
    # the single source of truth; no per-evaluator default).

    judge_name = (
        os.environ.get('MMDR_JUDGE_PROMPT')
        or judge_kwargs_local.pop('judge_prompt', None)
        or 'hle'
    ).strip().lower()
    if judge_name not in ('hle', 'judge_rs', 'judge_vdr'):
        raise ValueError(
            f"Unknown judge_prompt {judge_name!r}; expected 'hle', 'judge_rs', or 'judge_vdr'"
        )
    judge_kwargs_local.pop('judge_prompt', None)

    judge = build_judge(**judge_kwargs_local)
    assert judge.working(), "MMSearch-Plus LLM judge requires a working OpenAI API"

    if judge_name == 'judge_rs':
        judge_one_fn = _judge_one_judge_rs
        cache_stem = "_mmsearchplus_llm_judge_judge_rs_cache"
    elif judge_name == 'judge_vdr':
        judge_one_fn = _judge_one_judge_vdr
        cache_stem = "_mmsearchplus_llm_judge_judge_vdr_cache"
    else:
        judge_one_fn = _judge_one
        cache_stem = "_mmsearchplus_llm_judge_cache"

    cache_file = get_intermediate_file_path(eval_file, cache_stem, "pkl")
    cached: Dict[Any, Dict[str, Any]] = {}
    if osp.exists(cache_file):
        cached = load(cache_file)

    tups, indices = [], []
    for _, row in data.iterrows():
        idx = row["index"]
        if idx in cached:
            continue
        tups.append((judge, row))
        indices.append(idx)

    if indices:
        track_progress_rich(
            judge_one_fn, tups, nproc=nproc, chunksize=nproc,
            keys=indices, save=cache_file,
        )
        cached = load(cache_file)

    verdicts = {idx: cached[idx] for idx in data["index"] if idx in cached}
    n_total = len(data)
    n_correct = sum(1 for v in verdicts.values() if v.get("verdict") == "correct")
    n_incorrect = sum(1 for v in verdicts.values() if v.get("verdict") == "incorrect")
    n_error = n_total - n_correct - n_incorrect

    by_difficulty: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, Any]] = {}
    by_subtask: Dict[str, Dict[str, Any]] = {}
    for _, row in data.iterrows():
        idx = row["index"]
        ic = 1 if verdicts.get(idx, {}).get("verdict") == "correct" else 0
        for key, store in [("difficulty", by_difficulty), ("category", by_category), ("subtask", by_subtask)]:
            name = str(row.get(key, "unknown") or "unknown")
            store.setdefault(name, {"length": 0, "correct": 0})
            store[name]["length"] += 1
            store[name]["correct"] += ic
    for d in (by_difficulty, by_category, by_subtask):
        for v in d.values():
            v["accuracy"] = v["correct"] / v["length"] if v["length"] else 0.0

    summary = {
        "total": {"total_length": n_total, "accuracy": n_correct / n_total if n_total else 0.0},
        "correct": n_correct, "incorrect": n_incorrect, "error": n_error,
        "difficulty": by_difficulty, "category": by_category, "subtask": by_subtask,
    }
    if judge_name == 'judge_rs':
        score_stem = "_mmsearchplus_llm_judge_judge_rs"
    elif judge_name == 'judge_vdr':
        score_stem = "_mmsearchplus_llm_judge_judge_vdr"
    else:
        score_stem = "_mmsearchplus_llm_judge"
    dump(summary, get_intermediate_file_path(eval_file, score_stem, "json"))
    return summary, verdicts


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MMSearchPlus(ImageBaseDataset):
    """
    MMSearch-Plus VQA benchmark — harder, provenance-aware version of MMSearch.

    311 curated samples with 1-5 images per sample. Encrypted dataset
    (canary: "MMSearch-Plus"). Evaluation: token-F1 + optional LLM judge,
    broken down by difficulty (easy/difficult), category, and subtask.
    """

    TYPE = "VQA"

    DATASET_URL = {
        "MMSearchPlus": "https://huggingface.co/datasets/Cie1/MMSearch-Plus",
    }
    DATASET_MD5 = {
        "MMSearchPlus": "",
    }

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["MMSearchPlus"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        hf_ds = _load_hf_dataset()
        df = hf_ds.to_pandas()

        img_root = osp.join(lmu_root, "images", "MMSearchPlus")
        os.makedirs(img_root, exist_ok=True)

        # Cache images (up to 5 per sample)
        image_paths_col = []
        for i in range(len(df)):
            row = df.iloc[i]
            num_imgs = int(row.get("num_images", 0) or 0)
            paths = []
            for j in range(1, num_imgs + 1):
                img_key = f"img_{j}"
                img = row.get(img_key)
                if img is None:
                    continue
                filename = f"{i}_img{j}.jpg"
                path = osp.join(img_root, filename)
                if not osp.exists(path):
                    if not _save_image(img, path):
                        continue
                paths.append(filename)
            image_paths_col.append(paths)

        data = pd.DataFrame()
        data["index"] = [str(i) for i in range(len(df))]
        data["question"] = df["question"].astype(str)

        # Primary answer + alternatives
        primary = []
        alts = []
        for answers in df["answer"]:
            # HF .to_pandas() may convert lists to numpy arrays
            if hasattr(answers, "tolist"):
                answers = answers.tolist()
            if isinstance(answers, list) and answers:
                primary.append(str(answers[0]))
                alts.append([str(a) for a in answers[1:]])
            else:
                primary.append(str(answers))
                alts.append([])
        data["answer"] = primary
        data["alternative_gt_answers"] = alts

        # Metadata (guard against missing columns)
        for col in ("difficulty", "category", "subtask"):
            if col in df.columns:
                data[col] = df[col].astype(str)
            else:
                data[col] = "unknown"

        data["image_path"] = image_paths_col

        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]

        msgs: List[Dict[str, Any]] = []

        # Handle multiple images
        val = line.get("image_path", [])
        if isinstance(val, str):
            # TSV round-trip may serialize list as string
            import ast
            import json
            for parser in (json.loads, ast.literal_eval):
                try:
                    val = parser(val)
                    break
                except Exception:
                    continue

        if isinstance(val, list):
            for img_rel in val:
                if not img_rel:
                    continue
                img_path = str(img_rel)
                if not osp.isabs(img_path):
                    img_path = osp.join(self.img_root, img_rel)
                msgs.append({"type": "image", "value": img_path})
        elif val:
            img_path = str(val)
            if not osp.isabs(img_path):
                img_path = osp.join(self.img_root, img_path)
            msgs.append({"type": "image", "value": img_path})

        question = str(line.get("question", "") or "").strip()
        msgs.append({"type": "text", "value": question})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data and "answer" in data

        # Parse alternative answers
        def _to_alt_list(val):
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
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(s)
                        if isinstance(parsed, list):
                            return [str(x) for x in parsed]
                    except Exception:
                        continue
            return [s]

        data = data.copy()
        if "alternative_gt_answers" in data.columns:
            data["alternative_gt_answers"] = data["alternative_gt_answers"].apply(_to_alt_list)
        else:
            data["alternative_gt_answers"] = [[] for _ in range(len(data))]

        f1_scores = []
        em_scores = []
        for _, row in data.iterrows():
            pred = _extract_answer(str(row["prediction"]))
            gold = str(row["answer"])
            best_f1 = _f1_score(pred, gold)
            best_em = _exact_match(pred, gold)
            for alt in row["alternative_gt_answers"]:
                f1_alt = _f1_score(pred, alt)
                em_alt = _exact_match(pred, alt)
                if f1_alt > best_f1:
                    best_f1 = f1_alt
                if em_alt > best_em:
                    best_em = em_alt
            f1_scores.append(best_f1)
            em_scores.append(best_em)

        data["f1_score"] = f1_scores
        data["em_score"] = em_scores
        data["is_correct"] = [s >= 0.5 for s in f1_scores]

        def _agg_by_key(key, scores, score_name):
            if key not in data:
                return {}
            groups = {}
            for i, (_, row) in enumerate(data.iterrows()):
                name = str(row.get(key, "") or "")
                if not name:
                    continue
                groups.setdefault(name, []).append(scores[i])
            return {
                n: {"length": len(v), f"average_{score_name}": float(np.mean(v)) if v else 0.0}
                for n, v in groups.items()
            }

        summary: Dict[str, Any] = {
            "total": {
                "total_length": len(f1_scores),
                "average_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
                "average_em": float(np.mean(em_scores)) if em_scores else 0.0,
            },
            "difficulty": _agg_by_key("difficulty", f1_scores, "f1"),
            "category": _agg_by_key("category", f1_scores, "f1"),
            "subtask": _agg_by_key("subtask", f1_scores, "f1"),
        }

        dump(summary, get_intermediate_file_path(eval_file, "_mmsearchplus_f1", "json"))

        # LLM judge always runs alongside F1 (paper protocol). F1 above is
        # retained in `*_mmsearchplus_f1.json`.
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
