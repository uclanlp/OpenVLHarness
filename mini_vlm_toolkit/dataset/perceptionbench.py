"""PerceptionBench benchmark dataset.

PerceptionBench (MoonshotAI) evaluates *atomic* visual perception in MLLMs:
3,000 open-ended questions with short, uniquely-determined reference answers,
each isolating one of ten perceptual capabilities (visual relation, counting,
attribute, depth & 3D, localization, comparison, fine-grained recognition,
contextual integration, OCR, perception-related hallucination). Answers are
graded 0/1 by an LLM judge against the reference (see
``utils/judge_perceptionbench.py`` — the exact upstream teacher-grading prompt).

    Paper:   https://arxiv.org/abs/2607.24957
    Code:    https://github.com/MoonshotAI/PerceptionBench
    Dataset: https://huggingface.co/datasets/moonshotai/PerceptionBench

The TSV + on-disk images are auto-built from the HuggingFace dataset on first
use (see ``_ensure_perceptionbench_data``). Override the HF repo via env
``MMDR_PERCEPTIONBENCH_HF_REPO`` (default ``moonshotai/PerceptionBench``).

A question may contain ``<|image_N|>`` (1-indexed) placeholders marking where
each image is interleaved; images with no placeholder are appended after the
text. This mirrors upstream ``eval/eval.py`` ``build_messages`` so reported
numbers stay comparable to the public leaderboard.
"""

import csv
import json
import os
import os.path as osp
import re

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.judge_perceptionbench import grade_with_perceptionbench_judge
from ..smp import *
from ..smp.file import get_intermediate_file_path, LMUDataRoot
from ..utils import track_progress_rich

# 1-indexed image placeholder, e.g. "<|image_1|>", matching upstream PH regex.
_PLACEHOLDER = re.compile(r"<\|image_(\d+)\|>")
# Strips a "data:image/jpeg;base64," (or png/…) prefix off a data URI.
_DATA_URI_PREFIX = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)

_FAIL_MARKERS = ("[error]",)


def _strip_data_uri(s: str) -> str:
    return _DATA_URI_PREFIX.sub("", str(s or "").strip())


def _normalize_answer(ans):
    """Reference answer may be a bare string or a ``{"answer": ...}`` dict.

    Matches upstream ``run_one``: ``gt.get("answer") if isinstance(gt, dict)``.
    """
    if isinstance(ans, dict):
        return str(ans.get("answer", ""))
    return "" if ans is None else str(ans)


def _ensure_perceptionbench_data(lmu_root: str):
    """Idempotently materialise PerceptionBench TSV + on-disk images.

    Pulls ``moonshotai/PerceptionBench`` from HuggingFace, dumps each base64
    image to ``{lmu_root}/images/PerceptionBench/{index}_{k}.jpg`` (disk-friendly
    for the ~1.6 GB set, matching the FVQA auto-download precedent), and writes
    ``{lmu_root}/PerceptionBench.tsv``.

    TSV columns:
      index, question, answer, hint, error_category, source_bmk, source_idx,
      image_path
    where ``image_path`` is a JSON list of absolute image paths for the row and
    ``question`` keeps the upstream ``<|image_N|>`` placeholders verbatim.

    Override the HF repo via env ``MMDR_PERCEPTIONBENCH_HF_REPO``.
    """
    tsv_path = osp.join(lmu_root, "PerceptionBench.tsv")
    if osp.exists(tsv_path):
        return tsv_path

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "[perceptionbench/auto-download] the `datasets` package is required "
            "to build PerceptionBench. Install it with `pip install datasets`."
        ) from exc

    repo = os.environ.get("MMDR_PERCEPTIONBENCH_HF_REPO", "moonshotai/PerceptionBench")
    print(f"[perceptionbench/auto-download] loading {repo} (split=train) from HuggingFace")
    ds = load_dataset(repo, split="train")

    img_dir = osp.join(lmu_root, "images", "PerceptionBench")
    os.makedirs(img_dir, exist_ok=True)

    cols = ["index", "question", "answer", "hint", "error_category",
            "source_bmk", "source_idx", "image_path"]
    n = len(ds)
    print(f"[perceptionbench/auto-download] decoding images for {n} rows → {img_dir}")
    with open(tsv_path + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for i, rec in enumerate(ds):
            idx = rec.get("index", i)
            imgs = rec.get("image") or []
            if not isinstance(imgs, list):
                imgs = [imgs]
            paths = []
            for k, img in enumerate(imgs):
                path = osp.join(img_dir, f"{idx}_{k}.jpg")
                if not read_ok(path):
                    decode_base64_to_image_file(_strip_data_uri(img), path)
                paths.append(path)
            w.writerow([
                idx,
                str(rec.get("problem", "") or ""),
                _normalize_answer(rec.get("answer")),
                str(rec.get("hint", "") or ""),
                str(rec.get("error_category", "") or ""),
                str(rec.get("source_bmk", "") or ""),
                rec.get("source_idx", ""),
                json.dumps(paths, ensure_ascii=False),
            ])
            if (i + 1) % 200 == 0:
                print(f"[perceptionbench/auto-download]   {i + 1}/{n}")
    os.replace(tsv_path + ".tmp", tsv_path)
    print(f"[perceptionbench/auto-download] wrote TSV: {tsv_path} ({n} rows)")
    return tsv_path


def _is_failed(pred) -> bool:
    """Prediction failed to obtain an answer (upstream ``is_failed`` + our FAIL_MSG)."""
    if pred is None:
        return True
    s = str(pred)
    if not s.strip():
        return True
    if s.startswith(_FAIL_MARKERS):
        return True
    return "Failed to obtain answer via API" in s


def _grade_item(model, item):
    """Grade a single prediction with the PerceptionBench judge.

    Returns ``{correct: 0/1, reasoning: str}``. Failed predictions score 0
    without hitting the judge (matches upstream ``run_one``).
    """
    prediction = str(item.get("prediction", ""))
    if _is_failed(prediction):
        return dict(correct=0, reasoning="failed to obtain answer")
    reference = _normalize_answer(item.get("answer"))
    question = item.get("question", "")
    try:
        return grade_with_perceptionbench_judge(model, question, reference, prediction, retry=3)
    except Exception as e:  # noqa: BLE001 — never let one bad item abort the run
        return dict(correct=0, reasoning=f"judge error: {e}")


class PerceptionBenchDataset(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "PerceptionBench": "",   # empty => no remote TSV; auto-built on first use
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    def load_data(self, dataset):
        # First-use auto-prep: HF download + build TSV + dump images.
        if dataset == "PerceptionBench":
            _ensure_perceptionbench_data(LMUDataRoot())
        return super().load_data(dataset)

    # -- Prompt: interleave images into the question by <|image_N|> ----------

    def build_prompt(self, line):
        """Interleave images into the question exactly like upstream build_messages.

        For each ``<|image_N|>`` placeholder, splice ``image[N-1]`` in place;
        append any images without a placeholder after the trailing text.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_paths = self.dump_image(line)
        if not isinstance(tgt_paths, list):
            tgt_paths = [tgt_paths]

        question = str(line["question"] or "")
        msgs = []
        last, used = 0, set()
        for m in _PLACEHOLDER.finditer(question):
            n = int(m.group(1))
            if 1 <= n <= len(tgt_paths):
                used.add(n - 1)
                seg = question[last:m.start()]
                if seg:
                    msgs.append(dict(type="text", value=seg))
                msgs.append(dict(type="image", value=tgt_paths[n - 1]))
                last = m.end()
        if question[last:]:
            msgs.append(dict(type="text", value=question[last:]))
        for k, p in enumerate(tgt_paths):
            if k not in used:
                msgs.append(dict(type="image", value=p))
        if not msgs:
            msgs = [dict(type="text", value=question)]
        return msgs

    # -- Evaluation: 0/1 LLM-judge grading, overall + per-category ------------

    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger("Evaluation")
        data = load(eval_file)
        assert "prediction" in data, "prediction column missing from eval file"
        data["prediction"] = [str(x) for x in data["prediction"]]

        # Merge reference answer + category from dataset metadata by index.
        meta = self.data
        ans_map = {str(i): a for i, a in zip(meta["index"], meta["answer"])}
        cat_map = {str(i): c for i, c in zip(meta["index"], meta["error_category"])}
        q_map = {str(i): q for i, q in zip(meta["index"], meta["question"])}
        data["answer"] = [ans_map.get(str(i), "") for i in data["index"]]
        data["error_category"] = [cat_map.get(str(i), "") for i in data["index"]]
        if "question" not in data:
            data["question"] = [q_map.get(str(i), "") for i in data["index"]]

        nproc = judge_kwargs.pop("nproc", 4)

        # Judge tag for result filenames — identical derivation to
        # eval_injection._judge_tag so mcq_result_adapter can find the file.
        model_name = judge_kwargs.get("model", "exact_matching")
        name_str_map = {"chatgpt-0125": "openai", "gpt-4-0125": "gpt4"}
        name_str = name_str_map.get(model_name, model_name)
        if isinstance(name_str, str) and "/" in name_str:
            name_str = name_str.split("/")[-1]

        if model_name in (None, "exact_matching"):
            logger.error(
                "PerceptionBench is open-ended and requires a working LLM judge. "
                "Set judge_kwargs['model'] to a judge model (see configs/judge.yaml)."
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                "PerceptionBench evaluation requires a working judge model. "
                "Please configure judge_kwargs properly."
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        # Per-item grading with resumable cache.
        grade_file = get_intermediate_file_path(eval_file, f"_{name_str}_pb_grade", "pkl")
        result_cache = load(grade_file) if osp.exists(grade_file) else {}

        items_to_grade = [
            data.iloc[i] for i in range(len(data))
            if str(data.iloc[i]["index"]) not in result_cache
        ]
        if items_to_grade:
            tups = [dict(model=model, item=x) for x in items_to_grade]
            keys = [str(x["index"]) for x in items_to_grade]
            res = track_progress_rich(
                _grade_item, tups, nproc=nproc, chunksize=nproc,
                save=grade_file, keys=keys,
            )
            result_cache = load(grade_file)
            for k, v in zip(keys, res):
                if k not in result_cache:
                    result_cache[k] = v

        data["correct"] = [
            int(result_cache.get(str(i), {}).get("correct", 0)) for i in data["index"]
        ]
        data["reasoning"] = [
            result_cache.get(str(i), {}).get("reasoning", "") for i in data["index"]
        ]

        # Per-instance result table — filename is the contract with the
        # eval-injection mcq_result_adapter (correct_col='correct').
        eval_record = get_intermediate_file_path(eval_file, f"_{name_str}_result")
        dump(data, eval_record)

        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        dump(acc, score_file)
        logger.info(f"PerceptionBench accuracy results saved to {score_file}")
        return acc

    @staticmethod
    def _compute_accuracy(data):
        """Overall + per-``error_category`` accuracy (mirrors upstream reporting)."""
        rows = []

        def _acc_row(name, subset):
            total = len(subset)
            correct = int(subset["correct"].sum())
            return {
                "error_category": name,
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total * 100, 2) if total > 0 else 0.0,
            }

        rows.append(_acc_row("Overall", data))
        if "error_category" in data:
            for cat in sorted(x for x in data["error_category"].unique() if str(x).strip()):
                rows.append(_acc_row(cat, data[data["error_category"] == cat]))
        return pd.DataFrame(rows)
