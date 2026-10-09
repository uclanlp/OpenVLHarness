"""
MMSearch dataset integration: end2end, requery, rerank, summarization.

Input differences per task (HF split and model input):

- End2End (MMSearch): One question + optional one query image.
  Output: final answer. Eval: token-level F1.

- Requery (MMSearch_Requery): One question + optional query image + image_search_result
  (two images when image query exists). Output: search query text. Eval: (BLEU-1 + ROUGE-L F1) / 2.

- Rerank (MMSearch_Rerank): One question + optional two images (query + search result) + 8 website
  briefs (title, snippet, screenshot per site). Output: <Website K>. Eval: 0 / 0.5 / 1.

- Summarization (MMSearch_Summarization): One question + optional two images + one website
  (title, snippet, content, fullpage screenshot). Output: answer. Eval: token-level F1.

Final score (official): 0.75 * end2end + 0.05 * requery + 0.1 * rerank + 0.1 * summarization.
Use compute_mmsearch_final_score() or scripts/compute_mmsearch_final_score.py to aggregate.
"""

import collections
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from ..smp import *  # noqa: F401,F403 - provides LMUDataRoot, load, dump, osp
from ..smp.file import get_intermediate_file_path
from .utils.judge_cache import guard_cache, require_judge

logger = logging.getLogger(__name__)

_MMSEARCH_DATASET_ID = "CaraJ/MMSearch"


def _mmsearch_load_hf_dataset(name: str, split: str):
    """
    Load MMSearch from HuggingFace datasets with a cache fallback.

    Some environments can end up with a corrupted/incompatible HF datasets cache
    for this dataset which triggers:
        TypeError: must be called with a dataclass type or instance

    If that happens, retry using a fresh cache dir under LMUDataRoot().
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The `datasets` library is required to use MMSearch.\n"
            "Install via `pip install datasets`."
        ) from exc

    # Allow users to pin cache location explicitly.
    env_cache_dir = os.environ.get("MMSEARCH_HF_CACHE_DIR") or os.environ.get("HF_DATASETS_CACHE_DIR")
    if env_cache_dir:
        os.makedirs(env_cache_dir, exist_ok=True)
        return load_dataset(_MMSEARCH_DATASET_ID, name=name, split=split, cache_dir=env_cache_dir)

    try:
        return load_dataset(_MMSEARCH_DATASET_ID, name=name, split=split)
    except TypeError as e:
        # Known failure mode with incompatible cached dataset_info/features.
        if "must be called with a dataclass type or instance" not in str(e):
            raise
        fallback_cache_dir = osp.join(LMUDataRoot(), "hf_cache", "datasets")
        os.makedirs(fallback_cache_dir, exist_ok=True)
        logger.warning(
            "MMSearch: HF datasets cache seems incompatible; retrying with cache_dir=%s. "
            "If this persists, delete the old cache under ~/.cache/huggingface/datasets/ for %s.",
            fallback_cache_dir,
            _MMSEARCH_DATASET_ID,
        )
        return load_dataset(_MMSEARCH_DATASET_ID, name=name, split=split, cache_dir=fallback_cache_dir)


def _mmsearch_image_to_pil(img: Any) -> Optional[Any]:
    """Convert HF/dataset image value to PIL.Image if possible. Returns None if not convertible."""
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


def _mmsearch_save_image(img: Any, path: str, task_label: str = "image") -> bool:
    """Save image to path; log first failure per process. Returns True if saved."""
    pil = _mmsearch_image_to_pil(img)
    if pil is None:
        if not getattr(_mmsearch_save_image, "_warned", False):
            logger.warning(
                "MMSearch: could not convert %s to PIL (type=%s); skipping save to %s",
                task_label, type(img).__name__, path,
            )
            _mmsearch_save_image._warned = True  # type: ignore[attr-defined]
        return False
    try:
        pil.save(path)
        return True
    except Exception as e:
        if not getattr(_mmsearch_save_image, "_warned", False):
            logger.warning(
                "MMSearch: failed to save %s to %s: %s",
                task_label, path, e,
            )
            _mmsearch_save_image._warned = True  # type: ignore[attr-defined]
        return False


def _normalize_answer(text: str) -> str:
    """
    Normalize text for F1 scoring.

    This mirrors the official MMSearch implementation:
    - lowercase
    - remove punctuation
    - remove English articles
    - collapse whitespace
    """
    import re
    import string

    if text is None:
        text = ""
    text = str(text)

    def remove_articles(s: str) -> str:
        regex = re.compile(r"\\b(a|an|the)\\b", re.UNICODE)
        return re.sub(regex, " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _get_tokens(text: str) -> List[str]:
    if not text:
        return []
    return _normalize_answer(text).split()


def _f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 used by the MMSearch benchmark."""
    gold_tokens = _get_tokens(gold)
    pred_tokens = _get_tokens(prediction)
    common = (collections.Counter(gold_tokens) & collections.Counter(pred_tokens))
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # If either is empty, F1 is 1 iff both are empty.
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / float(len(pred_tokens))
    recall = num_same / float(len(gold_tokens))
    return (2 * precision * recall) / (precision + recall)


def _requery_score(prediction: str, gold: str) -> Dict[str, float]:
    """
    Requery metric: average of BLEU-1 and ROUGE-L F1.
    Requires: nltk (sentence_bleu), rouge-score. Optional; returns zeros if missing.
    """
    out: Dict[str, float] = {"bleu": 0.0, "rouge_l": 0.0, "score": 0.0}
    pred = (prediction or "").strip()
    gold = (gold or "").strip()
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(
            [gold.split()], pred.split(),
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing,
        )
        out["bleu"] = float(bleu)
    except Exception:
        pass
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(pred, gold)
        if scores and isinstance(scores, list) and scores[0]:
            out["rouge_l"] = float(scores[0].get("rouge-l", {}).get("f", 0.0))
    except Exception:
        pass
    out["score"] = (out["bleu"] + out["rouge_l"]) / 2.0
    return out


def _postprocess_rerank(prediction: str, rerank_num: int) -> Tuple[List[int], bool]:
    """
    Parse model output for <Website K> and return 0-based index list and validity.
    """
    pattern = r"<Website\s*(\d+)>"
    matches = re.findall(pattern, prediction, re.IGNORECASE)
    output_index = [int(x) - 1 for x in matches]
    valid = True
    if len(output_index) > rerank_num:
        output_index = output_index[:rerank_num]
        valid = False
    elif len(output_index) < rerank_num:
        if len(output_index) == 0:
            output_index = list(range(rerank_num))
        valid = False
    elif not all(0 <= x < rerank_num for x in output_index):
        output_index = list(range(rerank_num))
        valid = False
    return output_index, valid


# Fixed output format for MMSearch F1 tasks: model must end with "Answer: <concise answer>"
MMSEARCH_ANSWER_FORMAT_INSTRUCTION = (
    "\n\nYou may provide reasoning first, but you must end your response with exactly one line in this format: Answer: <your concise answer>"
    "\n\nImportant guidelines for this task:"
    "\n- Before giving your final answer, verify the question's factual claims against your search results."
    " If the question contains a false premise (e.g., wrong date, wrong person, event that never happened),"
    ' answer "invalid question" instead of guessing.'
    "\n- When search results mention specific dates, numbers, or technical details critical to the answer,"
    " use Webpage_Visit_Tool to read the full source page for precise information instead of relying"
    " solely on search snippets."
    "\n- When constructing search queries, include specific entities, names, dates, and context from"
    " the image and question. Prefer precise queries over broad ones."
)


def _extract_mmsearch_answer(prediction: str) -> str:
    """
    Extract the concise answer from model output for F1 scoring.
    Looks for a line starting with "Answer:" (case-insensitive) and returns the rest of that line.
    If multiple such lines exist, uses the last one. If none, returns the full prediction (backward compatible).
    """
    if not prediction or not isinstance(prediction, str):
        return (prediction or "").strip()
    pred = prediction.strip()
    if not pred:
        return pred
    last_answer = None
    for line in pred.splitlines():
        line_stripped = line.strip()
        if re.match(r"^Answer\s*:\s*", line_stripped, re.IGNORECASE):
            last_answer = re.sub(r"^Answer\s*:\s*", "", line_stripped, count=1, flags=re.IGNORECASE).strip()
    return last_answer if last_answer is not None else pred


# Official MMSearch final score: weighted average of four tasks
MMSEARCH_TASK_RATIO = {
    "end2end": 0.75,
    "requery": 0.05,
    "rerank": 0.1,
    "summarization": 0.1,
}
MMSEARCH_TASK_AVG_KEY = {
    "end2end": "average_f1",
    "requery": "average_requery_score",
    "rerank": "average_rerank_score",
    "summarization": "average_f1",
}


def compute_mmsearch_final_score(
    score_files: Dict[str, str],
    ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute MMSearch final score from four task score JSON files (same as official get_final_scores.py).

    Each score file should be the JSON produced by MMSearch* .evaluate() with keys:
    - total: { total_length, average_f1 | average_requery_score | average_rerank_score }
    - area: { area_name: { length, average } }
    - subfield: { subfield_name: { length, average } }

    Args:
        score_files: Dict mapping task name to path, e.g.
            {"end2end": "path/to/end2end_mmsearch_f1.json", "requery": "...", "rerank": "...", "summarization": "..."}
        ratios: Optional custom weights; default MMSEARCH_TASK_RATIO (end2end=0.75, requery=0.05, rerank=0.1, summarization=0.1).

    Returns:
        Dict with total_dict.average, area_dict, subfield_dict (weighted averages).
    """
    ratios = ratios or MMSEARCH_TASK_RATIO
    task_avg_key = MMSEARCH_TASK_AVG_KEY
    loaded: Dict[str, Dict[str, Any]] = {}
    for task in ratios:
        path = score_files.get(task)
        if not path or not osp.isfile(path):
            raise FileNotFoundError(f"MMSearch score file for task '{task}' not found: {path}")
        loaded[task] = load(path)
        if "total" not in loaded[task]:
            raise KeyError(f"MMSearch score file for '{task}' missing 'total': {path}")

    def get_avg(data: Dict[str, Any], task: str) -> float:
        total = data.get("total") or {}
        key = task_avg_key.get(task, "average_f1")
        return float(total.get(key, 0.0))

    # total_dict
    final: Dict[str, Any] = {
        "total_dict": {
            "average": sum(
                ratios[t] * get_avg(loaded[t], t) for t in ratios
            ),
        },
        "area_dict": {},
        "subfield_dict": {},
    }

    # area_dict: use end2end as template for area names; weighted sum of each task's area average
    end2end_data = loaded.get("end2end") or {}
    for area_name in (end2end_data.get("area") or {}):
        final["area_dict"][area_name] = {
            "average": sum(
                ratios[t] * (loaded[t].get("area") or {}).get(area_name, {}).get("average", 0.0)
                for t in ratios
            ),
        }
    for subfield_name in (end2end_data.get("subfield") or {}):
        final["subfield_dict"][subfield_name] = {
            "average": sum(
                ratios[t] * (loaded[t].get("subfield") or {}).get(subfield_name, {}).get("average", 0.0)
                for t in ratios
            ),
        }

    return final


class MMSearchEnd2End(ImageBaseDataset):
    """
    MMSearch end-to-end benchmark (summarization QA view).

    This dataset exposes the MMSearch end-to-end split as a VQA-style
    benchmark: each sample has a natural language query (question), an
    optional query image, and reference answers. The evaluation uses
    token-level F1 between the model prediction and ground-truth answers,
    following the official MMSearch implementation.

    Notes:
    - This integration focuses on the end-to-end QA objective (F1 on
      answers). The step-wise requery/rerank metrics are not included.
    - Data is pulled from the HuggingFace dataset `CaraJ/MMSearch`
      with `name='end2end', split='end2end'`.
    - Query images are cached to `LMUDataRoot()/images/MMSearch/`.
    """

    TYPE = "VQA"

    # URL/MD5 kept for consistency with ImageBaseDataset; actual loading
    # is implemented in `load_data` via HuggingFace `datasets`.
    DATASET_URL = {
        "MMSearch": "https://huggingface.co/datasets/CaraJ/MMSearch",
    }
    DATASET_MD5 = {
        "MMSearch": "",
    }

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["MMSearch"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        """
        Load MMSearch end-to-end split from HuggingFace and map it into
        the internal tabular format expected by ImageBaseDataset.

        Required columns:
        - index: string id
        - question: text query
        - answer: primary reference answer
        - alternative_gt_answers: list of additional acceptable answers
        - area / subfield: metadata used for breakdown statistics
        - image_path: optional relative path to cached query image
        """
        if dataset in HF_SPLITS:
            return load_hf_split(self, dataset)
        lmu_root = LMUDataRoot()
        # Reuse processed annotations if they already exist on disk, similar to MMStar, AI2D, etc.
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            # Ensure DataFrame for downstream usage.
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        if dataset != "MMSearch":
            raise ValueError(f"Unsupported dataset name for MMSearchEnd2End: {dataset}")

        hf_ds = _mmsearch_load_hf_dataset(name="end2end", split="end2end")
        df = hf_ds.to_pandas()

        img_root = osp.join(lmu_root, "images", "MMSearch")
        os.makedirs(img_root, exist_ok=True)

        def _cache_query_image(row: pd.Series):
            """Save query image (if any) to LMUData/images/MMSearch. Return filename (str) or [] if no image."""
            img = row.get("query_image", None)
            if img is None:
                return []
            filename = f"{row['sample_id']}.jpg"
            path = osp.join(img_root, filename)
            if not osp.exists(path):
                if not _mmsearch_save_image(img, path, "query_image"):
                    return []
            return filename

        # Build core columns
        data = pd.DataFrame()
        data["index"] = df["sample_id"].astype(str)
        data["question"] = df["query"].astype(str)
        data["answer"] = df["gt_answer"].astype(str)
        data["alternative_gt_answers"] = df.get("alternative_gt_answers", [[]])

        # Optional metadata (area/subfield) for breakdown statistics
        data["area"] = df.get("area", "")
        data["subfield"] = df.get("subfield", "")

        # Cache query images to disk. Use [] for no-image rows so image_base.toliststr() does not see "".
        data["image_path"] = df.apply(_cache_query_image, axis=1)

        # Persist processed annotations to LMUData root for reuse.
        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        """
        Build a multi-modal prompt for MMSearch.

        If a cached query image exists, it is attached as an image input
        followed by the textual question.
        """
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
        """
        Evaluate predictions using token-level F1 against reference answers.

        The input `eval_file` is expected to be the standard OpenVLHarness
        prediction file produced by `infer_image_job`, containing at least:
        - index
        - prediction
        - answer
        - alternative_gt_answers
        - area (optional)
        - subfield (optional)
        """
        require_judge(self.dataset_name, judge_kwargs)
        data = load(eval_file)
        assert "prediction" in data and "answer" in data, (
            f"MMSearchEnd2End.evaluate expects columns 'prediction' and 'answer' in {eval_file}"
        )

        # Normalize alternative answers into list-of-str
        def _to_alt_list(val: Any) -> List[str]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return []
            if isinstance(val, list):
                return [str(x) for x in val]
            # Some pipelines may serialize lists as strings.
            s = str(val).strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    import json

                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    return [s]
            return [s]

        data = data.copy()
        data["alternative_gt_answers"] = data.get("alternative_gt_answers", []).apply(_to_alt_list)

        f1_scores: List[float] = []
        for _, row in data.iterrows():
            pred = _extract_mmsearch_answer(str(row["prediction"]))
            gold = str(row["answer"])
            best = _f1_score(pred, gold)
            for alt in row["alternative_gt_answers"]:
                score_alt = _f1_score(pred, alt)
                if score_alt > best:
                    best = score_alt
            f1_scores.append(best)

        data["f1_score"] = f1_scores
        # Per-instance correctness based on F1 (used by experience learning
        # pipeline to label correct/incorrect trajectories). LLM judge will
        # override this below if enabled.
        data["is_correct"] = [s >= 0.5 for s in f1_scores]
        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        # Aggregate by area/subfield when available
        def _agg_by_key(key: str) -> Dict[str, Dict[str, float]]:
            if key not in data:
                return {}
            groups = {}
            for _, row in data.iterrows():
                name = str(row.get(key, "") or "")
                if not name:
                    continue
                groups.setdefault(name, []).append(row["f1_score"])
            return {
                name: {
                    "length": len(vals),
                    "average": float(np.mean(vals)) if vals else 0.0,
                }
                for name, vals in groups.items()
            }

        # Per-modality F1 breakdown (multimodal vs text-only).
        # Handle both list-typed and string-typed image_path values, since
        # serialization to TSV/xlsx flattens lists into strings like "[]".
        def _has_image(row):
            val = row.get("image_path", "")
            if isinstance(val, list):
                return len(val) > 0 and val[0] != ""
            s = str(val or "").strip()
            if not s:
                return False
            # Empty list serialized as a string
            if s in ("[]", "['']", "[\"\"]"):
                return False
            return True

        data["_has_image"] = data.apply(_has_image, axis=1)
        mm_f1 = [row["f1_score"] for _, row in data.iterrows() if row["_has_image"]]
        text_f1 = [row["f1_score"] for _, row in data.iterrows() if not row["_has_image"]]

        summary: Dict[str, Any] = {
            "total": {
                "total_length": len(f1_scores),
                "average_f1": avg_f1,
            },
            "modality": {
                "multimodal": {
                    "length": len(mm_f1),
                    "average_f1": float(np.mean(mm_f1)) if mm_f1 else 0.0,
                },
                "text_only": {
                    "length": len(text_f1),
                    "average_f1": float(np.mean(text_f1)) if text_f1 else 0.0,
                },
            },
            "area": _agg_by_key("area"),
            "subfield": _agg_by_key("subfield"),
        }

        score_file = get_intermediate_file_path(eval_file, "_mmsearch_f1", "json")
        dump(summary, score_file)

        # LLM judge evaluation always runs (in addition to token-F1) so the
        # reported metric matches the MMSearch paper protocol. F1 above is
        # also retained in `*_mmsearch_f1.json` for reference.
        try:
            judge_summary, judge_verdicts = _run_mmsearch_llm_judge(
                eval_file=eval_file,
                data=data,
                judge_kwargs=judge_kwargs,
            )
            summary["llm_judge"] = judge_summary
            # Override per-instance is_correct with LLM judge verdicts
            # (more reliable than token-F1 for semantic equivalence).
            data["is_correct"] = [
                judge_verdicts.get(idx, {}).get("verdict") == "correct"
                for idx in data["index"]
            ]
        except Exception as exc:
            logger.error(f"LLM judge evaluation failed: {exc}")

        # Persist updated data (with f1_score + is_correct columns) back to
        # the eval file so downstream consumers (e.g. experience learning
        # pipeline) can read per-instance correctness.
        data = data.drop(columns=["_has_image"], errors="ignore")
        try:
            dump(data, eval_file)
        except Exception as exc:
            logger.warning(f"Failed to write evaluated data back to {eval_file}: {exc}")

        return summary


def _run_mmsearch_llm_judge(eval_file: str, data: pd.DataFrame, judge_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLM judge evaluation on MMSearch predictions.

    Reuses the standard `build_judge` + `track_progress_rich` pattern from
    other datasets (e.g. MathVista in image_vqa.py).

    Judge prompt is selected by env var ``MMDR_JUDGE_PROMPT`` or
    ``judge_kwargs['judge_prompt']``: ``'hle'`` (default) | ``'judge_rs'`` |
    ``'judge_vdr'``. Cache + score files are suffixed with ``_judge_rs`` /
    ``_judge_vdr`` when those variants are active so the three results
    coexist on the same run dir.
    """
    from .utils import build_judge
    from ..utils.mp_util import track_progress_rich

    judge_kwargs_local = dict(judge_kwargs)
    judge_kwargs_local.pop("enable_llm_judge", None)
    nproc = judge_kwargs_local.pop("nproc", 4)
    # NOTE: ``judge_max_tokens`` is consumed centrally inside
    # ``build_judge()`` (translated to OpenAIWrapper's ``max_tokens``).
    # Yaml is the single source of truth; do NOT introduce a per-evaluator
    # default here.

    # Judge prompt selection: env var > judge_kwargs > default 'hle'
    judge_name = (
        os.environ.get('MMDR_JUDGE_PROMPT')
        or judge_kwargs_local.pop('judge_prompt', None)
        or 'hle'
    ).strip().lower()
    if judge_name not in ('hle', 'judge_rs', 'judge_vdr'):
        raise ValueError(
            f"Unknown judge_prompt {judge_name!r}; expected 'hle', 'judge_rs', or 'judge_vdr'"
        )
    # Drop the kwarg (if not popped above) so it doesn't reach OpenAIWrapper.
    judge_kwargs_local.pop('judge_prompt', None)

    judge = build_judge(**judge_kwargs_local)
    assert judge.working(), "MMSearch LLM judge requires a working OpenAI API"

    if judge_name == 'judge_rs':
        judge_one_fn = _mmsearch_judge_one_judge_rs
        cache_stem = "_mmsearch_llm_judge_judge_rs_cache"
    elif judge_name == 'judge_vdr':
        judge_one_fn = _mmsearch_judge_one_judge_vdr
        cache_stem = "_mmsearch_llm_judge_judge_vdr_cache"
    else:
        judge_one_fn = _mmsearch_judge_one
        cache_stem = "_mmsearch_llm_judge_cache"

    judge_cache_file = get_intermediate_file_path(eval_file, cache_stem, "pkl")
    guard_cache(judge_cache_file, data, judge_kwargs_local.get("model"), prompt=judge_name)
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

    # Aggregate
    verdicts = {idx: cached[idx] for idx in data["index"] if idx in cached}
    n_total = len(data)
    n_correct = sum(1 for v in verdicts.values() if v.get("verdict") == "correct")
    n_incorrect = sum(1 for v in verdicts.values() if v.get("verdict") == "incorrect")
    n_error = n_total - n_correct - n_incorrect
    accuracy = n_correct / n_total if n_total else 0.0

    # By area / subfield / modality
    by_subfield: Dict[str, Dict[str, Any]] = {}
    by_area: Dict[str, Dict[str, Any]] = {}
    by_modality: Dict[str, Dict[str, Any]] = {"multimodal": {"length": 0, "correct": 0}, "text_only": {"length": 0, "correct": 0}}
    for _, row in data.iterrows():
        idx = row["index"]
        v = verdicts.get(idx, {}).get("verdict")
        is_correct = 1 if v == "correct" else 0

        sf = str(row.get("subfield", "") or "unknown")
        by_subfield.setdefault(sf, {"length": 0, "correct": 0})
        by_subfield[sf]["length"] += 1
        by_subfield[sf]["correct"] += is_correct

        ar = str(row.get("area", "") or "unknown")
        by_area.setdefault(ar, {"length": 0, "correct": 0})
        by_area[ar]["length"] += 1
        by_area[ar]["correct"] += is_correct

        # Modality: check image_path. Handle both list and serialized string forms.
        img_val = row.get("image_path", "")
        if isinstance(img_val, list):
            has_img = len(img_val) > 0 and img_val[0] != ""
        else:
            s = str(img_val or "").strip()
            has_img = bool(s) and s not in ("[]", "['']", "[\"\"]")
        mod_key = "multimodal" if has_img else "text_only"
        by_modality[mod_key]["length"] += 1
        by_modality[mod_key]["correct"] += is_correct

    for d in (by_subfield, by_area, by_modality):
        for k, v in d.items():
            v["accuracy"] = v["correct"] / v["length"] if v["length"] else 0.0

    judge_summary = {
        "total": {"total_length": n_total, "accuracy": accuracy},
        "correct": n_correct,
        "incorrect": n_incorrect,
        "error": n_error,
        "modality": by_modality,
        "area": by_area,
        "subfield": by_subfield,
    }

    if judge_name == 'judge_rs':
        score_stem = "_mmsearch_llm_judge_judge_rs"
    elif judge_name == 'judge_vdr':
        score_stem = "_mmsearch_llm_judge_judge_vdr"
    else:
        score_stem = "_mmsearch_llm_judge"
    judge_score_file = get_intermediate_file_path(eval_file, score_stem, "json")
    dump(judge_summary, judge_score_file)
    return judge_summary, verdicts


# ---------------------------------------------------------------------------
# MMSearch end2end LLM judge
# ---------------------------------------------------------------------------
# Default judge: shared HLE-style prompt (extract_final_answer + binary
# correct yes/no JSON, see ``utils/hle_judge.py``). Same prompt as the
# other LLM-judge benchmarks in this project (mmsearch_plus, vdrbench,
# hle, simplevqa_en, bcvl) so cross-benchmark numbers are directly
# comparable, and matches WebWatcher's pipeline convention.
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement


def _mmsearch_judge_one(judge, row) -> Dict[str, Any]:
    """Judge a single MMSearch prediction. Returns dict with 'verdict' field."""
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")

    # Extract the parsed answer using existing helper
    pred_extracted = _extract_mmsearch_answer(pred)

    # Include alternative GTs — HLE prompt convention is a single
    # ``correct_answer`` field, so we OR-join the variants.
    alts_raw = row.get("alternative_gt_answers", [])
    alts: List[str] = []
    if isinstance(alts_raw, list):
        alts = [str(a) for a in alts_raw if a]
    elif isinstance(alts_raw, str) and alts_raw.strip():
        s = alts_raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                parsed = _json.loads(s)
                if isinstance(parsed, list):
                    alts = [str(a) for a in parsed if a]
            except Exception:
                alts = [s]
        else:
            alts = [s]

    gt_combined = gt
    if alts:
        gt_combined = " OR ".join([gt] + alts)

    if not pred_extracted.strip():
        return {"verdict": "incorrect", "raw_response": ""}

    msgs = [{
        "type": "text",
        "value": HLE_JUDGE_PROMPT.format(
            question=question[:1000],
            correct_answer=gt_combined[:500],
            response=pred_extracted[:2000],
        ),
    }]

    try:
        code, answer, _ = judge.generate_inner(msgs)
        if code != 0:
            return {"verdict": "error", "raw_response": str(answer)[:200]}
        parsed = parse_hle_judgement(str(answer))
        if parsed["correct"] == 1:
            return {"verdict": "correct", "raw_response": str(answer)[:200]}
        if parsed.get("reasoning") not in (
                "judge failed to parse", "judge unreachable after retries"):
            return {"verdict": "incorrect", "raw_response": str(answer)[:200]}
        return {"verdict": "error", "raw_response": str(answer)[:200]}
    except Exception as exc:
        return {"verdict": "error", "raw_response": f"exception: {exc}"}


# Judge_RS variant: same row signature, but uses the multimodal-search-r1
# judge prompt (system + user as separate roles) and the v2 XML output
# parser. Switched in via env var MMDR_JUDGE_PROMPT=judge_rs (per-call,
# NOT a global default — HLE remains the default judge for MMSearch).
from .utils.judge_rs import (
    JUDGE_RS_SYSTEM_PROMPT, JUDGE_RS_USER_PROMPT, parse_judge_rs_judgement,
)


def _mmsearch_judge_one_judge_rs(judge, row) -> Dict[str, Any]:
    """Judge a single MMSearch prediction using the Judge_RS v2 prompt.

    Sends system + user as separate role-tagged messages via
    ``judge.generate_inner`` (skips ``model.generate``'s side-effecting
    ``preprocess_message_with_role`` path so ``self.system_prompt`` on
    the OpenAIWrapper instance is not mutated).
    """
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")

    pred_extracted = _extract_mmsearch_answer(pred)

    alts_raw = row.get("alternative_gt_answers", [])
    alts: List[str] = []
    if isinstance(alts_raw, list):
        alts = [str(a) for a in alts_raw if a]
    elif isinstance(alts_raw, str) and alts_raw.strip():
        s = alts_raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                parsed_alts = _json.loads(s)
                if isinstance(parsed_alts, list):
                    alts = [str(a) for a in parsed_alts if a]
            except Exception:
                alts = [s]
        else:
            alts = [s]

    candidates = " | ".join(alts) if alts else "(none)"

    if not pred_extracted.strip():
        return {"verdict": "incorrect", "raw_response": ""}

    msgs_role_split = [
        {"role": "system", "content": [{"type": "text", "value": JUDGE_RS_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "value": JUDGE_RS_USER_PROMPT.format(
            question=question[:1000],
            ground_truth_answer=gt[:500],
            candidate_answers=candidates[:500],
            model_response=pred_extracted[:2000],
        )}]},
    ]

    try:
        code, answer, _ = judge.generate_inner(msgs_role_split)
        if code != 0:
            return {"verdict": "error", "raw_response": str(answer)[:200]}
        parsed = parse_judge_rs_judgement(str(answer))
        if parsed.get("correct") == 1:
            return {"verdict": "correct", "raw_response": str(answer)[:200]}
        if parsed.get("reasoning") != "judge_rs failed to parse":
            return {"verdict": "incorrect", "raw_response": str(answer)[:200]}
        return {"verdict": "error", "raw_response": str(answer)[:200]}
    except Exception as exc:
        return {"verdict": "error", "raw_response": f"exception: {exc}"}


# Optional alternate judge — VDR paper "contains-based" binary judge.
# Switched in via env MMDR_JUDGE_PROMPT=judge_vdr (per-call, NOT a global
# default — HLE remains MMSearch's default).
from .utils.judge_vdr import JUDGE_VDR_PROMPT, parse_judge_vdr_judgement


def _mmsearch_judge_one_judge_vdr(judge, row) -> Dict[str, Any]:
    """Judge a single MMSearch prediction using the Judge_VDR v1 prompt.

    Judge_VDR uses a single-turn user prompt (no system/user split — VDR
    paper's prompt is one flat instruction) and a plain-text
    ``correct: yes/no\\nreasoning: ...`` output format.
    """
    question = str(row.get("question", ""))
    gt = str(row.get("answer", ""))
    pred = str(row.get("prediction", "") or "")

    pred_extracted = _extract_mmsearch_answer(pred)

    alts_raw = row.get("alternative_gt_answers", [])
    alts: List[str] = []
    if isinstance(alts_raw, list):
        alts = [str(a) for a in alts_raw if a]
    elif isinstance(alts_raw, str) and alts_raw.strip():
        s = alts_raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import json as _json
                parsed_alts = _json.loads(s)
                if isinstance(parsed_alts, list):
                    alts = [str(a) for a in parsed_alts if a]
            except Exception:
                alts = [s]
        else:
            alts = [s]

    # VDR prompt has no candidate_answers slot — join all alternatives into
    # the single [Correct Answer] field with " | " separator.
    correct_text = " | ".join([gt[:500]] + [str(a)[:500] for a in alts]) if alts else gt[:500]

    if not pred_extracted.strip():
        return {"verdict": "incorrect", "raw_response": ""}

    user_prompt = JUDGE_VDR_PROMPT.format(
        question=question[:1000],
        correct_answer=correct_text,
        response=pred_extracted[:2000],
    )
    msgs = [{"role": "user", "content": [{"type": "text", "value": user_prompt}]}]

    try:
        code, answer, _ = judge.generate_inner(msgs)
        if code != 0:
            return {"verdict": "error", "raw_response": str(answer)[:200]}
        parsed = parse_judge_vdr_judgement(str(answer))
        if parsed.get("correct") == 1:
            return {"verdict": "correct", "raw_response": str(answer)[:200]}
        if parsed.get("reasoning") != "judge_vdr failed to parse":
            return {"verdict": "incorrect", "raw_response": str(answer)[:200]}
        return {"verdict": "error", "raw_response": str(answer)[:200]}
    except Exception as exc:
        return {"verdict": "error", "raw_response": f"exception: {exc}"}


# ----- Requery (stage1) -----

REQUERY_PROMPT_TEXT = """You are a helpful assistant. I am giving you a question, which cannot be solved without external knowledge.
Assume you have access to a text-only search engine (e.g., google). Please raise a query to the search engine to search for what is useful for you to answer the question correctly. Your query needs to consider the attribute of the query to search engine.
Here are 3 examples:
Question: Did Zheng Xiuwen wear a knee pad in the women's singles tennis final in 2024 Paris Olympics?
Query to the search engine: Images of Zheng Xiuwen in the women's singles tennis final in 2024 Paris Olympics

Question: When will Apple release iPhone16?
Query to the search engine: iPhone 16 release date

Question: Who will sing a French song at the Olympic Games closing ceremony?
Query to the search engine: Singers at the Olympic Games closing ceremony, French song

Question: {question}
Query to the search engine (do not involve any explanation): """

REQUERY_PROMPT_IMAGE = """You are a helpful assistant. I am giving you a question including an image, which cannot be solved without external knowledge.
Assume you have access to a search engine (e.g., google). Please raise a query to the search engine to search for what is useful for you to answer the question correctly. You need to consider the characteristics of asking questions to search engines when formulating your questions.
You are also provided with the search result of the image in the question. You should leverage the image search result to raise the text query.
Question: {question}
The image search result is: [see images below]
Query to the search engine (do not involve any explanation): """


class MMSearchRequery(ImageBaseDataset):
    """MMSearch requery task (stage1): predict search query. Eval: (BLEU-1 + ROUGE-L F1) / 2."""
    TYPE = "VQA"
    DATASET_URL = {"MMSearch_Requery": "https://huggingface.co/datasets/CaraJ/MMSearch"}
    DATASET_MD5 = {"MMSearch_Requery": ""}

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["MMSearch_Requery"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        if dataset != "MMSearch_Requery":
            raise ValueError(f"Unsupported: {dataset}")
        hf_ds = _mmsearch_load_hf_dataset(name="end2end", split="end2end")
        df = hf_ds.to_pandas()
        img_root = osp.join(lmu_root, "images", "MMSearch")
        os.makedirs(img_root, exist_ok=True)

        def _cache_row(row: pd.Series) -> Tuple:
            """Return (query_path or [], search_result_path or []) so image_base never sees ""."""
            q, s = [], []
            sid = str(row["sample_id"])
            if row.get("query_image") is not None:
                p = osp.join(img_root, f"{sid}_query.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["query_image"], p, "query_image")
                if osp.exists(p):
                    q = f"{sid}_query.jpg"
            if row.get("image_search_result") is not None:
                p = osp.join(img_root, f"{sid}_search.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["image_search_result"], p, "image_search_result")
                if osp.exists(p):
                    s = f"{sid}_search.jpg"
            return (q, s)

        paths = df.apply(_cache_row, axis=1)
        data = pd.DataFrame()
        data["index"] = df["sample_id"].astype(str)
        data["question"] = df["query"].astype(str)
        data["answer"] = df["gt_requery"].astype(str)
        data["image_path"] = [p[0] for p in paths]
        data["image_path_search_result"] = [p[1] for p in paths]
        data["area"] = df.get("area", "")
        data["subfield"] = df.get("subfield", "")
        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs: List[Dict[str, Any]] = []
        def _path_val(key: str) -> str:
            val = line.get(key, "")
            if isinstance(val, list):
                return val[0] if len(val) > 0 else ""
            return str(val or "").strip()
        q_img = _path_val("image_path")
        s_img = _path_val("image_path_search_result")
        has_image = bool(q_img or s_img)
        if q_img:
            msgs.append({"type": "image", "value": osp.join(self.img_root, q_img) if not osp.isabs(q_img) else q_img})
        if s_img:
            msgs.append({"type": "image", "value": osp.join(self.img_root, s_img) if not osp.isabs(s_img) else s_img})
        question = str(line.get("question", "") or "").strip()
        prompt = (REQUERY_PROMPT_IMAGE if has_image else REQUERY_PROMPT_TEXT).format(question=question)
        msgs.append({"type": "text", "value": prompt})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data and "answer" in data
        scores = [_requery_score(str(r["prediction"]), str(r["answer"]))["score"] for _, r in data.iterrows()]
        data = data.copy()
        data["requery_score"] = scores
        def _agg(k):
            if k not in data:
                return {}
            g = {}
            for _, r in data.iterrows():
                n = str(r.get(k, "") or "")
                if n:
                    g.setdefault(n, []).append(r["requery_score"])
            return {n: {"length": len(v), "average": float(np.mean(v))} for n, v in g.items()}
        summary = {"total": {"total_length": len(scores), "average_requery_score": float(np.mean(scores)) if scores else 0}, "area": _agg("area"), "subfield": _agg("subfield")}
        dump(summary, get_intermediate_file_path(eval_file, "_mmsearch_requery", "json"))
        return summary


# ----- Rerank (stage2) -----

RERANK_NUM_DEFAULT = 1
BRIEF_RESULT_NUM_DEFAULT = 8

RERANK_PROMPT_TEXT = """You are a helpful assistant. I am giving you a question and {brief_result_num} website information related to the question (including the screenshot, snippet and title).
You should now read the screenshots, snippets and titles. Select {rerank_num} website that are the most helpful for you to answer the question. Once you select it, the detailed content of them will be provided to help you correctly answer the question.
The question is: {question}
The website informations is:
{website_information}

You should directly output {rerank_num} website's index that can help you most, separated with ',', and enclose each website in angle brackets. The output format should be: <Website Index>.
An example of the output is: {incontext_example}
Your answer: """

RERANK_PROMPT_IMAGE = """You are a helpful assistant. I am giving you a question including an image. You are provided with the search result of the image in the question. And you are provided with {brief_result_num} website information related to the question (including the screenshot, snippet and title).
You should now read the screenshots, snippets and titles of these websites. Select {rerank_num} website that are the most helpful for you to answer the question.
The question is: {question}
The image search result is: [see images below]
The website informations is:
{website_information}

You should directly output {rerank_num} website's index that can help you most, separated with ',', and enclose each website in angle brackets. The output format should be: <Website Index>.
An example of the output is: {incontext_example}
Your answer: """


class MMSearchRerank(ImageBaseDataset):
    """MMSearch rerank task (stage2): output <Website K>. Score: 1 if in valid, 0.5 if not_sure, 0 else."""
    TYPE = "VQA"
    DATASET_URL = {"MMSearch_Rerank": "https://huggingface.co/datasets/CaraJ/MMSearch"}
    DATASET_MD5 = {"MMSearch_Rerank": ""}

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["MMSearch_Rerank"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        if dataset != "MMSearch_Rerank":
            raise ValueError(f"Unsupported: {dataset}")
        hf_ds = _mmsearch_load_hf_dataset(name="rerank", split="rerank")
        df = hf_ds.to_pandas()
        img_root = osp.join(lmu_root, "images", "MMSearch", "rerank")
        os.makedirs(img_root, exist_ok=True)
        n_brief = BRIEF_RESULT_NUM_DEFAULT
        rows = []
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            q_path, s_path = [], []
            if row.get("query_image") is not None:
                p = osp.join(img_root, f"{sid}_query.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["query_image"], p, "query_image")
                if osp.exists(p):
                    q_path = osp.join("rerank", f"{sid}_query.jpg")
            if row.get("image_search_result") is not None:
                p = osp.join(img_root, f"{sid}_search.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["image_search_result"], p, "image_search_result")
                if osp.exists(p):
                    s_path = osp.join("rerank", f"{sid}_search.jpg")
            website_parts = []
            image_paths = []
            for i in range(n_brief):
                info = row.get(f"website{i}_info") or {}
                title = info.get("title", "")
                snippet = info.get("snippet") or info.get("text", "")
                website_parts.append(
                    f"Website {i + 1} Title: {title};\nWebsite {i + 1} snippet: {snippet};\nWebsite {i + 1} Screenshot: [Image {i + 1}]"
                )
                img = row.get(f"website{i}_head_screenshot")
                if img is not None:
                    path = osp.join(img_root, f"{sid}_web{i}.jpg")
                    if not osp.exists(path):
                        _mmsearch_save_image(img, path, f"website{i}_head_screenshot")
                    if osp.exists(path):
                        image_paths.append(osp.join("rerank", f"{sid}_web{i}.jpg"))
                    else:
                        image_paths.append("")
                else:
                    image_paths.append("")
            website_information = "\n\n".join(website_parts)
            valid = row.get("valid", [])
            not_sure = row.get("not_sure", [])
            if not isinstance(valid, list):
                valid = list(valid) if hasattr(valid, "__iter__") else []
            if not isinstance(not_sure, list):
                not_sure = list(not_sure) if hasattr(not_sure, "__iter__") else []
            rows.append({
                "index": sid,
                "question": row["query"],
                "image_path": q_path,
                "image_path_search_result": s_path,
                "website_information": website_information,
                "image_paths": image_paths,
                "valid": valid,
                "not_sure": not_sure,
                "area": row.get("area", ""),
                "subfield": row.get("subfield", ""),
            })
        data = pd.DataFrame(rows)
        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = []
        img_root_abs = self.img_root
        for key in ["image_path", "image_path_search_result"]:
            p = line.get(key)
            if isinstance(p, list):
                p = p[0] if len(p) > 0 else ""
            if p:
                path = osp.join(img_root_abs, p) if not osp.isabs(str(p)) else p
                msgs.append({"type": "image", "value": path})
        paths = line.get("image_paths") or []
        if isinstance(paths, str):
            paths = [paths]
        for p in paths:
            if p:
                path = osp.join(img_root_abs, p) if not osp.isabs(str(p)) else p
                msgs.append({"type": "image", "value": path})
        brief_num = len([x for x in paths if x]) or BRIEF_RESULT_NUM_DEFAULT
        rerank_num = RERANK_NUM_DEFAULT
        incontext = ",".join([f"<Website {i + 1}>" for i in range(rerank_num)])
        has_image = bool(line.get("image_path") or line.get("image_path_search_result"))
        tpl = RERANK_PROMPT_IMAGE if has_image else RERANK_PROMPT_TEXT
        prompt = tpl.format(
            brief_result_num=brief_num,
            rerank_num=rerank_num,
            question=str(line.get("question", "")),
            website_information=line.get("website_information", ""),
            incontext_example=incontext,
        )
        msgs.append({"type": "text", "value": prompt})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data
        rerank_num = RERANK_NUM_DEFAULT
        scores = []
        for _, row in data.iterrows():
            indices, _ = _postprocess_rerank(str(row["prediction"]), rerank_num)
            selected = indices[0] if indices else 0
            valid_list = row.get("valid")
            not_sure_list = row.get("not_sure")
            if valid_list is None or (isinstance(valid_list, float) and np.isnan(valid_list)):
                valid_list = []
            if not_sure_list is None or (isinstance(not_sure_list, float) and np.isnan(not_sure_list)):
                not_sure_list = []
            if not isinstance(valid_list, list):
                valid_list = list(valid_list) if hasattr(valid_list, "__iter__") else []
            if not isinstance(not_sure_list, list):
                not_sure_list = list(not_sure_list) if hasattr(not_sure_list, "__iter__") else []
            if selected in valid_list:
                score = 1.0
            elif selected in not_sure_list:
                score = 0.5
            else:
                score = 0.0
            scores.append(score)
        data = data.copy()
        data["rerank_score"] = scores
        def _agg(k):
            if k not in data:
                return {}
            g = {}
            for _, r in data.iterrows():
                n = str(r.get(k, "") or "")
                if n:
                    g.setdefault(n, []).append(r["rerank_score"])
            return {n: {"length": len(v), "average": float(np.mean(v))} for n, v in g.items()}
        summary = {"total": {"total_length": len(scores), "average_rerank_score": float(np.mean(scores)) if scores else 0}, "area": _agg("area"), "subfield": _agg("subfield")}
        dump(summary, get_intermediate_file_path(eval_file, "_mmsearch_rerank", "json"))
        return summary


# ----- Summarization (stage3) -----

SUMMARIZATION_PROMPT_TEXT = """You are a helpful assistant. I am giving you a question and {rerank_num} website information related to the question.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. When answering questions about dates, use the yyyy-mm-dd format.
3. Answer the question with as few words as you can.

The website informations is {website_information}
The question is: {question}.
Please directly output the answer without any explanation: """

SUMMARIZATION_PROMPT_IMAGE = """You are a helpful assistant. I am giving you a question including an image. You are provided with the search result of the image in the question. And you are provided with {rerank_num} website information related to the question.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. When answering questions about dates, use the yyyy-mm-dd format.
3. Answer the question with as few words as you can.

The website informations is {website_information}
The image search result is: [see images below]
The question is: {question}.
Please directly output the answer without any explanation: """


class MMSearchSummarization(ImageBaseDataset):
    """MMSearch summarization task (stage3): answer given one website. Eval: token-level F1."""
    TYPE = "VQA"
    DATASET_URL = {"MMSearch_Summarization": "https://huggingface.co/datasets/CaraJ/MMSearch"}
    DATASET_MD5 = {"MMSearch_Summarization": ""}

    @classmethod
    def supported_datasets(cls) -> List[str]:
        return ["MMSearch_Summarization"]

    def load_data(self, dataset: str) -> pd.DataFrame:
        lmu_root = LMUDataRoot()
        ann_path = osp.join(lmu_root, f"{dataset}.tsv")
        if osp.exists(ann_path):
            data = load(ann_path)
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            return data

        if dataset != "MMSearch_Summarization":
            raise ValueError(f"Unsupported: {dataset}")
        hf_ds = _mmsearch_load_hf_dataset(name="summarization", split="summarization")
        df = hf_ds.to_pandas()
        img_root = osp.join(lmu_root, "images", "MMSearch", "summarization")
        os.makedirs(img_root, exist_ok=True)
        rows = []
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            fullpage_path = []
            img = row.get("website_fullpage_screenshot") or row.get("slimmed_website_fullpage_screenshot")
            if img is not None:
                path = osp.join(img_root, f"{sid}_fullpage.jpg")
                if not osp.exists(path):
                    _mmsearch_save_image(img, path, "website_fullpage_screenshot")
                if osp.exists(path):
                    fullpage_path = osp.join("summarization", f"{sid}_fullpage.jpg")
            q_path, s_path = [], []
            if row.get("query_image") is not None:
                p = osp.join(img_root, f"{sid}_query.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["query_image"], p, "query_image")
                if osp.exists(p):
                    q_path = osp.join("summarization", f"{sid}_query.jpg")
            if row.get("image_search_result") is not None:
                p = osp.join(img_root, f"{sid}_search.jpg")
                if not osp.exists(p):
                    _mmsearch_save_image(row["image_search_result"], p, "image_search_result")
                if osp.exists(p):
                    s_path = osp.join("summarization", f"{sid}_search.jpg")
            title = row.get("website_title", "")
            snippet = row.get("website_snippet", "")
            content = row.get("website_retrieved_content", "")
            website_information = (
                f"Website Title: {title};\nWebsite Snippet: {snippet};\nWebsite Content: {content};\n"
                "Website Screenshot: [see image below]"
            )
            rows.append({
                "index": sid,
                "question": row["query"],
                "answer": row["gt_answer"],
                "alternative_gt_answers": row.get("alternative_gt_answers", []),
                "area": row.get("area", ""),
                "subfield": row.get("subfield", ""),
                "website_information": website_information,
                "image_path_fullpage": fullpage_path,
                "image_path": q_path,
                "image_path_search_result": s_path,
            })
        data = pd.DataFrame(rows)
        dump(data, ann_path)
        return data

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = []
        img_root_abs = self.img_root
        for key in ["image_path_fullpage", "image_path_search_result", "image_path"]:
            p = line.get(key)
            if isinstance(p, list):
                p = p[0] if len(p) > 0 else ""
            if p:
                path = osp.join(img_root_abs, p) if not osp.isabs(str(p)) else p
                msgs.append({"type": "image", "value": path})
        has_image = bool(line.get("image_path") or line.get("image_path_search_result"))
        tpl = SUMMARIZATION_PROMPT_IMAGE if has_image else SUMMARIZATION_PROMPT_TEXT
        prompt = tpl.format(
            rerank_num=1,
            website_information=line.get("website_information", ""),
            question=str(line.get("question", "")),
        )
        prompt = prompt + MMSEARCH_ANSWER_FORMAT_INSTRUCTION
        msgs.append({"type": "text", "value": prompt})
        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> Dict[str, Any]:
        data = load(eval_file)
        assert "prediction" in data and "answer" in data

        def _to_alt_list(val: Any) -> List[str]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return []
            if isinstance(val, list):
                return [str(x) for x in val]
            s = str(val).strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    import json
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
            return [s]

        data = data.copy()
        if "alternative_gt_answers" in data:
            data["alternative_gt_answers"] = data["alternative_gt_answers"].apply(_to_alt_list)
        else:
            data["alternative_gt_answers"] = [[] for _ in range(len(data))]
        f1_scores = []
        for _, row in data.iterrows():
            pred = _extract_mmsearch_answer(str(row["prediction"]))
            gold = str(row["answer"])
            best = _f1_score(pred, gold)
            for alt in row["alternative_gt_answers"]:
                best = max(best, _f1_score(pred, alt))
            f1_scores.append(best)
        data["f1_score"] = f1_scores
        def _agg(k):
            if k not in data:
                return {}
            g = {}
            for _, r in data.iterrows():
                n = str(r.get(k, "") or "")
                if n:
                    g.setdefault(n, []).append(r["f1_score"])
            return {n: {"length": len(v), "average": float(np.mean(v))} for n, v in g.items()}
        summary = {"total": {"total_length": len(f1_scores), "average_f1": float(np.mean(f1_scores)) if f1_scores else 0}, "area": _agg("area"), "subfield": _agg("subfield")}
        dump(summary, get_intermediate_file_path(eval_file, "_mmsearch_summarization_f1", "json"))
        return summary

