"""MMSearch end-to-end task (MMSearch): one question plus an optional query image; the model
answers and is scored by the LLM judge (token-level F1 is also reported)."""

import collections
import logging
import os
import re
from typing import Any, Dict, List, Optional

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

    # Needs an LLM judge (or exact_matching); checked before inference.
    REQUIRES_LLM_JUDGE = True

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
    ``judge_kwargs['judge_prompt']``: ``'judge_vdr'`` (default) | ``'hle'`` |
    ``'judge_rs'``. Cache + score files are suffixed with ``_judge_rs`` /
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

    # Judge prompt selection: env var > judge_kwargs > default 'judge_vdr'
    judge_name = (
        os.environ.get('MMDR_JUDGE_PROMPT')
        or judge_kwargs_local.pop('judge_prompt', None)
        or 'judge_vdr'
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
# HLE-style judge (``judge_prompt: hle``; the default is judge_vdr): shared prompt (extract_final_answer + binary
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
# parser. Selected via env var MMDR_JUDGE_PROMPT=judge_rs.
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


# VDR paper "contains-based" binary judge, the default judge.
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
