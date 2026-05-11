"""Per-dataset eval-result injection into ``full_trajectories/<task_id>.json``.

After ``dataset.evaluate(result_file, ...)`` returns, the merger walks the
trajectory dir, looks up a per-dataset adapter by the dataset object's class
(MRO walk: most-specific class wins), and writes
``payload["eval"] = {"correct", "score", "raw"}`` into each JSON.

Adapters are registered by class **name** so the registry doesn't need to
import every dataset class up front (avoids cyclic imports and keeps startup
cheap).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

from ...smp import dump as _dump, load as _load
from . import adapters as _adapters


logger = logging.getLogger(__name__)


# Each entry: (adapter_callable, kwargs_for_adapter).
# Order within a class's MRO walk is bottom-up (most specific first), so a
# subclass entry overrides its parent's entry.
_REGISTRY: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
    # --- ImageMCQDataset family (image_mcq.py + spatial-bench MCQ subclasses) ---
    # Default columns: hit / log. Subclasses add their own raw cols below.
    "ImageMCQDataset": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["log", "extracted", "pred_extracted"]},
    ),
    # --- ImageYORNDataset family (MME / HallusionBench / POPE / AMBER / VSR-zeroshot) ---
    # evaluate() writes _<judge>_result.xlsx with a boolean `score` column
    # (answer == extracted); reuse the MCQ adapter pointed at that column.
    "ImageYORNDataset": (
        _adapters.mcq_result_adapter,
        {"correct_col": "score", "raw_cols": ["extracted", "answer"]},
    ),
    "FREAKMCQDataset": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["extracted"]},
    ),
    # MMSI / EmbSpatial / ERQA / MindCube / OmniSpatial subclass ImageMCQDataset
    # but write `pred_extracted` instead of `log`. Register them explicitly so
    # `raw` carries the right field names.
    "MMSIBench": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["pred_extracted", "question_type"]},
    ),
    "EmbSpatialBench": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["pred_extracted", "data_source"]},
    ),
    "ERQABench": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["pred_extracted", "question_type"]},
    ),
    "MindCubeBench": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["pred_extracted"]},
    ),
    "OmniSpatialBench": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": ["pred_extracted", "task_type", "sub_task_type"]},
    ),
    # FREAK_VQA: same xlsx pattern, only `hit` (no extracted/log).
    "FREAKVQADataset": (
        _adapters.mcq_result_adapter,
        {"correct_col": "hit", "raw_cols": []},
    ),
    # --- Spatial bench standalone classes (write _<judge>_result.xlsx via build_eval_paths) ---
    "RoboSpatialBench": (
        _adapters.mcq_result_adapter,
        {
            "correct_col": "is_correct",
            "raw_cols": ["category", "is_binary", "parsed_answer", "is_parsable"],
        },
    ),
    "RefSpatialBench": (
        _adapters.mcq_result_adapter,
        {
            "correct_col": None,
            "score_col": "score",
            "raw_cols": ["num_points", "is_parsable", "category"],
        },
    ),
    "SparBench": (
        _adapters.mcq_result_adapter,
        {
            "correct_col": "hit",
            "score_col_alt": "MRA:.5:.95:.05",
            "raw_cols": ["task", "task_type", "pred_extracted", "MRA:.5:.95:.05"],
        },
    ),
    # --- VSI-Bench (different filename: _result.xlsx without judge tag) ---
    "VSIBench": (_adapters.vsibench_adapter, {}),
    # --- Counting datasets (write _per_row.csv) ---
    "FSC147Dataset": (_adapters.counting_adapter, {}),
    "PixmoCount": (_adapters.counting_adapter, {}),
    "CountQA": (_adapters.counting_adapter, {"use_avg3_score": True}),
    # --- SimpleVQA (sibling gpt_eval.json) ---
    "SimpleVQA": (_adapters.simplevqa_adapter, {}),
    # --- MMSearch family (eval_file overwritten in place) ---
    "MMSearchEnd2End": (
        _adapters.inplace_score_adapter,
        {
            "correct_col": "is_correct",
            "score_from_correct": True,
            "raw_cols": ["f1_score"],
        },
    ),
    "MMSearchSummarization": (
        _adapters.inplace_score_adapter,
        {
            "score_col": "f1_score",
            "correct_threshold": 0.5,
            "raw_cols": ["f1_score"],
        },
    ),
    "MMSearchRequery": (
        _adapters.inplace_score_adapter,
        {
            "score_col": "requery_score",
            "raw_cols": ["requery_score"],
        },
    ),
    "MMSearchRerank": (
        _adapters.inplace_score_adapter,
        {
            "score_col": "rerank_score",
            "correct_threshold": 1.0,
            "raw_cols": ["rerank_score"],
        },
    ),
    # --- RealXBench (eval_file overwritten in place) ---
    "RealXBench": (
        _adapters.inplace_score_adapter,
        {
            "correct_col": "is_correct",
            "score_from_correct": True,
            "raw_cols": ["f1_score", "task_type"],
        },
    ),
}


def _lookup(dataset_obj) -> Optional[Tuple[Callable, Dict[str, Any]]]:
    for cls in type(dataset_obj).__mro__:
        entry = _REGISTRY.get(cls.__name__)
        if entry is not None:
            return entry
    return None


def merge_eval_into_trajectories(
    work_dir: str,
    dataset_name: str,
    dataset_obj,
    result_file: str,
    judge_kwargs: Optional[Dict[str, Any]] = None,
) -> int:
    """Merge per-instance eval results into ``<work_dir>/full_trajectories/<idx>.json``.

    Returns the number of trajectory JSONs updated. No-ops (returns 0) when:
      - the trajectory dir doesn't exist (e.g., legacy run / inference-only run);
      - no adapter is registered for the dataset's class hierarchy;
      - the adapter's source file is missing.

    Idempotent: re-running overwrites the ``eval`` block cleanly.
    """
    full_traj_dir = os.path.join(work_dir, "full_trajectories")
    if not os.path.isdir(full_traj_dir):
        return 0

    entry = _lookup(dataset_obj)
    if entry is None:
        logger.warning(
            "[eval_injection] no adapter registered for %s (class %s); "
            "skipping eval merge into full_trajectories",
            dataset_name,
            type(dataset_obj).__name__,
        )
        return 0

    adapter, kwargs = entry
    try:
        per_instance = adapter(result_file, dataset_obj, judge_kwargs or {}, **kwargs)
    except Exception as exc:
        logger.warning(
            "[eval_injection] adapter %s failed for %s: %s",
            adapter.__name__,
            dataset_name,
            exc,
        )
        return 0

    if not per_instance:
        logger.warning(
            "[eval_injection] adapter %s returned no per-instance results for %s "
            "(judge file likely missing)",
            adapter.__name__,
            dataset_name,
        )
        return 0

    updated = 0
    for task_id, eval_block in per_instance.items():
        json_path = os.path.join(full_traj_dir, f"{task_id}.json")
        if not os.path.exists(json_path):
            continue
        try:
            payload = _load(json_path)
        except Exception as exc:
            logger.warning(
                "[eval_injection] failed to read %s: %s", json_path, exc,
            )
            continue
        payload["eval"] = eval_block
        _dump(payload, json_path)
        updated += 1

    logger.info(
        "[eval_injection] merged eval into %d/%d trajectories for %s "
        "(adapter=%s)",
        updated,
        len(per_instance),
        dataset_name,
        adapter.__name__,
    )
    return updated


__all__ = ["merge_eval_into_trajectories"]
