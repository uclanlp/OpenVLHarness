"""Vision-DeepResearch subset loader.

Used by benchmarks that own a superset of Vision-DeepResearch's rows and want a dataset name that
IS their rows (currently ``WorldVQA_VDR300``). Mirrors ``webwatcher_subset.py``: one helper,
config-driven, no per-dataset logic here.

⛔ WHY A DEDICATED DATASET NAME RATHER THAN A POST-HOC RESCORE. Inferring the full superset and
  rescoring over their index list afterwards gives the same number but pays for every discarded
  rollout: WorldVQA is 3,000 rows for a 300-row answer. A ``*_VDR300`` name runs exactly their rows
  and needs no rescore.

⛔ WHY NOT SLICE THE ORIGINAL NAME IN PLACE. MMSearch (300 -> 171), SimpleVQA_EN (1013 -> 300) and
  hle (-> 330) are trimmed unconditionally inside their own classes via ``apply_webwatcher_subset``,
  which is fine because nobody reports their full sets. WorldVQA is different: runs exist against
  the full set, and silently redefining that name would make them irreproducible.

⛔ THE MAPPING IS BY CONTENT, NOT BY ID STRING. Their rows were matched onto ours by
  (question, answer), with a perceptual image hash to break ties, and the result is stored in
  ``configs/vdr_eval_subsets.json`` as ``ours_index``. Their published ids live in a different
  namespace (``worldvqa_test_1901`` against our positional ``0..2999``), so a naive intersection of
  the two id sets is empty. ``ours_index`` is the only correct join.
"""

from __future__ import annotations

import json
import os.path as osp
from typing import List

import pandas as pd

_SUBSETS_PATH = osp.normpath(osp.join(
    osp.dirname(__file__), "..", "..", "..", "configs", "vdr_eval_subsets.json"))


def vdr_subset_indices(bench: str, expected: int) -> List[str]:
    """The indices of OUR rows that correspond to Vision-DeepResearch's rows for *bench*.

    Raises rather than returning a short list. A silently-truncated subset produces a
    plausible-looking accuracy over the wrong denominator.
    """
    if not osp.exists(_SUBSETS_PATH):
        raise FileNotFoundError(
            f"vdr subset for {bench!r} needs {_SUBSETS_PATH}, which ships with the repository "
            f"and is missing from this checkout.")
    with open(_SUBSETS_PATH) as f:
        blob = json.load(f)
    try:
        ids = blob["benchmarks"][bench]["ours_index_list"]["ours_index"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"benchmarks.{bench}.ours_index_list.ours_index missing from {_SUBSETS_PATH} "
            f"({exc}).") from exc
    if not isinstance(ids, list) or len(ids) != expected:
        raise ValueError(
            f"vdr subset for {bench!r}: expected {expected} indices in ours_index_list, got "
            f"{len(ids) if isinstance(ids, list) else type(ids).__name__}. Refusing to build a set "
            f"that is not theirs.")
    return ids


def apply_vdr_subset(df: pd.DataFrame, bench: str, expected: int, dataset_name: str) -> pd.DataFrame:
    """Filter *df* to Vision-DeepResearch's rows for *bench*, or raise.

    ``dataset_name`` is only used in the error text, so the message names the dataset the caller
    asked for rather than the registry key.
    """
    keep = vdr_subset_indices(bench, expected)
    # Type-tolerant match: a TSV index column may load as int or str depending on pandas'
    # inference, while ours_index is written as str.
    mask = df["index"].isin(set(keep))
    if mask.sum() == 0:
        mask = df["index"].astype(str).isin({str(i) for i in keep})
    out = df[mask].reset_index(drop=True)
    if len(out) != expected:
        raise ValueError(
            f"{dataset_name}: sliced {len(out)} rows, expected {expected}. The local source table "
            f"does not contain every row Vision-DeepResearch evaluates on, so a score on it would "
            f"be over a different denominator than theirs. Refusing.")
    return out
