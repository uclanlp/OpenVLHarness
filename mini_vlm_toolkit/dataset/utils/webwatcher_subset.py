"""WebWatcher subset loader.

Reads ``configs/webwatcher_subsets.json`` and exposes a single helper
``apply_webwatcher_subset(df, dataset_name)`` that filters a DataFrame
to the indices listed in the subset config for the given dataset.

When invoked, the dataset class's ``load_data()`` should call this once
right before returning the data — by default the subset is applied
unconditionally so reported numbers stay aligned with WebWatcher's
evaluation. To bypass for a one-off full run, set the env var
``MVT_DISABLE_WEBWATCHER_SUBSET=1``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Optional

import pandas as pd


_CFG_PATH_DEFAULT = osp.normpath(osp.join(
    osp.dirname(__file__), "..", "..", "..", "configs", "webwatcher_subsets.json"
))


def _load_config(path: Optional[str] = None) -> Optional[dict]:
    p = path or _CFG_PATH_DEFAULT
    if not osp.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def get_webwatcher_subset_indices(dataset_name: str) -> Optional[list]:
    """Return the list of indices for `dataset_name`, or None if unsupported."""
    cfg = _load_config()
    if cfg is None:
        return None
    return cfg.get("subsets", {}).get(dataset_name, {}).get("indices")


def apply_webwatcher_subset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Filter `df` to the WebWatcher v2 subset for `dataset_name`.

    Returns a new DataFrame with rows whose ``index`` column matches one
    of the subset indices. When the env var
    ``MVT_DISABLE_WEBWATCHER_SUBSET=1`` is set, returns `df` unchanged.

    If `dataset_name` has no subset entry, returns `df` unchanged.
    """
    if os.environ.get("MVT_DISABLE_WEBWATCHER_SUBSET") == "1":
        return df

    indices = get_webwatcher_subset_indices(dataset_name)
    if not indices:
        return df

    # Match index column with type tolerance (TSV may load as int or str)
    idx_set = set(indices)
    # First try direct match
    mask = df["index"].isin(idx_set)
    if mask.sum() == 0:
        # Try string-cast match (handles int↔str dtype mismatch)
        mask = df["index"].astype(str).isin({str(i) for i in idx_set})

    n_before = len(df)
    out = df[mask].reset_index(drop=True)
    n_after = len(out)
    if n_after < len(idx_set):
        # Soft warning — don't fail loudly, but logger output makes it visible
        try:
            from mini_vlm_toolkit.smp import get_logger
            get_logger("WebWatcherSubset").warning(
                "WebWatcher subset for %s: matched %d/%d expected indices "
                "(local dataset has %d total rows)",
                dataset_name, n_after, len(idx_set), n_before,
            )
        except Exception:
            pass
    return out
