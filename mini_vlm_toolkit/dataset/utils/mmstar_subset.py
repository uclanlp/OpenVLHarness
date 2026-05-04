"""MMStar stratified subset sampling helper.

Produces a deterministic random sub-sample of MMStar that preserves
the natural distribution of both ``category`` (6 classes, 250 each) and
``l2_category`` (18 classes, 31..146 each; nests inside ``category``).

Stratification is on ``l2_category`` because every l2_category belongs
to exactly one category — sampling proportionally on the finer-grained
column automatically preserves the coarser one.

Opt-in via env var ``MVT_MMSTAR_SUBSET_N`` (e.g. ``300``) and
``MVT_MMSTAR_SUBSET_SEED`` (default ``42``). When ``MVT_MMSTAR_SUBSET_N``
is unset / empty / non-positive / >= len(df), the full set passes
through unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def apply_mmstar_stratified_subset(
    df: pd.DataFrame,
    n: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Apply a stratified random sub-sample to an MMStar DataFrame.

    Args:
        df: full MMStar DataFrame (1500 rows, with ``l2_category`` column)
        n: target sample size. If ``None``, reads from
           ``MVT_MMSTAR_SUBSET_N`` env (no-op when unset).
        seed: random_state for reproducibility. If ``None``, reads from
           ``MVT_MMSTAR_SUBSET_SEED`` env, default ``42``.

    Returns:
        Sub-sampled DataFrame (size ~= n) with original column schema and
        natural per-l2_category distribution preserved. Returned rows are
        sorted by ``index`` (stable order across re-runs).
    """
    if n is None:
        env_n = os.environ.get("MVT_MMSTAR_SUBSET_N", "").strip()
        if not env_n:
            return df
        try:
            n = int(env_n)
        except ValueError:
            logger.warning(
                "MMStar subset: ignoring invalid MVT_MMSTAR_SUBSET_N=%r", env_n
            )
            return df

    if seed is None:
        env_seed = os.environ.get("MVT_MMSTAR_SUBSET_SEED", "").strip()
        try:
            seed = int(env_seed) if env_seed else 42
        except ValueError:
            logger.warning(
                "MMStar subset: ignoring invalid MVT_MMSTAR_SUBSET_SEED=%r, using 42",
                env_seed,
            )
            seed = 42

    total = len(df)
    if n <= 0 or n >= total:
        return df
    if "l2_category" not in df.columns:
        logger.warning(
            "MMStar subset: 'l2_category' column missing — returning full set"
        )
        return df

    counts = df["l2_category"].value_counts()
    parts = []
    for l2_cat, group_n in counts.items():
        # Proportional rounding; keep at least 1 from each stratum.
        sample_n = int(round(int(group_n) * n / total))
        sample_n = max(1, min(sample_n, int(group_n)))
        sub = df[df["l2_category"] == l2_cat]
        parts.append(sub.sample(n=sample_n, random_state=seed))

    sampled = pd.concat(parts)
    # Stable order: sort by ``index`` if numeric, else fall back to original.
    try:
        sampled = sampled.iloc[
            sampled["index"].astype(int).argsort().values
        ].reset_index(drop=True)
    except (KeyError, ValueError):
        sampled = sampled.reset_index(drop=True)

    logger.info(
        "MMStar stratified subset applied: %d → %d (seed=%d, stratify=l2_category)",
        total,
        len(sampled),
        seed,
    )
    return sampled
