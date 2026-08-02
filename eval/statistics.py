"""Aggregate / paired statistics on per-instance metric arrays.

All functions take pandas Series (or numpy arrays) and return scalars or
small tuples. Designed to compose: compute per-instance metrics → store as
DataFrame columns → apply these aggregators to specific columns.

For method comparison vs the LN-global baseline, the typical flow is::

    deltas = paired_delta(steered=df_method["align_prob"],
                          base   =df_ln    ["align_prob"])
    headline = mean_delta(deltas)
    d_paired = cohens_d_paired(deltas)
    p        = wilcoxon_pvalue(deltas)
    lo, hi   = bootstrap_ci(deltas)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_delta(steered: pd.Series, base: pd.Series) -> pd.Series:
    """Element-wise paired difference: ``δ_i = steered_i − base_i``.

    Both Series must be aligned in instance order; this drops the index to
    avoid silent misalignment when one of the inputs has been reindexed.
    """
    return steered.reset_index(drop=True) - base.reset_index(drop=True)


def mean_delta(deltas: pd.Series) -> float:
    """Headline: arithmetic mean of paired deltas."""
    return float(deltas.mean())


def cohens_d_paired(deltas: pd.Series) -> float:
    """Cohen's d for paired data: ``mean(δ) / std(δ)``.

    Effect size; conventional cutoffs:  0.2 small, 0.5 medium, 0.8 large.
    Robust against the catastrophic-outlier blow-up that plagues mean(log P)
    because ``deltas`` is centered around 0 under H₀ (so a single outlier
    inflates ``std`` proportionally to its impact on ``mean``).
    """
    sd = float(deltas.std(ddof=1))
    return mean_delta(deltas) / max(sd, 1e-12)


def wilcoxon_pvalue(deltas: pd.Series) -> float:
    """Wilcoxon signed-rank test on paired deltas; H₀: ``median(δ) = 0``.

    Non-parametric; doesn't assume normality. Recommended as the headline
    significance test for method comparison because the distribution of
    paired deltas after the validity filter is rarely Gaussian.

    Returns ``1.0`` if every delta is exactly zero (no signal to test).
    """
    from scipy.stats import wilcoxon

    if (deltas == 0).all():
        return 1.0
    return float(wilcoxon(deltas).pvalue)


def bootstrap_ci(
    deltas: pd.Series,
    ci_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(deltas)``.

    Resamples ``deltas`` with replacement ``n_resamples`` times, takes the
    mean of each resample, returns the empirical ``(α/2, 1 − α/2)`` quantiles.
    Use this when ``deltas`` is non-Gaussian (which it usually is post-filter).

    ``seed`` defaults to the repo-wide ``globalenv.RANDOM_SEED`` for
    reproducibility — pass an int to override per-call.
    """
    if seed is None:
        from globalenv import RANDOM_SEED
        seed = RANDOM_SEED
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    means = np.empty(n_resamples, dtype=float)
    n = len(arr)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    alpha = 1.0 - ci_level
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high
