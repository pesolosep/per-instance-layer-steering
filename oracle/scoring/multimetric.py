"""Multi-metric scoring with gate-then-rank.

Per-instance ``score = rank_metric(row) if all_gates_pass(row) else -inf``.

Methods aggregating this column will demote subsets/layers that fail
gates on many instances naturally — no special-casing needed in the
methods themselves.

Validity gates (applied per-instance, only where data is available):

- ``yn_mass``:   ``steered_yn_mass > yn_mass_min``   (kills format collapse)
- ``rep_4gram``: ``rep_4gram_steered < rep_4gram_max`` (kills templating/loops)
- ``delta_ppl``: ``delta_ppl < delta_ppl_max``         (kills incoherence drift)

Rank metrics (extensible via ``register_rank_metric``):

- ``logit_margin`` (default): paired Δ of signed logit margin — saturation-
  resistant, monotone, recommended primary.
- ``alignment_prob``: paired Δ of renormalized alignment probability —
  paper-comparable, but saturates above 0.95.

Scoring is **column-presence-aware**: only gates whose columns are in the
DataFrame fire. This lets you score search-phase data (logits only,
``steered_yn_mass`` gate fires) and full evaluation data (all gates fire)
with the same function.

Usage::

    from oracle.scoring.multimetric import score, calibrate_baselines

    # 1. Calibrate gate thresholds from the unsteered baseline once per (task, model).
    baselines = calibrate_baselines(
        base_yn_mass=baseline_df["base_yn_mass"],
        base_rep_4gram=baseline_df["rep_4gram_base"],   # if generated
        base_ppl_mean=baseline_df["base_ppl"].mean(),    # if computed
    )

    # 2. Score the search-phase or full-eval DataFrame.
    scored = score(df, baselines=baselines, rank_metric="logit_margin")

    # 3. Hand to oracle/methods/* — they read the `score` column unchanged.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


# ─── Rank metric registry ───────────────────────────────────────────────

RankMetricFn = Callable[[pd.DataFrame], pd.Series]

RANK_METRICS: Dict[str, RankMetricFn] = {}


def register_rank_metric(name: str, fn: RankMetricFn) -> None:
    """Register a rank metric by name."""
    RANK_METRICS[name] = fn


def _logit_margin_delta(df: pd.DataFrame) -> pd.Series:
    lm_steered = df["gold_sign"] * (df["steered_logit_yes"] - df["steered_logit_no"])
    lm_base = df["gold_sign"] * (df["base_logit_yes"] - df["base_logit_no"])
    return lm_steered - lm_base


def _align_prob_delta(df: pd.DataFrame) -> pd.Series:
    return df["steered_prob"] - df["base_prob"]


register_rank_metric("logit_margin", _logit_margin_delta)
register_rank_metric("alignment_prob", _align_prob_delta)


# ─── Gating ─────────────────────────────────────────────────────────────

def _gates_pass(df: pd.DataFrame, baselines: Dict[str, float]) -> pd.Series:
    """Return a per-row boolean Series; True = all available gates pass.

    Only gates whose columns are present in ``df`` AND whose threshold
    is in ``baselines`` are applied. Missing gates are treated as
    automatically passing (don't reject in the absence of evidence).
    """
    valid = pd.Series(True, index=df.index)

    if "steered_yn_mass" in df.columns and "yn_mass_min" in baselines:
        valid &= df["steered_yn_mass"] > baselines["yn_mass_min"]

    if "rep_4gram_steered" in df.columns and "rep_4gram_max" in baselines:
        valid &= df["rep_4gram_steered"] < baselines["rep_4gram_max"]

    if "delta_ppl" in df.columns and "delta_ppl_max" in baselines:
        valid &= df["delta_ppl"] < baselines["delta_ppl_max"]

    return valid


# ─── Public API ─────────────────────────────────────────────────────────

def score(
    df: pd.DataFrame,
    baselines: Optional[Dict[str, float]] = None,
    *,
    rank_metric: str = "logit_margin",
    apply_gates: bool = True,
) -> pd.DataFrame:
    """Add a ``score`` column via gate-then-rank.

    Args:
        df: per-(instance, layer_set) DataFrame from the runner. Required
            columns depend on ``rank_metric``: ``logit_margin`` needs
            ``base_logit_yes``, ``base_logit_no``, ``steered_logit_yes``,
            ``steered_logit_no``, ``gold_sign``; ``alignment_prob`` needs
            ``base_prob``, ``steered_prob``. Gate columns
            (``steered_yn_mass``, ``rep_4gram_steered``, ``delta_ppl``)
            are optional.
        baselines: thresholds dict, e.g. ``{"yn_mass_min": 0.5,
            "rep_4gram_max": 0.4, "delta_ppl_max": 200.0}``. Required
            unless ``apply_gates=False``.
        rank_metric: name registered in ``RANK_METRICS``. Default
            ``"logit_margin"``.
        apply_gates: if False, rank-only score (ungated). Use for the
            "ungated multimetric" ablation cell — isolates gating
            contribution from rank-metric contribution.

    Returns:
        ``df`` with ``score`` column. ``-inf`` where gates fail (only
        when ``apply_gates=True``).
    """
    if rank_metric not in RANK_METRICS:
        raise ValueError(
            f"Unknown rank_metric {rank_metric!r}. Registered: "
            f"{list(RANK_METRICS)}. Use register_rank_metric() to add."
        )
    rank_score = RANK_METRICS[rank_metric](df)

    if not apply_gates:
        return df.assign(score=rank_score)

    if baselines is None:
        raise ValueError("baselines is required when apply_gates=True")

    valid = _gates_pass(df, baselines)
    score_col = np.where(valid, rank_score, -np.inf)
    return df.assign(score=score_col)


# ─── Threshold calibration ──────────────────────────────────────────────

def calibrate_baselines(
    base_yn_mass: Optional[pd.Series] = None,
    base_rep_4gram: Optional[pd.Series] = None,
    base_ppl_mean: Optional[float] = None,
    *,
    yn_mass_quantile: float = 0.05,
    rep_4gram_quantile: float = 0.95,
    ppl_factor: float = 2.0,
) -> Dict[str, float]:
    """Compute gate thresholds from the unsteered baseline distribution.

    Per-task, per-model. Saved once (as JSON) and reused across all
    steered runs for that (task, model) pair.

    Defaults follow the eval module's documented thresholds:

    - ``yn_mass_min`` = 5th percentile of baseline yn_mass (reject
      format-collapse below baseline-rare)
    - ``rep_4gram_max`` = 95th percentile of baseline 4-gram repetition
      (reject above baseline-rare repetition)
    - ``delta_ppl_max`` = 2 × baseline PPL (reject text 2× weirder than
      baseline under the unsteered model)

    Each input is optional; thresholds for missing inputs are not
    populated (so the corresponding gate auto-passes in ``score()``).

    Returns:
        baselines dict consumable by ``score()``.
    """
    out: Dict[str, float] = {}
    if base_yn_mass is not None:
        out["yn_mass_min"] = float(base_yn_mass.quantile(yn_mass_quantile))
    if base_rep_4gram is not None:
        out["rep_4gram_max"] = float(base_rep_4gram.quantile(rep_4gram_quantile))
    if base_ppl_mean is not None:
        out["delta_ppl_max"] = float(ppl_factor * base_ppl_mean)
    return out


__all__ = [
    "score",
    "calibrate_baselines",
    "register_rank_metric",
    "RANK_METRICS",
]
