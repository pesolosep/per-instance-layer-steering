"""Multi-metric scoring (v3): equal-weight z-scored sum of three signals.

Used by oracle methods (`joint_k`, `gt_global`, `tkm`) as a richer
alternative to `alignonly`. Combines three per-row signals:

  - alignment_lift  = steered_prob - base_prob
  - logit_margin_Δ  = gold_sign · ((s_yes - s_no) - (b_yes - b_no))
  - yn_mass_Δ       = steered_yn_mass - base_yn_mass

Each component is z-scored within the input DataFrame (across all
candidate (instance, layer_set) rows), then summed equal-weight. The
z-score normalization makes the score scale-invariant — `logit_margin_Δ`
would otherwise dominate due to its larger raw scale (±5 vs ±1 for
the others).

Returns DataFrame with `score` column added; **no gates, no
thresholds**. This is distinct from `oracle.scoring.multimetric` which
applies the gate-then-rank pipeline (broken on Yes/No data per
`docs/multimetric_calibration_plan.md`).

Use this scoring rule with the existing method selectors:

    from oracle.scoring import multimetric_3
    from oracle.methods import joint_k, gt_global, tkm

    df = multimetric_3.score(raw_search_phase_df)
    picks = joint_k.select(df, K)        # per-instance argmax over multi3
    picks = gt_global.select(df, K)      # global top-K by mean multi3
    picks = tkm.select(df, K)            # per-instance top-K by single-layer multi3
"""
from __future__ import annotations

import pandas as pd


def _z(series: pd.Series) -> pd.Series:
    """Z-score with degenerate-std guard."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Add `score` column = z(align_lift) + z(logit_margin_Δ) + z(yn_mass_Δ).

    Args:
        df: per-(instance, layer_set) DataFrame with required columns:
            base_prob, steered_prob,
            base_logit_yes, base_logit_no,
            steered_logit_yes, steered_logit_no,
            base_yn_mass, steered_yn_mass,
            gold_sign.

    Returns:
        Same DataFrame with `score` column added. Original columns are
        preserved (not overwritten) so downstream comparison can reuse them.
    """
    out = df.copy()
    align_lift = out["steered_prob"] - out["base_prob"]
    logit_margin_delta = out["gold_sign"] * (
        (out["steered_logit_yes"] - out["steered_logit_no"])
        - (out["base_logit_yes"] - out["base_logit_no"])
    )
    yn_mass_delta = out["steered_yn_mass"] - out["base_yn_mass"]

    out["score"] = _z(align_lift) + _z(logit_margin_delta) + _z(yn_mass_delta)
    return out
