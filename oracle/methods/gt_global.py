"""GT-global: one layer set for all instances.

Aggregate per-instance scores by mean per layer, then take top-K. This
is the LayerNavigator GT-global baseline (``pipeline.py`` line 235, 366).

Expected input: scored DataFrame with one row per (instance, layer) —
i.e., single-layer evaluations only. ``layer_set`` should be a length-1
tuple ``(layer_id,)`` for compatibility with the joint_k DataFrame
schema.
"""
from typing import List

import pandas as pd


def select(scored_df: pd.DataFrame, K: int) -> List[int]:
    """Return the K layers with the highest mean per-instance score.

    Args:
        scored_df: per-(instance, single-layer) DataFrame with ``score``,
            ``layer`` columns.
        K: number of layers to select.

    Returns:
        List of K layer ids, ranked descending by mean score.
    """
    per_layer_mean = scored_df.groupby("layer")["score"].mean()
    return per_layer_mean.nlargest(K).index.tolist()
