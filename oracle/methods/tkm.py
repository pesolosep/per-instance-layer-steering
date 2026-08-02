"""Top-K Marginal: per-instance top-K of single-layer effects.

For each instance, rank single-layer scores descending and take top-K.
Different layer set per instance. NOT deployable — needs behavioral
evaluation per test instance. Used as upper bound and for labeling.

Expected input: same as GT-global (per-instance, per-single-layer).
"""
from typing import Dict, List, Hashable

import pandas as pd


def select(scored_df: pd.DataFrame, K: int) -> Dict[Hashable, List[int]]:
    """Return per-instance top-K layer picks.

    Args:
        scored_df: per-(instance, single-layer) DataFrame with ``score``,
            ``layer``, ``instance_id`` columns.
        K: number of layers to select per instance.

    Returns:
        Mapping ``instance_id -> [layer_ids]`` of length K, ranked
        descending by score within each instance.
    """
    picks: Dict[Hashable, List[int]] = {}
    for iid, grp in scored_df.groupby("instance_id"):
        picks[iid] = grp.nlargest(K, "score")["layer"].tolist()
    return picks
