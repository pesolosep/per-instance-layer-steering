"""Joint-K oracle: per-instance argmax over K-sized layer subsets.

For each instance, evaluate the score of every K-sized subset present
in the DataFrame and pick the argmax. Subsets that the orchestrator
chose to evaluate (exhaustive for small K, beam for K ≥ 4 per
``oracle/config.py:SEARCH_BY_K``) are the candidate pool — this method
just picks the per-instance argmax from whatever's there.

Expected input: per-(instance, layer_set) DataFrame with ``score``,
``layer_set`` (tuple of layer ids), ``instance_id`` columns.

Note: this method does NOT itself perform the search — it consumes
whatever subsets the orchestrator already evaluated. ``runner.py`` is
responsible for choosing which subsets to evaluate (exhaustive vs beam).
"""
from typing import Dict, List, Hashable, Tuple

import pandas as pd


def select(scored_df: pd.DataFrame, K: int) -> Dict[Hashable, List[int]]:
    """Return per-instance argmax K-subset.

    Args:
        scored_df: per-(instance, layer_set) DataFrame with ``score``,
            ``layer_set``, ``instance_id`` columns. ``layer_set`` should
            be a tuple of layer ids of length K (filtered before calling).
        K: subset size; rows with ``len(layer_set) != K`` are filtered
            out before selection.

    Returns:
        Mapping ``instance_id -> [layer_ids]`` of length K (the
        argmax-scoring subset for that instance).
    """
    df_k = scored_df[scored_df["layer_set"].apply(lambda s: len(s) == K)]
    picks: Dict[Hashable, List[int]] = {}
    for iid, grp in df_k.groupby("instance_id"):
        best_row = grp.loc[grp["score"].idxmax()]
        picks[iid] = list(best_row["layer_set"])
    return picks
