"""Alignment-only scoring — faithful to LayerNavigator.

``score = steered_prob - base_prob`` per row. Single metric, probability
space, no validity gates. This is the recipe from
``LayerNavigator/layernav/pipeline.py`` (line 235) and
``LayerNavigator/layernav/joint_oracle.py`` (line 478).

Use this for replicating LayerNavigator baselines. Use ``multimetric``
(PR 2) for the gate-then-rank methodological extension.
"""
import pandas as pd


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with an added ``score`` column.

    Args:
        df: per-(instance, layer_set) raw eval DataFrame. Must contain
            ``base_prob`` and ``steered_prob`` columns.

    Returns:
        Same DataFrame with ``score = steered_prob - base_prob`` added.
        Original columns preserved so downstream comparison can reuse them.
    """
    return df.assign(score=df["steered_prob"] - df["base_prob"])
