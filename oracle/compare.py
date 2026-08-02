"""Comparison utilities across methods, metrics, and K values.

Three core comparisons enabled:

1. Method-vs-method at fixed metric: e.g., GT-global vs Joint-K under
   alignonly scoring → does joint search beat aggregate-then-select?
2. Metric-vs-metric at fixed method: e.g., Joint-K alignonly vs Joint-K
   multimetric → does gating change which subsets are picked?
3. Search-fidelity: beam-W4 picks vs exhaustive picks at K=3 →
   validates beam approximation before reporting K=4, 5 results.
"""
from typing import Dict, Hashable, Iterable, List

import pandas as pd


# ─── Per-instance pick comparisons ────────────────────────────────────

def jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> float:
    """Jaccard similarity of two layer sets."""
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_per_instance(
    picks_a: Dict[Hashable, List[int]],
    picks_b: Dict[Hashable, List[int]],
) -> Dict[Hashable, float]:
    """Jaccard agreement per instance between two per-instance pick maps."""
    common = set(picks_a) & set(picks_b)
    return {iid: jaccard(picks_a[iid], picks_b[iid]) for iid in common}


def mean_jaccard(
    picks_a: Dict[Hashable, List[int]],
    picks_b: Dict[Hashable, List[int]],
) -> float:
    """Mean Jaccard agreement across instances."""
    per = jaccard_per_instance(picks_a, picks_b)
    if not per:
        return float("nan")
    return sum(per.values()) / len(per)


# ─── Score-gap analysis ───────────────────────────────────────────────

def gap_per_instance(
    scored_df: pd.DataFrame,
    picks_a: Dict[Hashable, List[int]],
    picks_b: Dict[Hashable, List[int]],
) -> pd.Series:
    """Per-instance ``score(picks_a) - score(picks_b)`` (method-A advantage).

    Looks up the ``score`` column for each (instance, picked_set) pair
    in both methods and returns the per-instance difference.

    Requires that both sets-of-interest be present as rows in
    ``scored_df`` — i.e., the orchestrator evaluated both. If a method's
    pick is absent from the DataFrame the instance is dropped.
    """
    scored_df = scored_df.copy()
    scored_df["_set_key"] = scored_df["layer_set"].apply(lambda s: tuple(sorted(s)))
    keyed = scored_df.set_index(["instance_id", "_set_key"])["score"]

    rows = []
    for iid in set(picks_a) & set(picks_b):
        ka = tuple(sorted(picks_a[iid]))
        kb = tuple(sorted(picks_b[iid]))
        if (iid, ka) not in keyed.index or (iid, kb) not in keyed.index:
            continue
        rows.append((iid, keyed.loc[(iid, ka)] - keyed.loc[(iid, kb)]))
    if not rows:
        return pd.Series(dtype=float)
    iids, gaps = zip(*rows)
    return pd.Series(gaps, index=list(iids))


def mean_gap(
    scored_df: pd.DataFrame,
    picks_a: Dict[Hashable, List[int]],
    picks_b: Dict[Hashable, List[int]],
) -> float:
    """Mean per-instance score gap (method-A score − method-B score)."""
    g = gap_per_instance(scored_df, picks_a, picks_b)
    return float(g.mean()) if len(g) else float("nan")
