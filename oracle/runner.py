"""Steering-run orchestrator (Kaggle stub).

Defines the interface between the oracle harness and the Kaggle GPU
notebook that actually executes steering. Implement these functions in
the orchestrator notebook — they need the model, tokenizer, dataset,
and steering vectors which the notebook has bootstrapped.

Two functions to implement:

1. ``run_single_layer_evals``  → DataFrame for GT-global / TKM input
   (one row per (instance, single_layer))
2. ``run_subset_evals``        → DataFrame for Joint-K input
   (one row per (instance, layer_set), where ``layer_set`` is a tuple
    of length K)

Both emit DataFrames with the same schema (alignonly columns required;
multimetric columns optional and added if generation is enabled).

DataFrame schema
----------------

Required (for alignonly scoring):
- ``instance_id``     : Hashable, unique per test/val instance
- ``layer_set``       : Tuple[int, ...] — single-layer for runs 1, K-tuple for run 2
- ``layer``           : int — present only when ``len(layer_set) == 1``
                        (lets GT-global / TKM groupby cleanly without
                        unpacking the tuple)
- ``base_prob``       : float — P_align unsteered (renormalized over Yes/No)
- ``steered_prob``    : float — P_align steered (renormalized)

Optional (for multimetric scoring in PR 2):
- ``base_logit_yes``, ``base_logit_no``       : float
- ``steered_logit_yes``, ``steered_logit_no`` : float
- ``gold_sign``                                : int ∈ {+1, −1}
- ``yn_mass_steered``                          : float — P(yes) + P(no) full vocab
- ``rep_4gram_steered``                        : float — coherency
- ``delta_ppl``                                : float — coherency
- ``length_ratio``                             : float — sanity

Adding the optional columns at run time is cheap once generation is
enabled — they're computed from the same forward passes / generated
text. See ``eval/`` for the metric primitives to call.
"""
from typing import Hashable, Iterable, List, Sequence, Tuple

import pandas as pd


# ─── Public interface (Kaggle notebook implements these) ──────────────

def run_single_layer_evals(
    *,
    model,
    tokenizer,
    dataset,
    layers: Sequence[int],
    alpha: float = 1.0,
    generate_text: bool = False,
) -> pd.DataFrame:
    """Steer one layer at a time, return per-(instance, layer) DataFrame.

    Implement on Kaggle: for each instance × each layer, run an unsteered
    forward pass and a steered forward pass at that single layer, record
    base_prob and steered_prob (renormalized over Yes/No), and emit a
    row. Optionally generate text for coherency metrics.

    The output of this function feeds GT-global and TKM. It also serves
    as ``K=1`` rows for joint-K (single-layer subsets).

    Args:
        model:         steering-capable model wrapper.
        tokenizer:     for ``delta_ppl`` / generation when enabled.
        dataset:       iterable of test/val instances; each yields
                       (input_ids, gold_token_id, instance_id).
        layers:        layer indices to evaluate (e.g., ``range(32)``).
        alpha:         steering coefficient (uniform here; alpha-sweep
                       belongs in a wrapping loop).
        generate_text: if True, populate the multimetric optional columns.

    Returns:
        DataFrame with columns described in the module docstring.
    """
    raise NotImplementedError(
        "Implement in Kaggle orchestrator notebook. See module docstring "
        "for the required DataFrame schema."
    )


def run_subset_evals(
    *,
    model,
    tokenizer,
    dataset,
    layer_subsets: Iterable[Tuple[int, ...]],
    alpha_schedule="uniform",  # "uniform" → α=1 per layer; "k-norm" → α=1/K
    generate_text: bool = False,
) -> pd.DataFrame:
    """Steer joint K-layer subsets, return per-(instance, layer_set) DataFrame.

    Implement on Kaggle: for each instance × each subset in
    ``layer_subsets``, run unsteered and joint-steered forward passes
    (steering vector applied at each layer in the subset, scaled by
    ``alpha_schedule``) and emit a row.

    For exhaustive K=3 on N=32: ``layer_subsets =
    list(itertools.combinations(range(32), 3))`` — 4,960 subsets.

    For beam-W4 K=4, 5: the harness pre-computes a candidate pool via
    beam search over already-evaluated K-1 results (see
    ``oracle/config.py:SEARCH_BY_K``).

    Returns:
        DataFrame with the same schema as ``run_single_layer_evals``,
        except ``layer_set`` has ``len > 1`` and ``layer`` column is
        absent.
    """
    raise NotImplementedError(
        "Implement in Kaggle orchestrator notebook. See module docstring "
        "for the required DataFrame schema."
    )


# ─── Subset-pool helpers (no GPU needed; safe to call from harness) ────

def exhaustive_subsets(layers: Sequence[int], K: int) -> List[Tuple[int, ...]]:
    """All C(len(layers), K) subsets, as sorted tuples."""
    from itertools import combinations
    return [tuple(sorted(s)) for s in combinations(layers, K)]


def beam_extend(
    prev_picks: Iterable[Tuple[int, ...]],
    layers: Sequence[int],
    width: int,
) -> List[Tuple[int, ...]]:
    """Extend each surviving K−1 pick by every unused layer.

    The orchestrator should run this on the top-W picks from K−1, score
    the resulting K-subsets, and keep the top-W for the K+1 round.

    Args:
        prev_picks: top-W subsets from the previous K (length K−1).
        layers:     full pool of layer indices.
        width:      W; included for caller documentation, not used here
                    (caller does the top-W truncation after scoring).

    Returns:
        Deduplicated list of K-subsets to evaluate this round.
    """
    extensions = set()
    pool = set(layers)
    for pick in prev_picks:
        for l in pool - set(pick):
            extensions.add(tuple(sorted(pick + (l,))))
    return sorted(extensions)
