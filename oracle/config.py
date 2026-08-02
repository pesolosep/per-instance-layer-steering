"""K-sweep configuration for the oracle harness.

Edit these to adjust compute budget. ``SEARCH_BY_K`` selects the search
algorithm per K — exhaustive is feasible only for small K (cost is
combinatorial). Honest cost estimates per (task, model) at ~1s/subset
batched across instances:

| K | C(32, K) | exhaustive cost | beam-W4 cost   |
|---|----------|------------------|----------------|
| 1 |       32 | seconds          | n/a (trivial)  |
| 2 |      496 | minutes          | n/a            |
| 3 |    4,960 | ~1.4 hours       | minutes        |
| 4 |   35,960 | ~10 hours ❌      | minutes        |
| 5 |  201,376 | ~56 hours ❌❌     | minutes        |

Default policy: exhaustive for K ≤ 3, beam at width ``BEAM_WIDTH`` for
K ∈ {4, 5}. Validate the beam approximation against exhaustive at K=3
(Jaccard agreement) before reporting K=4, 5 results.

The ``"beam-W{N}"`` strings are derived from ``BEAM_WIDTH`` via f-string
so the width is single-source-of-truth: change ``BEAM_WIDTH`` and the
search-method labels (and downstream output filenames like
``K3_beam-W4.parquet``) update automatically.
"""
from typing import Dict, List

K_VALUES: List[int] = [1, 2, 3, 4, 5]

BEAM_WIDTH: int = 4
BEAM_LABEL: str = f"beam-W{BEAM_WIDTH}"

SEARCH_BY_K: Dict[int, List[str]] = {
    1: ["trivial"],                  # 32 single-layer evals, no search
    2: ["exhaustive", BEAM_LABEL],   # both: 496 pairs + beam W=4 (early beam validation)
    3: ["exhaustive", BEAM_LABEL],   # both: 4,960 triples + beam (paper-faithful + beam validation)
    4: [BEAM_LABEL],                 # exhaustive infeasible at K=4 over full layer pool
    5: [BEAM_LABEL],                 # exhaustive infeasible at K=5 over full layer pool
}
# Joint-K exhaustive at K=4, 5 is reintroduced separately (deferred) via a
# `joint-k-exhaustive-filtered` variant that restricts the layer pool to a
# manually-configured range, making C(|pool|, K) tractable at those K values.
