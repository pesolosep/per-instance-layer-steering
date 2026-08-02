"""Joint-K layer-selection strategies.

Selects K layers *jointly* — accounting for cross-layer interactions —
so the K=3 result need not be a prefix of the K=5 result. Search is
decoupled from scoring: edit ``_load_score_fn`` to wire up your
joint-score data source (pairwise matrix, oracle gap, online activation
evaluation, etc.) and the four search variants below all use it.

Variants registered:
- ``joint-k``                  — greedy forward selection
- ``joint-k-beam-W<int>``      — beam search, width parsed from name
- ``joint-k-exhaustive``       — evaluate every C(N, K) subset
- ``joint-k-exhaustive-filtered`` — exhaustive over EXHAUSTIVE_FILTER_RANGE only
"""
from itertools import combinations
import re
from typing import Callable, List, Optional, Tuple

from .base import LayerStrategy, register


# === MANUAL CONFIG: edit to change the layer range for exhaustive-filtered ===
# Inclusive [low, high]. Useful when you want to do exhaustive search but
# only over a region of interest (e.g., late-mid layers L21/L28 from your
# TKM analysis) — saves combinatorial blowup vs full-range exhaustive.
EXHAUSTIVE_FILTER_RANGE: Tuple[int, int] = (10, 25)


ScoreFn = Callable[[List[int]], float]


def _load_score_fn(task: str, method: str) -> ScoreFn:
    """Return a function score(subset) -> float for joint-K evaluation.

    TODO: wire this up to your data source — pairwise score matrix,
    oracle gap, online activation evaluation, etc. The four search
    algorithms below call this for every candidate subset.
    """
    raise NotImplementedError(
        "Implement _load_score_fn in strategies/joint_k.py to wire up "
        "your joint-score data source (pairwise matrix, oracle gap, ...)."
    )


# ─── Search algorithms (independent of how subsets are scored) ─────────

def _greedy_forward(
    score_fn: ScoreFn,
    layers: List[int],
    num_layers: int,
) -> List[Tuple[int, float]]:
    chosen: List[int] = []
    remaining = list(layers)
    out: List[Tuple[int, float]] = []
    for _ in range(num_layers):
        best_layer, best_score = max(
            ((l, score_fn(chosen + [l])) for l in remaining),
            key=lambda x: x[1],
        )
        chosen.append(best_layer)
        remaining.remove(best_layer)
        out.append((best_layer, best_score))
    return out


def _beam_search(
    score_fn: ScoreFn,
    layers: List[int],
    num_layers: int,
    width: int,
) -> List[Tuple[int, float]]:
    # Initial candidates: top-W single layers.
    candidates: List[Tuple[float, List[int]]] = sorted(
        ((score_fn([l]), [l]) for l in layers),
        key=lambda x: x[0],
        reverse=True,
    )[:width]

    for _ in range(num_layers - 1):
        next_candidates: List[Tuple[float, List[int]]] = []
        for _score, subset in candidates:
            for l in layers:
                if l in subset:
                    continue
                new_subset = subset + [l]
                next_candidates.append((score_fn(new_subset), new_subset))
        next_candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = next_candidates[:width]

    best_score, best_subset = candidates[0]
    return [(l, score_fn(best_subset[: i + 1])) for i, l in enumerate(best_subset)]


def _exhaustive(
    score_fn: ScoreFn,
    layers: List[int],
    num_layers: int,
) -> List[Tuple[int, float]]:
    if num_layers > len(layers):
        raise ValueError(
            f"num_layers={num_layers} exceeds available layer pool size {len(layers)}"
        )
    best_subset, best_score = max(
        ((list(s), score_fn(list(s))) for s in combinations(layers, num_layers)),
        key=lambda x: x[1],
    )
    return [(l, score_fn(best_subset[: i + 1])) for i, l in enumerate(best_subset)]


# ─── Registered strategies ─────────────────────────────────────────────

@register
class JointKGreedy(LayerStrategy):
    @classmethod
    def match(cls, name: str) -> Optional[dict]:
        return {} if name == "joint-k" else None

    def select(self, task, method, layers, num_layers):
        return _greedy_forward(_load_score_fn(task, method), layers, num_layers)


@register
class JointKBeam(LayerStrategy):
    _PATTERN = re.compile(r"^joint-k-beam-W(\d+)$")

    def __init__(self, width: int):
        assert width >= 1, f"width must be >= 1, got {width}"
        self.width = width

    @classmethod
    def match(cls, name: str) -> Optional[dict]:
        m = cls._PATTERN.fullmatch(name)
        return {"width": int(m.group(1))} if m else None

    def select(self, task, method, layers, num_layers):
        return _beam_search(
            _load_score_fn(task, method), layers, num_layers, self.width
        )


@register
class JointKExhaustive(LayerStrategy):
    @classmethod
    def match(cls, name: str) -> Optional[dict]:
        return {} if name == "joint-k-exhaustive" else None

    def select(self, task, method, layers, num_layers):
        return _exhaustive(_load_score_fn(task, method), layers, num_layers)


@register
class JointKExhaustiveFiltered(LayerStrategy):
    """Exhaustive search restricted to ``EXHAUSTIVE_FILTER_RANGE``.

    Edit the module-level constant to change the range — chosen by hand
    rather than by name so it's obvious in the file when set.
    """

    @classmethod
    def match(cls, name: str) -> Optional[dict]:
        return {} if name == "joint-k-exhaustive-filtered" else None

    def select(self, task, method, layers, num_layers):
        lo, hi = EXHAUSTIVE_FILTER_RANGE
        filtered = [l for l in layers if lo <= l <= hi]
        if num_layers > len(filtered):
            raise ValueError(
                f"num_layers={num_layers} exceeds filtered pool size "
                f"{len(filtered)} (range [{lo},{hi}]); widen the range."
            )
        return _exhaustive(_load_score_fn(task, method), filtered, num_layers)
