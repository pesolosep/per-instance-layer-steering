"""Base class, registry, and shared helpers for layer-selection strategies.

A `LayerStrategy` ranks layers (best first); `UniStrategy` (in
`strategy.py`) handles the top-K slicing. Each strategy owns its own
name parsing via the `match` classmethod, so parametric families
(e.g. ``joint-k-beam-W4``) work without central dispatch logic.
"""
from abc import ABC, abstractmethod
import json
from typing import List, Optional, Tuple, Type

_REGISTRY: List[Type["LayerStrategy"]] = []


def register(cls: Type["LayerStrategy"]) -> Type["LayerStrategy"]:
    _REGISTRY.append(cls)
    return cls


def get_strategy(name: str) -> "LayerStrategy":
    for cls in _REGISTRY:
        kwargs = cls.match(name)
        if kwargs is not None:
            return cls(**kwargs)
    raise ValueError(
        f"No strategy matches name {name!r}. "
        f"Registered: {[c.__name__ for c in _REGISTRY]}"
    )


def list_strategies() -> List[str]:
    return [c.__name__ for c in _REGISTRY]


class LayerStrategy(ABC):
    @classmethod
    @abstractmethod
    def match(cls, name: str) -> Optional[dict]:
        """Return constructor kwargs if this class handles `name`, else None."""

    @abstractmethod
    def select(
        self,
        task: str,
        method: str,
        layers: List[int],
        num_layers: int,
    ) -> List[Tuple[int, float]]:
        """Return the chosen K layers as [(layer, score), ...], best-first.

        Rank-based strategies sort all layers and return the top-K slice;
        K-sensitive strategies (e.g. Joint-K) run a K-specific search so
        the K=3 result need not be a prefix of the K=5 result.
        """


def load_layer_scores(
    task: str,
    method: str,
    layers: List[int],
    score_root: str = "./Score-standard",
) -> dict:
    """Load per-layer score JSONs written by ``get_score.py``."""
    out = {}
    for l in layers:
        with open(f"{score_root}/{task}/{task}+{method}/L{l}.json", "r") as f:
            out[l] = json.load(f)
    return out
