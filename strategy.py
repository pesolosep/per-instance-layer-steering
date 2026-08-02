"""Thin wrapper preserving the historical UniStrategy API.

Strategy implementations live in the ``strategies/`` package. Add new
selection methods by dropping a new module there; this file should not
need to change.

Callers pass ``layers`` explicitly (typically from a `ModelConfig`
preset's ``.layers`` field). No globalenv coupling.
"""
from typing import List

from strategies import get_strategy


class UniStrategy:
    def __init__(
        self,
        task: str,
        strategy: str,
        num_layers: int,
        method: str,
        layers: List[int],
    ):
        self.task = task
        self.strategy = strategy
        self.num_layers = num_layers
        self.method = method

        impl = get_strategy(strategy)
        chosen = impl.select(task, method, layers, num_layers)
        self.layers = [l for l, _ in chosen]
        self.scores = [s for _, s in chosen]
