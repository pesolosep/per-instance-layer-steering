"""Rank layers by ``s_score = d_score + c_score`` (the original "my")."""
from typing import List, Optional, Tuple

from .base import LayerStrategy, load_layer_scores, register


@register
class ScoreSumStrategy(LayerStrategy):
    NAME = "my"

    @classmethod
    def match(cls, name: str) -> Optional[dict]:
        return {} if name == cls.NAME else None

    def select(
        self,
        task: str,
        method: str,
        layers: List[int],
        num_layers: int,
    ) -> List[Tuple[int, float]]:
        scores = load_layer_scores(task, method, layers)
        ranked = {l: info["s_score"] for l, info in scores.items()}
        return sorted(ranked.items(), key=lambda x: x[1], reverse=True)[:num_layers]
