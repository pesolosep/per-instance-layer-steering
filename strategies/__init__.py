"""Layer-selection strategies.

To add a new strategy:
1. Create ``strategies/<your_strategy>.py`` defining a class that
   subclasses ``LayerStrategy`` and is decorated with ``@register``.
2. Import the new module below so it loads on package import.
"""
from .base import (
    LayerStrategy,
    get_strategy,
    list_strategies,
    load_layer_scores,
    register,
)

# Import strategy modules to populate the registry.
from . import score_sum  # noqa: F401
from . import joint_k  # noqa: F401

__all__ = [
    "LayerStrategy",
    "get_strategy",
    "list_strategies",
    "load_layer_scores",
    "register",
]
