"""Oracle selection methods.

Each method consumes a *scored* DataFrame (output of one of the
``scoring/`` backends) and returns picks. Methods are metric-agnostic
— they only read the ``score`` column.

| Method | Granularity | Search | DataFrame shape expected |
|---|---|---|---|
| ``gt_global`` | one set, all instances | sort + slice | per (instance, layer) — single-layer evals only |
| ``tkm``       | per-instance set       | sort + slice | per (instance, layer) — single-layer evals only |
| ``joint_k``   | per-instance set       | argmax over subsets | per (instance, layer_set) — multi-layer evals |
"""
from . import gt_global  # noqa: F401
from . import tkm  # noqa: F401
from . import joint_k  # noqa: F401

__all__ = ["gt_global", "tkm", "joint_k"]
