"""Behavioral oracle harness for layer selection.

Population-level oracles (GT-global, TKM, Joint-K) that use test-set
behavioral evaluation to find optimal layer subsets. NOT deployable —
they serve as upper bounds and as labeling sources for predictors.

Strategies in ``strategies/`` (e.g. ``ln-global``) are deployable
selectors and live separately. This module is for measurement, not
inference-time selection.

Layout:
- ``config``        : K-sweep range and search algorithm per K
- ``runner``        : orchestrator that runs steering + emits raw
                      per-(instance, set) DataFrame (Kaggle stub)
- ``scoring/``      : transform raw DataFrame → scored DataFrame
                      (one scalar score per row)
- ``methods/``      : pick layers from scored DataFrame
                      (GT-global, TKM, Joint-K — all metric-agnostic)
- ``compare``       : Jaccard / gap analysis across methods or metrics
"""
from .config import K_VALUES, SEARCH_BY_K

__all__ = ["K_VALUES", "SEARCH_BY_K"]
