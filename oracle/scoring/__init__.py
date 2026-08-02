"""Scoring backends — transform raw eval DataFrame to scored DataFrame.

A scoring function takes a DataFrame with raw evaluation columns
(``base_prob``, ``steered_prob``, logits, generated text, etc.) and
returns the same DataFrame with an added ``score`` column. Methods in
``oracle/methods/`` consume the ``score`` column and don't care which
metric produced it — that's how we run the same method against
different scoring backends without changing method code.

Backends:

- ``alignonly``  : ``score = steered_prob - base_prob``  (LayerNavigator-faithful)
- ``multimetric``: gate-then-rank with paired Δ logit_margin (your contribution).
                   Validity gates from ``eval/`` (yn_mass, rep_4gram, delta_ppl).
                   Rank metric pluggable via the registry (``register_rank_metric``);
                   gate behavior pluggable via ``apply_gates`` for ablation.
"""
from . import alignonly  # noqa: F401
from . import multimetric  # noqa: F401

__all__ = ["alignonly", "multimetric"]
