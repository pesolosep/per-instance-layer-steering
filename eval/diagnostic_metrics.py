"""Diagnostic metrics — direction-AGNOSTIC, computed for free during the
marginal layer sweep.

These quantities measure *how much* steering changed the model's binary
prediction, NOT *whether* it helped. They have known limitations:

- Direction-agnostic: ``logit_shift = 0.5`` could mean the model moved
  toward gold by 0.5 OR away by 0.5. They cannot answer "did steering
  help?" — that's what ``logit_margin`` and ``alignment_prob`` are for.

- Not valid as picker classifier features (chicken-and-egg): they require
  ``z_steered`` at the candidate layer, which is exactly the per-layer
  computation the picker is trying to avoid.

Their legitimate uses:

1. **Pipeline sanity check** during development: if ``logit_shift ≈ 0`` for
   ALL layers, something's broken (steering vector isn't being applied).

2. **Free byproduct** of the marginal layer sweep: when constructing oracle
   labels by trying every layer, ``z_steered`` is already computed. Compute
   these once and store them in the per-(instance, layer) DataFrame for any
   later analysis. No extra forward passes required.

3. **Methodological / exploratory analysis**: studying steering mechanisms
   as a research question — e.g., "do high-``s_score`` layers also produce
   large ``logit_shift``?" — without using the result as a headline number.

DO NOT promote any of these to a headline eval metric or to a picker feature.
The decision-relevant analogues already in the bundle are:

- For headline / sidecar:    ``logit_margin Δ``,  ``alignment_prob Δ``
- For picker features:        ``proj_on_sv``,  ``base_prob``
"""
from __future__ import annotations

import torch


def logit_shift(
    logits_steered: torch.Tensor,
    logits_base: torch.Tensor,
    t_yes: int,
    t_no: int,
) -> torch.Tensor:
    """L2 norm of the change in {yes, no} logits caused by steering.

    .. math::
        \\text{logit\\_shift} = \\|(z^{\\text{st}}_{\\text{yes}}, z^{\\text{st}}_{\\text{no}}) -
                                 (z^{\\text{b}}_{\\text{yes}}, z^{\\text{b}}_{\\text{no}})\\|_2

    Direction-agnostic; see module docstring for caveats.

    Args:
        logits_steered: ``[B, V]`` answer-position logits with steering applied.
        logits_base:    ``[B, V]`` answer-position logits without steering.
        t_yes:          token id of " Yes".
        t_no:           token id of " No".

    Returns:
        ``[B]`` L2 magnitude of the (Δyes, Δno) shift.
    """
    dy = logits_steered[..., t_yes] - logits_base[..., t_yes]
    dn = logits_steered[..., t_no] - logits_base[..., t_no]
    return torch.sqrt(dy * dy + dn * dn)


def kl_divergence_yn(
    logits_steered: torch.Tensor,
    logits_base: torch.Tensor,
    t_yes: int,
    t_no: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``KL(P_base ‖ P_steered)`` over the {yes, no} subspace, both renormalized.

    .. math::
        \\text{KL}_{\\text{yn}} = \\sum_{t \\in \\{\\text{yes}, \\text{no}\\}}
            P_b(t) \\log \\frac{P_b(t)}{P_s(t)}

    where ``P_b`` and ``P_s`` are the renormalized probabilities (binary
    softmax over the {yes, no} pair). Direction-agnostic; see module docstring.

    Returns:
        ``[B]`` KL divergence per instance, in nats.
    """
    z_b = torch.stack([logits_base[..., t_yes], logits_base[..., t_no]], dim=-1)
    z_s = torch.stack([logits_steered[..., t_yes], logits_steered[..., t_no]], dim=-1)
    p_b = torch.softmax(z_b, dim=-1)
    p_s = torch.softmax(z_s, dim=-1)
    log_ratio = torch.log(p_b.clamp(min=eps)) - torch.log(p_s.clamp(min=eps))
    return (p_b * log_ratio).sum(dim=-1)


def yn_flip(
    logits_steered: torch.Tensor,
    logits_base: torch.Tensor,
    t_yes: int,
    t_no: int,
) -> torch.Tensor:
    """Did the top-1 of {yes, no} switch under steering?

    Returns ``True`` for instances where ``argmax_{t ∈ {yes, no}}`` changed.
    Discrete; loses all magnitude info — `align_prob > 0.5` is exactly
    equivalent. Useful only as a "% flipped" sidecar count, never as a metric.

    Returns:
        ``[B]`` boolean tensor.
    """
    base_yes_won = logits_base[..., t_yes] > logits_base[..., t_no]
    steered_yes_won = logits_steered[..., t_yes] > logits_steered[..., t_no]
    return base_yes_won != steered_yes_won
