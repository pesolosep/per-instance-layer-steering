"""Coherency / format diagnostics for steering evaluation.

These need actual generated text. ``delta_ppl`` additionally needs one extra
forward pass through a *scoring model* (typically the unsteered base model) to
score the steered output's perplexity in the natural distribution.

Roles in the eval bundle:

- ``delta_ppl``:        steered text drifted into low-probability region of
                        the unsteered model (catches subtle incoherence).
- ``repetition_rate``:  degenerate repetition (the high-α / K=10 failure mode).
- ``length_ratio``:     runaway generation or premature stop.
"""
from __future__ import annotations

import torch


def perplexity(model, tokenizer, text: str) -> float:
    """Per-token perplexity of ``text`` under ``model``.

    One forward pass with ``labels = input_ids`` so HF returns the
    next-token cross-entropy; we exponentiate to get perplexity.

    Edge cases:
    - Empty text → returns ``+inf`` (treat as maximally unlikely).
    """
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    if enc.input_ids.numel() == 0:
        return float("inf")
    with torch.inference_mode():
        out = model(**enc, labels=enc.input_ids)
    return float(torch.exp(out.loss))


def delta_ppl(
    scoring_model,
    tokenizer,
    R_steered: str,
    R_unsteered: str,
) -> float:
    """``PPL(R_steered) − PPL(R_unsteered)``  under the SAME ``scoring_model``.

    Use the unsteered base model as ``scoring_model`` to measure how far the
    steered output drifted from the natural distribution. Positive ΔPPL means
    the steered text looks "weirder" to the original model.
    """
    return perplexity(scoring_model, tokenizer, R_steered) - perplexity(
        scoring_model, tokenizer, R_unsteered
    )


def repetition_rate(text: str, n: int = 4) -> float:
    """``1 − (unique_n_grams / total_n_grams)``  ∈ [0, 1].

    Computed on whitespace-tokenized text. Conventional ``n=4`` (4-gram
    repetition) is the standard "degenerate generation" detector (Holtzman
    et al. 2019; the same cutoff used in the Hugging Face generation guide).
    Returns ``0.0`` if the text is shorter than ``n`` tokens.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def length_ratio(R_steered: str, R_unsteered: str) -> float:
    """``len(R_steered) / max(1, len(R_unsteered))``.

    Catches runaway generation (``>>1``) or premature stop (``<<1``). Operates
    on character length; switch to token length if you have the tokenizer at
    hand and care about tokenization parity.
    """
    return len(R_steered) / max(1, len(R_unsteered))
