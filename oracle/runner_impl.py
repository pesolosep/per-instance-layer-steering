"""GPU-using implementations of the oracle runner interface.

Two-phase computation:

- ``search_phase()`` — cheap, logit-only. Compute alignonly + multimetric
  raw logit columns for every candidate subset. Used by methods to PICK
  subsets. Cost per call: one forward pass per (instance batch, subset).

- ``evaluate_chosen()`` — expensive, generates text. Adds coherency
  columns (yn_mass_steered, rep_4gram_steered, delta_ppl, length_ratio)
  for the WINNING subset only. Cost: one generation per instance per
  selected subset.

This split is what makes K=3 exhaustive (~4,960 subsets × 300 instances
× alpha-1 only = ~1.5M forward passes) tractable: only the ~4 chosen
subsets per (method, K, alpha) get the expensive text-generation pass.

Schema of the returned DataFrame matches ``oracle/runner.py`` docstring.

Usage::

    from oracle.runner_impl import (
        compute_baseline, search_phase, evaluate_chosen,
        run_subset_evals_via_phases,
    )

Requires:
  - ``model``: LlamaWrapper or QwenWrapper from ``model_wrapper.py``
  - ``dataset``: UniDataset(set='val', train=False)
  - Steering vectors at ``./Vectors/{task}/{method}/L{l}.pt``
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch

from eval.coherency_metrics import delta_ppl, length_ratio, repetition_rate
from eval.logit_metrics import alignment_prob, logit_margin, yn_mass


# ─── Helper: load + scale steering vectors ─────────────────────────────

def _load_vectors(
    layer_subset: Sequence[int],
    alpha_per_layer: Sequence[float],
    task: str,
    method: str,
    device,
    vector_root: str = "./Vectors",
) -> Dict[int, torch.Tensor]:
    """Load α-scaled steering vectors for a subset of layers."""
    assert len(layer_subset) == len(alpha_per_layer)
    vects = {}
    for l, alpha in zip(layer_subset, alpha_per_layer):
        path = Path(f"{vector_root}/{task}/{method}/L{l}.pt")
        v = torch.load(path, map_location=device)
        vects[l] = v * alpha
    return vects


# ─── Helper: per-instance forward pass collecting full-vocab logits ────

def _forward_collect_logits(model, dataset) -> torch.Tensor:
    """Run forward pass over the dataset, return [N_total, V] last-pos logits.

    Concatenates across the dataset's internal batching. Uses
    ``model.get_last_logits(input)`` which already squeezes the
    sequence dim.
    """
    chunks = []
    for d in dataset:
        d_dev = d.to(model.device)
        last_logits = model.get_last_logits(d_dev).detach().cpu()
        # get_last_logits returns [B, V] (or [V] for B=1 after squeeze).
        if last_logits.dim() == 1:
            last_logits = last_logits.unsqueeze(0)
        chunks.append(last_logits)
    return torch.cat(chunks, dim=0)  # [N_total, V]


def _generate_collect_texts(model, dataset, max_new_tokens: int = 200) -> List[str]:
    """Run generation over the dataset, return list of decoded strings."""
    texts: List[str] = []
    for d in dataset:
        d_dev = d.to(model.device)
        out = model.generate(d_dev, max_new_tokens=max_new_tokens)
        texts.extend(out)
    return texts


# ─── Baseline (cached per task) ────────────────────────────────────────

def compute_baseline(
    model,
    dataset,
    t_yes: int,
    t_no: int,
    *,
    generate_text: bool = True,
    max_new_tokens: int = 200,
) -> Dict:
    """Compute unsteered per-instance signals once per (task, model).

    Returns a dict with:
      - ``logits``: [N_total, V] full-vocab last-pos logits
      - ``texts``:  List[str] of unsteered generations (if generate_text)
      - ``align_prob``: [N_total] tensor
      - ``logit_margin``: [N_total] tensor
      - ``yn_mass``: [N_total] tensor
      - ``gold_signs``: [N_total] tensor of {+1, -1}

    Call once per (task, model) and reuse across all subsets.
    """
    model.reset_all()
    base_logits = _forward_collect_logits(model, dataset)  # [N, V]
    gold_signs = torch.tensor(dataset.keys, dtype=torch.long)  # +1 or -1
    z_yes = base_logits[:, t_yes]
    z_no = base_logits[:, t_no]

    out = {
        "logits": base_logits,
        "z_yes": z_yes,
        "z_no": z_no,
        "align_prob": alignment_prob(z_yes, z_no, gold_signs),
        "logit_margin": logit_margin(z_yes, z_no, gold_signs),
        "yn_mass": yn_mass(base_logits, t_yes, t_no),
        "gold_signs": gold_signs,
    }
    if generate_text:
        out["texts"] = _generate_collect_texts(model, dataset, max_new_tokens=max_new_tokens)
    return out


# ─── Search phase: cheap, per-subset logit-only ────────────────────────

def search_phase(
    model,
    dataset,
    *,
    layer_subsets: Iterable[Sequence[int]],
    alpha_schedule: str,        # "uniform" / "sqrt-norm" / "k-norm" / "pow-<β>"
    task: str,
    method: str,
    t_yes: int,
    t_no: int,
    baseline: Dict,
    search_method: str,         # "exhaustive" / "beam-W4" / "trivial" — for the column
    K: int,
    vector_root: str = "./Vectors",
) -> pd.DataFrame:
    """Cheap pass: compute alignonly + raw logit columns for each subset.

    Generates NO text. Multimetric coherency columns will be filled by
    ``evaluate_chosen`` for the winning subset only.

    Returns DataFrame with columns:
      instance_id, layer_set, K, search_method, alpha_schedule,
      base_prob, steered_prob,
      base_logit_yes, base_logit_no,
      steered_logit_yes, steered_logit_no,
      gold_sign,
      base_yn_mass, steered_yn_mass.

    The ``layer`` column (single-layer convenience for GT-global / TKM)
    is added when len(layer_set) == 1.
    """
    rows = []
    # Full-vocab `baseline["logits"]` is intentionally NOT persisted to parquet
    # (would be N×V floats ≈ 25 MB per task). We only use the Yes/No slices
    # which are stored separately as z_yes / z_no.
    gold_signs = baseline["gold_signs"]
    base_align = baseline["align_prob"]
    base_yn = baseline["yn_mass"]
    base_z_yes = baseline["z_yes"]
    base_z_no = baseline["z_no"]

    # Materialize so we can len() / enumerate for the per-subset timing log
    # below. Callers already pass lists in practice (see oracle_runner main),
    # so this is a no-op cast in the common case.
    subsets_list = list(layer_subsets)
    n_subsets = len(subsets_list)
    phase_t0 = time.perf_counter()

    for subset_idx, subset in enumerate(subsets_list):
        t_subset_start = time.perf_counter()

        subset = tuple(sorted(int(l) for l in subset))
        K_subset = len(subset)
        alphas = _alpha_schedule(K_subset, alpha_schedule)

        t0 = time.perf_counter()
        vects = _load_vectors(subset, alphas, task, method, model.device, vector_root=vector_root)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.reset_all()
        for l, v in vects.items():
            model.set_add_activations(l, v)
        t_hook = time.perf_counter() - t0

        t0 = time.perf_counter()
        steered_logits = _forward_collect_logits(model, dataset)  # [N, V]
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_z_yes = steered_logits[:, t_yes]
        s_z_no = steered_logits[:, t_no]
        s_align = alignment_prob(s_z_yes, s_z_no, gold_signs)
        s_yn = yn_mass(steered_logits, t_yes, t_no)

        for i in range(len(gold_signs)):
            row = {
                "instance_id": i,
                "layer_set": subset,
                "K": K_subset,
                "search_method": search_method,
                "alpha_schedule": alpha_schedule,
                "base_prob": float(base_align[i]),
                "steered_prob": float(s_align[i]),
                "base_logit_yes": float(base_z_yes[i]),
                "base_logit_no": float(base_z_no[i]),
                "steered_logit_yes": float(s_z_yes[i]),
                "steered_logit_no": float(s_z_no[i]),
                "gold_sign": int(gold_signs[i]),
                "base_yn_mass": float(base_yn[i]),
                "steered_yn_mass": float(s_yn[i]),
            }
            if K_subset == 1:
                row["layer"] = subset[0]
            rows.append(row)
        t_row = time.perf_counter() - t0

        t_total = time.perf_counter() - t_subset_start
        # Diagnostic: print first 20 subsets, then every 100, plus the last.
        # Lets you tell apart per-subset overhead growth (monotonic rise =
        # buffer/cache fragmentation) from constant overhead (flat rate but
        # higher than expected for K) at a glance.
        if subset_idx < 20 or (subset_idx + 1) % 100 == 0 or subset_idx == n_subsets - 1:
            elapsed = time.perf_counter() - phase_t0
            avg = elapsed / (subset_idx + 1)
            eta = avg * (n_subsets - subset_idx - 1)
            print(
                f"    [t] {subset_idx + 1}/{n_subsets} {subset}: "
                f"total={t_total:.2f}s "
                f"(vec={t_vec * 1000:.0f}ms hook={t_hook * 1000:.0f}ms "
                f"fwd={t_fwd:.2f}s row={t_row * 1000:.0f}ms) "
                f"| avg={avg:.2f}s eta={eta / 60:.1f}min rows={len(rows)}"
            )

    model.reset_all()
    return pd.DataFrame(rows)


# ─── Evaluation phase: text + coherency for chosen subsets only ────────

def evaluate_chosen(
    model,
    dataset,
    tokenizer,
    *,
    chosen_layer_sets: Iterable[Sequence[int]],
    alpha_schedule: str,
    task: str,
    method: str,
    baseline: Dict,
    t_yes: int,
    t_no: int,
    max_new_tokens: int = 200,
    scoring_model=None,
) -> pd.DataFrame:
    """Expensive pass: forward pass for logits + generate text + compute coherency.

    Call this ONLY for subsets selected by methods (winning K-subsets per
    method × K × alpha), not for all candidate subsets. Cost is
    proportional to len(chosen_layer_sets) × N_instances × max_new_tokens.

    Produces a complete per-(instance, picked_subset) row containing BOTH
    logit metrics (steered_logit_yes/no, steered_prob, steered_yn_mass) AND
    text/coherency metrics (rep_4gram, delta_ppl, length_ratio). This makes
    ``chosen_full_eval.parquet`` self-contained for per-pick analysis — no
    need to cross-join with search-phase parquets to recover after-steer
    logit values for the picked subset.

    Args:
        model:             steering wrapper (LlamaWrapper / QwenWrapper)
        dataset:           UniDataset (val)
        tokenizer:         for delta_ppl perplexity scoring
        chosen_layer_sets: list of selected subsets to evaluate fully
        alpha_schedule:    "uniform" / "sqrt-norm" / "k-norm" / "pow-<β>"
        baseline:          output of compute_baseline(generate_text=True)
        t_yes:             token id of " Yes" (extracts steered logit column)
        t_no:              token id of " No"
        scoring_model:     model used for delta_ppl scoring; defaults to
                           ``model.model`` (the unsteered base — note the
                           hooks are reset before delta_ppl call so the
                           forward pass actually IS unsteered there).

    Returns:
        DataFrame with columns:
          instance_id, layer_set, K, alpha_schedule,
          base_logit_yes, base_logit_no, base_prob, base_yn_mass, gold_sign,
          steered_logit_yes, steered_logit_no, steered_prob, steered_yn_mass,
          steered_text, base_text,
          rep_4gram_steered, rep_4gram_base, length_ratio, delta_ppl.
    """
    if scoring_model is None:
        scoring_model = model.model

    base_texts: List[str] = baseline.get("texts", [])
    if not base_texts:
        raise ValueError(
            "baseline['texts'] missing — call compute_baseline(generate_text=True)."
        )

    base_z_yes = baseline["z_yes"]
    base_z_no  = baseline["z_no"]
    base_align = baseline["align_prob"]
    base_yn    = baseline["yn_mass"]
    gold_signs = baseline["gold_signs"]

    rows = []
    for subset in chosen_layer_sets:
        subset = tuple(sorted(int(l) for l in subset))
        K_subset = len(subset)
        alphas = _alpha_schedule(K_subset, alpha_schedule)

        vects = _load_vectors(subset, alphas, task, method, model.device)
        model.reset_all()
        for l, v in vects.items():
            model.set_add_activations(l, v)

        # One forward pass to capture steered logits at the answer position.
        # Cheap relative to generation; gives us steered_prob / yn_mass /
        # logit_margin metrics for the picked subset without re-doing the
        # join with search-phase parquets (and covers gt_global K=4/5
        # picks that beam may not have visited).
        steered_logits = _forward_collect_logits(model, dataset)  # [N, V]
        s_z_yes = steered_logits[:, t_yes]
        s_z_no  = steered_logits[:, t_no]
        s_align = alignment_prob(s_z_yes, s_z_no, gold_signs)
        s_yn    = yn_mass(steered_logits, t_yes, t_no)

        # Then the expensive generation (drives rep_4gram / length_ratio / delta_ppl).
        steered_texts = _generate_collect_texts(model, dataset, max_new_tokens=max_new_tokens)

        # Reset hooks BEFORE delta_ppl. The scoring_model is model.model (same
        # underlying HF model used for generation), and the BlockOutputWrapper
        # hooks fire on every forward pass while add_activations is set.
        # Without this reset, delta_ppl would measure PPL(steered_text |
        # STEERED_model) instead of PPL(steered_text | UNSTEERED_model),
        # making the "drift from natural distribution" metric meaningless.
        model.reset_all()

        for i, (st, bt) in enumerate(zip(steered_texts, base_texts)):
            rows.append({
                "instance_id":      i,
                "layer_set":        subset,
                "K":                K_subset,
                "alpha_schedule":   alpha_schedule,
                # Baseline columns (alpha-agnostic; repeated per row for self-containment)
                "base_logit_yes":   float(base_z_yes[i]),
                "base_logit_no":    float(base_z_no[i]),
                "base_prob":        float(base_align[i]),
                "base_yn_mass":     float(base_yn[i]),
                "gold_sign":        int(gold_signs[i]),
                # Steered logit columns (the missing piece this PR closes)
                "steered_logit_yes": float(s_z_yes[i]),
                "steered_logit_no":  float(s_z_no[i]),
                "steered_prob":      float(s_align[i]),
                "steered_yn_mass":   float(s_yn[i]),
                # Steered text columns (existing)
                "steered_text":      st,
                "base_text":         bt,
                "rep_4gram_steered": repetition_rate(st),
                "rep_4gram_base":    repetition_rate(bt),
                "length_ratio":      length_ratio(st, bt),
                "delta_ppl":         delta_ppl(scoring_model, tokenizer, st, bt),
            })

    model.reset_all()
    return pd.DataFrame(rows)


# ─── Alpha schedules ───────────────────────────────────────────────────

def _alpha_schedule(K: int, schedule: str) -> List[float]:
    """Return per-layer α list of length K.

    Supported schedules (parametric family ``α(K) = 1 / K^β``):

    - ``uniform``    : β=0   → α=1                 (LN-faithful, total magnitude grows linearly)
    - ``sqrt-norm``  : β=0.5 → α=1/√K              (canonical magnitude-controlled comparison)
    - ``k-norm``     : β=1   → α=1/K               (strict magnitude conservation, available but
                                                    not recommended — too aggressive at high K)
    - ``pow-<float>``: β=<float> ∈ [0, 1]          (parametric for sensitivity sweeps)

    `sqrt-norm` is recommended as the principled "softer" alternative to
    `k-norm`: it matches the random-walk expectation under decorrelated
    steering vectors (E[‖Σ v_l‖] grows as √K, not K), so total expected
    magnitude stays approximately constant. At K=10 each layer still gets
    α ≈ 0.32 — meaningful per-layer contribution rather than the α=0.1
    whisper from k-norm.
    """
    if schedule == "uniform":
        return [1.0] * K
    if schedule == "sqrt-norm":
        return [1.0 / (K ** 0.5)] * K
    if schedule == "k-norm":
        return [1.0 / K] * K
    if schedule.startswith("pow-"):
        beta = float(schedule[len("pow-"):])
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"pow-<β>: β must be in [0, 1], got {beta}")
        return [1.0 / (K ** beta)] * K
    raise ValueError(f"Unknown alpha_schedule: {schedule}")
