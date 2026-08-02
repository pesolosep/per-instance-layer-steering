# `eval/` — Evaluation metrics for steering experiments

Pure-function metric library, organized by concern and cost. Replaces the
all-in-one `get_results.py` (kept at the repo root for reference until the
full eval pipeline is migrated).

## Layout

| File | Purpose | Cost per instance |
|---|---|---|
| `logit_metrics.py`      | `alignment_prob`, `logit_margin`, `yn_mass` | 1 forward pass at answer position (free; same forward used by upstream) |
| `coherency_metrics.py`  | `delta_ppl`, `repetition_rate`, `length_ratio` | 1 generation + 1 scoring forward pass |
| `statistics.py`         | `paired_delta`, `mean_delta`, `cohens_d_paired`, `wilcoxon_pvalue`, `bootstrap_ci` | Cheap, runs on per-instance arrays |
| `diagnostic_metrics.py` | `logit_shift`, `kl_divergence_yn`, `yn_flip` | Free during the marginal sweep — **NOT** a headline metric, **NOT** a picker feature |

## What each metric measures

| Metric | Formula | Catches | Range |
|---|---|---|---|
| `alignment_prob`  | `P(gold) / (P(yes) + P(no))`               | Direction preference (the *headline* number) | [0, 1] |
| `logit_margin`    | `σ · (z_yes − z_no)`, signed by gold       | Sensitivity at saturated regimes; linear in α | (-∞, +∞) |
| `yn_mass`         | `P(yes) + P(no)` over full vocab           | Format collapse out of Yes/No mode | [0, 1] |
| `delta_ppl`       | `PPL_unsteered(R_steered) − PPL_unsteered(R_unsteered)` | Steered text drifted from natural distribution | (-∞, +∞) |
| `repetition_rate` | `1 − unique_4grams / total_4grams`         | Degenerate repetition (high-α / K=10 collapse) | [0, 1] |
| `length_ratio`    | `len(R_steered) / len(R_unsteered)`        | Runaway or premature stop | (0, ∞) |

`alignment_prob` and `logit_margin` carry the same per-instance information
on different scales (logit is the inverse-sigmoid of the prob); aggregated
mean differs because of saturation.

## `diagnostic_metrics.py` — what it's for, what it isn't for

These three are **direction-agnostic** byproducts of the marginal layer sweep:

| Metric | Formula | Catches |
|---|---|---|
| `logit_shift`      | `‖(z_yes, z_no)_steered − (z_yes, z_no)_base‖_2`        | Magnitude of binary-pair shift |
| `kl_divergence_yn` | `KL(P_base ‖ P_steered)` over renormalized {yes, no}    | Distribution-level distance over the binary subspace |
| `yn_flip`          | `argmax{yes, no}_steered ≠ argmax{yes, no}_base` (bool)  | Did the top-1 binary preference switch |

**They are NOT for**:

- **Headline / sidecar eval metrics**: direction-agnostic — `logit_shift = 0.5`
  could be "moved toward gold by 0.5" or "moved AWAY by 0.5". Use the directional
  versions (`logit_margin Δ`, `alignment_prob Δ`) for method comparison.
- **Picker classifier features**: chicken-and-egg — they require `z_steered` at
  the candidate layer, which is the per-layer computation the picker is supposed
  to *avoid*. Use pre-steering features (`proj_on_sv`, `base_prob`) instead.

**They ARE for**:

- **Pipeline sanity check**: `logit_shift ≈ 0` for ALL layers means steering
  isn't being applied (broken).
- **Free byproduct of oracle-label construction**: when you sweep all layers
  to compute `align_prob_lift`, `z_steered` is already in hand — store these
  alongside in the per-(instance, layer) DataFrame for later analysis.
- **Methodological / exploratory analysis**: e.g., "do high-`s_score` layers
  produce large `logit_shift`?" — research-question answers, not headline numbers.

## Usage pattern

The metrics are deliberately small pure functions; the orchestration is up
to the caller. A typical per-method evaluation loop:

```python
import pandas as pd
import torch
from eval.logit_metrics     import alignment_prob, logit_margin, yn_mass
from eval.coherency_metrics import delta_ppl, repetition_rate, length_ratio
from eval.statistics        import paired_delta, mean_delta, cohens_d_paired, wilcoxon_pvalue

# Run forward pass once per instance, get answer-position logits.
# logits: [B, V],  z_yes = logits[:, t_yes],  z_no = logits[:, t_no],
# gold_signs: [B] of {+1, -1}
df = pd.DataFrame({
    "instance_id":   instance_ids,
    "align_prob":    alignment_prob(z_yes, z_no, gold_signs).cpu().numpy(),
    "logit_margin":  logit_margin(z_yes,  z_no, gold_signs).cpu().numpy(),
    "yn_mass":       yn_mass(logits, t_yes, t_no).cpu().numpy(),
})

# Optional: add coherency columns from generated text (one extra fwd pass for ΔPPL).
df["rep_4gram"]    = [repetition_rate(r) for r in steered_responses]
df["length_ratio"] = [length_ratio(rs, ru) for rs, ru in zip(steered_responses, unsteered_responses)]
df["delta_ppl"]    = [delta_ppl(unsteered_model, tokenizer, rs, ru)
                      for rs, ru in zip(steered_responses, unsteered_responses)]

# Compare method M vs LN baseline (paired Δ).
deltas   = paired_delta(df_method["align_prob"], df_ln["align_prob"])
headline = mean_delta(deltas)
d        = cohens_d_paired(deltas)
p        = wilcoxon_pvalue(deltas)
```

## Why modular instead of one big function

- Each metric is independently testable and inspectable.
- Mixing logit-based (cheap, always-on) and generation-based (expensive,
  optional) metrics in one function couples their cost models.
- Statistics live separately so the same aggregators apply to any metric
  column (alignment_prob Δ, logit_margin Δ, etc.) without duplicated code.

## Why no LLM judge here yet

LLM-judge-based behavioral compliance is intentionally deferred — see the
broader eval discussion in the project memory. It would land as a separate
`eval/judge.py` operating on a subsample of generated responses, using a
strong external judge model (NOT the steered model). For now the bundle is
fully automatic and reproducible.

## Relationship to upstream `get_results.py`

The upstream file at the repo root computes only `mean(alignment_prob)` and
returns a scalar; it discards per-instance values and emits no other
metrics. This module gives you a per-instance DataFrame with the full
bundle from the same compute, plus paired statistics for method comparison.
When the full eval pipeline is wired up (a Kaggle notebook orchestrator),
`get_results.py` can be retired.
