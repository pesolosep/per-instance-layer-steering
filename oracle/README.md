# `oracle/` — Behavioral oracle harness

Population-level oracles that use test-set behavioral evaluation to find
optimal layer subsets. **Not deployable** at inference time — they serve as
upper bounds, as labeling sources for predictors, and as baselines for
deployable selectors (in `strategies/`).

## Method × Metric matrix

| Method        | Granularity                | LayerNavigator equiv     | This module |
|---------------|----------------------------|--------------------------|-------------|
| `gt_global`   | one set, all instances     | GT-global (`pipeline.py:235,366`) | ✅ |
| `tkm`         | per-instance set           | Top-K Marginal           | ✅ |
| `joint_k`     | per-instance set, joint    | Joint K=3 Oracle (`joint_oracle.py:478`) | ✅ |

| Scoring backend  | Recipe                              | Status |
|------------------|--------------------------------------|--------|
| `alignonly`      | `score = steered_prob - base_prob` | ✅ LayerNavigator-faithful |
| `multimetric`    | gate-then-rank with paired Δ `logit_margin`, gates from `eval/` | ✅ your contribution |

So **3 methods × 2 metrics = 6 oracle outputs** per (task, model, K). Both
backends share the same orchestrator runs — once steering output is
generated, both metrics are computed from the same data. The `multimetric`
backend is column-presence-aware: gates only fire when their columns are
present, so the same `score()` works on search-phase data (logit gates only)
and on full evaluation data (all gates).

## K-search policy (`config.py`)

| K | Strategy   | Cost per (task, model) at ~1s/subset |
|---|------------|---------------------------------------|
| 1 | trivial    | seconds   |
| 2 | exhaustive | minutes   |
| 3 | exhaustive | ~1.4 hours (matches LayerNavigator paper) |
| 4 | beam-W4    | minutes   |
| 5 | beam-W4    | minutes   |

Beam-W4 is validated against exhaustive at K=3 (Jaccard agreement) before
its K=4, 5 results are reported.

## Layout

```
oracle/
  __init__.py
  config.py          # K_VALUES, SEARCH_BY_K, BEAM_WIDTH
  runner.py          # Kaggle stub: run_single_layer_evals, run_subset_evals
  scoring/
    alignonly.py     # score = steered_prob - base_prob
    multimetric.py   # gate-then-rank with paired Δ logit_margin
  methods/
    gt_global.py     # mean per-layer score → top-K (one global set)
    tkm.py           # per-instance top-K of single-layer scores
    joint_k.py       # per-instance argmax over K-subsets
  compare.py         # Jaccard, gap, beam-vs-exhaustive validation
```

## Pipeline

```python
from oracle.runner    import run_single_layer_evals, run_subset_evals, exhaustive_subsets
from oracle.scoring   import alignonly
from oracle.methods   import gt_global, tkm, joint_k
from oracle.compare   import mean_jaccard, mean_gap

# 1. Orchestrator (Kaggle): produce raw eval DataFrames
df_k1 = run_single_layer_evals(model=..., tokenizer=..., dataset=val, layers=range(32))
df_k3 = run_subset_evals(model=..., tokenizer=..., dataset=val,
                         layer_subsets=exhaustive_subsets(range(32), K=3))

# 2. Scoring: alignonly first, multimetric in PR 2
scored_k1 = alignonly.score(df_k1)
scored_k3 = alignonly.score(df_k3)

# 3. Methods: pick layers from scored DataFrames
gt_picks    = gt_global.select(scored_k1, K=3)             # one set: [L21, L13, L28]
tkm_picks   = tkm.select(scored_k1, K=3)                   # {iid: [...]} per instance
joint_picks = joint_k.select(scored_k3, K=3)               # {iid: [...]} per instance

# 4. Compare
gt_vs_joint_J = mean_jaccard(
    {iid: gt_picks for iid in joint_picks}, joint_picks,
)
print(f"GT-global vs Joint-K agreement: J = {gt_vs_joint_J:.3f}")
```

## What's NOT in this module

- **Steering execution itself.** `runner.py` is a stub; the Kaggle
  orchestrator notebook implements `run_single_layer_evals` and
  `run_subset_evals` using the model wrapper + dataset bootstrap.
- **Multimetric scoring (PR 2).** Will use `eval/logit_metrics.py`,
  `eval/coherency_metrics.py`, plus a calibrated baselines JSON for
  validity gates.
- **LLM judge.** Explicitly deferred per session decisions.
- **α schedule comparison.** The `alpha_schedule` arg in
  `run_subset_evals` is the hook; running `α=1` and `α=1/K` and comparing
  picks happens in the Kaggle notebook, not here.

## Why outside `strategies/`

`strategies/` contains *deployable* layer selectors — methods that pick
layers using only train-set / inference-time information (e.g., the
geometric `ln-global` heuristic). Behavioral oracles need test-set
evaluation, so they don't satisfy the deployability contract. Keeping
them in `oracle/` makes the distinction obvious in code structure:

- `strategies/select(...) → layers` is what runs at inference time.
- `oracle/methods/.select(scored_df, K) → layers` is what the harness
  computes once per experimental run; the result becomes labels or an
  upper-bound benchmark.
