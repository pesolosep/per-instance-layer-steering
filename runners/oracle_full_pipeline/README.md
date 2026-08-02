# `oracle_full_pipeline/` — single-(task, model) end-to-end pipeline

One notebook that produces **every parquet the downstream analysis consumes** for a single
`(task, model)` cell. Subsumes the work of four prior notebooks:

| Phase | Replaces | Output |
|---|---|---|
| A — Search | `oracle_runner` (K=1 trivial + K=2,3 exhaustive + K=2..5 beam-W4) + `ln_global_runner` (K=1..5 lnglobal) | `K{K}_{exhaustive,beam-W4,trivial,lnglobal}.parquet`, `baseline.parquet` |
| B — Picks | manual `method_picks.parquet` script | `method_picks.parquet` (5+ methods × 5 K × 2 α × 200 inst) |
| C — Steered full-eval + explain | `oracle_runner.evaluate_chosen` + `prep_explain_for_ppl` | `method_picks_eval.parquet` (matches K_exhaustive schema), `explain_texts_picks.parquet` |
| D — GPT-2 PPL | `prep_ppl_scoring` | `ppl_scores_picks.parquet` |

## Scope

| Axis | Values |
|---|---|
| K | 1, 2, 3, 4, 5 |
| Exhaustive | K ∈ {1, 2, 3} only (K≥4 infeasible) |
| Beam-W4 | K ∈ {2, 3, 4, 5} |
| LN-global | K ∈ {1, 2, 3, 4, 5} (1 global pick per K) |
| TKM, GT-Global | derived from K=1 trivial (no extra forwards), all K=1..5 |
| α | uniform + sqrt-norm |
| Scorings | alignonly + multimetric_3 |
| Methods iterated for picks | tkm × {align, multi3}, gt_global × {align, multi3}, ln_global, exhaustive × {align, multi3} (K≤3), beam × {align, multi3} (K≥2) |

## Configurability

Cell 2 is the only edit-per-session cell:

```python
TASK            = "conscientiousness"      # any task in Anth_MAIN
MODEL_PRESET    = LLAMA_8B                 # LLAMA_8B / AYA_8B / QWEN_7B / QWEN_32B
ALPHA_SCHEDULES = ["uniform"]              # ← edit per session
```

### Two-session pattern (recommended for budget)

A full `(task × model)` cell at K=1..5 + both α takes ~16-18 h, exceeding
the 12 h Kaggle limit. The notebook is designed for **one α per session**
with automatic carry-forward of prior session data:

**Session 1** (uniform):
```python
ALPHA_SCHEDULES = ["uniform"]
```
- Fresh run, no prior outputs to stage
- Produces: `baseline.parquet`, `uniform/K{1..5}_*.parquet`, `uniform/method_picks_eval.parquet`, `uniform/explain_texts_picks.parquet`, `uniform/ppl_scores_picks.parquet`, `ln_global_outputs/.../uniform/K{1..5}_lnglobal.parquet`
- Wall: ~8 h

**Session 2** (sqrt-norm):
1. Attach Session 1's output dataset to this notebook (Kaggle Datasets sidebar)
2. Edit Cell 2:
```python
ALPHA_SCHEDULES = ["sqrt-norm"]
```
3. Run All Cells
- Bootstrap detects prior `oracle_outputs*` / `ln_global*` dirs in `/kaggle/input`, **copies** them into working dir
- Idempotent skips prevent re-running ANY Session 1 work:
  - `baseline.parquet` (α-agnostic) — cached
  - `ln_global_outputs/.../picks_K{K}.json` (α-agnostic LN-global picks) — cached
  - `uniform/K{1..5}_*.parquet` (all uniform search results) — cached
  - `uniform/method_picks_eval.parquet`, `uniform/explain_texts_picks.parquet`, `uniform/ppl_scores_picks.parquet` — cached
- Generates only sqrt-norm data: `sqrt-norm/K{1..5}_*.parquet`, `sqrt-norm/method_picks_eval.parquet`, etc.
- **Phase B rewrites `method_picks.parquet`** to contain BOTH α schedules
- Wall: ~8 h
- Final output dataset has both α schedules merged

### Cross-(task, model) reuse

The pipeline runs once per `(TASK, MODEL_PRESET)` combination. For the full
matrix (multiple tasks / models), re-run with each pair. There is no
cross-task carry-forward (different tasks have different vectors, baselines,
picks) — each (task, model) is a fresh sequence of Session 1 + Session 2.

## Cost (Blackwell, conscientiousness × Llama-3-8B, both α, all K)

Approximate, per `(task, model)`:

| Phase | Wall |
|---|---|
| A — search (K=1 trivial + K=2 exh + **K=3 exh** + K=2..5 beam + LN-global K=1..5) | ~14-16 h (K=3 exh dominates: ~7 h × 2 α) |
| B — picks (CPU, derivation only) | < 1 min |
| C — steered full-eval + explain (Llama, both passes) | ~1-2 h |
| D — GPT-2 PPL | ~5 min |
| **Total** | **~16-18 h** |

**Will not fit in a single 12-hour Kaggle session.** The notebook is fully
idempotent / resume-safe — every phase checks if its output parquet exists
and skips. Run the same notebook across 2-3 Kaggle sessions and it picks up
where preemption left off.

## Resume protocol

If Kaggle preempts mid-K=3-exhaustive:

1. Download the working directory (parquets that were completed)
2. Re-stage the dataset on next session (or commit notebook to use the working dir)
3. Re-run the notebook — every phase skips completed outputs
4. K=3 exhaustive resumes from its last subset (via `oracle/io.is_complete` row count check)

Other phases (LN-global, beam, picks, eval, PPL) all idempotent at the
parquet level — already-written files are not regenerated.

## Memory protocol

Between Phase C (Llama steering) and Phase D (GPT-2 judge): explicit
`del model; gc.collect(); torch.cuda.empty_cache()`. Even on Blackwell's
102 GB VRAM, sequential loads prevent fragmentation from torch.compile
recompilation buffers accumulating.

## Skip-K controls (when budget tight)

Cell 2 also exposes `K_VALUES_RUN` to restrict the K-sweep. Examples:

```python
K_VALUES_RUN = [1, 2, 3]      # K≤3 only — finishes in ~7-8 h
K_VALUES_RUN = [3, 5]         # headline (K=3) + cliff onset (K=5) only
K_VALUES_RUN = [1, 2, 3, 4, 5]   # default (full sweep)
```

## Output structure

```
oracle_outputs/{task}/{model}/
├── baseline.parquet
├── {alpha_schedule}/
│   ├── K1_trivial.parquet
│   ├── K2_exhaustive.parquet
│   ├── K2_beam-W4.parquet
│   ├── K3_exhaustive.parquet
│   ├── K3_beam-W4.parquet
│   ├── K4_beam-W4.parquet
│   ├── K5_beam-W4.parquet
│   ├── method_picks.parquet
│   ├── method_picks_eval.parquet
│   ├── explain_texts_picks.parquet
│   └── ppl_scores_picks.parquet
└── ln_global_outputs/{task}/{model}/{alpha_schedule}/
    ├── K1_lnglobal.parquet
    ├── K2_lnglobal.parquet
    ├── K3_lnglobal.parquet
    ├── K4_lnglobal.parquet
    └── K5_lnglobal.parquet
```

Output schemas match the canonical formats consumed by
the downstream analysis layer — no schema adaptation needed.

## Source of truth

```bash
python runners/oracle_full_pipeline/build_notebook.py
```

regenerates `oracle_full_pipeline.ipynb` from the canonical Python source.
