# Method implementation 

**Scope: method + deployable recipe.** This repository runs the steering
pipeline end to end (raw per-instance parquets) and then trains and evaluates
the deployable recipe on them (`scripts/`). It deliberately does **not**
contain the table/figure-building code that turns results into the paper's
numbers — see [Scope of this release](#scope-of-this-release).

---

## Layout

| Path | What it is |
|---|---|
| `globalenv.py` | Model presets (`LLAMA_8B`, `AYA_8B`, …), task registry, seed |
| `dataset.py` | Dataset loading + tokenisation (train / val / test splits) |
| `model_wrapper.py` | Wraps the frozen model: activation capture + vector injection |
| `get_vec.py` | Steering-vector extraction (CAA mean-difference / PCA) |
| `get_score.py` | LayerNavigator steerability score per layer (discriminability, consistency) |
| `strategy.py` | Layer selection from those scores (the LN-global rule) |
| `get_results.py` | Logit readout and text generation |
| `main.py` | Minimal local driver, upstream-LayerNavigator-style; see note below |
| `utils.py` | Shared helpers |
| `prep/` | Offline prep: persona data, model download, vector building, GPT-2 PPL judge |
| `oracle/` | Per-instance oracle harness: `gt_global`, `tkm`, `joint_k` × `alignonly`/`multimetric` |
| `strategies/` | Deployable selection strategies (`base`, `joint_k`, `score_sum`) |
| `eval/` | Metrics: logit, coherency, diagnostic, statistics |
| `models/` | Model loading |
| `Dataset/` | `Anth_Persona_ALL` — the six Anthropic-Persona traits |
| `runners/oracle_full_pipeline/` | The end-to-end GPU driver (see below) |
| `scripts/` | The deployable recipe: ranker training, direction classifier, adaptive-K gate, compass baseline (see Scope) |

### Why the root `.py` files are not inside a `core/` package

`eval/` does `import globalenv`, and the root modules import each other as
top-level modules. Nesting them under `core/` would break every one of those
imports. The original flat layout is preserved so the copy runs unmodified.

Verified: all files byte-compile, and the folder is **import-closed** — every
non-third-party import resolves to a module inside this folder.

---

## Running it

The heavy compute was designed for Kaggle GPU sessions, not a laptop.

**`runners/oracle_full_pipeline/`** is the real entry point. One notebook produces
every parquet for a single `(task, model)` cell:

| Phase | Output |
|---|---|
| A — Search | `K{1,2,3}_exhaustive`, `K{2..5}_beam-W4`, `K{1..5}_lnglobal`, `baseline` |
| B — Picks | `method_picks.parquet` (methods × K × α × instances) |
| C — Steered eval + explain | `method_picks_eval.parquet`, `explain_texts_picks.parquet` |
| D — GPT-2 PPL | `ppl_scores_picks.parquet` |

Covers K ∈ {1..5}, exhaustive only where feasible (K ≤ 3), beam-W4 for K ∈ {2..5},
both α schedules (uniform, sqrt-norm). TKM and GT-global are derived from the K=1
trivial pass with no extra forward passes.

`build_notebook.py` is the **source of truth**; the `.ipynb` is generated from it:

```bash
python runners/oracle_full_pipeline/build_notebook.py
```

Never hand-edit the `.ipynb` — it is a derived artifact.

Before that, `prep/` stages the model weights, the codebase, and the CAA vectors as
Kaggle Datasets. Start at `prep/README.md`.

`main.py` is **not** the pipeline entry point. It is a minimal local reproduction of
upstream LayerNavigator's `main.py`, kept for smoke-testing on one model.

Set the model checkpoint path in `globalenv.py` before anything else.

### Dependencies

`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `transformers`,
`huggingface_hub`, `tqdm`, `nbformat`. See `requirements.txt`.

---

## Scope of this release

This repository contains the steering method, its baselines, and the deployable
recipe. `scripts/` holds the recipe layer, which consumes the parquets the
pipeline produces:

| Script | What it is |
|---|---|
| `train_final_predictors.py` (+ `final_predictor_models.py`, `train_mlp_predictors.py`) | trains the ranker variants, incl. the deployed PCA-25 listwise MLP |
| `run_final_predictor_full_eval.py` | deployment driver: predicted picks, embedding-logreg direction classifier, adaptive-K gate walk, steered evaluation |
| `eval_final_predictors_static_k.py`, `eval_final_predictors_adaptive.py` | fixed-K and gated evaluation (gate config in `GATE_CFG`) |
| `adaptive_k_across_predictors.py` | `gate_walk` reference implementation |
| `backfill_adaptive_gate.py` | gate reconstruction from K1-5 trajectories (validated ~99% exact-K match) |
| `compass_all_cells_final.py` | probe-based compass baseline (appendix comparison) |

**Run order:** the GPU pipeline (`runners/`) produces the per-instance parquets;
`scripts/` then trains and evaluates the recipe on them. The `scripts/` layer is
not runnable before the pipeline stages have produced their outputs.

Deliberately not part of this release:

| What | Why excluded |
|---|---|
| Paper table/figure builders | produces the paper's numbers, not the method |
| Result artifacts (~35 GB of parquets/CSVs) | data, not code |
