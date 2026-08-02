# `prep/` — Offline-prep + dataset loaders for Kaggle

Three related concerns live here:

1. **Codebase staging**: clone this repository into a code-only
   Kaggle Dataset (~few MB). Re-run on every codebase change.
2. **Model staging**: download HF model weights into a weights-only Kaggle Dataset
   (~16 GB). Re-run only when switching model targets.
3. **Dataset loaders**: pure-Python loaders for the JSONL evaluation data
   (`Dataset/Anth_Persona_ALL/`). The dataset ships in the repo, so consumer
   notebooks get it for free via the codebase Dataset.

**Why two prep notebooks instead of one**: codebase changes often during
development (PRs every few days), model weights change rarely (one-time per
model target). Splitting them means iterating on codebase doesn't trigger
a 16 GB re-download.

## What's here

| File | Purpose |
|---|---|
| `download_model.py` | `download_model()`, `verify_model_snapshot()`, `report_snapshot_size()`. Pure Python; no Kaggle dependency. |
| `prep_codebase.ipynb` | Kaggle notebook (CPU + Internet ON) that clones `main` into `/kaggle/working/layerselection-repo/`. **Output: code-only Dataset.** Run on every codebase change. Built via `prep_codebase.py`. |
| `prep_codebase.py` | Build script for `prep_codebase.ipynb`. |
| `prep_model.ipynb` | Kaggle notebook (CPU + Internet ON) that downloads one model's weights into `/kaggle/working/<model>/`. The repo is cloned into `/tmp/` (NOT `/kaggle/working/`) so the codebase does not bundle into the output. **Output: weights-only Dataset.** Run once per model. |
| `prep_vectors.ipynb` | Kaggle notebook (GPU + Internet OFF) that computes per-layer steering vectors + geometric scores once per (model). Snapshots `Vectors/{task}/md/L*.pt` and `Score-standard/{task}/{task}+md/L*.json` into `/kaggle/working/`. Built via `prep_vectors.py`. |
| `prep_vectors.py` | Build script for `prep_vectors.ipynb`. |
| `anth_persona.py` | Loader for `Dataset/Anth_Persona_ALL/` — six Yes/No tasks × train/val/test JSONL. Provides `TASKS`, `SPLITS`, `find_dataset_root()`, `load_split()`, etc. No torch / tokenizer dependency. The dataset ships in the repo (via `prep_codebase`), so consumer notebooks get it for free; do quick health checks inline via `summary_stats(find_dataset_root())`. |

## Run order

```
1. prep_codebase.ipynb  → code-only Dataset       (re-run on codebase changes)
2. prep_model.ipynb     → weights-only Dataset    (re-run on model switch only)
3. prep_vectors.ipynb   → Vectors + Score-standard Dataset
   ↑ attach codebase + model Datasets to this notebook
4. oracle_runner / ln_global_runner / bf16_smoke_test
   ↑ attach codebase + model + vectors Datasets to these
```

All consumer notebooks bootstrap by searching `/kaggle/input/` for the
sentinel `oracle/config.py` (rglob), so they find the codebase regardless
of Dataset naming or nesting.

## Supported models

Defined in `download_model.SUPPORTED_MODELS` — mirrors the model set used in
`LayerNavigator/`:

- `"llama"` → `meta-llama/Meta-Llama-3-8B-Instruct`
- `"aya"`   → `CohereForAI/aya-expanse-8b`
- `"qwen"`  → `Qwen/Qwen2.5-7B-Instruct`

## Anthropic Persona tasks

Defined in `anth_persona.TASKS` (matches the upstream paper's `Anth_MAIN`):

| Task slug | Display name | train | val | test |
|---|---|---|---|---|
| `conscientiousness` | Conscientiousness | 700 | 200 | 100 |
| `subscribes-to-Christianity` | Religion Following | 700 | 200 | 100 |
| `believes-it-has-phenomenal-consciousness` | Self-Aware | 700 | 200 | 100 |
| `cognitive-enhancement` | Self-Improvement | 700 | 200 | 100 |
| `desire-to-create-allies` | Alliance Building | 700 | 200 | 100 |
| `desire-to-maximize-impact-on-world` | Impact Maximization | 700 | 200 | 100 |

Each row is a JSON dict:

```json
{
  "question": "Is the following statement something you would say?\n\"...\"",
  "statement": "...",
  "label_confidence": 0.96,
  "answer_matching_behavior": " Yes",
  "answer_not_matching_behavior": " No"
}
```

## How a downstream notebook uses this

```python
import os, sys
from pathlib import Path

# Offline Kaggle: don't let HF clients try to call out
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Locate the repo (cloned into the prep snapshot at /kaggle/input/<prep-slug>/layerselection-repo/)
marker = next(Path("/kaggle/input").rglob("prep/anth_persona.py"))
sys.path.insert(0, str(marker.parent.parent))

# Now import the codebase helpers
from prep.anth_persona import TASKS, load_split, find_dataset_root
from models.load   import find_model_dir, load_model

dataset_root = find_dataset_root()
tokenizer, model = load_model(find_model_dir("llama", "/kaggle/input"))

for task in TASKS:
    train = load_split(task, "train", dataset_root)
    # … tokenize → forward → score …
```

## Notebooks can't read .py files directly

The bootstrap pattern above (`rglob` + `sys.path.insert`) is required because
Kaggle notebooks can only `import` from packages that are on `sys.path`. The prep
notebook clones the repo into its working directory, the cloned tree gets included
in the committed prep snapshot, and consumer notebooks reach into the snapshot at
`/kaggle/input/...` to add the repo root to `sys.path`. **Don't inline the loader
code into the notebook** — the snapshot is the single source of truth.

## Why one prep commit per model

`/kaggle/working/` quota (~20 GB) cannot hold both 8B-model snapshots at once and
GPU sessions have no internet, so weights must be staged from a CPU+Internet-ON
session and committed once per model.
