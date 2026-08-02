"""Build prep_vectors.ipynb for staging steering vectors + scores.

This is the Kaggle notebook that runs ONCE per (model) on a CPU+Internet-OFF
GPU session — it computes:

1. Mean-difference steering vectors per layer per task → ``Vectors/{task}/md/L*.pt``
2. Per-layer geometric scores (D, C, S) per task → ``Score-standard/{task}/{task}+md/L*.json``

Both outputs are saved into ``/kaggle/working/`` for snapshotting as a
Kaggle dataset, attachable as input to consumer notebooks (oracle_runner,
ln_global_runner) which assume ``Vectors/`` and ``Score-standard/`` exist
on disk.

This mirrors the pattern of ``prep_model.ipynb``: do the expensive
one-time computation in a prep session, snapshot the outputs, attach
them as read-only inputs to consumer sessions.

Usage::

    python prep/prep_vectors.py    # builds prep_vectors.ipynb
"""
from pathlib import Path

import nbformat as nbf


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = []

    nb.cells.append(md("""
# `prep_vectors` — Stage steering vectors + scores (Kaggle, GPU)

Computes per-layer mean-difference steering vectors and geometric scores
ONCE per (model). Outputs land in `/kaggle/working/Vectors/` and
`/kaggle/working/Score-standard/` for snapshotting.

**This is a prep notebook, not a consumer notebook.** It runs:
1. On a Kaggle GPU session
2. Internet OFF (model weights attached from `prep_model` snapshot)
3. Once per (model) — vectors are deterministic given model + train data

After running:
- **Commit the notebook** → produces a Kaggle output dataset
- Attach that dataset as input to `oracle_runner` and `ln_global_runner`
- Those consumers will see `Vectors/` and `Score-standard/` populated

## Pre-requisites

- Model weights attached at the path in `globalenv.MODEL` (from `prep_model` snapshot)
- this repository cloned (which includes `Dataset/Anth_Persona_ALL/`)

## What's computed

| Output | Location | Purpose |
|---|---|---|
| Steering vectors | `Vectors/{task}/md/L{l}.pt` | Used by all consumer notebooks for steering |
| Geometric scores | `Score-standard/{task}/{task}+md/L{l}.json` (D, C, S) | Used by `ln_global` heuristic + analysis |
"""))

    nb.cells.append(md("""
## 1. Bootstrap

Pre-req: attach the **prep-codebase** output dataset (`prep_codebase.ipynb`,
which clones the repository) AND the **prep-model** output dataset
(which contains the model snapshot).
"""))
    nb.cells.append(code('''
import os, sys, subprocess
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
LOCAL_DEV  = Path("/path/to/repo")
SENTINEL   = "oracle/config.py"

matches = list(INPUT_ROOT.rglob(SENTINEL)) if INPUT_ROOT.exists() else []
if matches:
    REPO_ROOT = matches[0].parent.parent
elif LOCAL_DEV.exists():
    REPO_ROOT = LOCAL_DEV
else:
    raise RuntimeError(
        f"Codebase not found. Attach the prep-codebase output dataset."
    )

sys.path.insert(0, str(REPO_ROOT))
os.chdir("/kaggle/working")  # write Vectors/ + Score-standard/ here for snapshotting

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Install flash-attn from the offline-packages Kaggle Dataset.
# Attach: kaggle.com/datasets/dennisfong/nvidia-nemotron-offline-packages
flash_whl = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
if os.path.exists(flash_whl):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", flash_whl])
        print("Installed flash_attn wheel (used by internal kernels)")
    except Exception as e:
        print(f"flash_attn install skipped: {e}")

print(f"Codebase root: {REPO_ROOT}")
print(f"cwd (output dir): {os.getcwd()}")
'''))

    nb.cells.append(md("""
## 2. Configuration

Pick `MODEL_PRESET` (a named `ModelConfig` from `globalenv`) and the
TASKS list. Switch models by changing `MODEL_PRESET` here + attaching
the corresponding prep-model dataset.
"""))
    nb.cells.append(code('''
from globalenv import LLAMA_8B, AYA_8B, QWEN_7B, QWEN_32B, Anth_MAIN, RANDOM_SEED

# ─── EDIT PER RUN ─────────────────────────────────────────────────
MODEL_PRESET = LLAMA_8B
TASKS  = list(Anth_MAIN)
METHOD = "md"            # mean-difference vectors (only one currently used downstream)
# ──────────────────────────────────────────────────────────────────

print(f"MODEL: {MODEL_PRESET.name}  ({MODEL_PRESET.num_layers} layers)")
print(f"Tasks: {len(TASKS)}")
print(f"Method: {METHOD}")
print(f"RANDOM_SEED: {RANDOM_SEED}  (pipeline is deterministic by design — vectors don't depend on this)")
'''))

    nb.cells.append(md("## 3. Imports + load model"))
    nb.cells.append(code('''
from model_wrapper import ModelWrapper
from dataset import UniDataset
from get_vec import uni_generate_vectors
from get_score import get_score

model = ModelWrapper(MODEL_PRESET)   # bf16 + FA2 + KV cache + device_map="auto"
LAYERS = MODEL_PRESET.layers

print(f"device: {model.device} | dtype: {next(model.model.parameters()).dtype}")
'''))

    nb.cells.append(md("""
## 5. Per-task: generate vectors + scores

For each task:
1. Load train dataset (used for vector computation)
2. `uni_generate_vectors` → saves `Vectors/{task}/md/L{l}.pt` per layer
3. `get_score` → saves `Score-standard/{task}/{task}+md/L{l}.json` (D, C, S per layer)

Both steps are idempotent — they overwrite existing outputs. Skip-on-exist
isn't built in because the cost is per-task ~10 min × 6 tasks = ~1 hour total
(within Kaggle session limits) and re-running guarantees fresh artifacts.
"""))
    nb.cells.append(code('''
for task in TASKS:
    print(f"\\n========== Task: {task} ==========")

    # 1. Load train dataset
    train_dataset = UniDataset(MODEL_PRESET, task=task, train=True, set="train", tokenizer=model.tokenizer)
    print(f"  train instances: {len(train_dataset.keys)}")

    # 2. Generate steering vectors
    print(f"  generating {METHOD} vectors...")
    uni_generate_vectors(
        method=METHOD,
        model=model,
        layers=LAYERS,
        dataset=train_dataset,
    )
    print(f"  vectors saved to ./Vectors/{task}/{METHOD}/")

    # 3. Compute geometric scores (D, C, S per layer)
    print(f"  computing scores...")
    get_score(
        layers=LAYERS,
        dataset=train_dataset,
        vec_task=task,
        vec_method=METHOD,
        acts_pre="standard",
    )
    print(f"  scores saved to ./Score-standard/{task}/{task}+{METHOD}/")

    del train_dataset
'''))

    nb.cells.append(md("""
## 6. Verify outputs

Quick sanity check that all expected files exist before committing.
"""))
    nb.cells.append(code('''
import json

print("Vectors:")
for task in TASKS:
    d = Path(f"./Vectors/{task}/{METHOD}")
    pt_files = sorted(d.glob("L*.pt"))
    print(f"  {task}: {len(pt_files)} .pt files (expected {len(LAYERS)})")
    assert len(pt_files) == len(LAYERS), f"Missing vectors for {task}"

print("\\nScore-standard:")
for task in TASKS:
    d = Path(f"./Score-standard/{task}/{task}+{METHOD}")
    json_files = sorted(d.glob("L*.json"))
    print(f"  {task}: {len(json_files)} .json files (expected {len(LAYERS)})")
    assert len(json_files) == len(LAYERS), f"Missing scores for {task}"

# Spot-check one score file
sample = json.load(open(f"./Score-standard/{TASKS[0]}/{TASKS[0]}+{METHOD}/L0.json"))
print(f"\\nSample score keys: {list(sample.keys())}")  # expect: s_score, d_score, c_score
'''))

    nb.cells.append(md("""
## 7. Done — commit this notebook

Commit the Kaggle notebook to snapshot `/kaggle/working/Vectors/` and
`/kaggle/working/Score-standard/` as an output dataset. Then attach that
dataset as input to consumer notebooks:

- `oracle_runner.ipynb` — uses `Vectors/{task}/md/L{l}.pt`
- `ln_global_runner.ipynb` — uses `Score-standard/{task}/{task}+md/L*.json`

After this prep, both consumer notebooks become fully self-contained
on the GPU side: weights + dataset + vectors + scores all attached
as read-only inputs, no on-the-fly computation needed.
"""))

    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).parent / "prep_vectors.ipynb"
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {out_path} ({len(nb.cells)} cells)")
