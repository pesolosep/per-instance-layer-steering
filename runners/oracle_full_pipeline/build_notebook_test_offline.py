"""Build oracle_full_pipeline_test_offline.ipynb from canonical Python source.

Test-split + Kaggle-offline variant. Cross-product of:
  - `build_notebook_test.py` (test split: UniDataset(set="test"),
    `{TASK}_test.jsonl`, output dirs `oracle_outputs_test/` and
    `ln_global_outputs_test/`, monitoring kit, Phase B clf_* preservation)
  - `build_notebook.py` (Kaggle offline: rglob-based dataset staging for
    Vectors/Score-standard/Dataset + prefix-match copy of prior
    oracle_outputs_test/ln_global_outputs_test, TRANSFORMERS_OFFLINE=1,
    flash_attn wheel install, _locate_gpt2_dir for offline judge model)

Only the bootstrap (cell 1) and the GPT-2 loader (cell 11) differ from
`build_notebook_test.py`. Everything else — configuration, imports,
monitoring kit, model load, dataset load, baseline, Phases A/B/C/D —
is identical to the test-split internet variant.

Run after editing:

    python runners/oracle_full_pipeline/build_notebook_test_offline.py
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

    # ─── Title ─────────────────────────────────────────────────────────
    nb.cells.append(md("""
# `oracle_full_pipeline_test_offline` — single (task, model) end-to-end (test split, Kaggle offline)

Test-split + Kaggle-offline variant of `oracle_full_pipeline.ipynb`. Same
four-phase pipeline, same monitoring kit and Phase B clf_* preservation as
`oracle_full_pipeline_test.ipynb`, but with the offline-Kaggle bootstrap:
stages Vectors/Score-standard/Dataset/oracle_outputs_test/ln_global_outputs_test
from attached Kaggle datasets, installs flash_attn from the offline wheel,
and locates GPT-2 medium under `/kaggle/input` or the local HF cache (no
internet `from_pretrained`).

| Phase | Output (under `oracle_outputs_test/{task}/{model}/`) |
|---|---|
| **A — Search** | `K{1..3}_exhaustive.parquet`, `K{2..5}_beam-W4.parquet`, `K1_trivial.parquet`, `K{1..5}_lnglobal.parquet`, `baseline.parquet` |
| **B — Picks** | `method_picks.parquet` (preserves prior clf_* picks if present) |
| **C — Steered eval + explain** | `method_picks_eval.parquet`, `explain_texts_picks.parquet` |
| **D — GPT-2 PPL** | `ppl_scores_picks.parquet` |

Idempotent at the parquet level — every phase checks if its output exists
and skips. Resumable across sessions, including across Kaggle preemption.
"""))

    # ─── Cell 1: Bootstrap (Kaggle offline) ────────────────────────────
    nb.cells.append(md("""
## 1. Bootstrap

Locate the codebase via sentinel; stage Vectors / Score-standard /
Dataset / oracle_outputs_test / ln_global_outputs_test from attached Kaggle
datasets; install flash-attn from the offline wheel. Same pattern as the
val-split prod notebook, but the prefix-match stagers target the
`oracle_outputs_test*` / `ln_global_outputs_test*` Kaggle datasets so prior
test-split sessions carry forward correctly.
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
        "Codebase not found. Attach the prep-codebase output dataset "
        "(the cloned repository) or set KAGGLE_INPUT_DIR."
    )
sys.path.insert(0, str(REPO_ROOT))

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
os.chdir(WORK_DIR)

# Stage Vectors / Score-standard / Dataset (single source, symlink)
if WORK_DIR == Path("/kaggle/working"):
    for name in ("Vectors", "Score-standard", "Dataset"):
        local = Path(name)
        if local.exists():
            continue
        kag_in = Path("/kaggle/input")
        match = next((p for p in kag_in.rglob(name) if p.is_dir() and any(p.iterdir())), None) if kag_in.exists() else None
        if match is not None:
            os.symlink(match, local)
            print(f"  staged {name}/ -> {match}")
        else:
            print(f"  WARN: {name}/ not found under /kaggle/input — attach prep-vectors dataset")

    # Stage oracle_outputs_test / ln_global_outputs_test by prefix-match.
    # CRITICAL: copy (not symlink) so this session\\'s output dataset contains
    # both prior-session data (carried forward) AND new data (this session\\'s α).
    # Symlinks would not commit cleanly when Kaggle packages the output dataset.
    # Test-split datasets share filenames with val-split datasets (baseline.parquet,
    # method_picks.parquet, etc.) — keeping them under distinct *_test prefixes
    # prevents accidental cross-contamination.
    import shutil
    def _stage_by_prefix(local_name, prefix):
        local = Path(local_name)
        kag_in = Path("/kaggle/input")
        if not kag_in.exists():
            return
        matches = sorted({
            p for p in kag_in.rglob(f"{prefix}*")
            if p.is_dir() and p.name.startswith(prefix) and any(p.iterdir())
        })
        if not matches:
            if local.exists() and any(local.iterdir()):
                n_pq = sum(1 for _ in local.rglob("*.parquet"))
                print(f"  {local_name}/ already in working dir ({n_pq} parquets)")
            else:
                print(f"  (none) no '{prefix}*' under /kaggle/input — fresh run, will write outputs locally")
            return

        local.mkdir(exist_ok=True)
        copied = 0
        for src in matches:
            for sp in src.rglob("*"):
                if sp.is_file():
                    rel = sp.relative_to(src)
                    dst = local / rel
                    if dst.exists():
                        continue   # local wins on conflict (idempotent re-run)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sp, dst)
                    copied += 1
        n_pq = sum(1 for _ in local.rglob("*.parquet"))
        src_str = ", ".join(str(p) for p in matches)
        print(f"  staged {local_name}/ from [{src_str}] (+{copied} new files, {n_pq} parquets total)")

    _stage_by_prefix("oracle_outputs_test",    "oracle_outputs_test")
    _stage_by_prefix("ln_global_outputs_test", "ln_global_outputs_test")

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

flash_whl = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
if os.path.exists(flash_whl):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", flash_whl])
        print("Installed flash_attn wheel")
    except Exception as e:
        print(f"flash_attn install skipped: {e}")

print(f"\\nCodebase root: {REPO_ROOT}")
print(f"cwd: {os.getcwd()}")
'''))

    # ─── Cell 2: Configuration ─────────────────────────────────────────
    nb.cells.append(md("""
## 2. Configuration — single edit per session

`(TASK, MODEL_PRESET, ALPHA_SCHEDULES)` defines the cell this notebook runs.

Set `ALPHA_SCHEDULES` to the α schedules you want this session to produce.
Phase B / C / D iterate exactly these — every phase\\'s idempotent
parquet-level skips prevent re-running completed work.

To smoke-test in ~minutes, set `K_VALUES_RUN = [1]` for a single run, then
change to `[1, 2, 3, 4, 5]` for the full sweep.
"""))
    nb.cells.append(code('''
from globalenv import LLAMA_8B, AYA_8B, QWEN_7B, QWEN_32B, Anth_MAIN

# ─── EDIT PER RUN ─────────────────────────────────────────────────
TASK            = "desire-to-create-allies"   # any task in Anth_MAIN
MODEL_PRESET    = LLAMA_8B                     # LLAMA_8B / AYA_8B / QWEN_7B / QWEN_32B
K_VALUES_RUN    = [1, 2, 3, 4, 5]              # full K-sweep

ALPHA_SCHEDULES = ["uniform"]                  # α schedules to produce this session
METHOD          = "md"                         # steering vector method
BATCH_SIZE      = 1012                         # forward-pass batch (Blackwell-tuned)
GEN_BATCH_SIZE  = 512                          # explain-text generation batch
CLASSIFY_BATCH  = 64                           # Y/N classify batch
PPL_BATCH_SIZE  = 512                          # GPT-2 PPL batch
PPL_MAX_LENGTH  = 200
MAX_NEW_TOKENS  = 200                          # explain text length
JUDGE_MODEL     = "gpt2-medium"
USE_TORCH_COMPILE = True
COMPILE_MODE      = "reduce-overhead"
# ──────────────────────────────────────────────────────────────────

OUTPUT_DIR    = Path("./oracle_outputs_test")
LN_OUTPUT_DIR = Path("./ln_global_outputs_test")

def _model_tag_from_preset(preset):
    n = preset.name.lower()
    if "llama" in n: return "llama-3-8b"
    if "aya"   in n: return "aya-expanse-8b"
    if "qwen"  in n and "32b" in n: return "qwen-2.5-32b"
    if "qwen"  in n: return "qwen-2.5-7b"
    return n.replace("/", "-")

MODEL_TAG = _model_tag_from_preset(MODEL_PRESET)
print(f"TASK={TASK}  MODEL={MODEL_TAG}  ({MODEL_PRESET.num_layers} layers)  SPLIT=test")
print(f"K_VALUES={K_VALUES_RUN}  ALPHA={ALPHA_SCHEDULES}")
print(f"BATCH={BATCH_SIZE}  GEN_BATCH={GEN_BATCH_SIZE}  CLASSIFY_BATCH={CLASSIFY_BATCH}")
print(f"OUTPUT_DIR={OUTPUT_DIR}  LN_OUTPUT_DIR={LN_OUTPUT_DIR}")
'''))

    # ─── Cell 3: Imports ───────────────────────────────────────────────
    nb.cells.append(md("## 3. Imports"))
    nb.cells.append(code('''
import json, time, gc
from typing import Dict, List, Sequence, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model_wrapper import ModelWrapper
from dataset import UniDataset
from prep.anth_persona import find_dataset_root
from globalenv import Anth_YN_QUESTION_PROFIX, INST_SYS_ANS_PREFIX_alt1

from oracle.config import K_VALUES, SEARCH_BY_K, BEAM_WIDTH, BEAM_LABEL
from oracle.runner import exhaustive_subsets, beam_extend
from oracle.runner_impl import compute_baseline, search_phase, _alpha_schedule, _load_vectors
from oracle.scoring import alignonly, multimetric_3
from oracle.methods import gt_global, tkm, joint_k
from oracle.io import save_df, load_df

from strategies import get_strategy

T_YES = MODEL_PRESET.yn_token_ids[" Yes"]
T_NO  = MODEL_PRESET.yn_token_ids[" No"]
N_LAYERS = MODEL_PRESET.num_layers

print(f"T_YES={T_YES}  T_NO={T_NO}  N_LAYERS={N_LAYERS}")
print(f"SEARCH_BY_K (canonical, will be filtered to K_VALUES_RUN): {SEARCH_BY_K}")
'''))

    # ─── Cell 3a: Monitoring kit ───────────────────────────────────────
    nb.cells.append(md("""
## 3a. Monitoring kit (GPU sidecar + peak-mem helpers)

Spawns a background `nvidia-smi -l 5` writing CSV samples to
`logs/{RUN_ID}/gpu.log`. Defines `print_gpu_state(label)` and
`print_peak_mem(label, reset=True)` used at the tail of each phase to
verify the GPU is being pressed (high `utilization.gpu` + peak VRAM
within a reasonable fraction of total) and that batch sizes are tuned.

Use `tools/monitor.sh` from the repo root for a one-shot live status
view: latest GPU samples + tail of the notebook execution log.
"""))
    nb.cells.append(code('''
import subprocess, atexit, datetime
from pathlib import Path as _P

# Tag this run with a timestamp; all monitoring logs land under logs/{run_id}/
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = _P("logs") / RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR.parent / "latest").unlink(missing_ok=True)
try:
    (LOG_DIR.parent / "latest").symlink_to(RUN_ID)
except (OSError, NotImplementedError):
    pass   # Windows / FS without symlinks — skip

def print_gpu_state(label=""):
    """One-shot GPU snapshot: util%, mem used / total, power."""
    if not torch.cuda.is_available():
        print(f"  [gpu {label}] CUDA unavailable"); return
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False).stdout.strip()
    print(f"  [gpu {label}] {out}")

def print_peak_mem(label, reset=True):
    """Peak allocated VRAM since last reset (GB)."""
    if not torch.cuda.is_available():
        print(f"  [peak-mem {label}] CUDA unavailable"); return
    peak_gb  = torch.cuda.max_memory_allocated() / 1e9
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [peak-mem {label}] peak={peak_gb:.2f}GB / {total_gb:.0f}GB "
          f"({100*peak_gb/total_gb:.0f}%)  current={alloc_gb:.2f}GB")
    if reset:
        torch.cuda.reset_peak_memory_stats()

def start_gpu_logger():
    """Spawn `nvidia-smi -l 5` -> logs/{RUN_ID}/gpu.log."""
    log_path = LOG_DIR / "gpu.log"
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
         "--format=csv", "-l", "5"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT
    )
    atexit.register(lambda: (proc.terminate() if proc.poll() is None else None))
    (LOG_DIR / "gpu.pid").write_text(str(proc.pid))
    print(f"  GPU logger pid={proc.pid} -> {log_path}")
    print(f"  view live: tail -f {log_path}")
    print(f"  one-shot status: ./tools/monitor.sh {RUN_ID}")
    return proc

_gpu_logger_proc = start_gpu_logger()
print_gpu_state("startup")
print(f"\\n  RUN_ID = {RUN_ID}")
print(f"  log dir: {LOG_DIR.resolve()}")
'''))

    # ─── Cell 4: Load Llama ────────────────────────────────────────────
    nb.cells.append(md("""
## 4. Load steering model (Llama / Aya / Qwen)

`ModelWrapper` handles bf16 + FA2 + KV-cache + device_map='auto'. Optional
torch.compile for Phase A throughput. Left-padded tokenizer for batched
generation in Phase C.
"""))
    nb.cells.append(code('''
model = ModelWrapper(MODEL_PRESET)
tokenizer = model.tokenizer
device = model.device

# Left-pad for Phase C batched generation. Right-pad would corrupt logits[:, -1, :].
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USE_TORCH_COMPILE:
    model.model = torch.compile(model.model, mode=COMPILE_MODE, fullgraph=False)
    print(f"torch.compile applied (mode={COMPILE_MODE}, fullgraph=False)")

print(f"model: device={device}  dtype={next(model.model.parameters()).dtype}")
print(f"padding_side={tokenizer.padding_side}  pad_token={tokenizer.pad_token!r}")
print_gpu_state("after model load")
print_peak_mem("model load", reset=True)
'''))

    # ─── Cell 5: Load test dataset ────────────────────────────────────
    nb.cells.append(md("""
## 5. Load test dataset

`UniDataset(set='test')` — held-out split used for predictor validation
(Phase 6 of the validation plan). Raw questions read from `{TASK}_test.jsonl`.
"""))
    nb.cells.append(code('''
val_dataset = UniDataset(MODEL_PRESET, task=TASK, train=False, set="test",
                          batch_size=BATCH_SIZE, tokenizer=tokenizer)
n_instances = len(val_dataset.keys)
print(f"test instances: {n_instances}")

# Also load raw questions (needed for explain prompt in Phase C)
data_path = find_dataset_root() / f"{TASK}_test.jsonl"
with open(data_path) as f:
    raw_data = [json.loads(line) for line in f]
questions = [d["question"] for d in raw_data]
print(f"raw questions loaded: {len(questions)}")
'''))

    # ─── Cell 6: Compute baseline ──────────────────────────────────────
    nb.cells.append(md("""
## 6. Compute baseline (cached, α-agnostic)

Per-instance unsteered metrics. Lives at the task level — same baseline used
across both α schedules (since baseline is unsteered, schedule doesn't matter).
"""))
    nb.cells.append(code('''
base_pq = OUTPUT_DIR / TASK / MODEL_TAG / "baseline.parquet"
base_pq.parent.mkdir(parents=True, exist_ok=True)

if base_pq.exists():
    print(f"baseline cached at {base_pq.name}")
else:
    print("computing baseline (unsteered forward + generation)...")
    bl = compute_baseline(model=model, dataset=val_dataset,
                          t_yes=T_YES, t_no=T_NO, generate_text=True)
    baseline_df = pd.DataFrame({
        "instance_id":       list(range(len(bl["gold_signs"]))),
        "base_logit_yes":    bl["z_yes"].tolist(),
        "base_logit_no":     bl["z_no"].tolist(),
        "base_align_prob":   bl["align_prob"].tolist(),
        "base_logit_margin": bl["logit_margin"].tolist(),
        "base_yn_mass":      bl["yn_mass"].tolist(),
        "gold_sign":         bl["gold_signs"].tolist(),
        "base_text":         bl.get("texts", [""] * len(bl["gold_signs"])),
    })
    pq, csv = save_df(baseline_df, base_pq)
    print(f"saved baseline → {pq.name} (+ {csv.name})")
    print_peak_mem("baseline", reset=True)

# Reload as torch tensors for downstream search_phase calls
_bdf = load_df(base_pq)
baseline = {
    "z_yes":        torch.tensor(_bdf["base_logit_yes"].values),
    "z_no":         torch.tensor(_bdf["base_logit_no"].values),
    "align_prob":   torch.tensor(_bdf["base_align_prob"].values),
    "logit_margin": torch.tensor(_bdf["base_logit_margin"].values),
    "yn_mass":      torch.tensor(_bdf["base_yn_mass"].values),
    "gold_signs":   torch.tensor(_bdf["gold_sign"].values),
    "texts":        _bdf["base_text"].tolist(),
}
print(f"baseline reloaded: {len(_bdf)} instances")
'''))

    # ─── Cell 7: Phase A.1 — search phase ───────────────────────────────
    nb.cells.append(md("""
## 7. Phase A.1 — Search phase (exhaustive + beam-W4)

Per (α, K, search_method) writes one parquet:
- `K1_trivial.parquet` — all single layers
- `K{2,3}_exhaustive.parquet` — full C(N,K) enumeration
- `K{2..5}_beam-W4.parquet` — beam-W4 trajectory

K=3 exhaustive is the slow one (~7h per α). Other phases are minutes.
Idempotent: existing parquets are skipped, so resuming after preemption
just picks up where it left off.
"""))
    nb.cells.append(code('''
def _output_path(alpha, K, search_method):
    p = OUTPUT_DIR / TASK / MODEL_TAG / alpha
    p.mkdir(parents=True, exist_ok=True)
    return p / f"K{K}_{search_method}.parquet"

def _top_w_from_parquet(pq_path, width):
    """Best-W K-subsets from a scored parquet (used to seed beam at next K)."""
    df = alignonly.score(load_df(pq_path))
    grouped = df.groupby(df["layer_set"].apply(tuple))["score"].mean()
    return [tuple(s) for s in grouped.nlargest(width).index]

def _candidate_subsets(K, search_method, layers_pool, beam_prev_picks):
    if search_method == "trivial":
        return [(l,) for l in layers_pool]
    if search_method == "exhaustive":
        return exhaustive_subsets(layers_pool, K)
    if search_method.startswith("beam-"):
        if K == 1:
            return [(l,) for l in layers_pool]
        return beam_extend(beam_prev_picks, layers_pool, BEAM_WIDTH)
    raise ValueError(f"Unknown search_method: {search_method}")

# Build the K × search_method matrix to actually run, filtered to K_VALUES_RUN
search_jobs = []
for alpha in ALPHA_SCHEDULES:
    for K in K_VALUES_RUN:
        for sm in SEARCH_BY_K[K]:
            search_jobs.append((alpha, K, sm))

print(f"Phase A.1 — {len(search_jobs)} (α, K, search_method) jobs to consider")
layers_pool = list(range(N_LAYERS))
t0 = time.perf_counter()

for alpha, K, search_method in search_jobs:
    out_pq = _output_path(alpha, K, search_method)
    if out_pq.exists():
        print(f"  skip α={alpha} K={K} {search_method} — exists at {out_pq.name}")
        continue

    # Beam at K>1 bootstraps from previous K
    beam_prev_picks = []
    if search_method.startswith("beam-") and K > 1:
        prev_pq = _output_path(alpha, K - 1, BEAM_LABEL)
        if not prev_pq.exists():
            for sm in SEARCH_BY_K[K - 1]:
                candidate = _output_path(alpha, K - 1, sm)
                if candidate.exists():
                    prev_pq = candidate
                    break
        if not prev_pq.exists():
            print(f"  WARN α={alpha} K={K} beam: no K-1 parquet to bootstrap; skipping")
            continue
        beam_prev_picks = _top_w_from_parquet(prev_pq, BEAM_WIDTH)
        print(f"  bootstrapped beam K={K} from {prev_pq.name}: {beam_prev_picks}")

    print(f"\\n  === α={alpha} K={K} {search_method} ===")
    subsets = _candidate_subsets(K, search_method, layers_pool, beam_prev_picks)
    print(f"  {len(subsets)} candidate subsets")

    df = search_phase(
        model=model, dataset=val_dataset,
        layer_subsets=subsets,
        alpha_schedule=alpha,
        task=TASK, method=METHOD,
        t_yes=T_YES, t_no=T_NO,
        baseline=baseline,
        search_method=search_method, K=K,
    )
    pq, csv = save_df(df, out_pq)
    elapsed = time.perf_counter() - t0
    print(f"  wrote {len(df)} rows → {pq.name}  (cumulative phase A.1 elapsed={elapsed/60:.1f}min)")
    print_peak_mem(f"A.1 K={K} {search_method}", reset=True)

print(f"\\nPhase A.1 done.")
print_gpu_state("end of Phase A.1")
'''))

    # ─── Cell 8: Phase A.2 — LN-global ───────────────────────────────
    nb.cells.append(md("""
## 8. Phase A.2 — LN-global (heuristic picks K=1..5)

LN-global ranks layers by their geometric `s_score = D + C` (loaded from
`Score-standard/{task}/{task}+md/L*.json`) and picks top-K. Per K, one
global subset. Per α, one steered-eval parquet `K{K}_lnglobal.parquet`.

Same `search_phase` machinery as Phase A.1, just with a single-subset
input per K.
"""))
    nb.cells.append(code('''
def _ln_alpha_dir(alpha):
    p = LN_OUTPUT_DIR / TASK / MODEL_TAG / alpha
    p.mkdir(parents=True, exist_ok=True)
    return p

# Picks are α-agnostic — store at task level
ln_task_dir = LN_OUTPUT_DIR / TASK / MODEL_TAG
ln_task_dir.mkdir(parents=True, exist_ok=True)

ln_strategy = get_strategy("my")    # LN paper baseline = s_score = D + C
ln_picks_by_K = {}
for K in K_VALUES_RUN:
    picks_path = ln_task_dir / f"picks_K{K}.json"
    if picks_path.exists():
        info = json.load(open(picks_path))
        ln_picks_by_K[K] = info["layers"]
        print(f"  K={K} ln_global picks cached: {info['layers']}")
    else:
        ranked = ln_strategy.select(TASK, METHOD, list(range(N_LAYERS)), K)
        layers = [int(l) for l, _ in ranked]
        scores = [float(s) for _, s in ranked]
        json.dump({"layers": layers, "scores": scores}, open(picks_path, "w"))
        ln_picks_by_K[K] = layers
        print(f"  K={K} ln_global picks (new): {layers}")

print(f"\\nPhase A.2 — eval-pass on LN picks per α:")
for alpha in ALPHA_SCHEDULES:
    α_dir = _ln_alpha_dir(alpha)
    for K in K_VALUES_RUN:
        search_pq = α_dir / f"K{K}_lnglobal.parquet"
        if search_pq.exists():
            print(f"  skip α={alpha} K={K} — exists at {search_pq.name}")
            continue
        picked_subset = tuple(sorted(ln_picks_by_K[K]))
        df_search = search_phase(
            model=model, dataset=val_dataset,
            layer_subsets=[picked_subset],
            alpha_schedule=alpha,
            task=TASK, method=METHOD,
            t_yes=T_YES, t_no=T_NO,
            baseline=baseline,
            search_method="ln-global", K=K,
        )
        pq, csv = save_df(df_search, search_pq)
        print(f"  wrote α={alpha} K={K} {len(df_search)} rows → {pq.name}")

print(f"\\nPhase A.2 done.")
print_peak_mem("Phase A.2", reset=True)
'''))

    # ─── Cell 9: Phase B — derive method picks ────────────────────────
    nb.cells.append(md("""
## 9. Phase B — Derive method picks → `method_picks.parquet`

Pure pandas, no GPU. From the search-phase parquets (Phase A.1 + A.2),
derive picks for every method × scoring × K × α.

Methods produced:
- `tkm_align`, `tkm_multi3` (per-instance from K1_trivial)
- `gt_global_align`, `gt_global_multi3` (global from K1_trivial)
- `ln_global` (from K{K}_lnglobal)
- `exhaustive_align`, `exhaustive_multi3` (per-instance from K{1..3}_exhaustive)
- `beam_align`, `beam_multi3` (per-instance from K{2..5}_beam-W4)

Preserves any externally-merged non-oracle picks (e.g. `clf_*` from
`predictor_pilot`) from a prior `method_picks.parquet` — Phase B otherwise
regenerates oracle picks from scratch and would wipe them.
"""))
    nb.cells.append(code('''
def _to_tup(ls):
    return tuple(sorted(int(l) for l in ls))

print(f"Phase B: α schedules to derive picks for: {ALPHA_SCHEDULES}")

rows = []

for alpha in ALPHA_SCHEDULES:
    α_dir = OUTPUT_DIR / TASK / MODEL_TAG / alpha
    ln_α_dir = LN_OUTPUT_DIR / TASK / MODEL_TAG / alpha

    # K=1 trivial (drives tkm + gt_global + exhaustive_align K=1)
    k1_pq = α_dir / "K1_trivial.parquet"
    if not k1_pq.exists():
        print(f"  α={alpha}: K1_trivial.parquet missing — skipping picks for this α")
        continue
    k1_raw = pd.read_parquet(k1_pq)
    k1_a = alignonly.score(k1_raw)
    k1_m = multimetric_3.score(k1_raw)
    all_iids = sorted(int(i) for i in k1_raw["instance_id"].unique())

    for K in K_VALUES_RUN:
        # tkm (per-instance)
        for iid, layers in tkm.select(k1_a, K).items():
            rows.append({"method_id": "tkm_align", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})
        for iid, layers in tkm.select(k1_m, K).items():
            rows.append({"method_id": "tkm_multi3", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})

        # gt_global (one global per K, broadcast to all iids)
        gg_a = _to_tup(gt_global.select(k1_a, K))
        gg_m = _to_tup(gt_global.select(k1_m, K))
        for iid in all_iids:
            rows.append({"method_id": "gt_global_align", "alpha_schedule": alpha, "K": K,
                         "instance_id": iid, "layer_set": gg_a})
            rows.append({"method_id": "gt_global_multi3", "alpha_schedule": alpha, "K": K,
                         "instance_id": iid, "layer_set": gg_m})

        # ln_global (one global per K, from Phase A.2 parquet)
        ln_pq = ln_α_dir / f"K{K}_lnglobal.parquet"
        if ln_pq.exists():
            ln_df = pd.read_parquet(ln_pq)
            ln_pick = _to_tup(ln_df.iloc[0]["layer_set"])
            for iid in all_iids:
                rows.append({"method_id": "ln_global", "alpha_schedule": alpha, "K": K,
                             "instance_id": iid, "layer_set": ln_pick})

    # exhaustive_align / exhaustive_multi3 (K=1 from k1, K=2,3 from K{K}_exhaustive)
    for iid, grp in k1_a.groupby("instance_id"):
        best = grp.loc[grp["score"].idxmax()]
        rows.append({"method_id": "exhaustive_align", "alpha_schedule": alpha, "K": 1,
                     "instance_id": int(iid), "layer_set": _to_tup(best["layer_set"])})
    for iid, grp in k1_m.groupby("instance_id"):
        best = grp.loc[grp["score"].idxmax()]
        rows.append({"method_id": "exhaustive_multi3", "alpha_schedule": alpha, "K": 1,
                     "instance_id": int(iid), "layer_set": _to_tup(best["layer_set"])})
    for K in (2, 3):
        if K not in K_VALUES_RUN: continue
        ex_pq = α_dir / f"K{K}_exhaustive.parquet"
        if not ex_pq.exists(): continue
        ex_raw = pd.read_parquet(ex_pq)
        ex_a = alignonly.score(ex_raw); ex_m = multimetric_3.score(ex_raw)
        for iid, layers in joint_k.select(ex_a, K).items():
            rows.append({"method_id": "exhaustive_align", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})
        for iid, layers in joint_k.select(ex_m, K).items():
            rows.append({"method_id": "exhaustive_multi3", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})

    # beam_align / beam_multi3 (K=2..5 from K{K}_beam-W4)
    for K in (2, 3, 4, 5):
        if K not in K_VALUES_RUN: continue
        bm_pq = α_dir / f"K{K}_beam-W4.parquet"
        if not bm_pq.exists(): continue
        bm_raw = pd.read_parquet(bm_pq)
        bm_a = alignonly.score(bm_raw); bm_m = multimetric_3.score(bm_raw)
        for iid, layers in joint_k.select(bm_a, K).items():
            rows.append({"method_id": "beam_align", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})
        for iid, layers in joint_k.select(bm_m, K).items():
            rows.append({"method_id": "beam_multi3", "alpha_schedule": alpha, "K": K,
                         "instance_id": int(iid), "layer_set": _to_tup(layers)})

picks_df = pd.DataFrame(rows)
# Drop rows where joint_k gave wrong-length layer_set (shouldn't happen but defensive)
picks_df = picks_df[picks_df.apply(lambda r: len(r["layer_set"]) == r["K"], axis=1)].reset_index(drop=True)

picks_out = OUTPUT_DIR / TASK / MODEL_TAG / "method_picks.parquet"
# Preserve any externally-merged non-oracle picks (e.g. clf_* from predictor_pilot
# Step 2). Phase B regenerates oracle methods from scratch each run, so without
# this clause those rows would be wiped on every test-pipeline re-run.
if picks_out.exists():
    prior = pd.read_parquet(picks_out)
    keep_mask = ~prior["method_id"].isin(picks_df["method_id"].unique())
    extra = prior[keep_mask].copy()
    if len(extra) > 0:
        extra["layer_set"] = extra["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
        picks_df = pd.concat([picks_df, extra], ignore_index=True)
        print(f"  preserved {len(extra)} non-oracle rows ({sorted(extra['method_id'].unique())}) from prior method_picks.parquet")

pq, csv = save_df(picks_df, picks_out)
print(f"wrote {len(picks_df)} rows → {pq.name} (+ {csv.name})")
print("\\nCoverage (rows per method × α × K):")
print(picks_df.groupby(["method_id", "alpha_schedule", "K"]).size().unstack(["alpha_schedule", "K"]).fillna(0).astype(int).to_string())
'''))

    # ─── Cell 10: Phase C — steered eval + explain ────────────────────
    nb.cells.append(md("""
## 10. Phase C — Steered full-eval + LN A.2.1 explain text generation

For each unique `(α, K, layer_set)` group from `method_picks.parquet`:
1. Apply steering hooks
2. **Pass 1**: forward on Y/N prompt with `My answer:` suffix → capture
   logits → compute steered_prob, steered_yn_mass, alignment_lift,
   logit_margin_delta. Also derive `picked_answer = argmax(logit_yes, logit_no)`.
3. **Pass 2**: generate ~200-token explain text under the same hooks,
   using the picked_answer in the prompt.
4. Reset hooks.

Writes per α:
- `method_picks_eval.parquet` — matches K_exhaustive schema (chain_lookup compatible)
- `explain_texts_picks.parquet` — Pass 3 (PPL) input

Also generates baseline (unsteered) explain texts per α as PPL reference.
"""))
    nb.cells.append(code('''
# Build prompts — MODEL_PRESET.yn_template / explain_template are the
# canonical per-model strings (Llama uses <|start_header_id|>…, Aya uses
# <|START_OF_TURN_TOKEN|>…, Qwen uses <|im_start|>…). YN_TEMPLATE matches
# the canonical UniDataset prompt (ends with "My answer:" → next token is
# " Yes" / " No", matching MODEL_PRESET.yn_token_ids).
#
# BOS convention: Llama/Aya templates carry the literal BOS (`<|begin_of_text|>`
# or `<BOS_TOKEN>`) AND the tokenizer call below uses `add_special_tokens=True`,
# so the tokenizer auto-prepends another BOS. Result is two BOS tokens at
# positions 0,1 — matching the double-BOS regime UniDataset emits at
# `dataset.py:_tok`. Qwen templates have no literal BOS (its tokenizer also
# defaults to `add_bos_token=False`, so Qwen stays single-BOS). This keeps
# Phase C's eval/explain prompts identical to the prompts under which
# Phase A.1 search-phase scored the candidate subsets, so picks chosen by
# Phase B remain valid for the eval context.

YN_TEMPLATE      = MODEL_PRESET.yn_template
EXPLAIN_TEMPLATE = MODEL_PRESET.explain_template

def build_yn_prompt(iid):
    return YN_TEMPLATE.format(question=questions[iid] + Anth_YN_QUESTION_PROFIX)

def build_explain_prompt(iid, picked_answer):
    return EXPLAIN_TEMPLATE.format(
        question=questions[iid] + Anth_YN_QUESTION_PROFIX,
        answer=picked_answer,
    )

@torch.no_grad()
def steered_logits_batch(prompts):
    """One forward; return parallel (yes_logit, no_logit, yn_mass) lists."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=512, add_special_tokens=True)
    ids  = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model.model(input_ids=ids, attention_mask=attn, use_cache=False)
    last_logits = out.logits[:, -1, :].float()
    yes_logits = last_logits[:, T_YES]
    no_logits  = last_logits[:, T_NO]
    yn_pair = torch.stack([yes_logits, no_logits], dim=-1)
    log_sum_yn  = yn_pair.logsumexp(dim=-1)
    log_sum_all = last_logits.logsumexp(dim=-1)
    yn_mass = torch.exp(log_sum_yn - log_sum_all)
    return (yes_logits.cpu().tolist(), no_logits.cpu().tolist(), yn_mass.cpu().tolist())

@torch.no_grad()
def generate_explain_batch(prompts, max_new_tokens=MAX_NEW_TOKENS):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                     add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    out = model.model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(MODEL_PRESET.terminator_token),
    )
    new_tokens = out[:, in_len:]
    n_new = (new_tokens != tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return list(zip(texts, n_new))

import math
def _steered_prob_aligned(yes_l, no_l, gold_sign):
    m = max(yes_l, no_l)
    p_yes = math.exp(yes_l - m) / (math.exp(yes_l - m) + math.exp(no_l - m))
    return p_yes if int(gold_sign) == 1 else (1.0 - p_yes)

# Base lookup (instance_id → row) for baseline metrics
base_df_for_lookup = load_df(base_pq)
base_lookup = {int(r["instance_id"]): r for _, r in base_df_for_lookup.iterrows()}
print(f"baseline lookup: {len(base_lookup)} instances")
'''))
    nb.cells.append(code('''
# Load picks file
picks_pq = OUTPUT_DIR / TASK / MODEL_TAG / "method_picks.parquet"
picks_df = pd.read_parquet(picks_pq)
picks_df["layer_set"] = picks_df["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))

# Group by (α, K, subset)
groups = {}
for _, row in picks_df.iterrows():
    key = (row["alpha_schedule"], int(row["K"]), row["layer_set"])
    g = groups.setdefault(key, {"iids": set(), "method_ids": set()})
    g["iids"].add(int(row["instance_id"]))
    g["method_ids"].add(row["method_id"])

print(f"Total unique (α, K, subset) groups: {len(groups)}")
per_α = defaultdict(int)
for (a, _, _) in groups: per_α[a] += 1
print(f"  by α: {dict(per_α)}")
'''))
    nb.cells.append(code('''
from oracle.io import load_or_empty, is_complete

# Per-group checkpoint cadence. Smaller = safer under preemption, slower
# (more parquet rewrites). 20 = ~1-2 min of work at risk on Blackwell.
CHECKPOINT_EVERY = 20

def _tup(x):
    """Canonicalize a layer-set (numpy array / list / tuple) to sorted-int tuple."""
    if x is None:
        return ()
    return tuple(sorted(int(l) for l in x))


def _process_alpha(alpha):
    """Phase C for one α — resumable from on-disk checkpoints."""
    print(f"\\n=== α = {alpha} ===")
    explain_pq = OUTPUT_DIR / TASK / MODEL_TAG / alpha / "explain_texts_picks.parquet"
    eval_pq    = OUTPUT_DIR / TASK / MODEL_TAG / alpha / "method_picks_eval.parquet"

    # ── Resume: seed records from any prior partial output ───────────
    prior_eval    = load_or_empty(eval_pq)
    prior_explain = load_or_empty(explain_pq)
    eval_records    = prior_eval.to_dict("records") if len(prior_eval) else []
    explain_records = prior_explain.to_dict("records") if len(prior_explain) else []
    for r in eval_records:
        r["layer_set"] = list(_tup(r["layer_set"]))
    for r in explain_records:
        r["layer_set"] = _tup(r["layer_set"])
    done_eval    = {(int(r["instance_id"]), _tup(r["layer_set"])) for r in eval_records}
    done_explain = {(int(r["instance_id"]), _tup(r["layer_set"])) for r in explain_records}
    seen_eval_keys = set(done_eval)
    print(f"  resume: prior eval rows={len(eval_records)}  explain rows={len(explain_records)}  "
          f"done eval keys={len(done_eval)}  done explain keys={len(done_explain)}")

    phase_start = time.perf_counter()
    def _checkpoint(label=""):
        if eval_records:
            save_df(pd.DataFrame(eval_records), eval_pq)
        if explain_records:
            save_df(pd.DataFrame(explain_records), explain_pq)
        wall = time.perf_counter() - phase_start
        rate = (len(eval_records) + len(explain_records)) / max(wall, 1e-6)
        print(f"    [checkpoint{label}] eval={len(eval_records)} explain={len(explain_records)} "
              f"wall={wall/60:.1f}min rate={rate:.1f}rows/s")
        print_gpu_state(f"checkpoint{label}")

    # ── Baseline pass (unsteered) — only iids still missing from explain ─
    base_iids_missing = [i for i in range(n_instances) if (i, ()) not in done_explain]
    t0 = time.perf_counter()
    if base_iids_missing:
        model.reset_all()
        base_answers = {}
        print(f"  baseline Y/N classify ({len(base_iids_missing)}/{n_instances} missing)...")
        for cs in range(0, len(base_iids_missing), CLASSIFY_BATCH):
            chunk = base_iids_missing[cs : cs + CLASSIFY_BATCH]
            yes_l, no_l, _ = steered_logits_batch([build_yn_prompt(i) for i in chunk])
            for i, y, n in zip(chunk, yes_l, no_l):
                base_answers[i] = "Yes" if y >= n else "No"
        print(f"    done ({time.perf_counter()-t0:.1f}s)")

        print("  baseline explain generation...")
        for cs in range(0, len(base_iids_missing), GEN_BATCH_SIZE):
            chunk = base_iids_missing[cs : cs + GEN_BATCH_SIZE]
            outs = generate_explain_batch([build_explain_prompt(i, base_answers[i]) for i in chunk])
            for i, (text, nn) in zip(chunk, outs):
                explain_records.append({
                    "instance_id":        i,
                    "layer_set":          (),
                    "alpha_schedule":     alpha,
                    "picked_answer":      base_answers[i],
                    "explain_text":       text,
                    "n_tokens_generated": int(nn),
                })
                done_explain.add((i, ()))
        print(f"    done ({time.perf_counter()-t0:.1f}s)")
        _checkpoint(" baseline")
    else:
        print(f"  baseline: all {n_instances} unsteered explain rows already present — skipping baseline pass")

    # ── Steered groups — filter to remaining work ────────────────────
    α_groups = sorted(((k, ls, info) for (a, k, ls), info in groups.items() if a == alpha),
                       key=lambda x: (x[0], x[1]))
    print(f"  steered groups: {len(α_groups)}")
    grp_t0 = time.perf_counter()
    processed_since_ckpt = 0

    for gi, (K, subset, info) in enumerate(α_groups):
        all_iids = sorted(info["iids"])
        iids_need_eval    = [i for i in all_iids if (i, subset) not in done_eval]
        iids_need_explain = [i for i in all_iids if (i, subset) not in done_explain]
        iids_active = sorted(set(iids_need_eval) | set(iids_need_explain))
        if not iids_active:
            if gi < 5 or (gi + 1) % 25 == 0 or gi == len(α_groups) - 1:
                print(f"    grp {gi+1:>4}/{len(α_groups)}  K={K} size={len(all_iids):>3}  "
                      f"resume-skip (all done)")
            continue

        alphas_per_layer = _alpha_schedule(K, alpha)
        vects = _load_vectors(list(subset), alphas_per_layer, TASK, METHOD, device)
        model.reset_all()
        for l, v in vects.items():
            model.set_add_activations(l, v)

        # Pass 1: Y/N classify + capture full logits for alignment metrics
        ans_by_iid = {}
        for cs in range(0, len(iids_active), CLASSIFY_BATCH):
            chunk = iids_active[cs : cs + CLASSIFY_BATCH]
            yes_l, no_l, yn_m = steered_logits_batch([build_yn_prompt(i) for i in chunk])
            for iid, y, n, m in zip(chunk, yes_l, no_l, yn_m):
                ans = "Yes" if y >= n else "No"
                ans_by_iid[iid] = ans
                key = (iid, subset)
                if key in seen_eval_keys:
                    continue
                seen_eval_keys.add(key)
                br = base_lookup.get(iid)
                if br is None: continue
                bp = float(br["base_align_prob"])
                sp = _steered_prob_aligned(y, n, int(br["gold_sign"]))
                eval_records.append({
                    "instance_id":        iid,
                    "layer_set":          list(subset),
                    "K":                  K,
                    "search_method":      "method_picks",
                    "alpha_schedule":     alpha,
                    "base_prob":          bp,
                    "steered_prob":       sp,
                    "base_logit_yes":     float(br["base_logit_yes"]),
                    "base_logit_no":      float(br["base_logit_no"]),
                    "steered_logit_yes":  float(y),
                    "steered_logit_no":   float(n),
                    "gold_sign":          int(br["gold_sign"]),
                    "base_yn_mass":       float(br["base_yn_mass"]),
                    "steered_yn_mass":    float(m),
                })
                done_eval.add((iid, subset))

        # Pass 2: explain generation under same hooks — only missing iids
        explain_chunk_iids = [i for i in iids_active if (i, subset) not in done_explain]
        for cs in range(0, len(explain_chunk_iids), GEN_BATCH_SIZE):
            chunk = explain_chunk_iids[cs : cs + GEN_BATCH_SIZE]
            outs = generate_explain_batch([build_explain_prompt(i, ans_by_iid[i]) for i in chunk])
            for iid, (text, nn) in zip(chunk, outs):
                explain_records.append({
                    "instance_id":        iid,
                    "layer_set":          subset,
                    "alpha_schedule":     alpha,
                    "picked_answer":      ans_by_iid[iid],
                    "explain_text":       text,
                    "n_tokens_generated": int(nn),
                })
                done_explain.add((iid, subset))

        processed_since_ckpt += 1
        if processed_since_ckpt >= CHECKPOINT_EVERY:
            _checkpoint(f" grp {gi+1}")
            processed_since_ckpt = 0

        if gi < 5 or (gi + 1) % 25 == 0 or gi == len(α_groups) - 1:
            elapsed = time.perf_counter() - grp_t0
            avg = elapsed / (gi + 1)
            eta = avg * (len(α_groups) - gi - 1)
            print(f"    grp {gi+1:>4}/{len(α_groups)}  K={K} size={len(iids_active):>3}  "
                  f"elapsed={elapsed:>6.1f}s  eta={eta/60:>5.1f}min  "
                  f"explain={len(explain_records)}  eval={len(eval_records)}")

    model.reset_all()
    _checkpoint(" final")
    print(f"  α={alpha} done in {time.perf_counter()-grp_t0:.1f}s "
          f"(explain={len(explain_records)}, eval={len(eval_records)})")
    return explain_records, eval_records


# ── Per-α completion gating ──────────────────────────────────────────
def _expected_counts(alpha):
    sub = picks_df[picks_df["alpha_schedule"] == alpha]
    if len(sub) == 0:
        return 0, 0
    n_pairs = len({(int(r["instance_id"]), _tup(r["layer_set"]))
                   for _, r in sub.iterrows()})
    return n_pairs, n_pairs + n_instances


explain_dfs = {}
print(f"Phase C will run α schedules: {ALPHA_SCHEDULES}")

for alpha in ALPHA_SCHEDULES:
    explain_pq = OUTPUT_DIR / TASK / MODEL_TAG / alpha / "explain_texts_picks.parquet"
    eval_pq    = OUTPUT_DIR / TASK / MODEL_TAG / alpha / "method_picks_eval.parquet"

    if not any(a == alpha for (a, _, _) in groups):
        print(f"  α={alpha}: no picks groups; skipping")
        continue

    expected_eval, expected_explain = _expected_counts(alpha)
    if is_complete(eval_pq, expected_eval) and is_complete(explain_pq, expected_explain):
        print(f"  α={alpha}: both Phase C outputs complete "
              f"(eval≥{expected_eval}, explain≥{expected_explain}); skipping")
        explain_dfs[alpha] = pd.read_parquet(explain_pq)
        continue

    print(f"  α={alpha}: expected eval={expected_eval}  explain={expected_explain}  "
          f"(entering resumable Phase C)")
    explain_recs, eval_recs = _process_alpha(alpha)

    df_explain = pd.DataFrame(explain_recs)
    pq, csv = save_df(df_explain, explain_pq)
    print(f"  α={alpha} final wrote {len(df_explain)} rows → {pq.name}")
    explain_dfs[alpha] = df_explain

    df_eval = pd.DataFrame(eval_recs)
    pq, csv = save_df(df_eval, eval_pq)
    print(f"  α={alpha} final wrote {len(df_eval)} rows → {pq.name}")
    if len(df_eval) > 0:
        align_lift = df_eval["steered_prob"] - df_eval["base_prob"]
        lm_delta = df_eval["gold_sign"] * ((df_eval["steered_logit_yes"] - df_eval["steered_logit_no"])
                                            - (df_eval["base_logit_yes"] - df_eval["base_logit_no"]))
        print(f"    alignment_lift  mean={align_lift.mean():+.4f}  median={align_lift.median():+.4f}")
        print(f"    logit_margin Δ  mean={lm_delta.mean():+.4f}  median={lm_delta.median():+.4f}")
        print(f"    steered_yn_mass mean={df_eval['steered_yn_mass'].mean():.4f}")

print(f"\\nPhase C done. {len(explain_dfs)} α processed.")
print_peak_mem("Phase C", reset=True)
'''))

    # ─── Cell 11: Free Llama + load GPT-2 (offline) ──────────────────
    nb.cells.append(md("""
## 11. Free Llama, load GPT-2 medium (offline)

Sequential model load. `del model; gc.collect(); torch.cuda.empty_cache()`
prevents fragmentation from torch.compile recompilation buffers
accumulating, even on Blackwell.

Offline locator: searches `/kaggle/input` (any attached Kaggle dataset
containing gpt2-medium weights) and the local HF cache for a
`config.json` with `model_type == "gpt2"` and the right model name in
its path. No internet `from_pretrained`.
"""))
    nb.cells.append(code('''
del model, tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Llama freed.")

from transformers import AutoModelForCausalLM, AutoTokenizer

def _locate_gpt2_dir():
    candidates = []
    kag = Path("/kaggle/input")
    if kag.exists():
        candidates += list(kag.rglob("config.json"))
    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_home.exists():
        candidates += list(hf_home.rglob("config.json"))
    for c in candidates:
        try:
            with open(c) as f:
                cfg = json.load(f)
            if cfg.get("model_type", "") == "gpt2" and \\
               JUDGE_MODEL.replace("/", "-").lower() in str(c.parent).lower():
                return c.parent
        except Exception:
            continue
    raise RuntimeError(
        f"No local snapshot for {JUDGE_MODEL!r}. Attach the prep_gpt2_judge "
        "output dataset (or any dataset containing gpt2-medium weights)."
    )

judge_path = _locate_gpt2_dir()
print(f"judge dir: {judge_path}")

judge_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
judge_tokenizer = AutoTokenizer.from_pretrained(judge_path)
if judge_tokenizer.pad_token is None:
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
judge_model = AutoModelForCausalLM.from_pretrained(
    judge_path, torch_dtype=judge_dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)
judge_model.train(False)
judge_device = next(judge_model.parameters()).device
print(f"judge: {sum(p.numel() for p in judge_model.parameters())/1e6:.0f}M params, "
      f"dtype={next(judge_model.parameters()).dtype}, device={judge_device}")
print_gpu_state("after GPT-2 load")
print_peak_mem("GPT-2 load", reset=True)
'''))

    # ─── Cell 12: PPL helper ──────────────────────────────────────────
    nb.cells.append(md("## 12. PPL helper (causal-LM next-token NLL average)"))
    nb.cells.append(code('''
@torch.no_grad()
def compute_ppl_batch(texts, jmodel, jtok, jdevice,
                      max_length=PPL_MAX_LENGTH, batch_size=PPL_BATCH_SIZE):
    ppls, n_tokens = [], []
    for cs in range(0, len(texts), batch_size):
        chunk = texts[cs : cs + batch_size]
        enc = jtok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        ids  = enc["input_ids"].to(jdevice)
        attn = enc["attention_mask"].to(jdevice)
        if ids.size(1) < 2:
            ppls.extend([float("nan")] * len(chunk))
            n_tokens.extend([int(attn[i].sum().item()) for i in range(len(chunk))])
            continue
        out = jmodel(input_ids=ids, attention_mask=attn)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask   = attn[:, 1:].contiguous().float()
        loss_per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="none",
        ).view(shift_labels.size())
        nll_sum = (loss_per_tok * shift_mask).sum(dim=1)
        denom   = shift_mask.sum(dim=1).clamp(min=1)
        ppl_per_sample = torch.exp(nll_sum / denom).cpu().tolist()
        n_tokens_per_sample = attn.sum(dim=1).cpu().tolist()
        for i in range(len(chunk)):
            ppls.append(float(ppl_per_sample[i]) if n_tokens_per_sample[i] >= 2 else float("nan"))
            n_tokens.append(int(n_tokens_per_sample[i]))
    return ppls, n_tokens
'''))

    # ─── Cell 13: Phase D — PPL ────────────────────────────────────────
    nb.cells.append(md("""
## 13. Phase D — GPT-2 PPL scoring per α → `ppl_scores_picks.parquet`

For each α: dedup unique explain_text strings, score under GPT-2 medium,
join with per-instance baseline PPL.
"""))
    nb.cells.append(code('''
for alpha, df in explain_dfs.items():
    ppl_pq = OUTPUT_DIR / TASK / MODEL_TAG / alpha / "ppl_scores_picks.parquet"
    if ppl_pq.exists():
        print(f"  α={alpha}: ppl_scores_picks cached at {ppl_pq.name}")
        continue

    print(f"\\n=== α={alpha} PPL scoring ===")
    et = df.copy()
    et["layer_set"] = et["layer_set"].apply(lambda x: tuple(int(l) for l in x) if x is not None else ())

    all_texts = et["explain_text"].fillna("").tolist()
    uniq_texts = list(dict.fromkeys(all_texts))
    print(f"  unique explain texts: {len(uniq_texts):,d} (from {len(all_texts):,d} rows)")

    t0 = time.perf_counter()
    ppls, ntoks = compute_ppl_batch(uniq_texts, judge_model, judge_tokenizer, judge_device,
                                      max_length=PPL_MAX_LENGTH, batch_size=PPL_BATCH_SIZE)
    elapsed = time.perf_counter() - t0
    print(f"  scored in {elapsed:.1f}s ({len(uniq_texts)/max(elapsed,1):.0f}/s)")

    text_map = dict(zip(uniq_texts, zip(ppls, ntoks)))

    base_rows = et[et["layer_set"].apply(lambda x: len(x) == 0)]
    base_ppl_by_iid = {}
    for _, row in base_rows.iterrows():
        ppl, _ = text_map[row["explain_text"] or ""]
        base_ppl_by_iid[int(row["instance_id"])] = ppl
    print(f"  baseline coverage: {len(base_ppl_by_iid)}/{et['instance_id'].nunique()} instances")

    out_rows = []
    for _, row in et.iterrows():
        ppl, ntok = text_map[row["explain_text"] or ""]
        iid = int(row["instance_id"])
        layer_set = row["layer_set"]
        is_baseline = (len(layer_set) == 0)
        base_ppl = base_ppl_by_iid.get(iid, float("nan"))
        out_rows.append({
            "instance_id":    iid,
            "layer_set":      layer_set,
            "K":              len(layer_set),
            "alpha_schedule": alpha,
            "judge_model":    JUDGE_MODEL,
            "picked_answer":  row.get("picked_answer", ""),
            "ppl":            ppl,
            "ppl_base":       base_ppl,
            "ppl_delta":      ppl - base_ppl if pd.notna(base_ppl) else float("nan"),
            "n_tokens":       ntok,
            "is_baseline":    is_baseline,
        })
    out_df = pd.DataFrame(out_rows)
    pq, csv = save_df(out_df, ppl_pq)
    print(f"  wrote {len(out_df)} rows → {pq.name}")

    steered = out_df[~out_df["is_baseline"]]
    if len(steered) > 0:
        print(f"    baseline ppl  mean={out_df[out_df['is_baseline']]['ppl'].mean():>8.2f}  "
              f"median={out_df[out_df['is_baseline']]['ppl'].median():>8.2f}")
        print(f"    steered ppl   mean={steered['ppl'].mean():>8.2f}  median={steered['ppl'].median():>8.2f}")
        print(f"    steered Δppl  mean={steered['ppl_delta'].mean():>+8.2f}  median={steered['ppl_delta'].median():>+8.2f}")
'''))

    # ─── Cell 14: Done ─────────────────────────────────────────────────
    nb.cells.append(md("""
## 14. Done

For `(task, model)` you should now have under
`oracle_outputs_test/{task}/{model}/`:

```
├── baseline.parquet
├── method_picks.parquet
├── {α}/
│   ├── K1_trivial.parquet
│   ├── K{2,3}_exhaustive.parquet
│   ├── K{2..5}_beam-W4.parquet
│   ├── method_picks_eval.parquet
│   ├── explain_texts_picks.parquet
│   └── ppl_scores_picks.parquet
└── (LN-global picks/eval under `ln_global_outputs_test/`)
```

Schemas match canonical the downstream analysis expectations.

**Next**: feed to the downstream analysis for the final ranking, pointing at
`oracle_outputs_test/` instead of `oracle_outputs/`.

**If preempted mid-K=3-exhaustive**: just re-run this notebook on the
next Kaggle session. Every phase checks for existing outputs and skips.
Bootstrap re-stages prior `oracle_outputs_test*` / `ln_global_outputs_test*`
datasets so completed work from prior sessions carries forward.
"""))

    return nb


if __name__ == "__main__":
    out_path = Path(__file__).parent / "oracle_full_pipeline_test_offline.ipynb"
    nb = build()
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {out_path} ({len(nb.cells)} cells)")
