"""End-to-end GPU pipeline: predict picks for 12 deployable variants × 24 cells,
gap-fill steering eval (logits + 200-token explain text + PPL).

Pipeline per (task, model_tag) — model loaded ONCE per task+model:
  1. Compute baseline on test (cached at oracle/{task}/{model_tag}/baseline_test.parquet)
  2. For each alpha (uniform, sqrt-norm):
     a. Load 12 variant joblibs from results/final_predictor/models/{task}/{model_tag}/{alpha}/
     b. Train Emb-LogReg direction classifier on val (offline)
     c. For each variant: predict scores on test, derive top-K layer sets (K=1..5)
        and adaptive-K via gate_walk with Emb-LogReg inferred direction
     d. Collect unique (instance, layer_set) pairs needed → diff against existing parquets
     e. Call evaluate_chosen with the missing layer_sets (one batched eval per layer_set:
        full-vocab forward + 200-token generation + PPL)
     f. Append rows to results/final_predictor/gapfill/{task}/{model_tag}/{alpha}.parquet

Idempotent: re-runs only what's missing in BOTH the original method_picks_eval_with_gap.parquet
AND the gapfill parquet.

Discord progress: writes JSON progress to /tmp/final_predictor_progress.json; a separate
discord_progress_worker.py picks it up.

Env vars:
  DISCORD_WEBHOOK_URL  Discord webhook for progress (optional)
  LAYER_SELECTION_ROOT (default: cwd-detected) — repo root

Run on remote GPU:
    DISCORD_WEBHOOK_URL=https://... python scripts/run_final_predictor_full_eval.py
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get(
    "LAYER_SELECTION_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from globalenv import LLAMA_8B, AYA_8B
from model_wrapper import ModelWrapper
from dataset import UniDataset
from oracle.io import load_df, save_df
from oracle.runner_impl import compute_baseline, evaluate_chosen
from oracle.scoring import multimetric_3

from scripts.predictor_eda._util import (
    cells, load_features, load_embeddings, load_baseline, load_k1_trivial,
    N_LAYERS, cell_label, softmax2,
)
from scripts.adaptive_k_across_predictors import predict_top_K_picks, gate_walk
from scripts.eval_final_predictors_static_k import predict_scores, load_eval_lookup
from scripts.eval_final_predictors_adaptive import train_emb_direction_classifier, GATE_CFG
from scripts.final_predictor_models import MLPRanker, LinearRanker

MODEL_PRESETS = {"llama-3-8b": LLAMA_8B, "aya-expanse-8b": AYA_8B}

OUT_MODELS = ROOT / "results/final_predictor/models"
OUT_GAPFILL = ROOT / "results/final_predictor/gapfill"
ORACLE_TEST = ROOT / "results/analysis_inputs_test/oracle_outputs"
ANALYSIS_INPUTS_TEST = ROOT / "results/analysis_inputs_test"

VARIANTS_ARCH = [
    "embed_mlp_full",
    "embed_mlp_pca25",
    "embed_linear",
    "unsteered_full",
    "unsteered_plus_pca25",
    "unsteered_lite_filtered_plus_pca25",
]
TARGETS = ["align", "multi3"]
VARIANTS = [f"{arch}_listwise_{t}" for arch in VARIANTS_ARCH for t in TARGETS]  # 12 IDs

K_VALUES = [1, 2, 3, 4, 5]

PROGRESS_FILE = Path("/tmp/final_predictor_progress.json")
MAX_NEW_TOKENS = 200


# ────────── Progress reporting ──────────
_start_time = time.time()


def write_progress(phase: str, message: str = "", **progress_kwargs) -> None:
    """Write JSON progress for the Discord worker to pick up."""
    elapsed_min = (time.time() - _start_time) / 60
    payload = {
        "phase": phase,
        "message": message,
        "host": socket.gethostname(),
        "progress": {
            "elapsed_min": round(elapsed_min, 1),
            **progress_kwargs,
        },
        "last_update_ts": time.time(),
    }
    try:
        PROGRESS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"[progress] write failed: {e}", flush=True)
    print(f"[{phase}] {message}", flush=True)


# ────────── Adapter: predict_scores signature reuses test infra ──────────
def _normalise_layer_set(ls) -> tuple:
    return tuple(sorted(int(l) for l in ls))


def _load_variant_predictions(variant_id, jpath, emb_t, feat_t, bl_t, k1_t, iids):
    """Returns (n_instances, 32) predicted layer-score matrix."""
    # predict_scores takes a `variant` keyword matching the architecture id
    arch = "_".join(variant_id.split("_")[:-2])  # strip "_listwise_<target>"
    # Map our arch IDs to the names predict_scores understands
    arch_map = {
        "embed_mlp_full": "embed_mlp_full_listwise",
        "embed_mlp_pca25": "embed_mlp_pca25_listwise",
        "embed_linear": "embed_linear_listwise",
        "unsteered_full": "unsteered_full_listwise",
        "unsteered_plus_pca25": "unsteered_plus_pca25_listwise",
        "unsteered_lite_filtered_plus_pca25": "unsteered_lite_filtered_plus_pca25_listwise",
    }
    legacy_arch = arch_map[arch]
    return predict_scores(legacy_arch, jpath, emb_t, feat_t, bl_t, k1_t, iids)


# ────────── Predict all picks per cell ──────────
def collect_picks_for_cell(task, model_tag, alpha):
    """Returns dict {variant_id: {"top_k_picks": {K: {iid: layer_set}}, "adaptive": {iid: layer_set}}}.

    Also returns the set of all unique layer_sets across all variants × K + adaptive.
    """
    cell_models = OUT_MODELS / task / model_tag / alpha
    if not cell_models.exists():
        return {}, set()

    feat_t = load_features(task, model_tag, alpha, "test")
    bl_t   = load_baseline(task, model_tag, alpha, "test")
    k1_t   = load_k1_trivial(task, model_tag, alpha, "test")
    emb_t  = load_embeddings(task, model_tag, alpha, "test")
    if any(x is None for x in [feat_t, bl_t, k1_t, emb_t]):
        return {}, set()

    bl_by = bl_t.set_index("instance_id")
    iids_emb = set(emb_t["instance_id"].astype(int).values)
    iids = sorted([int(i) for i in bl_by.index.astype(int).unique() if int(i) in iids_emb])

    # Train Emb-LogReg direction classifier (gold-free at inference)
    try:
        dir_clf, dir_pca, dir_sc = train_emb_direction_classifier(task, model_tag, alpha)
    except Exception as e:
        print(f"[predict] DIR-LR FAIL {task}/{model_tag}/{alpha}: {e}", flush=True)
        return {}, set()
    emb_t_mat = emb_t.set_index("instance_id").loc[iids].values.astype(np.float32)
    emb_t_pc  = dir_pca.transform(emb_t_mat)
    dir_pred  = dir_clf.predict(dir_sc.transform(emb_t_pc))
    inferred_gold = {int(iid): int(g) for iid, g in zip(iids, dir_pred)}

    # Need ev_lookup for the gate to function (it consults eval rows for stopping rules);
    # use the existing parquet — gate_walk will mark missing_eval where coverage gaps exist.
    ev_lookup = load_eval_lookup(task, model_tag, alpha) or {}

    picks_by_variant = {}
    unique_layer_sets = set()
    for variant_id in VARIANTS:
        jpath = cell_models / f"{variant_id}.joblib"
        if not jpath.exists():
            print(f"[predict] MISSING joblib: {variant_id} @ {task}/{model_tag}/{alpha}", flush=True)
            continue
        try:
            scores = _load_variant_predictions(variant_id, jpath, emb_t, feat_t, bl_t, k1_t, iids)
        except Exception as e:
            print(f"[predict] FAIL {variant_id}: {e}", flush=True)
            continue

        picks_top_k = {K: {} for K in K_VALUES}
        for i, iid in enumerate(iids):
            ranked = np.argsort(-scores[i])
            for K in K_VALUES:
                ls = _normalise_layer_set(ranked[:K])
                picks_top_k[K][int(iid)] = ls
                unique_layer_sets.add(ls)

        # Adaptive-K picks via gate_walk with Emb-LogReg inferred direction
        picks_by_K_for_gate = picks_top_k
        adaptive_picks = {}
        if ev_lookup:
            walk_df = gate_walk(picks_by_K_for_gate, ev_lookup, bl_t, inferred_gold, GATE_CFG)
            for _, w in walk_df.iterrows():
                iid = int(w["instance_id"])
                K_chosen = int(w["K_chosen"]) if w["K_chosen"] != -1 else 1
                ls = picks_by_K_for_gate[K_chosen].get(iid)
                if ls is not None:
                    adaptive_picks[iid] = ls
                    unique_layer_sets.add(ls)

        picks_by_variant[variant_id] = {
            "top_k": picks_top_k,
            "adaptive": adaptive_picks,
        }

    return picks_by_variant, unique_layer_sets


# ────────── Gap detection ──────────
def existing_layer_sets(task, model_tag, alpha):
    """Returns set of (iid, layer_set_tuple) already evaluated in either parquet."""
    covered = set()

    # 1. method_picks_eval_with_gap.parquet (original)
    pq1 = ORACLE_TEST / task / model_tag / alpha / "method_picks_eval_with_gap.parquet"
    if pq1.exists():
        df = pd.read_parquet(pq1)
        df["layer_set"] = df["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
        for iid, ls in df[["instance_id", "layer_set"]].itertuples(index=False):
            covered.add((int(iid), ls))

    # 2. Our own gapfill parquet (resume support)
    pq2 = OUT_GAPFILL / task / model_tag / f"{alpha}.parquet"
    if pq2.exists():
        df = pd.read_parquet(pq2)
        df["layer_set"] = df["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
        for iid, ls in df[["instance_id", "layer_set"]].itertuples(index=False):
            covered.add((int(iid), ls))

    return covered


def compute_missing_layer_sets(picks_by_variant, covered):
    """Returns sorted list of unique layer_sets that need evaluation.

    A layer_set is missing if there exists ANY (iid, layer_set) pair predicted
    by any variant that isn't in `covered`. Since evaluate_chosen runs ALL
    instances per layer_set at once, we only need the unique layer_sets.
    """
    needed_pairs = set()
    for variant_id, picks in picks_by_variant.items():
        for K, k_picks in picks["top_k"].items():
            for iid, ls in k_picks.items():
                needed_pairs.add((int(iid), ls))
        for iid, ls in picks["adaptive"].items():
            needed_pairs.add((int(iid), ls))

    missing_pairs = needed_pairs - covered
    missing_layer_sets = sorted({ls for _, ls in missing_pairs}, key=lambda x: (len(x), x))
    return missing_layer_sets, len(needed_pairs), len(missing_pairs)


# ────────── Per-cell gap-fill ──────────
def run_cell_gapfill(task, model_tag, alpha, model, t_yes, t_no, baseline, test_dataset):
    """Identify missing layer_sets and call evaluate_chosen for them. Append to gapfill parquet."""
    picks_by_variant, unique_layer_sets = collect_picks_for_cell(task, model_tag, alpha)
    if not picks_by_variant:
        print(f"[gapfill] no picks for {task}/{model_tag}/{alpha} — skip", flush=True)
        return None

    covered = existing_layer_sets(task, model_tag, alpha)
    missing_ls, n_needed, n_missing_pairs = compute_missing_layer_sets(picks_by_variant, covered)

    print(f"[gapfill] {task}/{model_tag}/{alpha}: "
          f"{len(unique_layer_sets)} unique layer_sets predicted, "
          f"{n_needed} (iid, layer_set) pairs needed, "
          f"{n_missing_pairs} pairs missing, "
          f"{len(missing_ls)} unique layer_sets to evaluate", flush=True)

    if not missing_ls:
        print(f"[gapfill] all layer_sets covered — nothing to do", flush=True)
        return picks_by_variant

    # Call evaluate_chosen for the missing layer_sets in CHUNKS so we save
    # intermediate progress (recoverable on crash) and can monitor per-chunk
    # throughput live. Chunk size kept small so first save lands within minutes.
    out_dir = OUT_GAPFILL / task / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pq = out_dir / f"{alpha}.parquet"

    CHUNK_SIZE = 20  # ~ minutes per chunk on Blackwell at FA2 + bs=100
    chunks = [missing_ls[i:i+CHUNK_SIZE] for i in range(0, len(missing_ls), CHUNK_SIZE)]
    print(f"[gapfill] splitting {len(missing_ls)} layer_sets into {len(chunks)} chunks of {CHUNK_SIZE}", flush=True)

    t_cell_start = time.time()
    total_rows = 0
    for chunk_idx, chunk in enumerate(chunks):
        t_chunk = time.time()
        try:
            chunk_df = evaluate_chosen(
                model=model, dataset=test_dataset, tokenizer=model.tokenizer,
                chosen_layer_sets=chunk,
                alpha_schedule=alpha,
                task=task, method="md",
                baseline=baseline,
                t_yes=t_yes, t_no=t_no,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[gapfill] chunk {chunk_idx+1}/{len(chunks)} FAILED: {e}", flush=True)
            continue

        chunk_df["layer_set"] = chunk_df["layer_set"].apply(_normalise_layer_set)
        # Append to gapfill parquet
        if out_pq.exists():
            existing = pd.read_parquet(out_pq)
            existing["layer_set"] = existing["layer_set"].apply(_normalise_layer_set)
            combined = pd.concat([existing, chunk_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["instance_id", "layer_set"], keep="last")
        else:
            combined = chunk_df
        save_df(combined, out_pq)
        total_rows = len(combined)

        chunk_elapsed = time.time() - t_chunk
        cell_elapsed = (time.time() - t_cell_start) / 60
        per_ls = chunk_elapsed / max(len(chunk), 1)
        remaining_ls = len(missing_ls) - (chunk_idx + 1) * CHUNK_SIZE
        eta_cell_min = remaining_ls * per_ls / 60 if remaining_ls > 0 else 0
        print(f"[gapfill] chunk {chunk_idx+1:3d}/{len(chunks)} ({len(chunk)} layer_sets) "
              f"done in {chunk_elapsed:.0f}s ({per_ls:.1f}s/ls) | "
              f"cell {cell_elapsed:.1f}m | ETA-cell {eta_cell_min:.1f}m | "
              f"parquet {total_rows} rows", flush=True)
        write_progress("gapfill",
                       f"{task}/{model_tag}/{alpha} chunk {chunk_idx+1}/{len(chunks)}",
                       task=task, model=model_tag, alpha=alpha,
                       chunks_done=chunk_idx+1, chunks_total=len(chunks),
                       per_layer_set_s=round(per_ls, 1),
                       cell_elapsed_min=round(cell_elapsed, 1),
                       eta_cell_min=round(eta_cell_min, 1))

    return picks_by_variant


# ────────── Main per-task loop ──────────
def run_task_model(task, model_tag, cfg_alphas=("uniform", "sqrt-norm"),
                    attn_impl: str = "flash_attention_2", dtype=torch.bfloat16,
                    batch_size: int = 100):
    """Load model once, iterate over alphas."""
    preset = MODEL_PRESETS[model_tag]
    write_progress("loading_model",
                   f"loading {preset.name} ({attn_impl}, bs={batch_size}) on {task}/{model_tag}",
                   task=task, model=model_tag, attn=attn_impl, batch_size=batch_size)

    model = ModelWrapper(preset, dtype=dtype, attn_impl=attn_impl)
    t_yes = preset.yn_token_ids[" Yes"]
    t_no  = preset.yn_token_ids[" No"]

    test_dataset = UniDataset(preset, task=task, train=False, set="test",
                              batch_size=batch_size, tokenizer=model.tokenizer)
    n_test = len(test_dataset.keys)
    write_progress("test_baseline",
                   f"computing test baseline on {task}/{model_tag} (n_test={n_test})",
                   task=task, model=model_tag, n_test=n_test)

    # Test baseline (cached)
    test_baseline_pq = ANALYSIS_INPUTS_TEST / "oracle_outputs" / task / model_tag / "baseline_test_full.parquet"
    if test_baseline_pq.exists():
        # Reload baseline from disk
        bdf = pd.read_parquet(test_baseline_pq)
        baseline = {
            "z_yes": torch.tensor(bdf["base_logit_yes"].values),
            "z_no":  torch.tensor(bdf["base_logit_no"].values),
            "align_prob": torch.tensor(bdf["base_align_prob"].values),
            "logit_margin": torch.tensor(bdf["base_logit_margin"].values),
            "yn_mass": torch.tensor(bdf["base_yn_mass"].values),
            "gold_signs": torch.tensor(bdf["gold_sign"].values),
            "texts": bdf["base_text"].tolist(),
        }
        print(f"[baseline] reloaded from cache: {test_baseline_pq}", flush=True)
    else:
        baseline = compute_baseline(
            model=model, dataset=test_dataset,
            t_yes=t_yes, t_no=t_no, generate_text=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        baseline_df = pd.DataFrame({
            "instance_id": list(range(len(baseline["gold_signs"]))),
            "base_logit_yes": baseline["z_yes"].tolist(),
            "base_logit_no":  baseline["z_no"].tolist(),
            "base_align_prob": baseline["align_prob"].tolist(),
            "base_logit_margin": baseline["logit_margin"].tolist(),
            "base_yn_mass": baseline["yn_mass"].tolist(),
            "gold_sign": baseline["gold_signs"].tolist(),
            "base_text": baseline.get("texts", [""] * len(baseline["gold_signs"])),
        })
        save_df(baseline_df, test_baseline_pq)
        print(f"[baseline] computed and saved: {test_baseline_pq}", flush=True)

    # Per-alpha gap-fill
    for alpha in cfg_alphas:
        write_progress("gapfill",
                       f"gapfill {task}/{model_tag}/{alpha}",
                       task=task, model=model_tag, alpha=alpha)
        run_cell_gapfill(task, model_tag, alpha, model, t_yes, t_no, baseline, test_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="subset of tasks; default all 6")
    parser.add_argument("--models", nargs="+", default=["llama-3-8b", "aya-expanse-8b"])
    parser.add_argument("--alphas", nargs="+", default=["uniform", "sqrt-norm"])
    parser.add_argument("--attn", default="flash_attention_2",
                        help="attn implementation; flash_attention_2 (default) | sdpa | eager")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="UniDataset batch size; default 100 (== test_100 single batch)")
    args = parser.parse_args()

    from scripts.predictor_eda._util import TASKS
    tasks = args.tasks or TASKS

    write_progress("starting",
                   f"final_predictor full eval across {len(tasks)} tasks × {len(args.models)} models",
                   tasks=str(tasks), models=str(args.models), alphas=str(args.alphas))

    try:
        for task in tasks:
            for model_tag in args.models:
                try:
                    run_task_model(task, model_tag, cfg_alphas=args.alphas,
                                   attn_impl=args.attn, batch_size=args.batch_size)
                except Exception as e:
                    traceback.print_exc()
                    write_progress("error",
                                   f"FAILED {task}/{model_tag}: {e}",
                                   task=task, model=model_tag, error=str(e))
                    continue
    except KeyboardInterrupt:
        write_progress("error", "Interrupted by user", final=True)
        raise
    except Exception as e:
        traceback.print_exc()
        write_progress("error", f"Unhandled exception: {e}", final=True)
        raise

    write_progress("done", "all tasks complete", final=True)


if __name__ == "__main__":
    main()
