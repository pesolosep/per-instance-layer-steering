"""Adaptive-K gate evaluation for the 6 final predictors.

For each variant × cell:
  1. Train an Emb-LogReg direction classifier on val Qwen3 PCA-25 → val gold_sign
     (OFFLINE supervision; identical to §4 direction-inference description).
  2. Apply to test → inferred direction per instance (NO test gold leak).
  3. Predict the variant's 32 layer scores on test.
  4. Walk the adaptive-K gate using INFERRED direction (gold-free at runtime).
  5. Compute alignment lift using TRUE gold (for the metric only — not for any decision).

Outputs:
  results/final_predictor/adaptive_per_cell.csv
  results/final_predictor/adaptive_aggregate.csv

Run:
    python scripts/eval_final_predictors_adaptive.py
"""
import sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predictor_eda._util import (
    cells, load_features, load_embeddings, load_baseline, load_k1_trivial,
    N_LAYERS, cell_label, softmax2, build_X_no_gold,
)
from scripts.adaptive_k_across_predictors import predict_top_K_picks, gate_walk
from scripts.eval_final_predictors_static_k import (
    predict_scores, load_eval_lookup, VARIANTS,
)

OUT_MODELS  = ROOT / "results/final_predictor/models"
OUT_DIR     = ROOT / "results/final_predictor"
ORACLE_VAL  = ROOT / "results/analysis_inputs/oracle_outputs"

# Gate config (matches eval_emb_direction_vs_compass.py)
GATE_CFG = {
    "k_min": 1, "k_max_high_bp": 5, "k_max_low_bp": 3,
    "base_prob_cutoff": 0.5, "yn_mass_floor": 0.3,
    "delta_yn": 0.30, "eps_lift": 0.001,
}

# STEP2: accumulator for the DEPLOYED direction-classifier probabilities (per instance,
# per cell). Captured at inference time so AUC/calibration need no offline re-fit.
DIR_PROBA_ROWS = []


def train_emb_direction_classifier(task, model_tag, alpha, n_pc=25):
    """Train LogReg(Qwen3 PCA-25 → gold_sign) on val. OFFLINE supervision only."""
    emb_v = pd.read_parquet(ROOT / f"results/predictor_pilot/{task}/{model_tag}/{alpha}/embeddings_val.parquet")
    bl_v  = pd.read_parquet(ORACLE_VAL / task / model_tag / "baseline.parquet")
    iids_v = sorted(emb_v["instance_id"].astype(int).unique().tolist())
    X_v = emb_v.set_index("instance_id").loc[iids_v].values.astype(np.float32)
    y_v = bl_v.set_index("instance_id").loc[iids_v]["gold_sign"].astype(int).values
    pca = PCA(n_components=min(n_pc, X_v.shape[1], X_v.shape[0] - 1)).fit(X_v)
    X_v_pc = pca.transform(X_v)
    sc = StandardScaler().fit(X_v_pc)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X_v_pc), y_v)
    return clf, pca, sc


def run_cell(task, model_tag, alpha):
    cell_models = OUT_MODELS / task / model_tag / alpha
    if not cell_models.exists():
        return []

    feat_t = load_features(task, model_tag, alpha, "test")
    bl_t   = load_baseline(task, model_tag, alpha, "test")
    k1_t   = load_k1_trivial(task, model_tag, alpha, "test")
    emb_t  = load_embeddings(task, model_tag, alpha, "test")
    if any(x is None for x in [feat_t, bl_t, k1_t, emb_t]):
        return []

    ev_lookup = load_eval_lookup(task, model_tag, alpha)
    if ev_lookup is None:
        return []

    bl_by = bl_t.set_index("instance_id")
    iids_emb_set = set(emb_t["instance_id"].astype(int).values)
    iids = sorted([int(i) for i in bl_by.index.astype(int).unique() if int(i) in iids_emb_set])
    true_gold = bl_by["gold_sign"].astype(int).to_dict()

    # ── Train Emb-LogReg direction classifier (offline, val) ──
    try:
        dir_clf, dir_pca, dir_sc = train_emb_direction_classifier(task, model_tag, alpha)
    except Exception as e:
        print(f"    DIR-LR FAIL: {e}")
        return []

    # ── Apply direction classifier to test (gold-free at inference) ──
    emb_t_mat = emb_t.set_index("instance_id").loc[iids].values.astype(np.float32)
    emb_t_pc  = dir_pca.transform(emb_t_mat)
    dir_pred  = dir_clf.predict(dir_sc.transform(emb_t_pc))
    inferred_gold = {int(iid): int(g) for iid, g in zip(iids, dir_pred)}

    # STEP2: persist the deployed direction-classifier probabilities (per instance, per cell).
    # Once per cell (classifier is method-independent), not per variant.
    try:
        _pos = list(dir_clf.classes_).index(1)
        _proba = dir_clf.predict_proba(dir_sc.transform(emb_t_pc))[:, _pos]
        for _iid, _g, _p in zip(iids, dir_pred, _proba):
            DIR_PROBA_ROWS.append({"task": task, "model": model_tag, "alpha": alpha,
                "split": "test", "instance_id": int(_iid), "inferred_gold": int(_g),
                "proba_pos": float(_p), "true_gold": int(true_gold[_iid])})
    except Exception:
        pass

    rows = []
    for variant in VARIANTS:
        jpath = cell_models / f"{variant}.joblib"
        if not jpath.exists():
            continue
        try:
            scores = predict_scores(variant, jpath, emb_t, feat_t, bl_t, k1_t, iids)
        except Exception as e:
            print(f"    PRED FAIL {variant}: {e}")
            continue

        picks_by_K = predict_top_K_picks(scores, iids, alpha)

        # Gate walk uses INFERRED gold (no leak at inference)
        walk_df = gate_walk(picks_by_K, ev_lookup, bl_t, inferred_gold, GATE_CFG)

        # Compute honest lift using TRUE gold (metric only)
        # Also tally fired-reason to expose missing-eval forced fallback
        honest_lifts, K_chosens, dir_correct = [], [], []
        fired_counts = {"plateau": 0, "yn_floor": 0, "delta_yn_backoff": 0,
                        "k_max_reached": 0, "missing_eval": 0, "none": 0}
        for _, w in walk_df.iterrows():
            iid = int(w["instance_id"])
            K_chosen = int(w["K_chosen"]) if w["K_chosen"] != -1 else None
            fired = w.get("fired", "none")
            fired_counts[fired] = fired_counts.get(fired, 0) + 1
            if K_chosen is None:
                continue
            gate_set = picks_by_K[K_chosen].get(iid)
            ev = ev_lookup.get((iid, gate_set))
            if ev is None:
                continue
            g_true = true_gold[iid]
            g_inf  = inferred_gold[iid]
            py_b, pn_b = softmax2(ev["base_logit_yes"], ev["base_logit_no"])
            py_s, pn_s = softmax2(ev["steered_logit_yes"], ev["steered_logit_no"])
            base_prob_true = py_b if g_true == 1 else pn_b
            p_t_steered    = py_s if g_true == 1 else pn_s
            honest_lifts.append(p_t_steered - base_prob_true)
            K_chosens.append(K_chosen)
            dir_correct.append(g_inf == g_true)

        n_eval = len(honest_lifts)
        rows.append({
            "variant": variant,
            "task": task, "model": model_tag, "alpha": alpha,
            "n_instances": len(iids),
            "n_resolved": n_eval,
            "n_missing": len(iids) - n_eval,
            "mean_K_adaptive":  round(float(np.mean(K_chosens)), 3) if K_chosens else None,
            "mean_lift_adaptive": round(float(np.mean(honest_lifts)), 4) if honest_lifts else None,
            "sem_lift_adaptive": round(float(np.std(honest_lifts, ddof=1) / np.sqrt(n_eval)), 4) if n_eval > 1 else None,
            "emb_dir_acc": round(float(np.mean(dir_correct)), 4) if dir_correct else None,
            "fired_plateau": fired_counts["plateau"],
            "fired_yn_floor": fired_counts["yn_floor"],
            "fired_backoff": fired_counts["delta_yn_backoff"],
            "fired_k_max": fired_counts["k_max_reached"],
            "fired_missing_eval": fired_counts["missing_eval"],
        })
    return rows


def main():
    t0 = time.time()
    all_rows = []
    cell_list = list(cells())
    for i, (task, model_tag, alpha) in enumerate(cell_list):
        try:
            rs = run_cell(task, model_tag, alpha)
        except Exception as e:
            import traceback; traceback.print_exc()
            continue
        all_rows.extend(rs)
        elapsed = (time.time() - t0) / 60
        print(f"  {i+1}/{len(cell_list)}  {cell_label(task, model_tag, alpha)}  +{len(rs)} rows  ({elapsed:.1f}m)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "adaptive_per_cell.csv", index=False)
    print(f"\nWrote per-cell: {OUT_DIR / 'adaptive_per_cell.csv'}  ({len(df)} rows)")

    # STEP2: persist deployed direction-classifier probabilities (per instance) so
    # AUC/calibration are exact next run (no offline re-fit needed).
    if DIR_PROBA_ROWS:
        dp = pd.DataFrame(DIR_PROBA_ROWS).drop_duplicates(["task", "model", "alpha", "instance_id"])
        dp.to_parquet(OUT_DIR / "direction_classifier_proba_per_instance.parquet", index=False)
        print(f"Wrote direction-classifier proba: {OUT_DIR / 'direction_classifier_proba_per_instance.parquet'}  ({len(dp)} rows)")

    def sem(s):
        n = len(s)
        return s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

    agg = (df.groupby("variant")
             .agg(n_cells=("task", "count"),
                  mean_K_adaptive=("mean_K_adaptive", "mean"),
                  mean_lift_adaptive=("mean_lift_adaptive", "mean"),
                  sem_lift_adaptive=("mean_lift_adaptive", sem),
                  mean_emb_dir_acc=("emb_dir_acc", "mean"))
             .reindex(VARIANTS)
             .round(4))
    agg.to_csv(OUT_DIR / "adaptive_aggregate.csv")
    print(f"Wrote aggregate: {OUT_DIR / 'adaptive_aggregate.csv'}  ({len(agg)} rows)")
    print("\n=== Adaptive-K with Emb-LogReg direction (24-cell mean) ===")
    print(agg.to_string())
    print(f"\nWall: {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
