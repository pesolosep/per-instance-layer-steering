"""Static K-sweep evaluation for the 6 final predictors at K ∈ {1..5}.

For each (variant, K, cell), pick the predictor's top-K layer set per instance,
look up its steered eval in method_picks_eval_with_gap.parquet, compute the
alignment lift = P_steered[target] - P_base[target] using TRUE gold (for the
metric only; no decisions depend on gold).

Outputs:
  results/final_predictor/static_k_per_cell.csv
  results/final_predictor/static_k_aggregate.csv

Coverage: K=1 is complete (all 32 single-layer picks evaluated upstream).
K=2..5 coverage is partial; per-cell missing-rate is reported.

Run:
    python scripts/eval_final_predictors_static_k.py
"""
import sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predictor_eda._util import (
    cells, pilot_dir, load_features, load_embeddings,
    load_baseline, load_k1_trivial, N_LAYERS, cell_label, softmax2,
    build_X_no_gold,
)

OUT_MODELS  = ROOT / "results/final_predictor/models"
OUT_DIR     = ROOT / "results/final_predictor"
ORACLE_TEST = ROOT / "results/analysis_inputs_test/oracle_outputs"

VARIANTS = [
    "embed_mlp_full_listwise",
    "embed_mlp_pca25_listwise",
    "embed_linear_listwise",
    "unsteered_full_listwise",
    "unsteered_plus_pca25_listwise",
    "unsteered_lite_filtered_plus_pca25_listwise",
]
K_VALUES = [1, 2, 3, 4, 5]


# ────────── Feature builders for test (must mirror training) ──────────
def features_embed_full_test(emb_t, iids):
    return emb_t.set_index("instance_id").loc[iids].values.astype(np.float32)


def features_embed_pca25_test(emb_t, iids, pca):
    emb_mat = emb_t.set_index("instance_id").loc[iids].values.astype(np.float32)
    return pca.transform(emb_mat).astype(np.float32)


def features_unsteered_full_test(feat_t, bl_t, k1_t, iids):
    X, _ = build_X_no_gold(
        feat_t, bl_t, k1_t, iids,
        include_geom=("proj_on_sv", "act_norm", "act_var", "steering_cosine"),
        include_d=(),
        include_proj_rank=True,
        include_inst=("p_yes", "p_no", "yn_mass", "entropy", "max_p", "abs_diff"),
    )
    return X.astype(np.float32)


def features_unsteered_lite_filtered_test(feat_t, bl_t, k1_t, iids):
    X, _ = build_X_no_gold(
        feat_t, bl_t, k1_t, iids,
        include_geom=("proj_on_sv", "act_norm"),
        include_d=(),
        include_proj_rank=False,
        include_inst=(),
    )
    return X.astype(np.float32)


# ────────── Predict per variant ──────────
def predict_scores(variant, jpath, emb_t, feat_t, bl_t, k1_t, iids):
    """Returns (n_instances, 32) predicted layer-score matrix."""
    payload = joblib.load(jpath)
    sc, reg = payload["scaler"], payload["regressor"]

    if variant == "embed_mlp_full_listwise":
        X = features_embed_full_test(emb_t, iids)
    elif variant == "embed_mlp_pca25_listwise":
        pca = payload["pca"]
        X = features_embed_pca25_test(emb_t, iids, pca)
    elif variant == "embed_linear_listwise":
        X = features_embed_full_test(emb_t, iids)
    elif variant == "unsteered_full_listwise":
        X = features_unsteered_full_test(feat_t, bl_t, k1_t, iids)
    elif variant == "unsteered_plus_pca25_listwise":
        pca = payload["pca"]
        X_un = features_unsteered_full_test(feat_t, bl_t, k1_t, iids)
        X_pca = features_embed_pca25_test(emb_t, iids, pca)
        X = np.concatenate([X_un, X_pca], axis=1)
    elif variant == "unsteered_lite_filtered_plus_pca25_listwise":
        pca = payload["pca"]
        X_lite = features_unsteered_lite_filtered_test(feat_t, bl_t, k1_t, iids)
        X_pca = features_embed_pca25_test(emb_t, iids, pca)
        X = np.concatenate([X_lite, X_pca], axis=1)
    else:
        raise ValueError(f"unknown variant: {variant}")

    Xs = sc.transform(X)

    # Handle PyTorch (MLPRanker, LinearRanker) and sklearn estimators uniformly
    import torch
    if hasattr(reg, "forward"):
        with torch.no_grad():
            scores = reg(torch.from_numpy(Xs).float()).cpu().numpy()
    else:
        scores = reg.predict(Xs)
    return scores


# ────────── Eval lookup ──────────
def load_eval_lookup(task, model, alpha):
    """Returns dict[(iid, layer_set_tuple)] → eval row (or None if missing)."""
    pq = ORACLE_TEST / task / model / alpha / "method_picks_eval_with_gap.parquet"
    if not pq.exists():
        return None
    ev = pd.read_parquet(pq)
    ev["layer_set"] = ev["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    ev = ev.drop_duplicates(subset=["instance_id", "layer_set"], keep="first")
    return ev.set_index(["instance_id", "layer_set"]).to_dict("index")


# ────────── Per-cell K-sweep ──────────
def run_cell(task, model_tag, alpha):
    cell_models = OUT_MODELS / task / model_tag / alpha
    if not cell_models.exists():
        return []

    feat_t = load_features(task, model_tag, alpha, "test")
    bl_t   = load_baseline(task, model_tag, alpha, "test")
    k1_t   = load_k1_trivial(task, model_tag, alpha, "test")  # used only by build_X_no_gold for instance feats
    emb_t  = load_embeddings(task, model_tag, alpha, "test")
    if any(x is None for x in [feat_t, bl_t, k1_t, emb_t]):
        return []

    ev_lookup = load_eval_lookup(task, model_tag, alpha)
    if ev_lookup is None:
        return []

    bl_by = bl_t.set_index("instance_id")
    iids = sorted(bl_by.index.astype(int).unique().tolist())
    iids = [i for i in iids if i in emb_t["instance_id"].astype(int).values]
    true_gold = bl_by["gold_sign"].astype(int).to_dict()

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

        for K in K_VALUES:
            lifts = []
            missing = 0
            for i, iid in enumerate(iids):
                ranked = np.argsort(-scores[i])
                ls = tuple(sorted(int(l) for l in ranked[:K]))
                ev = ev_lookup.get((iid, ls))
                if ev is None:
                    missing += 1
                    continue
                g = true_gold[iid]
                py_s, pn_s = softmax2(ev["steered_logit_yes"], ev["steered_logit_no"])
                py_b, pn_b = softmax2(ev["base_logit_yes"], ev["base_logit_no"])
                p_t_s = py_s if g == 1 else pn_s
                p_t_b = py_b if g == 1 else pn_b
                lifts.append(p_t_s - p_t_b)
            n_eval = len(lifts)
            mean_lift = float(np.mean(lifts)) if lifts else None
            sem_lift = float(np.std(lifts, ddof=1) / np.sqrt(n_eval)) if n_eval > 1 else None
            rows.append({
                "variant": variant, "K": K,
                "task": task, "model": model_tag, "alpha": alpha,
                "n_instances": len(iids),
                "n_eval_resolved": n_eval,
                "n_eval_missing": missing,
                "mean_lift": round(mean_lift, 4) if mean_lift is not None else None,
                "sem_lift": round(sem_lift, 4) if sem_lift is not None else None,
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
    df.to_csv(OUT_DIR / "static_k_per_cell.csv", index=False)
    print(f"\nWrote per-cell: {OUT_DIR / 'static_k_per_cell.csv'}  ({len(df)} rows)")

    # ── Aggregate per (variant, K) across 24 cells ──
    def sem(s):
        n = len(s)
        return s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

    agg = (df.groupby(["variant", "K"])
             .agg(n_cells=("task", "count"),
                  mean_lift=("mean_lift", "mean"),
                  sem_lift=("mean_lift", sem),
                  mean_missing_rate=("n_eval_missing", lambda s: s.sum() / (df.loc[s.index, "n_instances"].sum() or 1)))
             .round(4)
             .reset_index()
             .sort_values(["variant", "K"]))
    agg.to_csv(OUT_DIR / "static_k_aggregate.csv", index=False)
    print(f"Wrote aggregate: {OUT_DIR / 'static_k_aggregate.csv'}  ({len(agg)} rows)")

    # Print headline pivot
    pivot = agg.pivot(index="variant", columns="K", values="mean_lift")
    pivot = pivot.reindex(VARIANTS)
    print("\n=== Mean lift across 24 cells (apples-to-apples at each K) ===")
    print(pivot.to_string())

    miss_pivot = agg.pivot(index="variant", columns="K", values="mean_missing_rate")
    miss_pivot = miss_pivot.reindex(VARIANTS)
    print("\n=== Missing-eval rate per (variant, K) ===")
    print(miss_pivot.applymap(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "-").to_string())

    print(f"\nWall: {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
