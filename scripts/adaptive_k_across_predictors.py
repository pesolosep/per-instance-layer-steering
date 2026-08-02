"""Run adaptive-K gate walking for all 6 predictor variants across 24 cells.

For each (cell, predictor):
  1. Compute the predictor's K=1..5 top-K picks per test instance
     - clf_reg_full / clf_reg_cheap: use labeled base_prob (oracle config)
     - clf_reg_full_compass / clf_reg_cheap_compass: same predictor with compass-inferred base_prob
     - clf_reg_w2s / clf_reg_no_gold: target-agnostic
  2. Walk the adaptive-K gate using gate_config.json from compass-calibrated thresholds
  3. Record chosen K + lift per instance
  4. Aggregate per (cell, predictor) → mean adaptive K used, mean adaptive lift

Compare against the fixed-K=1 result from the prior session.
"""
import sys, json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import os
ROOT = Path(os.environ.get(
    "LAYER_SELECTION_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
sys.path.insert(0, str(ROOT))

TASKS = ["desire-to-create-allies", "conscientiousness", "cognitive-enhancement",
         "believes-it-has-phenomenal-consciousness", "desire-to-maximize-impact-on-world",
         "subscribes-to-Christianity"]
MODELS = ["llama-3-8b", "aya-expanse-8b"]
ALPHAS = ["uniform", "sqrt-norm"]
N_LAYERS = 32
PREDICTORS = ["clf_reg_full", "clf_reg_full_compass",
              "clf_reg_cheap", "clf_reg_cheap_compass",
              "clf_reg_w2s", "clf_reg_no_gold"]

OUT = ROOT / "results/predictor_pilot/adaptive_k_across_predictors"
OUT.mkdir(parents=True, exist_ok=True)


def softmax2(a, b):
    m = max(a, b); ea = np.exp(a-m); eb = np.exp(b-m); z = ea+eb
    return ea/z, eb/z


def calibrate_compass(K1_val_pq, bl_val_pq):
    K1 = pd.read_parquet(K1_val_pq); bl = pd.read_parquet(bl_val_pq)
    g = bl.set_index("instance_id")["gold_sign"].to_dict()
    K1["layer"] = K1["layer_set"].apply(lambda x: int(x[0]))
    K1["true_gold"] = K1["instance_id"].map(g)
    K1["compass"] = np.sign((K1["steered_logit_yes"] - K1["steered_logit_no"])
                              - (K1["base_logit_yes"] - K1["base_logit_no"]))
    acc = K1.groupby("layer").apply(
        lambda s: (s["compass"] == s["true_gold"]).mean(), include_groups=False)
    return [int(l) for l in acc.sort_values(ascending=False).head(3).index]


def infer_compass_gold(K1_test_pq, bl_test, top3):
    K1 = pd.read_parquet(K1_test_pq)
    K1["layer"] = K1["layer_set"].apply(lambda x: int(x[0]))
    K1["compass"] = np.sign((K1["steered_logit_yes"] - K1["steered_logit_no"])
                              - (K1["base_logit_yes"] - K1["base_logit_no"]))
    vp = K1.pivot_table(index="instance_id", columns="layer", values="compass", aggfunc="first")
    bl_w = bl_test.copy()
    bl_w["argmax_g"] = np.where(bl_w["base_logit_yes"] > bl_w["base_logit_no"], 1, -1)
    argmax = bl_w.set_index("instance_id")["argmax_g"].to_dict()
    out = {}
    for iid in vp.index:
        votes = vp.loc[iid, top3].values.astype(int)
        s = int(np.sum(votes))
        out[int(iid)] = 1 if s > 0 else (-1 if s < 0 else argmax.get(int(iid), 1))
    return out


def pivot_geom_bp(feat_df, iids, layer_cols, bp_lookup):
    blocks = []
    for col in layer_cols:
        b = (feat_df.pivot_table(index="instance_id", columns="layer", values=col, aggfunc="first")
                    .reindex(index=iids, columns=list(range(N_LAYERS))).fillna(0.0).values)
        blocks.append(b)
    proj_block = blocks[0]
    proj_rank = (-proj_block).argsort(axis=1).argsort(axis=1).astype(np.float32)
    blocks.append(proj_rank)
    base_prob = np.array([bp_lookup[iid] for iid in iids], dtype=np.float32)
    headroom = 1.0 - base_prob
    return np.column_stack([base_prob, headroom, np.concatenate(blocks, axis=1)]).astype(np.float32)


def predict_top_K_picks(scores, iids, alpha):
    """scores: (n_inst, 32). Return picks dict {K: {iid: layer_set}}."""
    out = {K: {} for K in [1, 2, 3, 4, 5]}
    for i, iid in enumerate(iids):
        ranked = np.argsort(-scores[i])
        for K in [1, 2, 3, 4, 5]:
            out[K][int(iid)] = tuple(sorted(int(l) for l in ranked[:K]))
    return out


def gate_walk(picks_by_K, ev_lookup, bl_test, gold_lookup, cfg):
    """For each instance, walk K=1..K_max with the gate rules. Return per-instance result."""
    K_MIN, K_MAX_HI, K_MAX_LO = cfg["k_min"], cfg["k_max_high_bp"], cfg["k_max_low_bp"]
    BP_CUT, YN_FLOOR = cfg["base_prob_cutoff"], cfg["yn_mass_floor"]
    DELTA_YN, EPS_LIFT = cfg["delta_yn"], cfg["eps_lift"]
    bl_idx = bl_test.set_index("instance_id")
    results = []
    iids = sorted(picks_by_K[1].keys())
    for iid in iids:
        g = gold_lookup[iid]
        bp_row = bl_idx.loc[iid]
        py_b, pn_b = softmax2(bp_row["base_logit_yes"], bp_row["base_logit_no"])
        base_prob = py_b if g == 1 else pn_b
        k_max = K_MAX_HI if base_prob >= BP_CUT else K_MAX_LO
        best_k, best_lift, best_yn = None, -np.inf, None
        prev_yn, prev_lift = None, None
        fired = "none"
        for k in range(K_MIN, k_max + 1):
            layer_set = picks_by_K[k].get(iid)
            if layer_set is None: break
            ev = ev_lookup.get((iid, layer_set))
            if ev is None:
                fired = "missing_eval"; break
            py_s, pn_s = softmax2(ev["steered_logit_yes"], ev["steered_logit_no"])
            p_t = py_s if g == 1 else pn_s
            lift = p_t - base_prob
            yn = ev["steered_yn_mass"]
            if yn < YN_FLOOR: fired = "yn_floor"; break
            if prev_yn is not None and (yn - prev_yn) < -DELTA_YN: fired = "delta_yn_backoff"; break
            if prev_lift is not None and (lift - prev_lift) < EPS_LIFT:
                if lift > best_lift: best_k, best_lift, best_yn = k, lift, yn
                fired = "plateau"; break
            if lift > best_lift: best_k, best_lift, best_yn = k, lift, yn
            prev_yn, prev_lift = yn, lift
        else:
            fired = "k_max_reached"
        if best_k is None:
            r0 = picks_by_K[K_MIN].get(iid)
            ev0 = ev_lookup.get((iid, r0)) if r0 else None
            if ev0 is not None:
                py0, pn0 = softmax2(ev0["steered_logit_yes"], ev0["steered_logit_no"])
                best_k, best_lift = K_MIN, ((py0 if g==1 else pn0) - base_prob)
        results.append({"instance_id": iid, "K_chosen": best_k if best_k else -1,
                        "lift": best_lift if best_k else np.nan, "fired": fired})
    return pd.DataFrame(results)


def run_cell(task, model, alpha):
    VAL = ROOT / f"results/analysis_inputs/oracle_outputs/{task}/{model}"
    TEST = ROOT / f"results/analysis_inputs_test/oracle_outputs/{task}/{model}"
    PILOT = ROOT / f"results/predictor_pilot/{task}/{model}/{alpha}"
    GATE = ROOT / f"results/adaptive_k_gate/{task}/{model}/{alpha}"
    if not (PILOT / "features_test.parquet").exists():
        return None

    # Use gate_config from prior calibration if exists, else compute fresh σ
    cfg_path = GATE / "gate_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        # Minimal calibration-on-fly
        cfg = {"k_min": 1, "k_max_high_bp": 2, "k_max_low_bp": 3,
                "base_prob_cutoff": 0.95, "yn_mass_floor": 0.5,
                "eps_yn": 0.05, "delta_yn": 0.075, "eps_lift": 0.08}

    # Compass calibration + test inference
    top3 = calibrate_compass(VAL / alpha / "K1_trivial.parquet", VAL / "baseline.parquet")
    bl_test = pd.read_parquet(TEST / "baseline.parquet")
    true_gold = bl_test.set_index("instance_id")["gold_sign"].to_dict()
    inferred_gold = infer_compass_gold(TEST / alpha / "K1_trivial.parquet", bl_test, top3)

    bp_labeled = bl_test.set_index("instance_id")["base_align_prob"].to_dict()
    bp_compass = {}
    for _, r in bl_test.iterrows():
        py, pn = softmax2(r["base_logit_yes"], r["base_logit_no"])
        g = inferred_gold.get(int(r["instance_id"]), 1)
        bp_compass[int(r["instance_id"])] = float((py if g == 1 else pn) * r["base_yn_mass"])

    feat_test = pd.read_parquet(PILOT / "features_test.parquet")
    iids_full = sorted(feat_test["instance_id"].unique().astype(int).tolist())

    eval_df = pd.read_parquet(TEST / alpha / "method_picks_eval.parquet")
    eval_df["layer_set"] = eval_df["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    ev_a = eval_df.drop_duplicates(subset=["instance_id", "layer_set"], keep="first")
    ev_lookup = ev_a.set_index(["instance_id", "layer_set"]).to_dict("index")

    # Build predictor → predicted scores per test instance
    pred_scores = {}
    # full / full_compass
    p_full = PILOT / "models" / "clf_reg_full.joblib"
    if p_full.exists():
        m = joblib.load(p_full)
        X_lab = pivot_geom_bp(feat_test, iids_full, ["proj_on_sv", "logit_shift"], bp_labeled)
        X_cmp = pivot_geom_bp(feat_test, iids_full, ["proj_on_sv", "logit_shift"], bp_compass)
        pred_scores["clf_reg_full"]         = m["regressor"].predict(m["scaler"].transform(X_lab))
        pred_scores["clf_reg_full_compass"] = m["regressor"].predict(m["scaler"].transform(X_cmp))
    # cheap / cheap_compass
    p_cheap = PILOT / "models" / "clf_reg_cheap.joblib"
    if p_cheap.exists():
        m = joblib.load(p_cheap)
        X_lab = pivot_geom_bp(feat_test, iids_full, ["proj_on_sv"], bp_labeled)
        X_cmp = pivot_geom_bp(feat_test, iids_full, ["proj_on_sv"], bp_compass)
        pred_scores["clf_reg_cheap"]         = m["regressor"].predict(m["scaler"].transform(X_lab))
        pred_scores["clf_reg_cheap_compass"] = m["regressor"].predict(m["scaler"].transform(X_cmp))
    # w2s
    p_w2s = PILOT / "models" / "clf_reg_w2s.joblib"
    if p_w2s.exists():
        m = joblib.load(p_w2s)
        emb_test = pd.read_parquet(PILOT / "embeddings_test.parquet")
        X = emb_test.drop(columns=["instance_id"]).values.astype(np.float32)
        scores = m["regressor"].predict(m["scaler"].transform(X))
        # Reindex to iids_full order
        emb_iids = emb_test["instance_id"].tolist()
        score_lookup = {int(emb_iids[i]): scores[i] for i in range(len(emb_iids))}
        pred_scores["clf_reg_w2s"] = np.array([score_lookup[int(iid)] for iid in iids_full])
    # no_gold
    p_ng = PILOT / "models" / "clf_reg_no_gold.joblib"
    if p_ng.exists():
        m = joblib.load(p_ng)
        # Rebuild no_gold features from feat_test + bl_test
        # (must reproduce the same feature pipeline)
        feat = feat_test.copy()
        K1_test = pd.read_parquet(TEST / alpha / "K1_trivial.parquet")
        K1_test["layer"] = K1_test["layer_set"].apply(lambda x: int(x[0]))
        K1_test["d_yes"] = K1_test["steered_logit_yes"] - K1_test["base_logit_yes"]
        K1_test["d_no"] = K1_test["steered_logit_no"] - K1_test["base_logit_no"]
        blocks = []
        for col in ["proj_on_sv", "act_norm", "act_var", "steering_cosine"]:
            b = (feat.pivot_table(index="instance_id", columns="layer", values=col, aggfunc="first")
                      .reindex(index=iids_full, columns=list(range(N_LAYERS))).fillna(0.0).values)
            blocks.append(b)
        for col in ("d_yes", "d_no"):
            b = (K1_test.pivot_table(index="instance_id", columns="layer", values=col, aggfunc="first")
                    .reindex(index=iids_full, columns=list(range(N_LAYERS))).fillna(0.0).values)
            blocks.append(b)
        proj_rank = (-blocks[0]).argsort(axis=1).argsort(axis=1).astype(np.float32)
        blocks.append(proj_rank)
        bl_by = bl_test.set_index("instance_id")
        py_a, pn_a, ynm = [], [], []
        for iid in iids_full:
            py, pn = softmax2(bl_by.at[iid, "base_logit_yes"], bl_by.at[iid, "base_logit_no"])
            py_a.append(py); pn_a.append(pn); ynm.append(bl_by.at[iid, "base_yn_mass"])
        py_a, pn_a, ynm = np.array(py_a), np.array(pn_a), np.array(ynm)
        e = -py_a * np.log(py_a + 1e-9) - pn_a * np.log(pn_a + 1e-9)
        inst = np.column_stack([py_a, pn_a, ynm, e, np.maximum(py_a, pn_a), np.abs(py_a-pn_a)])
        X = np.column_stack([inst, np.concatenate(blocks, axis=1)]).astype(np.float32)
        pred_scores["clf_reg_no_gold"] = m["regressor"].predict(m["scaler"].transform(X))

    # For each predictor: build top-K picks, walk gate, record per-instance + summary
    summary_rows = []
    for pred in PREDICTORS:
        if pred not in pred_scores: continue
        scores = pred_scores[pred]
        picks = predict_top_K_picks(scores, iids_full, alpha)
        # Use TRUE gold for lift measurement, compass-inferred for compass variants' K_max bucket
        gold = inferred_gold if pred.endswith("_compass") else true_gold
        per_inst = gate_walk(picks, ev_lookup, bl_test, gold, cfg)
        n = len(per_inst)
        missing_eval_count = (per_inst["fired"] == "missing_eval").sum()
        valid = per_inst[per_inst["lift"].notna()]
        # Also evaluate FIXED K=1 for comparison
        k1_lifts = []
        for iid in iids_full:
            ls1 = picks[1][iid]
            ev1 = ev_lookup.get((iid, ls1))
            if ev1 is None: continue
            g = true_gold.get(iid, 1)
            py_s, pn_s = softmax2(ev1["steered_logit_yes"], ev1["steered_logit_no"])
            bp_row = bl_test[bl_test["instance_id"] == iid].iloc[0]
            py_b, pn_b = softmax2(bp_row["base_logit_yes"], bp_row["base_logit_no"])
            base_prob_true = py_b if g == 1 else pn_b
            k1_lifts.append((py_s if g == 1 else pn_s) - base_prob_true)
        summary_rows.append({
            "task": task, "model": model, "alpha": alpha, "predictor": pred,
            "n_instances": n,
            "n_eval_missing": int(missing_eval_count),
            "mean_K_adaptive": round(float(valid["K_chosen"].mean()), 3) if len(valid) else None,
            "mean_lift_adaptive": round(float(valid["lift"].mean()), 4) if len(valid) else None,
            "mean_lift_fixed_K1": round(float(np.mean(k1_lifts)), 4) if k1_lifts else None,
            "n_plateau": int((per_inst["fired"] == "plateau").sum()),
            "n_yn_floor": int((per_inst["fired"] == "yn_floor").sum()),
            "n_backoff": int((per_inst["fired"] == "delta_yn_backoff").sum()),
            "n_k_max":  int((per_inst["fired"] == "k_max_reached").sum()),
        })
    return summary_rows


def main():
    all_rows = []
    for task in TASKS:
        for model in MODELS:
            for alpha in ALPHAS:
                rows = run_cell(task, model, alpha)
                if rows:
                    all_rows.extend(rows)
                    print(f"OK {task[:25]:25s} × {model:15s} × {alpha:10s}  ({len(rows)} predictors)")
                else:
                    print(f"SKIP {task} × {model} × {alpha}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "adaptive_k_results.csv", index=False)
    print(f"\nwrote {OUT/'adaptive_k_results.csv'} ({len(df)} rows)")

    # Aggregate per predictor (24-cell mean)
    print("\n=== Per-predictor mean across 24 cells (× α) ===")
    agg = (df.groupby("predictor")
             .agg(n_cells=("task", "size"),
                   mean_K_adaptive=("mean_K_adaptive", "mean"),
                   mean_lift_adaptive=("mean_lift_adaptive", "mean"),
                   mean_lift_K1=("mean_lift_fixed_K1", "mean"),
                   median_lift_adaptive=("mean_lift_adaptive", "median"),
                   total_eval_missing=("n_eval_missing", "sum"))
             .round(4))
    agg["adaptive_gain_vs_K1"] = (agg["mean_lift_adaptive"] - agg["mean_lift_K1"]).round(4)
    print(agg.to_string())


if __name__ == "__main__":
    main()
