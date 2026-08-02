"""Compass-ensemble validation on ALL 24 cells (6 tasks × 2 models × 2 α schedules).

For each cell:
1. Inline calibration: σ thresholds + top-3 compass layers from val_200.
2. 3-layer ensemble compass on test_100.
3. Walk gate twice (labeled gold vs compass) on 9 baseline methods.
4. Compute lift_recovery_ratio and supporting stats.

Output: 24-cell cross-table + per-method aggregation + universal anchor analysis.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

TASKS = [
    "desire-to-create-allies",
    "conscientiousness",
    "cognitive-enhancement",
    "believes-it-has-phenomenal-consciousness",
    "desire-to-maximize-impact-on-world",
    "subscribes-to-Christianity",
]
MODELS = ["llama-3-8b", "aya-expanse-8b"]
ALPHAS = ["uniform", "sqrt-norm"]
BASELINES = [
    "tkm_align", "tkm_multi3", "ln_global",
    "exhaustive_align", "exhaustive_multi3",
    "beam_align", "beam_multi3",
    "gt_global_align", "gt_global_multi3",
]
OUT = Path("results/adaptive_k_gate/compass_vs_labeled/all_cells_final")
OUT.mkdir(parents=True, exist_ok=True)


def softmax2(a, b):
    m = max(a, b); ea = np.exp(a-m); eb = np.exp(b-m); z = ea+eb
    return ea/z, eb/z


def calibrate(task, model, alpha):
    ROOT_VAL = Path(f"results/analysis_inputs/oracle_outputs/{task}/{model}")
    K1_val_pq = ROOT_VAL / alpha / "K1_trivial.parquet"
    bl_val_pq = ROOT_VAL / "baseline.parquet"
    eval_val_pq = ROOT_VAL / alpha / "method_picks_eval.parquet"
    picks_val_pq = ROOT_VAL / "method_picks.parquet"
    if not all(p.exists() for p in [K1_val_pq, bl_val_pq, eval_val_pq, picks_val_pq]):
        return None
    K1_val = pd.read_parquet(K1_val_pq)
    bl_val = pd.read_parquet(bl_val_pq)
    eval_val = pd.read_parquet(eval_val_pq)
    picks_val = pd.read_parquet(picks_val_pq)
    picks_val = picks_val[picks_val["alpha_schedule"] == alpha].copy()
    if len(picks_val) == 0:
        return None
    eval_val["layer_set"] = eval_val["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    picks_val["layer_set"] = picks_val["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    cal_method = "tkm_multi3" if (picks_val["method_id"] == "tkm_multi3").any() else "tkm_align"
    cal_picks = picks_val[picks_val["method_id"] == cal_method].copy()
    cal_join = cal_picks.merge(
        eval_val[["instance_id", "layer_set", "steered_yn_mass", "steered_prob", "base_prob"]],
        on=["instance_id", "layer_set"], how="inner")
    cal_join["lift"] = cal_join["steered_prob"] - cal_join["base_prob"]
    if len(cal_join) < 10:
        sigma_yn = sigma_lift = 0.05
    else:
        yn_w = cal_join.pivot_table(index="instance_id", columns="K",
                                     values="steered_yn_mass", aggfunc="first")
        lf_w = cal_join.pivot_table(index="instance_id", columns="K",
                                     values="lift", aggfunc="first")
        deltas = []
        for k in sorted(yn_w.columns):
            if k == min(yn_w.columns): continue
            prev = k - 1
            if prev not in yn_w.columns: continue
            for iid in yn_w.index:
                if pd.isna(yn_w.at[iid, k]) or pd.isna(yn_w.at[iid, prev]): continue
                healthy = yn_w.at[iid, k] >= 0.5 and yn_w.at[iid, prev] >= 0.5
                deltas.append({"dyn": yn_w.at[iid, k]-yn_w.at[iid, prev],
                                "dlf": lf_w.at[iid, k]-lf_w.at[iid, prev],
                                "healthy": healthy})
        deltas = pd.DataFrame(deltas) if deltas else pd.DataFrame()
        healthy_d = deltas[deltas["healthy"]] if len(deltas) else deltas
        sigma_yn = float(healthy_d["dyn"].std()) if len(healthy_d) else 0.05
        sigma_lift = float(healthy_d["dlf"].std()) if len(healthy_d) else 0.05

    gold_val = bl_val.set_index("instance_id")["gold_sign"].to_dict()
    K1_val["layer"] = K1_val["layer_set"].apply(lambda x: int(x[0]))
    K1_val["true_gold"] = K1_val["instance_id"].map(gold_val)
    K1_val["compass"] = np.sign(
        (K1_val["steered_logit_yes"] - K1_val["steered_logit_no"])
        - (K1_val["base_logit_yes"] - K1_val["base_logit_no"]))
    layer_acc = K1_val.groupby("layer").apply(
        lambda s: (s["compass"] == s["true_gold"]).mean(), include_groups=False)
    top3 = [int(l) for l in layer_acc.sort_values(ascending=False).head(3).index]
    return {
        "sigma_yn": sigma_yn, "sigma_lift": sigma_lift,
        "eps_yn": sigma_yn, "delta_yn": 1.5 * sigma_yn, "eps_lift": sigma_lift,
        "k_min": 1, "k_max_high_bp": 2, "k_max_low_bp": 3,
        "base_prob_cutoff": 0.95, "yn_mass_floor": 0.5,
        "compass_layers": top3,
        "val_top_acc": float(layer_acc.max()),
    }


def run_cell(task, model, alpha, cfg):
    ROOT_TEST = Path(f"results/analysis_inputs_test/oracle_outputs/{task}/{model}")
    bl_test_pq = ROOT_TEST / "baseline.parquet"
    K1_test_pq = ROOT_TEST / alpha / "K1_trivial.parquet"
    eval_test_pq = ROOT_TEST / alpha / "method_picks_eval.parquet"
    picks_test_pq = ROOT_TEST / "method_picks.parquet"
    if not all(p.exists() for p in [bl_test_pq, K1_test_pq, eval_test_pq, picks_test_pq]):
        return None, None
    bl_test = pd.read_parquet(bl_test_pq)
    true_gold = bl_test.set_index("instance_id")["gold_sign"].to_dict()
    bl_w = bl_test.copy()
    bl_w["argmax_gold"] = np.where(
        bl_w["base_logit_yes"] > bl_w["base_logit_no"], 1, -1)
    argmax_gold = bl_w.set_index("instance_id")["argmax_gold"].to_dict()

    K1_test = pd.read_parquet(K1_test_pq)
    K1_test["layer"] = K1_test["layer_set"].apply(lambda x: int(x[0]))
    K1_test["compass"] = np.sign(
        (K1_test["steered_logit_yes"] - K1_test["steered_logit_no"])
        - (K1_test["base_logit_yes"] - K1_test["base_logit_no"]))
    vote_pivot = K1_test.pivot_table(
        index="instance_id", columns="layer", values="compass", aggfunc="first")

    ensemble_gold = {}
    for iid in vote_pivot.index:
        votes = vote_pivot.loc[iid, cfg["compass_layers"]].values.astype(int)
        s = int(np.sum(votes))
        if s > 0: ensemble_gold[int(iid)] = 1
        elif s < 0: ensemble_gold[int(iid)] = -1
        else: ensemble_gold[int(iid)] = argmax_gold.get(int(iid), 1)
    test_acc = sum(1 for iid, g in ensemble_gold.items()
                   if g == true_gold.get(iid)) / len(ensemble_gold)

    eval_df = pd.read_parquet(eval_test_pq)
    picks = pd.read_parquet(picks_test_pq)
    picks = picks[picks["alpha_schedule"] == alpha].copy()
    eval_df["layer_set"] = eval_df["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    picks["layer_set"] = picks["layer_set"].apply(lambda x: tuple(sorted(int(l) for l in x)))
    ev_a = eval_df.drop_duplicates(subset=["instance_id", "layer_set"], keep="first")
    ev_lookup = ev_a.set_index(["instance_id", "layer_set"]).to_dict("index")

    K_MIN, K_MAX_HI, K_MAX_LO = cfg["k_min"], cfg["k_max_high_bp"], cfg["k_max_low_bp"]
    BP_CUT, YN_FLOOR = cfg["base_prob_cutoff"], cfg["yn_mass_floor"]
    DELTA_YN, EPS_LIFT = cfg["delta_yn"], cfg["eps_lift"]

    def gw(method, iid, gold_sign):
        bp_row = bl_test[bl_test["instance_id"] == iid].iloc[0]
        p_yb, p_nb = softmax2(bp_row["base_logit_yes"], bp_row["base_logit_no"])
        base_prob = p_yb if gold_sign == 1 else p_nb
        k_max = K_MAX_HI if base_prob >= BP_CUT else K_MAX_LO
        sub = picks[(picks["method_id"] == method) & (picks["instance_id"] == iid)].sort_values("K")
        by_k = {int(r["K"]): r for _, r in sub.iterrows()}
        if not by_k: return np.nan
        eff_k_min = K_MIN if K_MIN in by_k else min(by_k.keys())
        best_k, best_lift = None, -np.inf
        prev_yn, prev_lift = None, None
        for k in range(eff_k_min, k_max + 1):
            if k not in by_k: break
            r = by_k[k]
            ev = ev_lookup.get((int(iid), r["layer_set"]))
            if ev is None: break
            p_y, p_n = softmax2(ev["steered_logit_yes"], ev["steered_logit_no"])
            p_t = p_y if gold_sign == 1 else p_n
            lift = p_t - base_prob
            yn = ev["steered_yn_mass"]
            if yn < YN_FLOOR: break
            if prev_yn is not None and (yn - prev_yn) < -DELTA_YN: break
            if prev_lift is not None and (lift - prev_lift) < EPS_LIFT:
                if lift > best_lift: best_k, best_lift = k, lift
                break
            if lift > best_lift: best_k, best_lift = k, lift
            prev_yn, prev_lift = yn, lift
        if best_k is None:
            r0 = by_k.get(eff_k_min)
            if r0 is not None:
                ev0 = ev_lookup.get((int(iid), r0["layer_set"]))
                if ev0 is not None:
                    p_y0, p_n0 = softmax2(ev0["steered_logit_yes"], ev0["steered_logit_no"])
                    best_k = eff_k_min
                    best_lift = (p_y0 if gold_sign == 1 else p_n0) - base_prob
        return best_lift if best_k is not None else np.nan

    instances = sorted(picks["instance_id"].unique().astype(int).tolist())
    rows = []
    for method in BASELINES:
        present = picks[picks["method_id"] == method]
        if len(present) == 0: continue
        lab = [gw(method, iid, true_gold.get(iid, 1)) for iid in instances]
        cmp = [gw(method, iid, ensemble_gold.get(iid, 1)) for iid in instances]
        lab_a = np.array([x for x in lab if not np.isnan(x)])
        cmp_a = np.array([x for x in cmp if not np.isnan(x)])
        if len(lab_a) == 0 or lab_a.mean() == 0:
            ratio = float("nan"); ratio_clip = float("nan")
        else:
            ratio = float(cmp_a.mean() / lab_a.mean())
            ratio_clip = max(-1.0, min(2.0, ratio))  # clip extreme outliers from near-zero baselines
        rows.append({
            "task": task, "model": model, "alpha": alpha, "method": method,
            "n": len(lab_a), "compass_layers": tuple(cfg["compass_layers"]),
            "test_acc": round(test_acc, 4), "val_top_acc": round(cfg["val_top_acc"], 4),
            "labeled_lift": round(float(lab_a.mean()), 4),
            "compass_lift": round(float(cmp_a.mean()), 4),
            "ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
            "ratio_clipped": round(ratio_clip, 4) if not np.isnan(ratio_clip) else np.nan,
        })
    return rows, test_acc


def main():
    all_rows = []
    layers_by_cell = {}
    skipped = []
    for task in TASKS:
        for model in MODELS:
            for alpha in ALPHAS:
                cfg = calibrate(task, model, alpha)
                if cfg is None:
                    skipped.append((task, model, alpha, "calibration failed (missing val data)"))
                    continue
                rows, acc = run_cell(task, model, alpha, cfg)
                if rows is None:
                    skipped.append((task, model, alpha, "test side missing"))
                    continue
                all_rows.extend(rows)
                layers_by_cell[(task, model, alpha)] = cfg["compass_layers"]
                print(f"OK {task[:30]:30s} × {model:15s} × {alpha:10s}  "
                      f"layers={cfg['compass_layers']}  test_acc={acc:.3f}")
    if skipped:
        print("\n=== SKIPPED ===")
        for s in skipped:
            print(f"  {s}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "all_cells_compass.csv", index=False)
    print(f"\nwrote {OUT/'all_cells_compass.csv'}  ({len(df)} rows = cells × methods)")

    # Per-cell median across methods (robust against near-zero-baseline outliers)
    print("\n=== Per-cell median lift recovery (across 9 baseline methods) ===")
    cell_summary = (df.groupby(["task","model","alpha"])
                      .agg(test_acc=("test_acc","first"),
                           compass_layers=("compass_layers","first"),
                           median_ratio=("ratio","median"),
                           mean_ratio_clipped=("ratio_clipped","mean"),
                           mean_labeled_lift=("labeled_lift","mean"),
                           n_methods=("method","size"))
                      .round(4))
    print(cell_summary.to_string())

    # Per-method aggregation
    print("\n=== Per-method median ratio across 24 cells ===")
    method_summary = (df.groupby("method")
                        .agg(n_cells=("task","size"),
                             median_ratio=("ratio","median"),
                             mean_ratio_clipped=("ratio_clipped","mean"),
                             p10_ratio=("ratio_clipped", lambda s: s.quantile(0.1)),
                             p90_ratio=("ratio_clipped", lambda s: s.quantile(0.9)))
                        .round(4)
                        .sort_values("median_ratio", ascending=False))
    print(method_summary.to_string())

    # Universal anchor analysis
    print("\n=== Universal compass anchor analysis ===")
    from collections import Counter
    layer_counts_llama = Counter()
    layer_counts_aya = Counter()
    for (task, model, alpha), layers in layers_by_cell.items():
        ctr = layer_counts_llama if model == "llama-3-8b" else layer_counts_aya
        for L in layers:
            ctr[L] += 1
    n_llama = sum(1 for (_,m,_) in layers_by_cell if m == "llama-3-8b")
    n_aya = sum(1 for (_,m,_) in layers_by_cell if m == "aya-expanse-8b")
    print(f"\nLlama ({n_llama} cells): top compass-layer occurrence frequency")
    for L, n in layer_counts_llama.most_common(8):
        print(f"  L={L:>3d}: {n}/{n_llama} cells ({n/n_llama*100:.0f}%)")
    print(f"\nAya ({n_aya} cells): top compass-layer occurrence frequency")
    for L, n in layer_counts_aya.most_common(8):
        print(f"  L={L:>3d}: {n}/{n_aya} cells ({n/n_aya*100:.0f}%)")

    # Headline: median ratio summary
    print("\n=== HEADLINE ===")
    overall_median = df["ratio"].median()
    overall_p10 = df["ratio_clipped"].quantile(0.1)
    overall_p90 = df["ratio_clipped"].quantile(0.9)
    n_cells = len(layers_by_cell)
    print(f"  Cells tested: {n_cells}/24")
    print(f"  Total (cell × method) data points: {len(df)}")
    print(f"  Median ratio (across all data points): {overall_median:.4f}")
    print(f"  P10 ratio (clipped):                  {overall_p10:.4f}")
    print(f"  P90 ratio (clipped):                  {overall_p90:.4f}")
    print(f"  Cells with median ratio ≥ 0.90: "
          f"{(cell_summary['median_ratio'] >= 0.90).sum()}/{len(cell_summary)}")


if __name__ == "__main__":
    main()
