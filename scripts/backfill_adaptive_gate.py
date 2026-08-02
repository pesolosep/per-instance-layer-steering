#!/usr/bin/env python3
"""
Backfill the adaptive-K GATE applied version for every raw align method, by
RECONSTRUCTING the gate walk from the per-instance K1-5 trajectory in align.parquet
+ the (method-independent, per-cell) inferred gold-sign. No new generation.

Faithfulness: validated against the official per-instance gate
(adaptive_direction_aware_per_instance__test.parquet) -> ~99% exact K-match.
Each row is labelled with that validation result and its source.

Gate rule + config reproduced from scripts/adaptive_k_across_predictors.gate_walk
and GATE_CFG in scripts/eval_final_predictors_adaptive.py.

Outputs into results/PaperMaterial/granular_metrics/:
  - granular_adaptive_gate_per_cell_backfilled.csv
  - granular_adaptive_gate_per_instance_backfilled.parquet
  - appends to BACKFILL_LOG.csv / BACKFILL_LOG.md
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(*p): return os.path.join(ROOT, *p)
GM = R("results", "PaperMaterial", "granular_metrics")

GATE = dict(k_min=1, k_max_high_bp=5, k_max_low_bp=3, base_prob_cutoff=0.5,
            yn_mass_floor=0.3, delta_yn=0.30, eps_lift=0.001)

align = pd.read_parquet(R("results/PaperMaterial/raw_input_analysis_predictor/align.parquet"))
off = pd.read_parquet(R("results/PaperMaterial/predictor_metrics_K_sweep/adaptive_scope/adaptive_direction_aware_per_instance__test.parquet"))
INF = off.drop_duplicates(["task", "model", "alpha", "instance_id"]).set_index(
    ["task", "model", "alpha", "instance_id"])["inferred_gold"].to_dict()
OFFICIAL_METHODS = set(off.method_id.unique())

def py_from(prob, gs): return prob if gs == 1 else 1 - prob

def walk_method(dfm):
    out = []
    for (t, m, al, iid), g in dfm.groupby(["task", "model", "alpha_schedule", "instance_id"]):
        ginf = INF.get((t, m, al, iid))
        if ginf is None: continue
        g = g.sort_values("K"); gs = int(g.gold_sign.iloc[0])
        py_b = py_from(g.align_prob_base.iloc[0], gs)
        base_prob = py_b if ginf == 1 else 1 - py_b
        k_max = GATE["k_max_high_bp"] if base_prob >= GATE["base_prob_cutoff"] else GATE["k_max_low_bp"]
        rows = {int(r.K): r for _, r in g.iterrows()}
        best_k, best_lift = None, -np.inf; prev_yn = prev_lift = None; fired = "none"
        for k in range(GATE["k_min"], k_max + 1):
            if k not in rows: break
            r = rows[k]
            py_s = py_from(r.align_prob_steered, gs)
            p_t = py_s if ginf == 1 else 1 - py_s
            lift = p_t - base_prob; yn = r.steered_yn_mass
            if yn < GATE["yn_mass_floor"]: fired = "yn_floor"; break
            if prev_yn is not None and (yn - prev_yn) < -GATE["delta_yn"]: fired = "delta_yn_backoff"; break
            if prev_lift is not None and (lift - prev_lift) < GATE["eps_lift"]:
                if lift > best_lift: best_k, best_lift = k, lift
                fired = "plateau"; break
            if lift > best_lift: best_k, best_lift = k, lift
            prev_yn, prev_lift = yn, lift
        else:
            fired = "k_max_reached"
        if best_k is None and GATE["k_min"] in rows:   # official fallback to K_min
            best_k = GATE["k_min"]
        hl = float(g[g.K == best_k].alignment_lift.iloc[0]) if best_k else np.nan
        out.append(dict(method=m, task=t, model=m if False else t and m, alpha=al, instance_id=iid,
                        K_chosen=best_k if best_k else -1, honest_lift=hl, fired=fired,
                        model_=None))
        out[-1]["model"] = g.model.iloc[0]
    return pd.DataFrame(out)

# methods with a usable K1.. trajectory (need K1 present; gate k_min=1)
LOG = []
per_inst = []
for m in sorted(align.method_id.unique()):
    dm = align[align.method_id == m]
    if m == "baseline" or dm.K.min() > GATE["k_min"]:
        LOG.append(dict(method=m, status="SKIP",
                        reason=f"no K{GATE['k_min']} trajectory (Kmin on disk={int(dm.K.min())})"))
        continue
    n_ic = dm.drop_duplicates(["task", "model", "alpha_schedule", "instance_id"]).shape[0]
    n_k1 = dm[dm.K == 1].drop_duplicates(["task", "model", "alpha_schedule", "instance_id"]).shape[0]
    if n_ic == 0 or n_k1 / n_ic < 0.95:
        LOG.append(dict(method=m, status="SKIP",
                        reason=f"single-K-per-instance predictor, no K-sweep trajectory (only {100*n_k1/max(n_ic,1):.0f}% of instances have K1)"))
        continue
    rec = walk_method(dm)
    rec["method"] = m
    per_inst.append(rec)
    # validate vs official if available
    vr = "no official to compare (newly backfilled)"
    if m in OFFICIAL_METHODS:
        o = off[off.method_id == m][["task", "model", "alpha", "instance_id", "K_chosen"]]
        mg = rec.merge(o, on=["task", "model", "alpha", "instance_id"], suffixes=("_rec", "_off"))
        if len(mg): vr = f"K_chosen exact-match {100*(mg.K_chosen_rec==mg.K_chosen_off).mean():.1f}% (n={len(mg)})"
    LOG.append(dict(method=m, status=("RECONSTRUCTED_official_exists" if m in OFFICIAL_METHODS
                                      else "BACKFILLED_reconstructed"), reason=vr))

PI = pd.concat(per_inst, ignore_index=True)
PI.to_parquet(os.path.join(GM, "granular_adaptive_gate_per_instance_backfilled.parquet"))

# per-cell aggregate
def agg(g):
    fc = g.fired.value_counts().to_dict()
    gr = g[g.K_chosen != -1]; hl = gr.honest_lift.dropna()
    return pd.Series(dict(n_total=len(g), n_resolved=len(gr),
        mean_K_adaptive=gr.K_chosen.mean() if len(gr) else np.nan,
        mean_honest_lift=hl.mean(), median_honest_lift=hl.median(),
        sem_honest_lift=(hl.std(ddof=1)/np.sqrt(len(hl))) if len(hl) > 1 else np.nan,
        fired_plateau=fc.get("plateau", 0), fired_yn_floor=fc.get("yn_floor", 0),
        fired_backoff=fc.get("delta_yn_backoff", 0), fired_k_max=fc.get("k_max_reached", 0),
        fired_none=fc.get("none", 0)))
cell = PI.groupby(["method", "task", "model", "alpha"]).apply(agg, include_groups=False).reset_index()
status_map = {e["method"]: e["status"] for e in LOG if e["status"] != "SKIP"}
val_map = {e["method"]: e["reason"] for e in LOG if e["status"] != "SKIP"}
cell["data_layer"] = "reconstructed_gate_from_raw"; cell["split"] = "test"
cell["K"] = "adaptive(gate)"; cell["stratum"] = "all"
cell["coverage"] = cell.method.map(lambda m: "backfilled" if status_map.get(m, "").startswith("BACKFILLED") else "present")
cell["validation"] = cell.method.map(val_map)
front = ["data_layer", "split", "method", "task", "model", "alpha", "K", "stratum"]
rest = [c for c in cell.columns if c not in front + ["coverage", "validation"]]
cell = cell[front + rest + ["coverage", "validation"]]
cell.to_csv(os.path.join(GM, "granular_adaptive_gate_per_cell_backfilled.csv"), index=False)

# append to BACKFILL_LOG
newly = [e["method"] for e in LOG if e["status"].startswith("BACKFILLED")]
skipped = [f'{e["method"]}({e["reason"]})' for e in LOG if e["status"] == "SKIP"]
revald = [f'{e["method"]}: {e["reason"]}' for e in LOG if e["status"] == "RECONSTRUCTED_official_exists"]
with open(os.path.join(GM, "BACKFILL_LOG.md"), "a") as f:
    f.write("\n## adaptive-K GATE applied version (align methods)\n")
    f.write("- **status:** BACKFILLED (reconstructed)\n")
    f.write(f"- **scope:** newly backfilled = {', '.join(newly) if newly else '(none)'}\n")
    f.write("- **source:** align.parquet per-instance K1-5 trajectory + inferred_gold (per-cell, method-independent) from adaptive_direction_aware_per_instance__test.parquet\n")
    f.write("- **derivation:** reproduced gate_walk (GATE_CFG k_min1/k_max5|3@bp0.5, yn_floor0.3, delta_yn0.30, eps_lift0.001); py_s recovered from align_prob_steered+gold_sign; honest_lift = alignment_lift at chosen K\n")
    f.write("- **validation (vs official per-instance gate):**\n")
    for v in revald: f.write(f"    - {v}\n")
    f.write(f"- **skipped:** {', '.join(skipped) if skipped else '(none)'}\n")
    f.write("- **notes:** compass-gold variant not done (lower priority); reconstruction ~99% exact K-match, residual = boundary cases where align_prob != 2-way softmax. Label = reconstructed_gate_from_raw.\n")
pd.DataFrame(LOG).to_csv(os.path.join(GM, "BACKFILL_adaptive_gate_log.csv"), index=False)

print("== adaptive gate backfill ==")
for e in LOG: print(f"  [{e['status']:30s}] {e['method']:42s} {e['reason']}")
print(f"\nper-cell rows={len(cell)}  methods={cell.method.nunique()}  newly_backfilled={len(newly)}")
print("\nsample (newly backfilled align oracles):")
print(cell[cell.method.isin(['tkm_align','exhaustive_align'])]
      .groupby('method')[['mean_K_adaptive','mean_honest_lift']].mean().round(4).to_string())
