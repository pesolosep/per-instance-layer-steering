"""Shared utilities for the predictor EDA pipeline."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

# Auto-detect repo root: env var overrides, else infer from this file's location
# (scripts/predictor_eda/_util.py → repo root). Works on local + remote
# without per-machine config.
ROOT = Path(os.environ.get(
    "LAYER_SELECTION_ROOT",
    str(Path(__file__).resolve().parent.parent.parent),
))
TASKS = ["desire-to-create-allies", "conscientiousness", "cognitive-enhancement",
         "believes-it-has-phenomenal-consciousness", "desire-to-maximize-impact-on-world",
         "subscribes-to-Christianity"]
MODELS = ["llama-3-8b", "aya-expanse-8b"]
ALPHAS = ["uniform", "sqrt-norm"]
N_LAYERS = 32
GEOM_FEATURES = ["proj_on_sv", "steering_cosine", "act_norm", "act_var"]
FULL_FEATURES = GEOM_FEATURES + ["logit_shift"]
INSTANCE_FEATURES = ["base_prob"]
EDA_OUT = ROOT / "results/predictor_pilot_eda"


def cells():
    """Yield (task, model, alpha) for all 24 cells."""
    for t in TASKS:
        for m in MODELS:
            for a in ALPHAS:
                yield t, m, a


def cell_dir(task, model, alpha, eda_root=EDA_OUT):
    return eda_root / task / model / alpha


def pilot_dir(task, model, alpha):
    return ROOT / f"results/predictor_pilot/{task}/{model}/{alpha}"


def softmax2(z_yes, z_no):
    """Stable 2-way softmax → (p_yes, p_no). Accepts arrays."""
    m = np.maximum(z_yes, z_no)
    e_y = np.exp(z_yes - m); e_n = np.exp(z_no - m)
    z = e_y + e_n
    return e_y / z, e_n / z


def compute_targets_per_layer(task, model, alpha, split):
    """Compute per-(instance, layer) targets from K1_trivial.

    Returns DataFrame with columns:
      instance_id, layer, gold_sign,
      align_lift     := p_steered[target] - p_base[target]
      align_prob_s   := p_steered[target]
      logit_margin_s := (z_target - z_other)_steered
      yn_mass_s      := steered_yn_mass
      multi3_proxy   := ( max(0, align_prob_s - 0.5) * max(0, logit_margin_s) * yn_mass_s ) ** (1/3)
    """
    base = "analysis_inputs" if split == "val" else "analysis_inputs_test"
    pq = ROOT / f"results/{base}/oracle_outputs/{task}/{model}/{alpha}/K1_trivial.parquet"
    if not pq.exists():
        return None
    k1 = pd.read_parquet(pq)
    k1["layer"] = k1["layer_set"].apply(lambda x: int(x[0]))
    p_y_s, p_n_s = softmax2(k1["steered_logit_yes"].values, k1["steered_logit_no"].values)
    p_y_b, p_n_b = softmax2(k1["base_logit_yes"].values,    k1["base_logit_no"].values)
    g = k1["gold_sign"].values
    align_lift = np.where(g == 1, p_y_s - p_y_b, p_n_s - p_n_b)
    align_prob_s = np.where(g == 1, p_y_s, p_n_s)
    margin_s = (k1["steered_logit_yes"] - k1["steered_logit_no"]).values * g
    yn_s = k1["steered_yn_mass"].values
    multi3 = (np.maximum(0, align_prob_s - 0.5)
              * np.maximum(0, margin_s)
              * yn_s) ** (1.0 / 3.0)
    return pd.DataFrame({
        "instance_id":   k1["instance_id"].astype(int).values,
        "layer":         k1["layer"].astype(int).values,
        "gold_sign":     g.astype(int),
        "align_lift":    align_lift,
        "align_prob_s":  align_prob_s,
        "logit_margin_s": margin_s,
        "yn_mass_s":     yn_s,
        "multi3_proxy":  multi3,
    })


def load_features(task, model, alpha, split):
    """Load features parquet for the given split."""
    pq = pilot_dir(task, model, alpha) / f"features_{split}.parquet"
    if not pq.exists(): return None
    return pd.read_parquet(pq)


def load_embeddings(task, model, alpha, split):
    pq = pilot_dir(task, model, alpha) / f"embeddings_{split}.parquet"
    if not pq.exists(): return None
    return pd.read_parquet(pq)


def load_labels_reg_val(task, model, alpha):
    """The actual training target the RF was supervised on (val side only)."""
    pq = pilot_dir(task, model, alpha) / "labels_reg_val.parquet"
    if not pq.exists(): return None
    return pd.read_parquet(pq)


def join_features_targets(task, model, alpha, split):
    """Build per-cell unified DataFrame for EDA.

    Returns long-format DataFrame with:
      (instance_id, layer, <feature cols>, gold_sign, align_lift, multi3_proxy, …)
    """
    feat = load_features(task, model, alpha, split)
    if feat is None: return None
    tgt = compute_targets_per_layer(task, model, alpha, split)
    if tgt is None: return None
    j = feat.merge(tgt, on=["instance_id", "layer"], how="inner")
    return j


def per_instance_best_layer(target_long: pd.DataFrame, target_col="align_lift"):
    """For each instance, return argmax-layer (best K=1 layer for this prompt)."""
    return target_long.loc[target_long.groupby("instance_id")[target_col].idxmax()][
        ["instance_id", "layer", target_col]
    ].rename(columns={"layer": "best_layer", target_col: "best_score"})


def ndcg_at_k(scores_pred: np.ndarray, scores_true: np.ndarray, k: int = 3) -> float:
    """Normalized DCG@k between two score vectors using non-negative relevance.

    Relevance := max(0, score_true). This makes NDCG well-defined when
    score_true can be negative (e.g., align_lift for layers that hurt).
    Returns nan if iDCG = 0 (all layers have non-positive true score).
    """
    if len(scores_pred) != len(scores_true): return np.nan
    rel_true = np.maximum(0.0, scores_true)
    order = np.argsort(-scores_pred)[:k]
    dcg = float(np.sum([rel_true[order[i]] / np.log2(i + 2) for i in range(k)]))
    ideal_order = np.argsort(-rel_true)[:k]
    idcg = float(np.sum([rel_true[ideal_order[i]] / np.log2(i + 2) for i in range(k)]))
    return dcg / idcg if idcg > 0 else np.nan


def top_k_precision(scores_pred: np.ndarray, scores_true: np.ndarray, k: int = 3) -> float:
    """Fraction of predictor's top-K that overlaps with true top-K (of layers)."""
    if len(scores_pred) != len(scores_true): return np.nan
    pred_topk = set(np.argsort(-scores_pred)[:k].tolist())
    true_topk = set(np.argsort(-scores_true)[:k].tolist())
    return len(pred_topk & true_topk) / k


def hit_at_k(scores_pred: np.ndarray, scores_true: np.ndarray, k: int = 1) -> float:
    """Did the predictor's top-K contain the truly-best-layer (true argmax)?"""
    if len(scores_pred) != len(scores_true): return np.nan
    true_top1 = int(np.argmax(scores_true))
    pred_topk = set(np.argsort(-scores_pred)[:k].tolist())
    return 1.0 if true_top1 in pred_topk else 0.0


def spearman_safe(x, y):
    """Spearman ρ that returns nan on degenerate input."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3: return np.nan
    if np.std(x) == 0 or np.std(y) == 0: return np.nan
    r, _ = spearmanr(x, y)
    return float(r)


def cell_label(task, model, alpha):
    return f"{task[:18]:18s} {model:14s} {alpha:9s}"


NO_GOLD_GEOM_FEATURES = ("proj_on_sv", "act_norm", "act_var", "steering_cosine")
NO_GOLD_D_FEATURES = ("d_yes", "d_no")
NO_GOLD_INST_FEATURES = ("p_yes", "p_no", "yn_mass", "entropy", "max_p", "abs_diff")


def build_X_no_gold(feat_df, bl_df, k1_df, iids,
                    include_geom=NO_GOLD_GEOM_FEATURES,
                    include_d=NO_GOLD_D_FEATURES,
                    include_proj_rank=True,
                    include_inst=NO_GOLD_INST_FEATURES):
    """Replicates scripts/predictor_no_gold_with_compass.build_no_gold_features
    with toggles for ablation. Returns (X, names).

    Feature layout follows the no_gold script exactly when all toggles are on:
      [inst (6), proj_on_sv × 32, act_norm × 32, act_var × 32,
       steering_cosine × 32, d_yes × 32, d_no × 32, proj_rank × 32]  → 230 d
    """
    bl_by = bl_df.set_index("instance_id") if "instance_id" in bl_df.columns else bl_df

    # Per-layer geometry
    blocks = []
    names = []
    proj_block = None
    for col in include_geom:
        b = (feat_df.pivot_table(index="instance_id", columns="layer",
                                 values=col, aggfunc="first")
                    .reindex(index=iids, columns=list(range(N_LAYERS)))
                    .fillna(0.0).values)
        blocks.append(b)
        names.extend([f"{col}__L{l}" for l in range(N_LAYERS)])
        if col == "proj_on_sv":
            proj_block = b

    # Per-layer d_yes / d_no from K1_trivial
    if include_d:
        k1_local = k1_df.copy()
        if "d_yes" not in k1_local.columns:
            k1_local["d_yes"] = k1_local["steered_logit_yes"] - k1_local["base_logit_yes"]
            k1_local["d_no"]  = k1_local["steered_logit_no"]  - k1_local["base_logit_no"]
        if "layer" not in k1_local.columns:
            k1_local["layer"] = k1_local["layer_set"].apply(lambda x: int(x[0]))
        for col in include_d:
            b = (k1_local.pivot_table(index="instance_id", columns="layer",
                                      values=col, aggfunc="first")
                         .reindex(index=iids, columns=list(range(N_LAYERS)))
                         .fillna(0.0).values)
            blocks.append(b)
            names.extend([f"{col}__L{l}" for l in range(N_LAYERS)])

    # proj_rank (derived from proj_on_sv ordering)
    if include_proj_rank:
        if proj_block is None:
            # Need proj_on_sv even if not in geom features
            proj_block = (feat_df.pivot_table(index="instance_id", columns="layer",
                                              values="proj_on_sv", aggfunc="first")
                                  .reindex(index=iids, columns=list(range(N_LAYERS)))
                                  .fillna(0.0).values)
        proj_rank = (-proj_block).argsort(axis=1).argsort(axis=1).astype(np.float32)
        blocks.append(proj_rank)
        names.extend([f"proj_rank__L{l}" for l in range(N_LAYERS)])

    # Instance features
    py_a = np.empty(len(iids), dtype=np.float32)
    pn_a = np.empty(len(iids), dtype=np.float32)
    ynm  = np.empty(len(iids), dtype=np.float32)
    for i, iid in enumerate(iids):
        py, pn = softmax2(bl_by.at[iid, "base_logit_yes"], bl_by.at[iid, "base_logit_no"])
        py_a[i] = py; pn_a[i] = pn
        ynm[i]  = bl_by.at[iid, "base_yn_mass"]
    inst_lookup = {
        "p_yes":    py_a,
        "p_no":     pn_a,
        "yn_mass":  ynm,
        "entropy":  -py_a * np.log(py_a + 1e-9) - pn_a * np.log(pn_a + 1e-9),
        "max_p":    np.maximum(py_a, pn_a),
        "abs_diff": np.abs(py_a - pn_a),
    }
    inst_blocks = []
    inst_names = []
    for col in include_inst:
        inst_blocks.append(inst_lookup[col])
        inst_names.append(col)
    inst_arr = np.column_stack(inst_blocks) if inst_blocks else np.empty((len(iids), 0))

    if blocks:
        X = np.column_stack([inst_arr, np.concatenate(blocks, axis=1)]).astype(np.float32)
    else:
        X = inst_arr.astype(np.float32)
    full_names = inst_names + names
    return X, full_names


def load_k1_trivial(task, model, alpha, split):
    """Load K1_trivial.parquet for the given split."""
    base = "analysis_inputs" if split == "val" else "analysis_inputs_test"
    pq = ROOT / f"results/{base}/oracle_outputs/{task}/{model}/{alpha}/K1_trivial.parquet"
    if not pq.exists(): return None
    return pd.read_parquet(pq)


def load_baseline(task, model, alpha, split):
    """Load baseline.parquet (alpha-agnostic) for val or test."""
    base = "analysis_inputs" if split == "val" else "analysis_inputs_test"
    pq = ROOT / f"results/{base}/oracle_outputs/{task}/{model}/baseline.parquet"
    if not pq.exists(): return None
    return pd.read_parquet(pq)
