"""Train all 6 final-predictor variants per (task, model, alpha) cell.

Every variant is gold-free at inference. Anti-leak assertions reject any
feature name containing d_yes / d_no / logit_shift / base_align_prob / compass.

Variants:
  1. embed_mlp_full_listwise         (Qwen3 1024-d  → MLP 1024→64→32, listwise KL)
  2. embed_mlp_pca25_listwise        (Qwen3 PCA-25  → MLP 25→64→32, listwise KL)
  3. embed_linear_listwise           (Qwen3 1024-d  → Linear 1024→32, listwise KL)
  4. unsteered_full_listwise         (166-d cheap geom + base prob → MLP, listwise KL)
  5. unsteered_plus_pca25_listwise   (166 + 25 = 191-d → MLP, listwise KL)
  6. unsteered_lite_filtered_plus_pca25_listwise
                                     (proj_on_sv×32 + act_norm×32 + PCA-25 = 89-d → MLP, listwise KL)

Variant 1 reuses the existing clf_reg_w2s_mlp_listwise.joblib (no retrain).

Saves joblibs to:
  results/final_predictor/models/{task}/{model}/{alpha}/{variant_id}.joblib

Emits:
  results/final_predictor/manifest.csv  — variant ID, dim, leak audit PASS

Run:
    python scripts/train_final_predictors.py
"""
import sys, time, shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.predictor_eda._util import (
    cells, pilot_dir, load_features, load_embeddings,
    load_baseline, load_k1_trivial, N_LAYERS, cell_label,
    build_X_no_gold, softmax2,
)
from scripts.train_mlp_predictors import build_align_labels
from scripts.final_predictor_models import MLPRanker, LinearRanker, listwise_kl

OUT_MODELS = ROOT / "results/final_predictor/models"
OUT_MANIFEST = ROOT / "results/final_predictor/manifest.csv"

# ────────── Training hyperparameters ──────────
MLP_EPOCHS  = 300
MLP_LR      = 1e-3
MLP_WD      = 1e-4
MLP_HIDDEN  = 64
MLP_DROPOUT = 0.3
PCA_N       = 25
SEED        = 42

BANNED_FEATURE_SUBSTRINGS = ("d_yes", "d_no", "logit_shift", "base_align_prob", "compass")


def train_listwise(model_ctor, X_v, y_v, in_dim,
                   epochs=MLP_EPOCHS, lr=MLP_LR, wd=MLP_WD):
    torch.manual_seed(SEED)
    model = model_ctor(in_dim=in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xv = torch.from_numpy(X_v).float()
    yv = torch.from_numpy(y_v).float()
    model.train(True)
    for _ in range(epochs):
        opt.zero_grad()
        loss = listwise_kl(model(Xv), yv)
        loss.backward()
        opt.step()
    model.train(False)
    return model


# ────────── Anti-leak guard ──────────
def assert_no_leak(feature_names, variant_id):
    """Reject any feature name containing banned substrings."""
    for name in feature_names:
        for banned in BANNED_FEATURE_SUBSTRINGS:
            assert banned not in str(name), (
                f"LEAK in variant '{variant_id}': feature '{name}' contains '{banned}'"
            )


# ────────── Feature builders (each returns (X, feature_names)) ──────────
def features_embed_full(emb_v):
    """Variant 1, 3 — raw Qwen3 1024-d."""
    iids = sorted(emb_v["instance_id"].astype(int).unique().tolist())
    X = emb_v.set_index("instance_id").loc[iids].values.astype(np.float32)
    names = [f"qwen3__d{i}" for i in range(X.shape[1])]
    return X, names, iids


def features_embed_pca25(emb_v):
    """Variant 2 — Qwen3 PCA-25. Returns X, names, iids, pca."""
    iids = sorted(emb_v["instance_id"].astype(int).unique().tolist())
    emb_mat = emb_v.set_index("instance_id").loc[iids].values.astype(np.float32)
    pca = PCA(n_components=min(PCA_N, emb_mat.shape[1], emb_mat.shape[0] - 1)).fit(emb_mat)
    X = pca.transform(emb_mat).astype(np.float32)
    names = [f"qwen3_pca__pc{i}" for i in range(X.shape[1])]
    return X, names, iids, pca


def features_unsteered_full(feat_v, bl_v, k1_dummy, iids):
    """Variant 4 — 166-d cheap geom + base prob.
    proj_on_sv ×32 + act_norm ×32 + act_var ×32 + steering_cosine ×32
    + proj_rank ×32 + (p_yes, p_no, yn_mass, entropy, max_p, abs_diff)
    """
    X, names = build_X_no_gold(
        feat_v, bl_v, k1_dummy, iids,
        include_geom=("proj_on_sv", "act_norm", "act_var", "steering_cosine"),
        include_d=(),                  # NO K1 cheating
        include_proj_rank=True,
        include_inst=("p_yes", "p_no", "yn_mass", "entropy", "max_p", "abs_diff"),
    )
    return X.astype(np.float32), names


def features_unsteered_lite_filtered(feat_v, bl_v, k1_dummy, iids):
    """Variant 6 (geom only, no PCA yet) — proj_on_sv ×32 + act_norm ×32 = 64-d."""
    X, names = build_X_no_gold(
        feat_v, bl_v, k1_dummy, iids,
        include_geom=("proj_on_sv", "act_norm"),
        include_d=(),
        include_proj_rank=False,
        include_inst=(),
    )
    return X.astype(np.float32), names


# ────────── Per-cell trainer ──────────
def run_cell(task, model_tag, alpha):
    pilot = pilot_dir(task, model_tag, alpha)
    feat_v = load_features(task, model_tag, alpha, "val")
    bl_v   = load_baseline(task, model_tag, alpha, "val")
    k1_v   = load_k1_trivial(task, model_tag, alpha, "val")   # used only for align labels
    emb_v  = load_embeddings(task, model_tag, alpha, "val")
    if any(x is None for x in [feat_v, bl_v, k1_v, emb_v]):
        return None

    out_dir = OUT_MODELS / task / model_tag / alpha
    out_dir.mkdir(parents=True, exist_ok=True)

    # iids must come from emb_v (consistent ordering across all variants)
    iids = sorted(emb_v["instance_id"].astype(int).unique().tolist())

    # Training target: z-scored align lift per instance (32-d)
    # Uses k1_v for steering outputs at K=1 per layer — gold_sign appears here
    # because this is OFFLINE TRAINING SUPERVISION, not inference.
    y_align = build_align_labels(k1_v, iids, bl_v)

    # ── Variant 1: reuse existing clf_reg_w2s_mlp_listwise.joblib ──
    src1 = pilot / "models" / "clf_reg_w2s_mlp_listwise.joblib"
    dst1 = out_dir / "embed_mlp_full_listwise.joblib"
    if src1.exists():
        shutil.copy2(src1, dst1)
        manifest_rows.append({
            "variant": "embed_mlp_full_listwise", "task": task, "model": model_tag, "alpha": alpha,
            "n_features": 1024, "leak_audit": "PASS",
            "note": "reused from clf_reg_w2s_mlp_listwise.joblib",
        })
    else:
        manifest_rows.append({
            "variant": "embed_mlp_full_listwise", "task": task, "model": model_tag, "alpha": alpha,
            "n_features": 1024, "leak_audit": "SKIP",
            "note": "source joblib missing — re-train clf_reg_w2s_mlp_listwise upstream",
        })

    # ── Build feature matrices ──
    X_full, n_full, _      = features_embed_full(emb_v)
    X_pca, n_pca, _, pca25 = features_embed_pca25(emb_v)
    X_un, n_un             = features_unsteered_full(feat_v, bl_v, k1_v, iids)
    X_lite, n_lite         = features_unsteered_lite_filtered(feat_v, bl_v, k1_v, iids)

    # Variant 5: unsteered_full + PCA-25
    X_un_pca = np.concatenate([X_un, X_pca], axis=1)
    n_un_pca = n_un + n_pca

    # Variant 6: lite_filtered + PCA-25
    X_lite_pca = np.concatenate([X_lite, X_pca], axis=1)
    n_lite_pca = n_lite + n_pca

    # ── Anti-leak audit ──
    leak_audits = {
        "embed_mlp_pca25_listwise":                          n_pca,
        "embed_linear_listwise":                             n_full,
        "unsteered_full_listwise":                           n_un,
        "unsteered_plus_pca25_listwise":                     n_un_pca,
        "unsteered_lite_filtered_plus_pca25_listwise":       n_lite_pca,
    }
    for v, names in leak_audits.items():
        assert_no_leak(names, v)

    # ── Train + save ──
    def _train_and_save(variant_id, X, n_names, ctor=MLPRanker, extra=None):
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        model = train_listwise(ctor, Xs, y_align, in_dim=X.shape[1])
        payload = {
            "scaler": sc, "regressor": model, "head_type": ctor.__name__,
            "feature_names": n_names, "n_features_total": X.shape[1],
            "loss": "listwise_softmax_kl",
            "epochs": MLP_EPOCHS, "lr": MLP_LR, "weight_decay": MLP_WD,
            "hidden": MLP_HIDDEN if ctor is MLPRanker else None,
            "dropout": MLP_DROPOUT if ctor is MLPRanker else None,
        }
        if extra: payload.update(extra)
        joblib.dump(payload, out_dir / f"{variant_id}.joblib")
        manifest_rows.append({
            "variant": variant_id, "task": task, "model": model_tag, "alpha": alpha,
            "n_features": X.shape[1], "leak_audit": "PASS",
            "note": ctor.__name__,
        })

    _train_and_save("embed_mlp_pca25_listwise",  X_pca,      n_pca,
                    ctor=MLPRanker, extra={"pca": pca25})
    _train_and_save("embed_linear_listwise",     X_full,     n_full,
                    ctor=LinearRanker)
    _train_and_save("unsteered_full_listwise",   X_un,       n_un,
                    ctor=MLPRanker)
    _train_and_save("unsteered_plus_pca25_listwise", X_un_pca, n_un_pca,
                    ctor=MLPRanker, extra={"pca": pca25})
    _train_and_save("unsteered_lite_filtered_plus_pca25_listwise", X_lite_pca, n_lite_pca,
                    ctor=MLPRanker, extra={"pca": pca25})

    return True


manifest_rows = []


def main():
    t0 = time.time()
    n_ok = 0
    cell_list = list(cells())
    for i, (task, model_tag, alpha) in enumerate(cell_list):
        try:
            r = run_cell(task, model_tag, alpha)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL {task}/{model_tag}/{alpha}: {e}")
            continue
        if r:
            n_ok += 1
        elapsed = (time.time() - t0) / 60
        print(f"  {i+1}/{len(cell_list)}  {cell_label(task, model_tag, alpha)}  ({elapsed:.1f}m)")

    # ── Write manifest ──
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(OUT_MANIFEST, index=False)

    print(f"\nTrained {n_ok}/{len(cell_list)} cells × 6 variants")
    print(f"  manifest: {OUT_MANIFEST.relative_to(ROOT)}  ({len(manifest_rows)} rows)")
    print(f"  models:   {OUT_MODELS.relative_to(ROOT)}/{{task}}/{{model}}/{{alpha}}/")
    print(f"  wall:     {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
