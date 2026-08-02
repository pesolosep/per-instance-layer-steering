"""Train MLP-based predictor heads (faithful WhereToSteer + combined variants).

Variants:
  clf_reg_w2s_mlp        — MLP on raw Qwen3 1024-d embedding (faithful W2S).
  clf_reg_lite_w2s_mlp   — MLP on production 89-d (only_d_pair 64 + Qwen3 PCA-25).

Architecture: Linear(in_dim -> 64) -> ReLU -> Dropout(0.3) -> Linear(64 -> 32)
Loss: MSE on align target (z-scored per instance)
Optimizer: Adam(lr=1e-3, weight_decay=1e-4), 200 epochs (fixed).
"""
import sys, time, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.predictor_eda._util import (
    cells, pilot_dir, load_features, load_embeddings, load_baseline,
    load_k1_trivial, N_LAYERS, cell_label, softmax2,
    build_X_no_gold, NO_GOLD_D_FEATURES,
)

ROOT = Path(__file__).resolve().parent.parent   # repo root, like the sibling scripts
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_HIDDEN = 64
MLP_DROPOUT = 0.3
SEED = 42


class MLPRanker(nn.Module):
    def __init__(self, in_dim, hidden=MLP_HIDDEN, out_dim=N_LAYERS, dropout=MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def zscore_per_instance(mat):
    m = mat.mean(axis=1, keepdims=True)
    s = mat.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (mat - m) / s


def build_align_labels(k1, iids, bl):
    k1_local = k1.copy()
    k1_local["layer"] = k1_local["layer_set"].apply(
        lambda x: int(x[0]) if hasattr(x, "__len__") else int(x))
    bl_by = bl.set_index("instance_id") if "instance_id" in bl.columns else bl
    align_mat = np.zeros((len(iids), N_LAYERS), dtype=np.float32)
    for i, iid in enumerate(iids):
        gold = int(bl_by.at[iid, "gold_sign"])
        py_b, pn_b = softmax2(bl_by.at[iid, "base_logit_yes"], bl_by.at[iid, "base_logit_no"])
        bp = py_b if gold == 1 else pn_b
        sub = k1_local[k1_local["instance_id"] == iid]
        for _, r in sub.iterrows():
            py_s, pn_s = softmax2(r["steered_logit_yes"], r["steered_logit_no"])
            p_t = py_s if gold == 1 else pn_s
            align_mat[i, int(r["layer"])] = p_t - bp
    return zscore_per_instance(align_mat)


def build_lite(feat, bl, k1, iids):
    X, _ = build_X_no_gold(feat, bl, k1, iids,
                            include_geom=(), include_d=NO_GOLD_D_FEATURES,
                            include_proj_rank=False, include_inst=())
    return X


def train_mlp(X_v, y_v, in_dim, epochs=MLP_EPOCHS, lr=MLP_LR, wd=MLP_WEIGHT_DECAY):
    torch.manual_seed(SEED)
    model = MLPRanker(in_dim=in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xv = torch.from_numpy(X_v).float()
    yv = torch.from_numpy(y_v).float()
    model.train(True)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xv)
        loss = F.mse_loss(pred, yv)
        loss.backward()
        opt.step()
    model.train(False)
    return model


def predict_mlp(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).cpu().numpy()


def run_cell(task, model_tag, alpha):
    pilot = pilot_dir(task, model_tag, alpha)
    feat_val = load_features(task, model_tag, alpha, "val")
    bl_v = load_baseline(task, model_tag, alpha, "val")
    k1_v = load_k1_trivial(task, model_tag, alpha, "val")
    emb_v = load_embeddings(task, model_tag, alpha, "val")
    if any(x is None for x in [feat_val, bl_v, k1_v, emb_v]):
        return None

    iids_v = sorted(emb_v["instance_id"].astype(int).unique().tolist())
    y_align = build_align_labels(k1_v, iids_v, bl_v)

    emb_v_mat = emb_v.set_index("instance_id").loc[iids_v].values.astype(np.float32)
    lite_v = build_lite(feat_val, bl_v, k1_v, iids_v)

    pca25 = PCA(n_components=min(25, emb_v_mat.shape[1], emb_v_mat.shape[0]-1)).fit(emb_v_mat)
    emb_v_pc25 = pca25.transform(emb_v_mat).astype(np.float32)
    X_combined = np.concatenate([lite_v, emb_v_pc25], axis=1)

    # Variant 1: faithful W2S (Qwen3 1024-d raw -> MLP)
    sc_emb = StandardScaler().fit(emb_v_mat)
    Xv_emb = sc_emb.transform(emb_v_mat)
    model_w2s = train_mlp(Xv_emb, y_align, in_dim=emb_v_mat.shape[1])
    joblib.dump({
        "scaler": sc_emb, "regressor": model_w2s, "head_type": "mlp",
        "feature_variant": "raw_qwen3_1024", "n_features_total": emb_v_mat.shape[1],
        "epochs": MLP_EPOCHS, "lr": MLP_LR, "weight_decay": MLP_WEIGHT_DECAY,
        "hidden": MLP_HIDDEN, "dropout": MLP_DROPOUT,
    }, pilot / "models" / "clf_reg_w2s_mlp.joblib")

    # Variant 2: production combined (89-d) -> MLP
    sc_comb = StandardScaler().fit(X_combined)
    Xv_comb = sc_comb.transform(X_combined)
    model_combo = train_mlp(Xv_comb, y_align, in_dim=X_combined.shape[1])
    joblib.dump({
        "scaler": sc_comb, "regressor": model_combo, "head_type": "mlp",
        "pca": pca25,
        "feature_variant": "lite_plus_emb_pca25", "n_features_total": X_combined.shape[1],
        "epochs": MLP_EPOCHS, "lr": MLP_LR, "weight_decay": MLP_WEIGHT_DECAY,
        "hidden": MLP_HIDDEN, "dropout": MLP_DROPOUT,
    }, pilot / "models" / "clf_reg_lite_w2s_mlp.joblib")

    return True


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
        print(f"  {i+1}/{len(cell_list)}  {cell_label(task, model_tag, alpha)}  ({(time.time()-t0)/60:.1f}m)")
    print(f"\nTrained {n_ok}/{len(cell_list)} cells x 2 MLP variants = {n_ok*2} joblibs")
    print(f"Total wall: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
