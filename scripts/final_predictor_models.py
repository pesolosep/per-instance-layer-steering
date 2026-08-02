"""Shared model classes for the final_predictor pipeline.

Lives in a stable module path so joblib can unpickle from any caller
(train script, eval scripts, notebook).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LAYERS = 32


def listwise_kl(pred, target):
    log_p = F.log_softmax(pred, dim=1)
    q     = F.softmax(target, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean")


class MLPRanker(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=N_LAYERS, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinearRanker(nn.Module):
    def __init__(self, in_dim, out_dim=N_LAYERS):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x)
