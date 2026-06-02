# -*- coding: utf-8 -*-
"""TQNet (ICML 2025) wrapper.

Reference:
    Lin, Lin, Wei, Wu, Xu, Wei, "Temporal Query Network for Efficient
    Multivariate Time Series Forecasting", ICML 2025.

Successor to CycleNet by the same SCUT lab. Defaults from
`TQNet-main/scripts/TQNet/etth1.sh` (cycle=24, dropout=0.5, lr=1e-3) and
paper §3.2 (single-layer attention + lightweight MLP).

The official model.forward expects an extra `cycle_index` tensor (each
sample's position within the dataset's daily / weekly cycle). Our
DataLoader does not expose this, so we pass `torch.zeros(B, dtype=long)`
which fixes the temporal queries to a single phase. Sub-optimal vs paper
but consistent across all batches.
"""

import argparse
import os
import torch
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "TQNet-main")


class _TQNetWrap(nn.Module):
    def __init__(self, inner, pred_len):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len

    def forward(self, x):
        B = x.shape[0]
        cycle_index = torch.zeros(B, dtype=torch.long, device=x.device)
        out = self.inner(x, cycle_index)
        if isinstance(out, tuple):
            out = out[0]
        return out[:, -self.pred_len:, :]


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.TQNet")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Cycle length — paper §3.2 + etth1.sh uses 24 (hourly).
        cycle=getattr(args, "tqnet_cycle", 24),
        # Head — paper §3.3 uses MLP variant for ETT (mlp default).
        model_type="mlp",
        d_model=getattr(args, "tqnet_d_model", 128),
        # ETTh1 script dropout=0.5 (high regularization to compensate for
        # parameter-rich MLP + attention combo).
        dropout=getattr(args, "tqnet_dropout", 0.5),
        # Reversible Instance Norm — paper §3.2 keeps it on.
        use_revin=getattr(args, "tqnet_use_revin", True),
    )
    inner = mod.Model(cfg)
    return _TQNetWrap(inner, pred_len)
