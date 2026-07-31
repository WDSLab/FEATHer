# -*- coding: utf-8 -*-
"""CycleNet (NeurIPS 2024) wrapper — Residual Cycle Forecasting.

Reference:
    "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic
    Patterns", NeurIPS 2024 (arXiv 2409.18479).

A learnable recurrent cycle removed from / added back to a linear (or shallow
MLP) backbone. We default to the LINEAR backbone (smaller, edge-relevant); the
upstream default is 'mlp' (d_model=512 ~ 100K params). `cycle` is the period,
set per-dataset like our other period-tied models (FFT-detected).

Upstream forward is `forward(x, cycle_index)`: it needs the per-sample cyclic
phase. Our DataLoader has no per-sample cycle position, so we feed zeros —
identical treatment to TQNet (see CLAUDE.md gotcha). This holds the cycle phase
fixed across all batches (a documented, consistent limitation).
"""

import argparse
import os

import torch
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "CycleNet-main")


class _CycleNetWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, idx)


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.CycleNet")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        cycle=int(getattr(args, "cyclenet_cycle", 24)),
        model_type=getattr(args, "cyclenet_model_type", "linear"),
        d_model=int(getattr(args, "d_model", 512)),
        use_revin=bool(getattr(args, "use_revin", True)),
    )
    return _CycleNetWrap(mod.Model(cfg))
