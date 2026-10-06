# -*- coding: utf-8 -*-
"""PhaseFormer (ICLR 2026) wrapper — phase tokens + cross-phase routing.

Reference:
    Niu, Deng, Tong, "PhaseFormer: From Patches to Phases for Efficient and
    Effective Time Series Forecasting", ICLR 2026 (arXiv 2510.04134).
    Code: github.com/neumyor/PhaseFormer_TSL (Time-Series-Library port of
    github.com/neumyor/PhaseFormer; commit ed1db61, 2026-03-10, MIT).

Added 2026-10-06 (user): newest peer-reviewed ~1K-parameter forecaster.
Channel-independent ("without cross-channel fusion": every channel is routed
on its own with shared weights).

Defaults = the official ETTh1 setting for H = 96/192/336 in the original repo's
run_etth1.py (the ~1K lightweight configuration): 3 layers, latent_dim 4,
8 routers, 1 attention head, attention dropout 0.1, phase positional embedding,
RevIN without affine, lr 1e-3.  The paper switches to a 10x wider setting at
ETTh1 H = 720; we keep the lightweight setting at every H (the comparison is
among lightweight models).  The paper trains with a Huber loss; we use MSE like
every other model (protocol).  period_len follows the SparseTSF FFT rule.
"""

import argparse
import os

import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "PhaseFormer-main")


class _PhaseFormerWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.PhaseFormer")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        period_len=int(getattr(args, "phaseformer_period_len", 24)),
        phase_layers=int(getattr(args, "phaseformer_layers", 3)),
        latent_dim=int(getattr(args, "phaseformer_latent_dim", 4)),
        phase_encoder_hidden=16,
        predictor_hidden=32,
        phase_num_routers=int(getattr(args, "phaseformer_routers", 8)),
        phase_attn_heads=int(getattr(args, "phaseformer_heads", 1)),
        phase_attn_dropout=float(getattr(args, "phaseformer_attn_dropout", 0.1)),
        phase_attn_use_relpos=True,
        phase_use_pos_embed=bool(getattr(args, "phaseformer_pos_embed", True)),   # Traffic: False
        use_revin=True,
        revin_affine=False,
    )
    return _PhaseFormerWrap(mod.Model(cfg))
