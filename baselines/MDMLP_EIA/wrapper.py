# -*- coding: utf-8 -*-
"""MDMLP-EIA (AAAI 2026) wrapper.

Reference:
    arXiv:2511.09924 — "MDMLP-EIA: Multi-domain Dynamic MLPs with Energy
    Invariant Attention for Time Series Forecasting" (Zhang et al., AAAI 2026).

Defaults from `MDMLP_EIA-main/run.py` argparse defaults:
    model_type='mlp', ma_type='ema', alpha=0.3, beta=0.3, revin=1,
    FreMLP_embed_size=64, FreMLP_hidden_size=256, individual=False.

The official forward(x) returns (B, pred_len, D) directly — no 4-input
wrapping needed.
"""

import argparse
import os

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "MDMLP_EIA-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.MDMLP_EIA")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Heads — paper main is 'mlp' (run.py default)
        model_type=getattr(args, "mdmlp_model_type", "mlp"),
        # AZCF channel fusion uses individual=False by default
        individual=getattr(args, "mdmlp_individual", False),
        # Decomposition — paper §3.1 (DEMA + EMA dual)
        ma_type=getattr(args, "mdmlp_ma_type", "ema"),
        alpha=getattr(args, "mdmlp_alpha", 0.3),
        beta=getattr(args, "mdmlp_beta", 0.3),
        # RevIN — paper §3 (on by default)
        revin=getattr(args, "mdmlp_revin", 1),
        # FreMLP seasonal head capacity — paper §3.2 (Dynamic Capacity Adjustment)
        FreMLP_embed_size=getattr(args, "mdmlp_FreMLP_embed_size", 64),
        FreMLP_hidden_size=getattr(args, "mdmlp_FreMLP_hidden_size", 256),
    )
    return mod.Model(cfg)
