# -*- coding: utf-8 -*-
"""TimeMixer (ICLR 2024) wrapper.

Reference:
    Wang, Wu, Shi, Hu, Luo, Ma, Zhang, Zhou, "TimeMixer: Decomposable
    Multiscale Mixing for Time Series Forecasting", ICLR 2024.

Defaults from `TimeMixer-main/scripts/long_term_forecast/ETT_script/
TimeMixer_ETTh1_unify.sh` (d_model=16, d_ff=32, e_layers=2,
down_sampling_layers=3, down_sampling_window=2).
"""

import argparse
import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "TimeMixer-main")


class _FourInWrap(nn.Module):
    def __init__(self, inner, pred_len):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len

    def forward(self, x):
        out = self.inner(x, None, None, None)
        if isinstance(out, tuple):
            out = out[0]
        return out[:, -self.pred_len:, :]


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.TimeMixer")
    cfg = argparse.Namespace(
        # Task + sizes
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=getattr(args, "label_len", seq_len),
        pred_len=pred_len,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        # Backbone — paper §4.2 unify-script defaults (intentionally small)
        d_model=getattr(args, "tm_d_model", 16),
        d_ff=getattr(args, "tm_d_ff", 32),
        e_layers=getattr(args, "tm_e_layers", 2),
        dropout=getattr(args, "tm_dropout", 0.1),
        embed="timeF",
        freq="h",
        # Multi-scale — paper §3.1 (PDM); unify-script: 3 layers ×2 window
        down_sampling_layers=getattr(args, "tm_down_sampling_layers", 3),
        down_sampling_window=getattr(args, "tm_down_sampling_window", 2),
        down_sampling_method=getattr(args, "tm_down_sampling_method", "avg"),
        # Mode — channel_independence=1 in unify script, but the ETTh1
        # multi-target script keeps it off; keep on for sub-1K parity
        # since channel-independent is the paper's main claim.
        channel_independence=getattr(args, "tm_channel_independence", 1),
        decomp_method=getattr(args, "tm_decomp_method", "moving_avg"),
        moving_avg=getattr(args, "tm_moving_avg", 25),
        top_k=getattr(args, "tm_top_k", 5),
        use_norm=getattr(args, "tm_use_norm", 1),
        use_future_temporal_feature=0,
    )
    inner = mod.Model(cfg)
    return _FourInWrap(inner, pred_len)
