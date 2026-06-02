# -*- coding: utf-8 -*-
"""TimesNet (ICLR 2023) wrapper.

Reference:
    Wu, Hu, Liu, Zhou, Long, "TimesNet: Temporal 2D-Variation Modeling
    for General Time Series Analysis", ICLR 2023.

Defaults from `TimesNet-main/scripts/long_term_forecast/ETT_script/
TimesNet_ETTh1.sh` (d_model=16, d_ff=32, e_layers=2, top_k=5).

cuFFT half-precision does not support non-power-of-two signal sizes, so
AMP is forcibly off in the worker — TimesNet calls `FFT_for_Period` on
length (seq_len + pred_len) which is rarely a power of two.
"""

import argparse
import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "TimesNet-main")


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
    mod = isolated_import(_REPO, "models.TimesNet")
    cfg = argparse.Namespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=getattr(args, "label_len", seq_len // 2),
        pred_len=pred_len,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        # Backbone — paper Table 13 + ETTh1 script (intentionally small)
        d_model=getattr(args, "tn_d_model", 16),
        d_ff=getattr(args, "tn_d_ff", 32),
        e_layers=getattr(args, "tn_e_layers", 2),
        # Inception block — paper §3.2
        num_kernels=getattr(args, "tn_num_kernels", 6),
        # Period detection top-k — paper §3.1 (FFT_for_Period)
        top_k=getattr(args, "tn_top_k", 5),
        dropout=getattr(args, "tn_dropout", 0.1),
        embed="timeF",
        freq="h",
    )
    inner = mod.Model(cfg)
    return _FourInWrap(inner, pred_len)
