# -*- coding: utf-8 -*-
"""LMS-AutoTSF (arXiv 2024) wrapper.

Reference:
    Delibaşoğlu, Chakraborty, Heintz, "LMS-AutoTSF: Learnable Multi-Scale
    Decomposition and Integrated Autocorrelation for Time Series
    Forecasting", arXiv 2412.06866 (2024).

Defaults from `LMS_AutoTSF-main/scripts/long_term_forecast/ETT_script/
LMSAutoTSF_ETTh1.sh` (channel_independence=0, e_layers=2) plus the
Time-Series-Library `run.py` defaults the script inherits silently
(d_model=128, dropout=0.1, lr=1e-4).

forward is the 4-input Time-Series-Library form; we feed Nones for the
time-mark and decoder inputs.
"""

import argparse
import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "LMS_AutoTSF-main")


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
    mod = isolated_import(_REPO, "models.LMSAutoTSF")
    cfg = argparse.Namespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=getattr(args, "label_len", seq_len // 2),
        pred_len=pred_len,
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        # TS-Library default for d_model — paper does not override.
        d_model=getattr(args, "lms_d_model", 128),
        dropout=getattr(args, "lms_dropout", 0.1),
        # use_norm — paper §3.3 (RevIN-style); ETTh1 script keeps it on.
        use_norm=getattr(args, "lms_use_norm", 1),
        # channel_independence=0 in ETTh1 script (multivariate joint head);
        # paper §3.4 reports this as the default for non-foundation runs.
        channel_independence=getattr(args, "lms_channel_independence", 0),
    )
    inner = mod.Model(cfg)
    return _FourInWrap(inner, pred_len)
