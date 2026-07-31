# -*- coding: utf-8 -*-
"""MixLinear (ICLR 2026) wrapper — ultra-low-resource dual-domain linear.

Reference:
    "MixLinear: Extreme Low Resource Multivariate Time Series Forecasting
    with 0.1K Parameters", ICLR 2026 (arXiv 2410.02081).

The sub-1K rival closest to FEATHer's niche (edge, time+frequency). Defaults
follow run_longExp.py: alpha=0.5 (time/freq mix), lpf=15, period_len=24.

Two upstream quirks handled here (upstream repo is re-clonable, so we adapt in
the wrapper rather than editing it):
  1. MixLinear.forward has leftover debug print() calls -> suppress stdout.
  2. The frequency branch takes the first `lpf` FFT bins of the SEGMENT axis
     (size seg_num_x = ceil(seq_len / period_len)); with our short seq_len=96
     lpf can exceed seg_num_x and crash the FLinear1 = Linear(lpf, 2) matmul.
     Clamp lpf <= seg_num_x.
"""

import argparse
import contextlib
import io
import math
import os

import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "MixLinear-main")


class _QuietMixLinear(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.model(x)


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.MixLinear")
    period_len = int(getattr(args, "mixlinear_period_len", 24))
    seg_num_x = math.ceil(seq_len / period_len)
    lpf = min(int(getattr(args, "mixlinear_lpf", 15)), seg_num_x)
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        period_len=period_len,
        lpf=lpf,
        alpha=float(getattr(args, "mixlinear_alpha", 0.5)),
    )
    return _QuietMixLinear(mod.Model(cfg))
