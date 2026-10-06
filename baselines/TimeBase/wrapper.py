# -*- coding: utf-8 -*-
"""TimeBase (ICML 2025 spotlight) wrapper — minimalist basis forecaster.

Reference:
    Huang et al., "TimeBase: The Power of Minimalism in Efficient Long-term Time
    Series Forecasting", ICML 2025 (PMLR 267:26227-26246).
    Code: github.com/hqh0728/TimeBase (commit 369b330, 2025-09-04, MIT).

Added 2026-10-06 (user): smallest peer-reviewed long-horizon forecaster with an
explicit edge claim; channel-independent (one shared segment->basis->segment map).

Defaults = the official ETTh1 script (scripts/etth1.sh): period_len 24,
basis_num 6, use_period_norm 1, use_orthogonal 1 with orthogonal_weight 0.16,
individual 0, Adam lr 1e-1 — the same "ETTh1 script = model default" rule as
SparseTSF.  period_len follows the per-dataset FFT rule of SparseTSF (divides
L and H).  The orthogonality regulariser is part of the method: the wrapper
stores weight * loss in `_aux_loss` during training and the worker adds it.
"""

import argparse
import os

import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "TimeBase-main")


class _TimeBaseWrap(nn.Module):
    def __init__(self, model, ortho_weight):
        super().__init__()
        self.model = model
        self.ortho_weight = float(ortho_weight)
        self._aux_loss = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out, ortho = out
            self._aux_loss = self.ortho_weight * ortho if self.training else None
        return out


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.TimeBase")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        period_len=int(getattr(args, "timebase_period_len", 24)),
        basis_num=int(getattr(args, "timebase_basis_num", 6)),
        use_period_norm=int(getattr(args, "timebase_use_period_norm", 1)),
        use_orthogonal=int(getattr(args, "timebase_use_orthogonal", 1)),
        individual=int(getattr(args, "timebase_individual", 0)),   # official Weather script: 1
    )
    return _TimeBaseWrap(mod.Model(cfg), getattr(args, "timebase_orthogonal_weight", 0.16))
