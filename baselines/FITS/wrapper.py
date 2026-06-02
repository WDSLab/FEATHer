# -*- coding: utf-8 -*-
"""FITS (ICLR 2024 Spotlight) wrapper.

Reference:
    Xu, Zeng, Xu, "FITS: Modeling Time Series with $10k$ Parameters",
    ICLR 2024.

Defaults from `FITS-main/scripts/FITS/etth1.sh` and paper §3 (instance
norm → rFFT → complex linear in the low-frequency subband → irFFT).

The official forward returns (xy, low_xy); we expose only xy. AMP is
disabled globally in the worker because the complex linear layer is
incompatible with `_amp_foreach_non_finite_check_and_unscale_cuda`.
"""

import argparse
import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "FITS-main")


class _FITSWrap(nn.Module):
    def __init__(self, inner, pred_len):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len

    def forward(self, x):
        xy, _ = self.inner(x)
        return xy[:, -self.pred_len:, :]


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.FITS")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Individual freq upsampler per channel — paper Table 1 keeps the
        # shared variant for compactness; ETT scripts use individual=True
        # on long sequences. We default to shared for sub-1K parameter
        # parity with our model.
        individual=getattr(args, "fits_individual", False),
        # Low-pass cut-off — paper §3.2 defines DCF "dominance frequency";
        # ETT scripts use cut_freq tuned per (seq_len, pred_len). seq_len/4
        # is the harmonic-centered default suggested in paper §4.1.
        cut_freq=getattr(args, "fits_cut_freq", 0) or max(1, seq_len // 4),
    )
    inner = mod.Model(cfg)
    return _FITSWrap(inner, pred_len)
