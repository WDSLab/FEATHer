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


def _cut_freq(args, seq_len):
    """Low-pass cut-off (number of rFFT bins kept).

    2026-10-07: official rule (FITS-main/run_longExp_F.py:112-113, every script leaves --cut_freq 0):
        cut_freq = (seq_len // base_T + 1) * H_order + 10
    with H_order / base_T from the dataset's final script. The official values were tuned for
    L = 360/720; when the formula exceeds what our input length allows it is set to that maximum
    (protocol rule: an out-of-range length-dependent value -> the largest value the input length
    allows).  The maximum is seq_len // 2, not seq_len // 2 + 1: FITS upsamples the kept bins by
    (L + H) / L and the result must fit the (L + H) // 2 + 1 output bins (49 * 2 = 98 > 97 at
    L = H = 96 crashes upstream as well). An explicit fits_cut_freq still wins; without
    H_order the old seq_len // 4 default is kept (byte-identical to the pre-10-07 runs).
    """
    explicit = int(getattr(args, "fits_cut_freq", 0) or 0)
    if explicit:
        return explicit
    h_order = getattr(args, "fits_h_order", None)
    if h_order is None:
        return max(1, seq_len // 4)
    base_t = int(getattr(args, "fits_base_t", 24))
    return min((seq_len // base_t + 1) * int(h_order) + 10, seq_len // 2)


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.FITS")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Individual freq upsampler per channel — the final scripts use it only for Weather
        # (scripts/FITS/weather_best.sh); shared elsewhere.
        individual=bool(getattr(args, "fits_individual", False)),
        cut_freq=_cut_freq(args, seq_len),
    )
    inner = mod.Model(cfg)
    return _FITSWrap(inner, pred_len)
