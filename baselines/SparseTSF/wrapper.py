# -*- coding: utf-8 -*-
"""SparseTSF (ICML 2024 Oral / TPAMI 2026) wrapper.

Reference:
    Lin, Lin, Xu, Zhao, Wei, "SparseTSF: Modeling Long-Term Time Series
    Forecasting with 1k Parameters", ICML 2024 Oral.
    Journal extension: TPAMI 2026 — "SparseTSF: Lightweight and Robust
    Time Series Forecasting via Sparse Modeling".

Defaults from `SparseTSF-main/scripts/SparseTSF/linear/etth1.sh`
(period_len=24, model_type=linear) plus paper §3.2.

⚠ TWO HEADS, TWO REGISTRY ENTRIES. This wrapper defaults to `linear`, which is
the paper's headline model — 41 parameters at ETTh1/H=96, the basis of the "1k
parameters" title, and the only variant ported to the board (47 ms). The `mlp`
head adds a d_model=128 hidden layer and costs ~30x more (up to 4,535 params,
essentially FEATHer's own footprint); it is registered separately as
`SparseTSF_MLP` and is NOT covered by the board measurement. Tables print them
as SparseTSF-Linear / SparseTSF-MLP — never as a bare "SparseTSF", since the
two differ by more than an order of magnitude in both size and cost.
"""

import argparse
import os

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "SparseTSF-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.SparseTSF")
    # period_len must divide both seq_len and pred_len. Fall back to the
    # largest valid divisor ≤ 24 (paper's hourly choice) when 24 is not
    # admissible (e.g. SML / Volatility with non-standard periodicities).
    period = getattr(args, "sparsetsf_period_len", 0) or _default_period(seq_len, pred_len)
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Period — paper §3.2 + etth1.sh uses 24 (hourly data).
        period_len=period,
        # MLP head capacity — only used when model_type='mlp'; paper Table 1
        # reports linear as default. ETT mlp script uses d_model=128.
        d_model=getattr(args, "sparsetsf_d_model", 128),
        # 'linear' = the 1K-param paper main variant; 'mlp' = the TPAMI
        # extension's lightweight non-linear head.
        model_type=getattr(args, "sparsetsf_model_type", "linear"),
    )
    return mod.Model(cfg)


def _default_period(seq_len, pred_len):
    for p in (24, 12, 6, 4, 2, 1):
        if seq_len % p == 0 and pred_len % p == 0:
            return p
    return 1


# ---------------------------------------------------------------------------
# 2026-10-06 control arm (user): "what does FEATHer do better than SparseTSF plus a small
# channel-mixing bus?"  This is the UNCHANGED upstream SparseTSF with FEATHer's residual
# channel mixer in front of it and nothing else of FEATHer (no bands, no FFT gate, no
# spectral loss, no SPK re-implementation).  Pre-declared rule and arms:
# manuscript/drafts/control_sparsetsf_mix_plan.md.
#
#   norm = "revin"  FEATHer's instance norm (window mean detached, std not - the same
#                   half-RevIN as FEATHer.py) -> mixer -> SparseTSF -> * std + mean
#   norm = "mean"   SparseTSF's own window-mean removal only (no scale) -> mixer ->
#                   SparseTSF -> + mean.  With the mixer zeroed this is byte-for-byte the
#                   upstream model (SparseTSF subtracts the window mean itself, so the
#                   second subtraction is a no-op).
#   mixer           h <- h + W_up(GELU(W_down h)), W_down: D -> rank, W_up: rank -> D,
#                   per time step over the channel axis = FEATHer.nl_down / nl_up (rank 4).
# The upstream model is built FIRST so its initialisation is identical to the plain
# SparseTSF run with the same seed; the mixer's weights are drawn afterwards.
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseTSFMix(nn.Module):
    def __init__(self, net, n_features, rank=4, norm="revin"):
        super().__init__()
        assert norm in ("revin", "mean"), norm
        self.net = net
        self.norm = norm
        self.rank = rank
        self.nl_down = nn.Linear(n_features, rank)
        self.nl_up = nn.Linear(rank, n_features)

    def forward(self, x):                                   # (B, L, D) -> (B, H, D)
        if self.norm == "revin":
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            h = (x - means) / stdev
        else:
            means = x.mean(1, keepdim=True)
            stdev = None
            h = x - means
        h = h + self.nl_up(F.gelu(self.nl_down(h)))         # the mixing bus
        y = self.net(h)
        if stdev is not None:
            y = y * stdev
        return y + means


def build_mix(args, n_features, seq_len, pred_len, model_type="linear", norm="revin"):
    import copy
    a = copy.copy(args)
    a.sparsetsf_model_type = model_type
    net = build(a, n_features, seq_len, pred_len)
    return SparseTSFMix(net, n_features, rank=getattr(args, "sparsetsf_mix_rank", 4), norm=norm)
