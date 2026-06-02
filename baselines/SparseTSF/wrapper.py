# -*- coding: utf-8 -*-
"""SparseTSF (ICML 2024 Oral / TPAMI 2026) wrapper.

Reference:
    Lin, Lin, Xu, Zhao, Wei, "SparseTSF: Modeling Long-Term Time Series
    Forecasting with 1k Parameters", ICML 2024 Oral.
    Journal extension: TPAMI 2026 — "SparseTSF: Lightweight and Robust
    Time Series Forecasting via Sparse Modeling".

Defaults from `SparseTSF-main/scripts/SparseTSF/linear/etth1.sh`
(period_len=24, model_type=linear) plus paper §3.2.

Note: SparseTSF claims robustness as an *implicit* property of the sparse
design but reports no explicit noise-robustness experiments. Our Phase 3
sweep evaluates this empirically alongside FEATHer.
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
