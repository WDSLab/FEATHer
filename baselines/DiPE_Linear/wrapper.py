# -*- coding: utf-8 -*-
"""DiPE-Linear (DASFAA 2026) wrapper.

Reference:
    arXiv:2411.17257 — "Disentangled Parameter-Efficient Linear Model for
    Long-Term Time Series Forecasting" (wintertee/DiPE-Linear, DASFAA 2026).

Defaults from `DiPE_Linear-main/configs/models/DiPE/base.yaml` (lr=1e-3,
50 epochs, MSE loss) and `timeprophet/models/DiPE.py` constructor:
use_revin=True, use_time_w=True, use_freq_w=True, num_experts=1.

The DiPE module is imported under `timeprophet.models.DiPE` and exposes a
class `DiPE` (not `Model`). forward(x) returns (B, L, D) directly.

Requires `einops` (already in the feather env via the iTransformer install).
"""

import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "DiPE_Linear-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "timeprophet.models.DiPE")
    DiPE = mod.DiPE
    return DiPE(
        input_len=seq_len,
        output_len=pred_len,
        input_features=n_features,
        output_features=n_features,
        # base.yaml + paper §3: instance norm wrapper on
        use_revin=getattr(args, "dipe_use_revin", True),
        # Static frequency / time weights — paper main variant
        use_time_w=getattr(args, "dipe_use_time_w", True),
        use_freq_w=getattr(args, "dipe_use_freq_w", True),
        # Single-expert routing by default (paper main); MoE variants
        # need >1 + routed loss recipe that doesn't fit our unified loop.
        num_experts=getattr(args, "dipe_num_experts", 1),
        individual_f=getattr(args, "dipe_individual_f", False),
        individual_t=getattr(args, "dipe_individual_t", False),
        individual_c=getattr(args, "dipe_individual_c", False),
        loss_alpha=0.0,
        t_loss="mse",
    )
