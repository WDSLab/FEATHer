# -*- coding: utf-8 -*-
"""PatchTST (ICLR 2023) wrapper.

Reference:
    Nie, Nguyen, Sinthong, Kalagnanam, "A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers", ICLR 2023.

Defaults from `PatchTST-main/PatchTST_supervised/scripts/PatchTST/*.sh`
(canonical ETT setup) and paper §4.2 Table 2.
"""

import argparse
import os

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "PatchTST-main", "PatchTST_supervised")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.PatchTST")
    cfg = argparse.Namespace(
        # Sizes
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        # Backbone — paper §4.2 Table 2 (Multivariate ETT)
        e_layers=getattr(args, "patchtst_e_layers", 3),
        n_heads=getattr(args, "patchtst_n_heads", 16),
        d_model=getattr(args, "patchtst_d_model", 128),
        d_ff=getattr(args, "patchtst_d_ff", 256),
        dropout=getattr(args, "patchtst_dropout", 0.2),
        fc_dropout=getattr(args, "patchtst_fc_dropout", 0.2),
        head_dropout=getattr(args, "patchtst_head_dropout", 0.0),
        individual=getattr(args, "patchtst_individual", False),
        # Patching — paper §3.2; etth1 script uses patch_len=16 stride=8
        patch_len=getattr(args, "patchtst_patch_len", 16),
        stride=getattr(args, "patchtst_stride", 8),
        padding_patch=getattr(args, "patchtst_padding_patch", "end"),
        # RevIN — paper §3.3 (instance-norm wrapper on)
        revin=getattr(args, "patchtst_revin", True),
        affine=getattr(args, "patchtst_affine", False),
        subtract_last=getattr(args, "patchtst_subtract_last", False),
        # Series decomposition (off by default in ETT scripts; on for some
        # weather/traffic settings in the official sweep)
        decomposition=getattr(args, "patchtst_decomposition", False),
        kernel_size=getattr(args, "patchtst_kernel_size", 25),
    )
    return mod.Model(cfg)
