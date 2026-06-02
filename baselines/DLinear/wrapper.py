# -*- coding: utf-8 -*-
"""DLinear (AAAI 2023) wrapper.

Reference:
    Zeng, Chen, Zhang, Xu, "Are Transformers Effective for Time Series
    Forecasting?", AAAI 2023.

Defaults follow `DLinear-main/scripts/EXP-LongForecasting/Linear/etth1.sh`
and `models/DLinear.py:48` (`kernel_size = 25` for the moving-average
decomposition).
"""

import argparse
import os

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "DLinear-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "models.DLinear")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        # Channel-wise input/output count (paper §3 — "individual" linear
        # heads per channel are switched on for D>1 in the official ETT
        # scripts; we keep the safer default of shared weights since some
        # datasets have D in the hundreds).
        enc_in=n_features,
        individual=getattr(args, "dlinear_individual", False),
    )
    return mod.Model(cfg)
