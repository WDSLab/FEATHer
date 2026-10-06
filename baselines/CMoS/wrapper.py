# -*- coding: utf-8 -*-
"""CMoS (ICML 2025) wrapper — chunk-wise spatial correlations + correlation mixing.

Reference:
    Si, Pei, Li, Pei, Xie, "CMoS: Rethinking Time Series Prediction Through the
    Lens of Chunk-wise Spatial Correlations", ICML 2025 (PMLR 267:55493-55510),
    arXiv 2505.19090.  Code: github.com/CSTCloudOps/CMoS (commit b696a0c, 2025-06-19).

Added 2026-10-06 (user): CMoS is a super-lightweight model with an edge-device
claim and is the paper we cite for "lightweight models avoid channel mixing".

Channel handling: CMoS does NOT mix channel values.  Each channel is forecast
from its own past; what is shared is K basis chunk-to-chunk mappings, mixed per
channel by a softmax gate computed from that channel's own (private) conv.

Defaults = the repository's model default `model/CMoS/config.toml`
(seg_size 8, num_map 4, kernel 8, stride 4, dropout 0.05, use_pi false,
lr 8e-4) — the same "model default" rule as every other mfg baseline.  The
per-dataset LTSF configs in `configs/*.toml` use L = 336/720 and searched lr;
our protocol fixes L = 96.  seg_size 8 divides L = 96 and every H.

Note: upstream builds num_map + 1 mapping layers but only uses the first
num_map in forward(); the spare one is counted by num_params but would not be
deployed (flag when reporting parameters).
"""

import argparse
import os

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "CMoS-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "model.CMoS.Model")
    seg = int(getattr(args, "cmos_seg_size", 8))
    assert seq_len % seg == 0 and pred_len % seg == 0, (seq_len, pred_len, seg)
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        seg_size=seg,
        num_map=int(getattr(args, "cmos_num_map", 4)),
        kernel_size=int(getattr(args, "cmos_kernel_size", 8)),
        conv_stride=int(getattr(args, "cmos_conv_stride", 4)),
        c=n_features,
        dropout=float(getattr(args, "cmos_dropout", 0.05)),
        use_pi=bool(getattr(args, "cmos_use_pi", False)),
        period=int(getattr(args, "cmos_period", 24)),
    )
    return mod.Model(cfg)
