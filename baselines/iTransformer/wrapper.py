# -*- coding: utf-8 -*-
"""iTransformer (ICLR 2024) wrapper.

Reference:
    Liu, Hu, Zhang, Lei, Zhou, Li, Liu, Yang, Long, "iTransformer:
    Inverted Transformers Are Effective for Time Series Forecasting",
    ICLR 2024.

Defaults from `iTransformer-main/scripts/multivariate_forecasting/ETT/
iTransformer_ETTh1.sh` (d_model=256, d_ff=256, e_layers=2) and
`iTransformer-main/run.py` argparse defaults for the rest.

The official model.forward takes (x_enc, x_mark_enc, x_dec, x_mark_dec).
Our worker passes only x; the adapter feeds None for the extras because
DataEmbedding_inverted accepts x_mark=None.
"""

import argparse
import os
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "iTransformer-main")


class _FourInWrap(nn.Module):
    def __init__(self, inner, pred_len):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len

    def forward(self, x):
        out = self.inner(x, None, None, None)
        if isinstance(out, tuple):
            out = out[0]
        return out[:, -self.pred_len:, :]


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "model.iTransformer")
    cfg = argparse.Namespace(
        seq_len=seq_len,
        pred_len=pred_len,
        # Embedding / norm — paper §3.2
        output_attention=False,
        use_norm=True,
        embed="timeF",
        freq="h",
        class_strategy="projection",
        # Backbone — official ETTh1 script (d_model=256, d_ff=256, e=2)
        d_model=getattr(args, "itrans_d_model", 256),
        d_ff=getattr(args, "itrans_d_ff", 256),
        e_layers=getattr(args, "itrans_e_layers", 2),
        n_heads=getattr(args, "itrans_n_heads", 8),
        dropout=getattr(args, "itrans_dropout", 0.1),
        factor=getattr(args, "itrans_factor", 1),
        activation="gelu",
    )
    inner = mod.Model(cfg)
    return _FourInWrap(inner, pred_len)
