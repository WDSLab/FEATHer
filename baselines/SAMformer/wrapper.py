# -*- coding: utf-8 -*-
"""SAMformer (ICML 2024 oral) wrapper — channel-wise attention + SAM training.

Reference:
    Ilbert et al., "SAMformer: Unlocking the Potential of Transformers in Time
    Series Forecasting with Sharpness-Aware Minimization and Channel-Wise
    Attention", ICML 2024 (arXiv 2402.10198).
    Code: github.com/romilbert/samformer, PyTorch port `samformer_pytorch/`
    (commit 71f10ea, 2025-06-05, Apache-2.0).

The established peer-reviewed lightweight model that
MIXES channel values (attention over channel tokens), about DLinear's size.

Defaults = the PyTorch trainer's defaults: hid_dim 16, RevIN (affine), Adam
lr 1e-3, weight_decay 1e-5, SAM rho 0.5.  SAM is part of the method; the worker
switches to SAM only for this model (two forward/backward passes per step).
"""

import os

import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "SAMformer-main", "samformer_pytorch")


class _SAMformerWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):                       # (B, L, D) -> (B, H, D)
        out = self.net(x.transpose(1, 2), flatten_output=False)   # (B, D, H)
        return out.transpose(1, 2)


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "samformer.samformer")
    net = mod.SAMFormerArchitecture(num_channels=n_features, seq_len=seq_len,
                                    hid_dim=int(getattr(args, "samformer_hid_dim", 16)),
                                    pred_horizon=pred_len, use_revin=True)
    return _SAMformerWrap(net)


def make_sam(params, lr, rho=0.5, weight_decay=1e-5):
    import torch
    sam = isolated_import(_REPO, "samformer.utils.sam")
    return sam.SAM(params, base_optimizer=torch.optim.Adam, rho=rho, lr=lr,
                   weight_decay=weight_decay)
