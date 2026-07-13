# -*- coding: utf-8 -*-
"""SCINet (NeurIPS 2022) wrapper.

Reference:
    Liu, Zeng, Chen, Xu, Lai, Ma, Xu, "SCINet: Time Series Modeling and
    Forecasting with Sample Convolution and Interaction", NeurIPS 2022.
    Official repo: cure-lab/SCINet.

Added 2026-07-13 as the 13th baseline. A TPAMI reviewer (R2 #4) named SCINet as
prior multi-scale work; benchmarking it directly (rather than only
differentiating in Related Work) pre-empts the "you didn't compare against the
multi-scale method" critique — SCINet is the explicit hierarchical
sample-convolution baseline against FEATHer's multi-band frequency decomposition.

Config = the paper's ETT long-horizon setup (run_ETTh.py): single stack, 3
interaction levels, hidden multiplier 4, kernel 5, dropout 0.5, RevIN (RIN) on.
  - num_stacks=1 -> forward returns a single (B, pred_len, D) tensor.
  - num_decoder_layer=1 -> avoids the upstream `.cuda()`-hardcoded decoder path
    (which would break --ngpu N on non-default devices).
  - RIN=True is paper-faithful for LTSF but prints `/// RIN ACTIVATED ///` on
    EVERY forward; the adapter silences that (cosmetic, would otherwise flood
    the sweep logs). We do NOT edit the gitignored upstream file.
"""

import contextlib
import io
import os

import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "SCINet-main")


class _SCINetWrapped(nn.Module):
    """Uniform forward(x) -> (B, pred_len, D); silences the upstream per-forward
    RIN print. Submodule `net` keeps all parameters visible to the worker's
    param count and checkpoint save/load."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.net(x)


def _levels_for(seq_len, want=3):
    # SCINet asserts input_len % 2**num_levels == 0 (binary interaction tree).
    # seq_len=96 admits 3 (96/8=12); guard keeps it valid for other seq_lens.
    for lv in range(want, 0, -1):
        if seq_len % (2 ** lv) == 0:
            return lv
    return 1


def build(args, n_features, seq_len, pred_len):
    SCINet = isolated_import(_REPO, "models.SCINet").SCINet
    net = SCINet(
        output_len=pred_len,
        input_len=seq_len,
        input_dim=n_features,
        hid_size=getattr(args, "scinet_hidden_size", 4),
        num_stacks=1,
        num_levels=getattr(args, "scinet_levels", 0) or _levels_for(seq_len, 3),
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=getattr(args, "scinet_kernel", 5),
        dropout=getattr(args, "scinet_dropout", 0.5),
        single_step_output_One=0,
        positionalE=False,
        modified=True,
        RIN=getattr(args, "scinet_rin", True),
    )
    return _SCINetWrapped(net)
