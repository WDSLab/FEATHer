# -*- coding: utf-8 -*-
"""TCN forecaster — dilated causal temporal conv net (Bai, Kolter & Koltun,
arXiv 2018), a common PdM / industrial-sensor practitioner baseline.

Stacked residual blocks of dilated CAUSAL 1D convs (dilation 2**i) so the
receptive field grows exponentially without lookahead; a direct multi-step
linear head reads the last timestep's representation. Channel-mixed.
"Practitioner tier" (not sub-1K). Self-contained (no upstream clone).

Contract: forward(x: (B, seq_len, D)) -> (B, pred_len, D).
"""

import torch.nn as nn


class _Chomp1d(nn.Module):
    """Trim the right padding so the conv stays causal (no future leakage)."""
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, n_features,
                 channels=64, levels=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        blocks, c_in = [], n_features
        for i in range(levels):
            blocks.append(_TCNBlock(c_in, channels, kernel_size, 2 ** i, dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, pred_len * n_features)

    def forward(self, x):                        # x: (B, seq_len, D)
        h = self.tcn(x.transpose(1, 2))          # (B, C, seq_len)
        last = h[:, :, -1]                       # (B, C) — last timestep
        y = self.head(last)                      # (B, pred_len * D)
        return y.view(-1, self.pred_len, self.n_features)


def build(args, n_features, seq_len, pred_len):
    return Model(
        seq_len=seq_len, pred_len=pred_len, n_features=n_features,
        channels=getattr(args, "tcn_channels", 64),
        levels=getattr(args, "tcn_levels", 4),
        kernel_size=getattr(args, "tcn_kernel_size", 3),
        dropout=getattr(args, "tcn_dropout", 0.1),
    )
