# -*- coding: utf-8 -*-
"""GRU forecaster — classic RNN practitioner baseline (JMS two-tier lineup).

Identical to the LSTM baseline but with a GRU cell (fewer gates, a common
lighter-weight practitioner choice). Channel-mixed, direct multi-step head off
the final hidden state. "Practitioner tier" (not sub-1K). Self-contained.

Contract: forward(x: (B, seq_len, D)) -> (B, pred_len, D).
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, n_features,
                 hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, pred_len * n_features)

    def forward(self, x):                        # x: (B, seq_len, D)
        out, _ = self.rnn(x)                     # (B, seq_len, H)
        last = out[:, -1]                        # (B, H)
        y = self.head(last)                      # (B, pred_len * D)
        return y.view(-1, self.pred_len, self.n_features)


def build(args, n_features, seq_len, pred_len):
    return Model(
        seq_len=seq_len, pred_len=pred_len, n_features=n_features,
        hidden_size=getattr(args, "rnn_hidden_size", 64),
        num_layers=getattr(args, "rnn_num_layers", 2),
        dropout=getattr(args, "rnn_dropout", 0.1),
    )
