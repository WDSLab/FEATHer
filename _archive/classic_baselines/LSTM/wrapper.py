# -*- coding: utf-8 -*-
"""LSTM forecaster — classic RNN practitioner baseline (JMS two-tier lineup).

Textbook multivariate LSTM: encode the lookback with a stacked LSTM, take the
final hidden state, and project it directly to the full horizon (direct
multi-step). Channel-mixed (all D channels in/out jointly) — how practitioners
typically deploy an LSTM forecaster. NOT parameter-budget constrained: this is
the "practitioner tier" accuracy reference contrasted against the sub-1K
edge-tier models (SparseTSF/FITS/FEATHer). Self-contained (no upstream clone).

Contract (uniform worker interface): forward(x: (B, seq_len, D)) -> (B, pred_len, D).
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, n_features,
                 hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.rnn = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, pred_len * n_features)

    def forward(self, x):                        # x: (B, seq_len, D)
        out, _ = self.rnn(x)                     # (B, seq_len, H)
        last = out[:, -1]                        # (B, H) — final hidden state
        y = self.head(last)                      # (B, pred_len * D)
        return y.view(-1, self.pred_len, self.n_features)


def build(args, n_features, seq_len, pred_len):
    return Model(
        seq_len=seq_len, pred_len=pred_len, n_features=n_features,
        hidden_size=getattr(args, "rnn_hidden_size", 64),
        num_layers=getattr(args, "rnn_num_layers", 2),
        dropout=getattr(args, "rnn_dropout", 0.1),
    )
