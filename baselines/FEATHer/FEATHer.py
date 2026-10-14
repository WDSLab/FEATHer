# -*- coding: utf-8 -*-
"""FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster (frozen model: docs/MODEL_FINAL.md).

    RevIN -> 3 bands (conv k=3 | conv k=5 | AvgPool(4) + interpolation) -> banded FFT gate -> weighted sum
          -> residual channel-mixing MLP (D -> R -> GELU -> D) -> SparseTSF-style head -> RevIN^-1

The defaults are the frozen model.  Arguments that change a value: num_bands, mlp_rank, use_residual_mlp, use_norm
(ablation), channel_mode (LTSF), head_seg / agg_shared (size-matched comparison).  Variants that swap a part are
subclasses: ablation.py (gate / head ablations), variants.py (control arms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.FEATHer.gate import FFTFrequencyGate
from baselines.FEATHer.heads import SparsePeriodKernel

# allowed values of each switch; the first one is the frozen model
OPTIONS = {
    "num_bands": (3, 2, 1),                  # POINT + MID + LOW | POINT + LOW | POINT only
    "channel_mode": ("mix", "ci_full"),      # channels mixed | every layer sees one channel (LTSF)
}


class FEATHer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int = 24, num_bands: int = 3,
                 point_kernel: int = 3, use_residual_mlp: bool = True, mlp_rank: int = 4, channel_mode: str = "mix",
                 use_norm: bool = True, head_seg: int = 0, agg_shared: bool = False):
        super().__init__()
        for name, allowed in OPTIONS.items():
            if locals()[name] not in allowed:
                raise ValueError(f"FEATHer {name}={locals()[name]!r}; allowed: {allowed}")
        self.seq_len, self.pred_len, self.num_bands = seq_len, pred_len, num_bands
        self.use_norm, self.use_residual_mlp, self.channel_mode = use_norm, use_residual_mlp, channel_mode
        d = 1 if channel_mode == "ci_full" else d_model

        # (construction order = the frozen model's RNG stream; do not reorder)
        if use_residual_mlp:                                   # residual channel-mixing MLP
            self.nl_down = nn.Linear(d, mlp_rank)
            self.nl_up = nn.Linear(mlp_rank, d)
        # bands: POINT = depthwise conv, MID = depthwise conv k=5, LOW = AvgPool(4) + interpolation
        self.conv_point = nn.Conv1d(d, d, kernel_size=point_kernel, padding=point_kernel // 2, groups=d)
        self.conv_mid = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d) if num_bands == 3 else None
        self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)
        self.freq_gate = self._make_gate(seq_len, num_bands)
        self.sparse_head = self._make_head(seq_len, pred_len, d, period, head_seg, agg_shared)

    def _make_gate(self, seq_len, num_bands):
        return FFTFrequencyGate(seq_len, num_bands)

    def _make_head(self, seq_len, pred_len, d, period, head_seg, agg_shared):
        return SparsePeriodKernel(seq_len, pred_len, d, period, seg=head_seg, agg_shared=agg_shared)

    def _normalize(self, x):
        """RevIN statistics; only the mean is detached (the frozen model was trained this way)."""
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        return (x - means) / stdev, means, stdev

    def _bands(self, x):
        """(B, L, D) -> [POINT, (MID), (LOW)], each (B, L, D)."""
        h = x.permute(0, 2, 1)
        bands = [self.conv_point(h)]
        if self.num_bands == 3:
            bands.append(self.conv_mid(h))
        if self.num_bands >= 2:
            bands.append(F.interpolate(self.pool_low(h), size=self.seq_len, mode="linear", align_corners=False))
        return [b.permute(0, 2, 1) for b in bands]

    def _gate(self, x):
        """(B, num_bands) band weights, or None for an unweighted sum."""
        return self.freq_gate(x)

    def _mixer(self, h):
        """Correction added by the residual MLP: W_up(GELU(W_down h)), over the channel axis."""
        return self.nl_up(F.gelu(self.nl_down(h)))

    def forward(self, x: torch.Tensor, return_components: bool = False):
        """x: (B, L, D) -> (B, pred_len, D); with return_components also the bands (num_bands, B, L, D)."""
        B, L, D = x.size()
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"
        ci = self.channel_mode == "ci_full"
        if ci:                                                 # fold channels into the batch
            x = x.permute(0, 2, 1).reshape(B * D, L, 1)
        if self.use_norm:
            x, means, stdev = self._normalize(x)

        bands_list = self._bands(x)
        bands = torch.stack(bands_list, dim=1)                 # (B, S, L, D)
        g = self._gate(x)
        hidden = bands.sum(dim=1) if g is None else (g.view(-1, bands.size(1), 1, 1) * bands).sum(dim=1)
        if self.use_residual_mlp:
            hidden = hidden + self._mixer(hidden)
        y = self.sparse_head(hidden)                           # (B, pred_len, D)

        if self.use_norm:                                      # denormalise
            y = y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        if ci:
            y = y.reshape(B, D, self.pred_len).permute(0, 2, 1)
        return (y, torch.stack(bands_list, dim=0)) if return_components else y
