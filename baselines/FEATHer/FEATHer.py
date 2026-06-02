# -*- coding: utf-8 -*-
"""
FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster

Components:
- FFT-based Frequency Gate (FFTFrequencyGate)
- Multi-band decomposition via 1D Conv (POINT / HIGH / MID / LOW)
- DenseTemporalKernel (depthwise temporal mixing + in/out projection)
- SparsePeriodKernel (period-aware sparse forecasting head)

Band configurations (num_bands):
    2: [POINT, LOW]
    3: [POINT, MID, LOW]
    4: [POINT, HIGH, MID, LOW]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_topk_gate(w: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Top-k sparsified gate (optional).

    Args:
        w: (B, C) gate weights
        k: number of active bands

    Returns:
        (B, C) sparse gate (only top-k non-zero, renormalized)
    """
    B, C = w.shape
    if k >= C:
        return w

    topk_vals, topk_idx = torch.topk(w, k=k, dim=-1)
    mask = torch.zeros_like(w)
    mask.scatter_(1, topk_idx, 1.0)

    w_sparse = w * mask
    denom = w_sparse.sum(dim=-1, keepdim=True) + 1e-8
    w_sparse = w_sparse / denom
    return w_sparse


class FFTFrequencyGate(nn.Module):
    """
    FFT-based frequency gate for adaptive band weighting.

    Computes FFT magnitude spectrum, applies Conv1d, and outputs
    softmax-normalized weights for each frequency band.

    Args:
        seq_len: Input sequence length
        num_bands: Number of frequency bands (2, 3, or 4)
        kernel_size: Kernel size for Conv1d
    """

    def __init__(self, seq_len: int, num_bands: int = 3, kernel_size: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.num_bands = num_bands

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_bands,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor

        Returns:
            (B, num_bands) softmax-normalized gate weights
        """
        B, L, D = x.shape

        # Convert to float32 to avoid cuFFT errors in autocast(fp16)
        x32 = x.float()
        Freq = torch.fft.rfft(x32, dim=1)
        mag = torch.abs(Freq)

        # Channel mean -> (B, L_f, 1) -> (B, 1, L_f)
        mag_mean = mag.mean(dim=2, keepdim=True).permute(0, 2, 1)

        # Conv1d + Global Pooling
        z = self.conv(mag_mean)
        z = self.pool(z).squeeze(-1)

        w = torch.softmax(z, dim=-1)
        return w


class DenseTemporalKernel(nn.Module):
    """
    Learned depthwise convolution kernel for dense temporal mixing.

    Projects input to latent state space, applies channel-wise convolution,
    then projects back to original dimension.

    Args:
        d_model: Model dimension
        d_state: Latent state dimension
        kernel_size: Convolution kernel size
    """

    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size

        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        self.kernel = nn.Parameter(torch.randn(d_state, kernel_size) * 0.01)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, L, D) input tensor

        Returns:
            (B, L, D) output tensor
        """
        B, L, D = u.size()

        # Project to latent space: (B, L, D) -> (B, S, L)
        x = self.in_proj(u).permute(0, 2, 1)

        # Depthwise conv: groups = S
        k = self.kernel.unsqueeze(1)
        y = F.conv1d(x, k, padding=self.kernel_size - 1, groups=self.d_state)
        y = y[:, :, :L]  # Trim to length L

        # Project back: (B, S, L) -> (B, L, D)
        y = self.out_proj(y.permute(0, 2, 1))
        return y


class SparsePeriodKernel(nn.Module):
    """
    Period-aware sparse forecasting kernel (inspired by SparseTSF).

    Reorganizes input by period phases, applies shared linear projection
    across periods, then reconstructs the output sequence.

    Args:
        seq_len: Input sequence length (must be divisible by period)
        pred_len: Prediction length (must be divisible by period)
        d_model: Model dimension
        period: Period size (e.g., 24 for hourly data with daily patterns)
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int):
        super().__init__()
        assert seq_len % period == 0, "seq_len must be divisible by period"
        assert pred_len % period == 0, "pred_len must be divisible by period"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.period = period

        self.n = seq_len // period   # input periods
        self.m = pred_len // period  # output periods

        # Sliding aggregation conv
        k = 2 * (period // 2) + 1
        self.agg_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=k, padding=k // 2,
            groups=d_model, bias=False
        )

        # Cross-period backbone
        self.backbone = nn.Linear(self.n, self.m, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) input tensor

        Returns:
            (B, pred_len, D) output tensor
        """
        B, L, D = h.size()
        w, n, m, H = self.period, self.n, self.m, self.pred_len

        # Residual + depthwise aggregation
        h_t = h.permute(0, 2, 1)
        h_agg = h_t + self.agg_conv(h_t)

        # Reshape for period-wise processing
        x_merged = h_agg.reshape(B * D, L)
        x_by_phase = x_merged.view(B * D, n, w).permute(0, 2, 1)

        # Shared backbone for each phase
        x_per_phase = x_by_phase.reshape(-1, n)
        y_per_phase = self.backbone(x_per_phase)
        y_by_phase = y_per_phase.view(B * D, w, m)

        # Reconstruct output
        y_reordered = y_by_phase.permute(0, 2, 1).reshape(B * D, H)
        y = y_reordered.view(B, D, H).permute(0, 2, 1)
        return y


class FEATHer(nn.Module):
    """
    FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster

    An ultra-lightweight model for long-term time series forecasting that
    combines multi-scale frequency decomposition with adaptive gating
    and period-aware sparse forecasting.

    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        d_model: Number of input features
        d_state: Latent state dimension for DenseTemporalKernel
        kernel_size: Kernel size for DenseTemporalKernel
        use_norm: Whether to use instance normalization
        period: Period for SparsePeriodKernel
        num_bands: Number of frequency bands (2, 3, or 4)
        use_topk_gate: Whether to use top-k sparse gating
        topk: Number of active bands when using top-k gating
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_state: int = 8,
        kernel_size: int = 7,
        use_norm: bool = True,
        period: int = 24,
        num_bands: int = 3,
        use_topk_gate: bool = False,
        topk: int = 2,
    ):
        super().__init__()

        assert num_bands in [2, 3, 4], "num_bands must be 2, 3, or 4"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.period = period
        self.num_bands = num_bands
        self.use_topk_gate = use_topk_gate
        self.topk = topk

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model)

        # Multi-band 1D Conv decomposition
        # POINT: kernel=1, captures high-frequency details
        self.conv_point = nn.Conv1d(
            d_model, d_model, kernel_size=1, padding=0, groups=d_model
        )

        # HIGH: kernel=3 (only when num_bands == 4)
        if num_bands == 4:
            self.conv_high = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
        else:
            self.conv_high = None

        # MID: kernel=5 (when num_bands >= 3)
        if num_bands >= 3:
            self.conv_mid = nn.Conv1d(
                d_model, d_model, kernel_size=5, padding=2, groups=d_model
            )
        else:
            self.conv_mid = None

        # LOW: avg pooling (L -> L/4) then upsample
        self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)

        # Shared temporal kernel
        self.dense_kernel = DenseTemporalKernel(
            d_model=d_model,
            d_state=d_state,
            kernel_size=kernel_size
        )

        # FFT-based frequency gate
        self.freq_gate = FFTFrequencyGate(
            seq_len=seq_len,
            num_bands=num_bands,
            kernel_size=5,
        )

        # Sparse period-aware forecasting head
        self.sparse_head = SparsePeriodKernel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            period=period,
        )

    def forward(self, x: torch.Tensor, return_components: bool = False):
        """
        Args:
            x: (B, L, D) input tensor
            return_components: If True, also return intermediate band outputs

        Returns:
            y: (B, pred_len, D) prediction
            H (optional): (num_bands, B, L, D) band components after DTK
        """
        B, L, D = x.size()
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # Instance normalization
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = (x - means) / stdev
        else:
            means, stdev = None, None
            x_norm = x

        # Input projection
        x_proj = self.in_proj(x_norm)
        h_t = x_proj.permute(0, 2, 1)  # (B, D, L)

        # Multi-band decomposition
        # POINT band
        fP = self.conv_point(h_t).permute(0, 2, 1)
        oP = self.dense_kernel(fP)

        # HIGH band (optional)
        oH = None
        if self.num_bands == 4 and self.conv_high is not None:
            fH = self.conv_high(h_t).permute(0, 2, 1)
            oH = self.dense_kernel(fH)

        # MID band (optional)
        oM = None
        if self.num_bands >= 3 and self.conv_mid is not None:
            fM = self.conv_mid(h_t).permute(0, 2, 1)
            oM = self.dense_kernel(fM)

        # LOW band
        fL_ds = self.pool_low(h_t).permute(0, 2, 1)
        fL = F.interpolate(
            fL_ds.permute(0, 2, 1),
            size=self.seq_len,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        oL = self.dense_kernel(fL)

        # Collect bands according to num_bands
        if self.num_bands == 2:
            bands_list = [oP, oL]
        elif self.num_bands == 3:
            bands_list = [oP, oM, oL]
        elif self.num_bands == 4:
            bands_list = [oP, oH, oM, oL]
        else:
            raise ValueError("num_bands must be 2, 3, or 4")

        bands = torch.stack(bands_list, dim=1)  # (B, num_bands, L, D)

        # FFT-based gate
        g = self.freq_gate(x_norm)  # (B, num_bands)

        if self.use_topk_gate:
            g = sparse_topk_gate(g, k=self.topk)

        g_expanded = g.view(B, self.num_bands, 1, 1)
        hidden = (g_expanded * bands).sum(dim=1)  # (B, L, D)

        # Forecasting head
        y = self.sparse_head(hidden)  # (B, pred_len, D)

        # Denormalization
        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        if return_components:
            H = torch.stack(bands_list, dim=0)  # (num_bands, B, L, D)
            return y, H

        return y
