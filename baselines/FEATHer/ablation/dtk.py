# -*- coding: utf-8 -*-
"""
FEATHer DTK (Dense Temporal Kernel) Ablation Model
===================================================

Variants:
- 'none': Remove DTK, identity (pass through)
- 'mlp': Projection + activation without conv (in_proj -> GELU -> out_proj)
- 'shallow': d_state halved
- 'full': Current structure (baseline)

Usage:
    model = FEATHer_DTK(
        seq_len=96,
        pred_len=96,
        d_model=7,
        d_state=10,
        kernel_size=6,
        period=12,
        num_bands=4,
        dtk_type='full',  # 'none', 'mlp', 'shallow', 'full'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FFT-based Frequency Gate
# =============================================================================
class FFTFrequencyGate(nn.Module):
    """FFT magnitude -> Conv1d -> num_bands-way gate"""

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
        B, L, D = x.shape
        x32 = x.float()
        Freq = torch.fft.rfft(x32, dim=1)
        mag = torch.abs(Freq)
        mag_mean = mag.mean(dim=2, keepdim=True)
        mag_mean = mag_mean.permute(0, 2, 1)
        z = self.conv(mag_mean)
        z = self.pool(z).squeeze(-1)
        w = torch.softmax(z, dim=-1)
        return w


# =============================================================================
# DTK Variants
# =============================================================================

class DTK_None(nn.Module):
    """No DTK - identity pass through"""
    
    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        # No parameters, just pass through
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u


class DTK_MLP(nn.Module):
    """MLP only - projection with activation, no conv"""
    
    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, L, S) -> (B, L, D)
        x = self.in_proj(u)
        x = F.gelu(x)
        y = self.out_proj(x)
        return y


class DTK_Shallow(nn.Module):
    """Shallow DTK - d_state halved"""

    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.d_state = max(1, d_state // 2)  # Half, minimum 1
        self.kernel_size = kernel_size
        
        self.in_proj = nn.Linear(d_model, self.d_state, bias=False)
        self.out_proj = nn.Linear(self.d_state, d_model, bias=False)
        self.kernel = nn.Parameter(torch.randn(self.d_state, kernel_size) * 0.01)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.size()
        x = self.in_proj(u).permute(0, 2, 1)
        k = self.kernel.unsqueeze(1)
        y = F.conv1d(x, k, padding=self.kernel_size - 1, groups=self.d_state)
        y = y[:, :, :L]
        y = self.out_proj(y.permute(0, 2, 1))
        return y


class DTK_Full(nn.Module):
    """Full DTK - baseline"""
    
    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        
        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        self.kernel = nn.Parameter(torch.randn(d_state, kernel_size) * 0.01)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.size()
        x = self.in_proj(u).permute(0, 2, 1)
        k = self.kernel.unsqueeze(1)
        y = F.conv1d(x, k, padding=self.kernel_size - 1, groups=self.d_state)
        y = y[:, :, :L]
        y = self.out_proj(y.permute(0, 2, 1))
        return y


# =============================================================================
# Sparse Period Kernel
# =============================================================================
class SparsePeriodKernel(nn.Module):
    """Period-aware sparse forecasting kernel"""

    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int):
        super().__init__()
        assert seq_len % period == 0
        assert pred_len % period == 0

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.period = period

        self.n = seq_len // period
        self.m = pred_len // period

        k = 2 * (period // 2) + 1
        self.agg_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=k, padding=k // 2,
            groups=d_model, bias=False
        )
        self.backbone = nn.Linear(self.n, self.m, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        w, n, m, H = self.period, self.n, self.m, self.pred_len

        h_t = h.permute(0, 2, 1)
        h_agg = h_t + self.agg_conv(h_t)

        x_merged = h_agg.reshape(B * D, L)
        x_by_phase = x_merged.view(B * D, n, w).permute(0, 2, 1)

        x_per_phase = x_by_phase.reshape(-1, n)
        y_per_phase = self.backbone(x_per_phase)
        y_by_phase = y_per_phase.view(B * D, w, m)

        y_reordered = y_by_phase.permute(0, 2, 1).reshape(B * D, H)
        y = y_reordered.view(B, D, H).permute(0, 2, 1)
        return y


# =============================================================================
# FEATHer DTK Ablation
# =============================================================================
class FEATHer_DTK(nn.Module):
    """
    FEATHer with configurable DTK
    
    Args:
        dtk_type: DTK variant
            - 'none': identity (no DTK)
            - 'mlp': projection + GELU (no conv)
            - 'shallow': d_state // 2
            - 'full': baseline
    """

    VARIANT_NAMES = {
        'none': 'wo_DTK',
        'mlp': 'MLP_only',
        'shallow': 'Shallow_DTK',
        'full': 'w_DTK',
    }

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_state: int = 8,
        kernel_size: int = 7,
        use_norm: bool = True,
        period: int = 24,
        num_bands: int = 4,
        dtk_type: str = 'full',
    ):
        super().__init__()

        assert dtk_type in ['none', 'mlp', 'shallow', 'full'], \
            f"Invalid dtk_type: {dtk_type}"
        assert num_bands in [2, 3, 4], "num_bands must be 2, 3, or 4"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.period = period
        self.num_bands = num_bands
        self.dtk_type = dtk_type

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model)

        # Multi-band Conv
        self.conv_point = nn.Conv1d(
            d_model, d_model, kernel_size=1, padding=0, groups=d_model
        )
        
        if num_bands == 4:
            self.conv_high = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
        else:
            self.conv_high = None
        
        if num_bands >= 3:
            self.conv_mid = nn.Conv1d(
                d_model, d_model, kernel_size=5, padding=2, groups=d_model
            )
        else:
            self.conv_mid = None
        
        self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)

        # DTK (different version depending on type)
        if dtk_type == 'none':
            self.dense_kernel = DTK_None(d_model, d_state, kernel_size)
        elif dtk_type == 'mlp':
            self.dense_kernel = DTK_MLP(d_model, d_state, kernel_size)
        elif dtk_type == 'shallow':
            self.dense_kernel = DTK_Shallow(d_model, d_state, kernel_size)
        elif dtk_type == 'full':
            self.dense_kernel = DTK_Full(d_model, d_state, kernel_size)

        # FFT Gate
        self.freq_gate = FFTFrequencyGate(
            seq_len=seq_len,
            num_bands=num_bands,
            kernel_size=5,
        )

        # SPK Head
        self.sparse_head = SparsePeriodKernel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            period=period,
        )

    def get_variant_name(self):
        return self.VARIANT_NAMES.get(self.dtk_type, f"Custom_{self.dtk_type}")

    def forward(self, x: torch.Tensor, return_components: bool = False):
        B, L, D = x.size()
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # Instance normalization
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = (x - means) / stdev
        else:
            means = None
            stdev = None
            x_norm = x

        # Input projection
        x_proj = self.in_proj(x_norm)
        h_t = x_proj.permute(0, 2, 1)  # (B, D, L)

        # Multi-band decomposition
        bands_list = []

        # POINT
        fP = self.conv_point(h_t).permute(0, 2, 1)
        oP = self.dense_kernel(fP)
        bands_list.append(oP)

        # HIGH (num_bands == 4)
        if self.num_bands == 4 and self.conv_high is not None:
            fH = self.conv_high(h_t).permute(0, 2, 1)
            oH = self.dense_kernel(fH)
            bands_list.append(oH)

        # MID (num_bands >= 3)
        if self.num_bands >= 3 and self.conv_mid is not None:
            fM = self.conv_mid(h_t).permute(0, 2, 1)
            oM = self.dense_kernel(fM)
            bands_list.append(oM)

        # LOW
        fL_ds = self.pool_low(h_t).permute(0, 2, 1)
        fL = F.interpolate(
            fL_ds.permute(0, 2, 1),
            size=self.seq_len,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        oL = self.dense_kernel(fL)
        bands_list.append(oL)

        # FFT Gate
        bands = torch.stack(bands_list, dim=1)  # (B, num_bands, L, D)
        g = self.freq_gate(x_norm)  # (B, num_bands)
        g_expanded = g.view(B, self.num_bands, 1, 1)
        hidden = (g_expanded * bands).sum(dim=1)  # (B, L, D)

        # SPK Head
        y = self.sparse_head(hidden)

        # Denormalization
        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        if return_components:
            H = torch.stack(bands_list, dim=0)
            return y, H

        return y


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_variants():
    """Return all DTK variants"""
    return ['none', 'mlp', 'shallow', 'full']


def get_variant_name(dtk_type):
    """Return variant name for given dtk_type"""
    return FEATHer_DTK.VARIANT_NAMES.get(dtk_type, f"Custom_{dtk_type}")