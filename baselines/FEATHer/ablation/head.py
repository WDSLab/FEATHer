# -*- coding: utf-8 -*-
"""
FEATHer Head Ablation Model
===========================

Variants:
- 'linear': Simple Linear projection
- 'mlp': 2-layer MLP
- 'conv': 1D Conv head
- 'spk': Sparse Period Kernel (baseline)

Usage:
    model = FEATHer_Head(
        seq_len=96,
        pred_len=96,
        d_model=7,
        d_state=10,
        kernel_size=6,
        period=12,
        num_bands=4,
        head_type='spk',  # 'linear', 'mlp', 'conv', 'spk'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Forecasting Heads
# =============================================================================

class LinearHead(nn.Module):
    """Simple Linear projection head"""
    
    def __init__(self, seq_len: int, pred_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        self.linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D)
        h_t = h.permute(0, 2, 1)  # (B, D, L)
        y_t = self.linear(h_t)    # (B, D, H)
        y = y_t.permute(0, 2, 1)  # (B, H, D)
        return y


class MLPHead(nn.Module):
    """2-layer MLP head"""
    
    def __init__(self, seq_len: int, pred_len: int, d_model: int, hidden_ratio: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        hidden_dim = seq_len * hidden_ratio
        
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D)
        h_t = h.permute(0, 2, 1)  # (B, D, L)
        y_t = self.mlp(h_t)       # (B, D, H)
        y = y_t.permute(0, 2, 1)  # (B, H, D)
        return y


class ConvHead(nn.Module):
    """1D Conv head with multiple layers"""
    
    def __init__(self, seq_len: int, pred_len: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Depthwise conv + pointwise projection
        self.conv1 = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=d_model
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        self.proj = nn.Linear(seq_len, pred_len)
        self.act = nn.GELU()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D)
        h_t = h.permute(0, 2, 1)  # (B, D, L)
        
        h_t = self.act(self.conv1(h_t))
        h_t = self.act(self.conv2(h_t))
        
        y_t = self.proj(h_t)      # (B, D, H)
        y = y_t.permute(0, 2, 1)  # (B, H, D)
        return y


class SparsePeriodKernel(nn.Module):
    """Period-aware sparse forecasting kernel (baseline)"""

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
# FFT Frequency Gate
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
# Dense Temporal Kernel
# =============================================================================

class DenseTemporalKernel(nn.Module):
    """Depthwise conv kernel for temporal mixing"""

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
# FEATHer Head Ablation
# =============================================================================

class FEATHer_Head(nn.Module):
    """
    FEATHer with configurable forecasting head
    
    Args:
        head_type: Head variant
            - 'linear': Simple Linear projection
            - 'mlp': 2-layer MLP
            - 'conv': 1D Conv head
            - 'spk': Sparse Period Kernel (baseline)
    """

    VARIANT_NAMES = {
        'linear': 'Linear_Head',
        'mlp': 'MLP_Head',
        'conv': 'Conv_Head',
        'spk': 'SPK_Head',
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
        head_type: str = 'spk',
    ):
        super().__init__()

        assert head_type in ['linear', 'mlp', 'conv', 'spk'], \
            f"Invalid head_type: {head_type}"
        assert num_bands in [2, 3, 4], "num_bands must be 2, 3, or 4"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.period = period
        self.num_bands = num_bands
        self.head_type = head_type

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

        # Shared DTK
        self.dense_kernel = DenseTemporalKernel(
            d_model=d_model,
            d_state=d_state,
            kernel_size=kernel_size
        )

        # FFT Frequency Gate
        self.freq_gate = FFTFrequencyGate(
            seq_len=seq_len,
            num_bands=num_bands,
            kernel_size=5,
        )

        # Forecasting Head (varies by type)
        if head_type == 'linear':
            self.head = LinearHead(seq_len, pred_len, d_model)
        elif head_type == 'mlp':
            self.head = MLPHead(seq_len, pred_len, d_model)
        elif head_type == 'conv':
            self.head = ConvHead(seq_len, pred_len, d_model)
        elif head_type == 'spk':
            self.head = SparsePeriodKernel(seq_len, pred_len, d_model, period)

    def get_variant_name(self):
        return self.VARIANT_NAMES.get(self.head_type, f"Custom_{self.head_type}")

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

        # Stack bands
        bands = torch.stack(bands_list, dim=1)  # (B, num_bands, L, D)

        # FFT Gating
        g = self.freq_gate(x_norm)  # (B, num_bands)
        g_expanded = g.view(B, self.num_bands, 1, 1)
        hidden = (g_expanded * bands).sum(dim=1)  # (B, L, D)

        # Forecasting Head
        y = self.head(hidden)

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
    """Return all Head variants"""
    return ['linear', 'mlp', 'conv', 'spk']


def get_variant_name(head_type):
    """Return variant name for given head_type"""
    return FEATHer_Head.VARIANT_NAMES.get(head_type, f"Custom_{head_type}")
