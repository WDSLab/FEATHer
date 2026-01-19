# -*- coding: utf-8 -*-
"""
FEATHer Multi-scale Ablation Model
==================================

Multi-scale ablation support based on base FEATHer.py

Variants:
- 'P': Single Point only
- 'L': Single Low only
- 'PL': Point + Low (num_bands=2)
- 'PML': Point + Mid + Low (num_bands=3)
- 'PHML': Point + High + Mid + Low (num_bands=4)

Usage:
    model = FEATHer_Multiscale(
        seq_len=96,
        pred_len=96,
        d_model=7,
        d_state=10,
        kernel_size=6,
        period=12,
        band_config='PHML',  # 'P', 'L', 'PL', 'PML', 'PHML'
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
# FEATHer Multi-scale Ablation
# =============================================================================
class FEATHer_Multiscale(nn.Module):
    """
    FEATHer with configurable multi-scale branches
    
    Args:
        band_config: Branch combination to use (combination of P, H, M, L)
            - Single: 'P', 'H', 'M', 'L'
            - Dual: 'PH', 'PM', 'PL', 'HM', 'HL', 'ML'
            - Tri: 'PHM', 'PHL', 'PML', 'HML'
            - Full: 'PHML'
    """

    # All possible combinations
    ALL_VARIANTS = [
        # Single (4)
        'P', 'H', 'M', 'L',
        # Dual (6)
        'PH', 'PM', 'PL', 'HM', 'HL', 'ML',
        # Tri (4)
        'PHM', 'PHL', 'PML', 'HML',
        # Full (1)
        'PHML',
    ]

    VARIANT_NAMES = {
        # Single
        'P': 'Single_P', 'H': 'Single_H', 'M': 'Single_M', 'L': 'Single_L',
        # Dual
        'PH': 'Dual_PH', 'PM': 'Dual_PM', 'PL': 'Dual_PL',
        'HM': 'Dual_HM', 'HL': 'Dual_HL', 'ML': 'Dual_ML',
        # Tri
        'PHM': 'Tri_PHM', 'PHL': 'Tri_PHL', 'PML': 'Tri_PML', 'HML': 'Tri_HML',
        # Full
        'PHML': 'Full_PHML',
    }

    SCALE_CATEGORY = {
        'P': 'Single', 'H': 'Single', 'M': 'Single', 'L': 'Single',
        'PH': 'Dual', 'PM': 'Dual', 'PL': 'Dual',
        'HM': 'Dual', 'HL': 'Dual', 'ML': 'Dual',
        'PHM': 'Tri', 'PHL': 'Tri', 'PML': 'Tri', 'HML': 'Tri',
        'PHML': 'Full',
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
        band_config: str = 'PHML',
    ):
        super().__init__()

        # Normalize band_config (sort in order: P, H, M, L)
        band_config = ''.join(sorted(band_config, key=lambda x: 'PHML'.index(x)))
        
        assert band_config in self.ALL_VARIANTS, \
            f"Invalid band_config: {band_config}. Must be one of {self.ALL_VARIANTS}"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.period = period
        self.band_config = band_config

        # band_config -> branches to use
        self.use_P = 'P' in band_config
        self.use_H = 'H' in band_config
        self.use_M = 'M' in band_config
        self.use_L = 'L' in band_config
        
        self.num_bands = sum([self.use_P, self.use_H, self.use_M, self.use_L])

        # Input projection
        self.in_proj = nn.Linear(d_model, d_model)

        # Multi-band Conv (create only the ones used)
        if self.use_P:
            self.conv_point = nn.Conv1d(
                d_model, d_model, kernel_size=1, padding=0, groups=d_model
            )
        if self.use_H:
            self.conv_high = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            )
        if self.use_M:
            self.conv_mid = nn.Conv1d(
                d_model, d_model, kernel_size=5, padding=2, groups=d_model
            )
        if self.use_L:
            self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)

        # Shared DTK
        self.dense_kernel = DenseTemporalKernel(
            d_model=d_model,
            d_state=d_state,
            kernel_size=kernel_size
        )

        # FFT Gate (not needed for single, but kept for consistency)
        if self.num_bands > 1:
            self.freq_gate = FFTFrequencyGate(
                seq_len=seq_len,
                num_bands=self.num_bands,
                kernel_size=5,
            )
        else:
            self.freq_gate = None

        # SPK Head
        self.sparse_head = SparsePeriodKernel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            period=period,
        )

    def get_variant_name(self):
        return self.VARIANT_NAMES.get(self.band_config, f"Custom_{self.band_config}")

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

        if self.use_P:
            fP = self.conv_point(h_t).permute(0, 2, 1)
            oP = self.dense_kernel(fP)
            bands_list.append(oP)

        if self.use_H:
            fH = self.conv_high(h_t).permute(0, 2, 1)
            oH = self.dense_kernel(fH)
            bands_list.append(oH)

        if self.use_M:
            fM = self.conv_mid(h_t).permute(0, 2, 1)
            oM = self.dense_kernel(fM)
            bands_list.append(oM)

        if self.use_L:
            fL_ds = self.pool_low(h_t).permute(0, 2, 1)
            fL = F.interpolate(
                fL_ds.permute(0, 2, 1),
                size=self.seq_len,
                mode="linear",
                align_corners=False
            ).permute(0, 2, 1)
            oL = self.dense_kernel(fL)
            bands_list.append(oL)

        # Gating
        if self.num_bands == 1:
            # Single band - no gating needed
            hidden = bands_list[0]
        else:
            # Multi-band - use FFT gate
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
    """Return all multi-scale variants (15 total)"""
    return FEATHer_Multiscale.ALL_VARIANTS.copy()


def get_single_variants():
    """Single-scale variants (4 total)"""
    return ['P', 'H', 'M', 'L']


def get_dual_variants():
    """Dual-scale variants (6 total)"""
    return ['PH', 'PM', 'PL', 'HM', 'HL', 'ML']


def get_tri_variants():
    """Tri-scale variants (4 total)"""
    return ['PHM', 'PHL', 'PML', 'HML']


def get_full_variants():
    """Full-scale variants (1 total)"""
    return ['PHML']


def get_variant_name(band_config):
    """Return variant name for given band_config"""
    # Normalize
    band_config = ''.join(sorted(band_config, key=lambda x: 'PHML'.index(x)))
    return FEATHer_Multiscale.VARIANT_NAMES.get(band_config, f"Custom_{band_config}")


def get_scale_category(band_config):
    """Return scale category for given band_config (Single, Dual, Tri, Full)"""
    band_config = ''.join(sorted(band_config, key=lambda x: 'PHML'.index(x)))
    return FEATHer_Multiscale.SCALE_CATEGORY.get(band_config, 'Unknown')