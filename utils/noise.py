# -*- coding: utf-8 -*-
"""
Input corruptions for SensorFault-Bench-style robustness evaluation.

Train once on clean data, then at test time replace `x` with `corrupt(x, ...)`
before passing it to the model. The reference protocol is Windmann et al.,
"Benchmarking Sensor-Fault Robustness in Forecasting" (2026), simplified to
four fault axes most relevant to industrial IoT edge deployment:

    Gaussian noise   thermal / electrical noise            σ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
    Missing values   communication drop / sensor outage    rate ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
    Impulse spikes   sensor glitch / EMI interference      rate ∈ {0.005, 0.01, 0.02, 0.05}
    Quantization     low-bit ADC (Cortex-M edge)           bits ∈ {16, 12, 8, 6, 4}

`apply_corruption(x, fault_type, severity)` is the single dispatch point —
both the worker and any future ablation script should call it.
"""

from typing import Dict, List
import torch


# -----------------------------------------------------------------------------
# Individual corruption functions
# -----------------------------------------------------------------------------

def add_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive Gaussian noise.

    Args:
        x: (B, L, D) input (already normalized by the model's InstanceNorm
            or by upstream StandardScaler — sigma is in those units).
        sigma: noise std in normalized units.

    Returns:
        (B, L, D) corrupted tensor.
    """
    return x + torch.randn_like(x) * sigma


def apply_missing(x: torch.Tensor, rate: float, fill: str = "zero") -> torch.Tensor:
    """Random per-element missingness.

    Args:
        x: (B, L, D) input.
        rate: per-element drop probability ∈ [0, 1].
        fill: "zero" sets missing entries to 0; "last" forward-fills along
            the time axis from the most recent observed value (first
            timestep falls back to 0).

    Returns:
        (B, L, D) corrupted tensor.
    """
    if rate <= 0:
        return x
    mask = torch.rand_like(x) < rate  # True == missing
    if fill == "zero":
        return x.masked_fill(mask, 0.0)
    if fill == "last":
        # Vectorized forward-fill: for each position t, replace masked entries
        # with the most recent unmasked value along dim=1.
        keep = (~mask).to(x.dtype)
        # Cumulative-max of timestep indices where the value is kept.
        idx = torch.arange(x.shape[1], device=x.device).view(1, -1, 1).expand_as(x)
        last_kept_idx = torch.cummax(idx * keep.long(), dim=1).values
        gathered = torch.gather(x, dim=1, index=last_kept_idx.clamp(min=0))
        # Where no previous observation exists (first timestep masked) → 0.
        first_seen = torch.cummax(keep, dim=1).values > 0
        return torch.where(first_seen.bool(), gathered, torch.zeros_like(x))
    raise ValueError(f"unknown fill mode: {fill!r}")


def inject_impulse(x: torch.Tensor, rate: float, magnitude_std: float = 3.0) -> torch.Tensor:
    """Sparse impulse / spike injection.

    Args:
        x: (B, L, D) input.
        rate: per-element spike probability ∈ [0, 1].
        magnitude_std: spike magnitude in σ units of x (for normalized x,
            σ ≈ 1 so magnitude_std=3.0 ≈ ±3 σ).

    Returns:
        (B, L, D) corrupted tensor.
    """
    if rate <= 0:
        return x
    mask = torch.rand_like(x) < rate
    signs = torch.sign(torch.randn_like(x))
    return torch.where(mask, x + signs * magnitude_std, x)


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform quantization to `bits`-bit resolution per (batch, channel).

    Mirrors a fixed-point ADC: range is taken per sequence per channel
    (max-min), divided into 2**bits levels, value rounded to the nearest.

    Args:
        x: (B, L, D) input.
        bits: ADC resolution. ≥ 32 → no-op.

    Returns:
        (B, L, D) corrupted tensor.
    """
    if bits is None or bits >= 32:
        return x
    levels = 2 ** int(bits)
    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    x_range = (x_max - x_min).clamp(min=1e-8)
    x_norm = (x - x_min) / x_range
    x_quant = torch.round(x_norm * (levels - 1)) / (levels - 1)
    return x_quant * x_range + x_min


# -----------------------------------------------------------------------------
# Dispatch + sweep tables
# -----------------------------------------------------------------------------

# Severity grids per fault. Keys must match what `apply_corruption` accepts.
SEVERITY_SWEEPS: Dict[str, List[float]] = {
    "gauss":   [0.05, 0.10, 0.20, 0.30, 0.50],
    "miss":    [0.05, 0.10, 0.20, 0.30, 0.50],
    "impulse": [0.005, 0.01, 0.02, 0.05],
    "quant":   [16, 12, 8, 6, 4],          # ADC bits
}

# Severity == 0 (or bits=32 for quant) is the "clean" reference baked into
# the sweep — convenient for plotting degradation curves.
CLEAN_SENTINEL: Dict[str, float] = {
    "gauss":   0.0,
    "miss":    0.0,
    "impulse": 0.0,
    "quant":   32,
}

FAULT_TYPES = list(SEVERITY_SWEEPS.keys())


def apply_corruption(x: torch.Tensor, fault_type: str, severity: float) -> torch.Tensor:
    """Single dispatch for all four fault axes.

    Args:
        x: (B, L, D) input.
        fault_type: one of FAULT_TYPES.
        severity: σ for gauss, drop-rate for miss, spike-rate for impulse,
            bit-depth for quant.

    Returns:
        (B, L, D) corrupted tensor.
    """
    if fault_type == "gauss":
        return add_gaussian(x, sigma=float(severity))
    if fault_type == "miss":
        return apply_missing(x, rate=float(severity), fill="zero")
    if fault_type == "impulse":
        return inject_impulse(x, rate=float(severity), magnitude_std=3.0)
    if fault_type == "quant":
        return quantize(x, bits=int(severity))
    raise ValueError(f"unknown fault_type: {fault_type!r}; expected one of {FAULT_TYPES}")
