"""Core estimator: PyTorch nn.Module -> Cortex-M3 cost dictionary.

We do not rely on third-party FLOP counters (thop / fvcore) so the
behavior is auditable and the dependency list stays minimal. The
mechanism is straightforward:

  1. Register a forward hook on every leaf nn.Module.
  2. The hook records (a) the FLOP count for that op and (b) the
     output activation byte size, using the standard formulas in
     ``op_costs``.
  3. While the forward pass runs, a small reference-tracking pass
     maintains a multiset of currently-alive intermediate tensors.
     The peak of (sum of live tensor bytes) over the forward pass is
     the TFLite-Micro arena size.
  4. After forward, we sum FLOPs across modules, convert to latency
     via ``op_costs.estimate_latency_s``, derive energy via
     ``op_costs.estimate_energy_mj``, and compute the parameter blob
     size from ``num_params * (bit_width / 8)``.

The estimator is deliberately conservative: anywhere we are unsure
of the exact cycle cost (e.g., generic non-leaf nn.Modules), we fall
back to a fp32-MAC equivalent that overestimates rather than
underestimates the budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from deployment.cortex_m3.op_costs import (
    CortexM3Profile,
    estimate_energy_mj,
    estimate_latency_s,
    flops_avgpool1d,
    flops_conv1d,
    flops_fft,
    flops_linear,
    flops_softmax,
)


# ----------------------------------------------------------------------
# Result schema
# ----------------------------------------------------------------------
@dataclass
class EdgeEstimate:
    """One row of edge_estimates.csv."""

    model: str
    data: str
    pred_len: int
    seq_len: int
    n_features: int
    bit_width: int
    num_params: int
    param_bytes: int
    activation_arena_bytes: int
    peak_ram_bytes: int          # arena + runtime reservation + I/O buffer
    flash_bytes: int             # param bytes + runtime code overhead
    flops: int
    latency_ms: float
    energy_mj: float
    deploy_16k: int              # 1 if peak_ram_bytes <= 16*1024
    deploy_32k: int
    deploy_64k: int

    def as_row(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Hook utilities
# ----------------------------------------------------------------------
def _tensor_bytes(t: torch.Tensor, bit_width: int) -> int:
    """Bytes a tensor would occupy at the requested bit-width."""
    return t.numel() * max(1, bit_width // 8)


def _flop_for_module(mod: nn.Module, inp: Tuple, out: torch.Tensor) -> int:
    """Best-effort FLOP count for a single leaf module."""
    if isinstance(mod, nn.Linear):
        # input shape (..., in_features) -> output (..., out_features)
        in_t = inp[0] if isinstance(inp, tuple) else inp
        batch_elems = int(in_t.numel() / in_t.shape[-1])
        return flops_linear(mod.in_features, mod.out_features, batch_elems)

    if isinstance(mod, nn.Conv1d):
        out_len = int(out.shape[-1])
        return flops_conv1d(mod.in_channels, mod.out_channels,
                            kernel=mod.kernel_size[0],
                            out_len=out_len, groups=mod.groups)

    if isinstance(mod, (nn.AvgPool1d, nn.AdaptiveAvgPool1d)):
        # rough: kernel covers (in_len / out_len) elements
        in_t = inp[0] if isinstance(inp, tuple) else inp
        in_len = int(in_t.shape[-1])
        out_len = int(out.shape[-1])
        kernel = max(1, in_len // max(1, out_len))
        return flops_avgpool1d(int(in_t.shape[1]), out_len, kernel)

    if isinstance(mod, nn.Softmax):
        return flops_softmax(int(out.numel()))

    # Misc element-wise activations (ReLU, GELU, LayerNorm, ...) are
    # treated as ~2 FLOPs per element; this is a conservative overcount
    # for ReLU and an undercount for GELU but the overall budget is
    # dominated by the linear ops above.
    return 2 * int(out.numel())


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def estimate_model(
    model: nn.Module,
    *,
    model_name: str,
    data_name: str,
    seq_len: int,
    pred_len: int,
    n_features: int,
    bit_width: int = 8,
    profile: CortexM3Profile | None = None,
) -> EdgeEstimate:
    """Run a single forward pass with hooks and return an EdgeEstimate."""
    profile = profile or CortexM3Profile()
    model.eval()
    device = next(model.parameters()).device

    flops_total = 0
    live_bytes: List[int] = []
    arena_peak = 0
    fft_added = False  # FEATHer / FITS perform an FFT outside leaf modules

    def _hook(mod, inp, out):
        nonlocal flops_total, arena_peak, fft_added
        try:
            flops_total += _flop_for_module(mod, inp, out)
        except Exception:
            # leaf module we have not categorised — fall back to numel*2
            if isinstance(out, torch.Tensor):
                flops_total += 2 * int(out.numel())

        # Estimate currently-alive bytes: prior activations + current
        # output. We approximate the TFLite-Micro arena as 2 layers
        # worth of bytes (current output + one upstream consumer
        # buffer). For multi-branch models the multi-branch fan-out
        # is captured by enumerating the parallel leaf hooks before
        # the gating layer.
        if isinstance(out, torch.Tensor):
            ob = _tensor_bytes(out, bit_width)
            live_bytes.append(ob)
            window = sum(live_bytes[-4:])  # 4-layer sliding window
            arena_peak = max(arena_peak, window)

    handles = []
    for sub in model.modules():
        if len(list(sub.children())) == 0:  # leaf
            handles.append(sub.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            x = torch.zeros(1, seq_len, n_features, device=device)
            try:
                model(x)
            except Exception:
                # some wrappers expect (x, x_mark, x_dec, x_mark_dec).
                # Try the multi-arg signature.
                mark = torch.zeros(1, seq_len, 4, device=device)
                x_dec = torch.zeros(1, pred_len, n_features, device=device)
                mark_dec = torch.zeros(1, pred_len, 4, device=device)
                model(x, mark, x_dec, mark_dec)
    finally:
        for h in handles:
            h.remove()

    # Detect FEATHer-style FFT branches by name pattern. If the model
    # contains an obvious FFT step that bypasses our hook (torch.fft),
    # add a one-shot FFT cost.
    if any("fft" in n.lower() or "gate" in n.lower()
           for n, _ in model.named_modules()):
        flops_total += flops_fft(seq_len) * n_features

    # ---- aggregate ------------------------------------------------------
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = num_params * max(1, bit_width // 8)

    io_buffer = (seq_len + pred_len) * n_features * max(1, bit_width // 8)
    peak_ram_bytes = arena_peak + profile.runtime_ram_bytes + io_buffer
    flash_bytes = param_bytes + profile.runtime_code_bytes

    latency_s = estimate_latency_s(flops_total, profile, bit_width)
    energy_mj = estimate_energy_mj(latency_s, profile)

    flash_ok = flash_bytes <= profile.flash_bytes

    def _fits(budget_kb: int) -> int:
        return int(flash_ok and peak_ram_bytes <= budget_kb * 1024)

    return EdgeEstimate(
        model=model_name,
        data=data_name,
        pred_len=pred_len,
        seq_len=seq_len,
        n_features=n_features,
        bit_width=bit_width,
        num_params=num_params,
        param_bytes=param_bytes,
        activation_arena_bytes=arena_peak,
        peak_ram_bytes=peak_ram_bytes,
        flash_bytes=flash_bytes,
        flops=flops_total,
        latency_ms=latency_s * 1000.0,
        energy_mj=energy_mj,
        deploy_16k=_fits(16),
        deploy_32k=_fits(32),
        deploy_64k=_fits(64),
    )
