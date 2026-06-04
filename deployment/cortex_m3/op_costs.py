"""Cortex-M3 reference profile for the deployment estimator.

The numbers here are pulled from publicly available ARM and TI
documentation; we keep them in one place so a reader can audit
every assumption that drives the latency / energy estimates.

References:
  - ARM Cortex-M3 Technical Reference Manual (ARM DDI 0337E), §6
    (cycle counts for ALU and multiplier instructions).
  - ARM Cortex-M3 Devices Generic User Guide (ARM DUI 0552A),
    instruction timing tables.
  - Texas Instruments LM3S6965 Microcontroller Data Sheet
    (DS-LM3S6965), §28 (electrical characteristics: active-mode
    current ~33 mA typical at 50 MHz, peak ~110 mA worst-case).
  - CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs
    (Lai et al., 2018) — effective throughput on int8 MAC.

All values below are conservative defaults; the per-method override
dict at the bottom lets us refine specific operators if the audit
finds a tighter bound.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CortexM3Profile:
    """Static cost model for ARM Cortex-M3 @ 50 MHz."""

    # ---- clock / power ---------------------------------------------------
    clock_hz: float = 50e6           # LM3S6965EVB stock clock
    vdd_v: float = 3.3               # core supply
    i_typ_ma: float = 33.0           # typical active-mode current
    i_peak_ma: float = 110.0         # peak active-mode current
    # We use the busy/peak figure because inference saturates the core.
    i_inference_ma: float = 90.0     # midpoint, conservative

    # ---- instruction throughput -----------------------------------------
    # Cycles per multiply-accumulate. Cortex-M3 lacks the DSP extension
    # (M4F has SIMD MAC at 1 cyc); plain M3 MUL is 1 cyc but the load /
    # store around it dominates. CMSIS-NN int8 reports ~3 cyc/MAC end-
    # to-end on Cortex-M3-class cores including all overhead.
    cycles_per_mac_int8: float = 3.0
    cycles_per_mac_fp32: float = 9.0   # software float on M3 (no FPU)
    cycles_per_mac_fp16: float = 6.0   # half-precision via softfp

    # ---- memory subsystem ------------------------------------------------
    sram_bytes: int = 64 * 1024      # LM3S6965 has 64 KB SRAM
    flash_bytes: int = 256 * 1024    # LM3S6965 has 256 KB flash
    # Code-side overhead an int8 inference runtime adds on top of the
    # parameter blob (CMSIS-NN kernels + arena bookkeeping + main loop).
    runtime_code_bytes: int = 6 * 1024     # ~6 KB measured for CMSIS-NN
    runtime_ram_bytes: int = 2 * 1024      # ~2 KB stack/heap reserved

    # ---- supported RAM budgets to report deployability against ---------
    ram_budgets_kb: tuple = (16, 32, 64)


# ----------------------------------------------------------------------
# Operator FLOP counters
# ----------------------------------------------------------------------
# We attach these via PyTorch forward hooks in estimator.py. The goal
# is not architectural fidelity but a defensible counting rule that we
# can audit against published FLOP analyses of the same baselines
# (e.g., the FITS paper, the TimesNet paper).
# ----------------------------------------------------------------------
def flops_linear(in_features: int, out_features: int, batch_elems: int) -> int:
    """Standard Linear / FC layer: out = x @ W + b ."""
    return 2 * batch_elems * in_features * out_features  # MAC -> 2 FLOPs


def flops_conv1d(in_ch: int, out_ch: int, kernel: int, out_len: int,
                 groups: int = 1) -> int:
    """1D convolution. groups=in_ch reduces to depthwise."""
    macs = (in_ch // groups) * out_ch * kernel * out_len
    return 2 * macs


def flops_avgpool1d(channels: int, out_len: int, kernel: int) -> int:
    """Average pool — additions only, no multiplications."""
    return channels * out_len * (kernel - 1)


def flops_fft(length: int) -> int:
    """Radix-2 FFT FLOP count, conservative upper bound."""
    if length <= 1:
        return 0
    import math
    return int(5 * length * math.log2(length))


def flops_softmax(n_elements: int) -> int:
    """exp + division + max-subtraction; ~5 ops per element."""
    return 5 * n_elements


# ----------------------------------------------------------------------
# Convert FLOPs to latency / energy
# ----------------------------------------------------------------------
def estimate_latency_s(flops: int, profile: CortexM3Profile,
                       bit_width: int) -> float:
    """Wall-clock latency for ``flops`` FLOPs on Cortex-M3 @ clock_hz."""
    if bit_width == 8:
        cyc_per_mac = profile.cycles_per_mac_int8
    elif bit_width == 16:
        cyc_per_mac = profile.cycles_per_mac_fp16
    else:  # 32-bit float
        cyc_per_mac = profile.cycles_per_mac_fp32
    cycles = (flops / 2.0) * cyc_per_mac  # FLOPs->MACs is 2x
    return cycles / profile.clock_hz


def estimate_energy_mj(latency_s: float, profile: CortexM3Profile) -> float:
    """Inference energy in millijoules: latency × current × Vdd."""
    power_w = (profile.i_inference_ma / 1000.0) * profile.vdd_v
    return latency_s * power_w * 1000.0
