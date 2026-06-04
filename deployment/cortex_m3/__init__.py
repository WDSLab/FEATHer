"""Cortex-M3 deployment estimator for FEATHer + baselines.

Simulation-based estimates of on-device cost (peak RAM, flash, FLOPs,
latency, energy) without requiring a physical board. Calibrated
against the LM3S6965EVB target (ARM Cortex-M3 @ 50 MHz, 64 KB SRAM).

Use:
    python -m deployment.cortex_m3.run --check
    python -m deployment.cortex_m3.run \
        --model FEATHer --data ETTh1 --pred_len 96 --bit_width 8

Outputs are appended to ``results/edge_estimates.csv`` with the
columns documented in ``estimator.estimate_model``.

These are *estimates*, not measurements. Section IX of the manuscript
flags them as such and points to the validation plan (calibration
against TFLite-Micro arena planner + cycle-accurate emulator) before
camera-ready submission.
"""
from deployment.cortex_m3.estimator import estimate_model  # noqa: F401
from deployment.cortex_m3.op_costs import CortexM3Profile   # noqa: F401
