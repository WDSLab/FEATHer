# -*- coding: utf-8 -*-
"""Severity grids + deterministic corruption seeding — torch-free.

AUDIT-FIX (2026-07-14, P3): these tables used to live in utils/noise.py with
a hand-copied mirror in tools/paper/robust_summary.py (which must stay
runnable without torch). A silent drift between the two would make the
summary's top-severity match find 0 rows (NaN p-values) without any error,
so the single source of truth moved here; utils/noise.py re-exports for the
torch-side callers (workers, gate_logging).
"""

from typing import Dict, List


# Severity grids per fault. Keys must match what noise.apply_corruption accepts.
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


def corruption_seed(base_seed: int, fault_type: str, severity: float) -> int:
    """Deterministic RNG seed for a (fault_type, severity) evaluation cell.

    AUDIT-FIX (2026-07-14, commit cfd90c5): replaces the workers' previous
    `hash((ft, sev))` derivation, which was PYTHONHASHSEED-salted and thus
    differed across processes. See code_audit_2026-07-14.md P0-2.

    Every process that corrupts the same (base_seed, fault, severity) cell
    must draw the SAME noise realization, so that (a) re-runs reproduce rows
    bit-for-bit and (b) different models are compared under matched noise
    (which is what makes the paired per-cell Wilcoxon tight). Do NOT use
    Python's `hash()` here — string hashing is salted per interpreter
    (PYTHONHASHSEED), so it differs across worker processes.
    """
    fi = FAULT_TYPES.index(fault_type)
    si = SEVERITY_SWEEPS[fault_type].index(severity)
    return base_seed * 1000 + fi * 100 + si
