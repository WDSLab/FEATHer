# -*- coding: utf-8 -*-
"""Chip verdict of one cell (model, dataset, H, setting) - no board needed.

one_cell builds the complete board firmware (board_prep.dry_build: the CubeIDE flags, the real linker script, HAL
included) at the dataset's true channel count; the tuning calls it for every drawn setting
(tools/results/tune_budget.py).  Without a setting it builds the model's base setting (period rules resolved with that
dataset).  The scratch image is linked against oversized regions (relax_ld) so an over-budget cell still reports both
sizes; they are compared with the real 128 KB flash / 20 KB SRAM here.

What is charged:
  flash = whole image MINUS the golden test vectors (input L x D + output H x D floats,
          harness-only constants a deployed node does not carry)
  ram   = .data + .bss + the linker's heap/stack reserve (0x600 B) - includes the
          caller's input window s_x (L x D floats).
Full-buffer ports (the harness owns an H x D output buffer s_y) also get ram_stream =
ram - H x D x 4: every forecast value is final when computed, so <pfx>_forward_emit hands
it out at once and no output buffer is held (the same way FEATHer emits its
blocks).  Internal arenas stay as built.

Stack is not measured here (only the reserve); cells that pass go to the board next.

    python deployment/board/tools/chip_verdict.py --models mixlinear,fits --data Steel --H 96     # base settings, printed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_prep as bp  # noqa: E402

SEQ_LEN = 96
DATA_D = {"Steel": 6, "MetroPT3": 7, "WindSCADA": 14, "N2O": 14, "EWELD": 14, "BlastFurnace": 28}
LTSF_D = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21, "Electricity": 321, "Traffic": 862}
DATA_D.update(LTSF_D)                     # tier ltsf: the chip rule on the LTSF data
BOARD_MODELS = ["feather_ci", "stsf", "stsf_mlp", "mixlinear", "fits", "cmos", "timebase", "phaseformer"]
# board model -> (registry model, override key) for the ports whose export does not resolve
# the dataset itself: the dataset's period is passed explicitly
GEO = {"feather_ci": ("FEATHer_CI", "period"), "stsf": ("SparseTSF", "sparsetsf_period_len")}


def geo_of(model: str, data: str, H: int) -> dict | None:
    if model not in GEO:
        return None
    sys.path.insert(0, str(bp.ROOT))
    from tools.results.protocol import mfg_main_config_id, resolve
    reg, key = GEO[model]
    ov = resolve(reg, "ltsf" if data in LTSF_D else "mfg", data, H, mfg_main_config_id(reg))[2]
    return {"period": int(ov[key])}


def period_of(model: str, data: str, H: int, config: str | None = None):
    """the period the cell is built with, for the CSV (config models resolve it inside
    their export; read it the same way)"""
    g = geo_of(model, data, H)
    if g:
        return next(iter(g.values()))
    keys = {"stsf_mlp": ("SparseTSF_MLP", "sparsetsf_period_len"),
            "mixlinear": ("MixLinear", "mixlinear_period_len"),
            "cmos": ("CMoS", "cmos_period"), "timebase": ("TimeBase", "timebase_period_len"),
            "phaseformer": ("PhaseFormer", "phaseformer_period_len")}
    if model not in keys:
        return None
    from tools.results.protocol import mfg_main_config_id, resolve
    reg, key = keys[model]
    cid = ("cand:" + config) if config else mfg_main_config_id(reg)
    return resolve(reg, "mfg", data, H, cid)[2].get(key)


def config_of(model: str) -> str:
    a = bp.config_args(model)
    return a[1] if a else ("frozen" if model.startswith("feather") else "default")


def one_cell(model: str, data: str, H: int, config: str | None = None, extra_geo: dict | None = None) -> dict:
    """extra_geo: geometry keys added to the dataset's (e.g. FEATHer seg / agg_shared)"""
    D = DATA_D[data]
    full = bp.MODELS[model][4]
    golden = (SEQ_LEN + H) * D * 4
    row = dict(model=model, config=config or config_of(model), data=data, D=D, H=H, golden_B=golden,
               period=period_of(model, data, H, config))
    try:
        r = bp.dry_build(bp.DEFAULT_PROJECT, model, H, D, relax_ld=True,
                         data=data if model in bp.CONFIG_MODELS else None,
                         geo={**(geo_of(model, data, H) or {}), **(extra_geo or {})} or None, config=config)
    except SystemExit as e:                                   # export refused the geometry
        return {**row, "build": "EXPORT_FAIL", "verdict": "?", "why": " ".join(str(e).split())[-200:]}
    if "flash" not in r:
        return {**row, "build": "BUILD_FAIL", "verdict": "?", "why": r["why"][-200:]}
    flash_dep = r["flash"] - golden
    ram_stream = r["ram"] - H * D * 4 if full else None      # emit API: no output buffer
    fits_flash = flash_dep <= bp.FLASH_BYTES
    fits_ram = (ram_stream if full else r["ram"]) <= bp.RAM_BYTES
    return {**row, "build": "LINK_OK", "flash_B": flash_dep, "ram_B": r["ram"],
            "ram_stream_B": ram_stream, "fits_flash": fits_flash, "fits_ram": fits_ram,
            "verdict": "FIT" if fits_flash and fits_ram else "OFF", "why": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(BOARD_MODELS))
    ap.add_argument("--data", default="MetroPT3")
    ap.add_argument("--H", default="96")
    a = ap.parse_args()
    for m in a.models.split(","):
        for d in a.data.split(","):
            for H in (int(h) for h in a.H.split(",")):
                t0 = time.time()
                row = one_cell(m, d, H)
                fl, ra = row.get("flash_B"), row.get("ram_stream_B") or row.get("ram_B")
                print(f"{m:<12} {row['config']:<12} {d:<12} D={DATA_D[d]:<3} H={H:<4} {row['build']:<13} "
                      f"flash {fl if fl is not None else '-':>7}  ram {ra if ra is not None else '-':>6}  "
                      f"{row['verdict']}  {row.get('why', '')[:80]}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
