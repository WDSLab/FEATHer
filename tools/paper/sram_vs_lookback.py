# -*- coding: utf-8 -*-
"""Chip budget of every lightweight forecaster against the input length L.

L = 96 is not a FEATHer-specific choice, so Table 12 shows every model.
Per model and horizon H (MetroPT3 cell, D = 7, NUCLEO-F103RB 128 KB / 20 KB):

  SRAM - an UPPER BOUND on the channels one chip can serve.  Every port holds the caller's input
         window, 4 L bytes per channel, and its own buffers only grow with L.  Keeping each model's
         measured L = 96 fixed part (tools/paper/table_deploy.streams_per_chip, 552 B stack) and
         its per-channel arena, and growing only the window, gives
             channels(L) <= floor((20 KB - fixed_96) / (4 L + arena_per_channel)),
         a bound that needs no model of the internals.  (FEATHer-mix's own buffers give 18 / 7 / 0
         at L = 192 / 336 / 720 - Table 12 note.)
  flash - the model built at (L, H) with the current registry config (parameter count exact), added
         to the measured L = 96 image: flash(L, H) = flash_96(H) + 4 * (params(L, H) - params(96, H)).
         Fits if <= 128 KB (firmware floor included in flash_96).

    python tools/paper/sram_vs_lookback.py      -> results/analysis/sram_vs_lookback_20261006.csv
"""
import contextlib
import io
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "paper"))
os.chdir(REPO)
import measure_complexity as mc  # noqa: E402
import table_deploy as tdp       # noqa: E402

MODELS = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS", "CycleNet", "DLinear",
          "FEATHer_CI", "FEATHer_mix"]
LS, HS = (96, 192, 336, 720), (96, 720)
SRAM, FLASH, STACK = 20 * 1024, 128 * 1024, 552
OUT = "results/analysis/sram_vs_lookback_20261006.csv"


def params(m, L, H):
    extra = {"seq_len": L}
    if m == "FEATHer_CI":
        extra["channel_mode"] = "ci_full"
    with contextlib.redirect_stdout(io.StringIO()):
        model, D, _ = mc.build("FEATHer" if m.startswith("FEATHer") else m, "MetroPT3", H, extra)
    assert D == 7
    return sum(p.numel() for p in model.parameters())


def main():
    fp = tdp.load_footprint()
    st = tdp.streams_per_chip(fp, STACK).set_index(["model", "H"])
    dep = pd.read_csv(os.path.join("analysis", "deployment", "deploy_cells.csv"))
    rows = []
    for m in MODELS:
        key = tdp.FP_KEY.get(m, m)
        for H in HS:
            fl96 = fp.get((key, 7, H))
            fl96 = fl96.total_flash if fl96 is not None else None
            if fl96 is None:                         # excluded cells keep their printed flash
                ex = dep[(dep.model == m) & (dep.H == H) & (dep.data == "MetroPT3")]
                fl96 = float(ex.flash_kb.iloc[0]) * 1024 if len(ex) else None
            p96 = params(m, 96, H)
            srow = st.loc[(m, H)] if (m, H) in st.index else None
            for L in LS:
                pL = params(m, L, H) if L != 96 else p96
                flash = None if fl96 is None else fl96 + 4 * (pL - p96)
                if srow is not None and not pd.isna(srow.streams):
                    arena = srow.per_stream_B - 4 * 96
                    ch = int((SRAM - srow.fixed_B) // (4 * L + arena))
                else:
                    ch = None
                ok_flash = flash is not None and flash <= FLASH
                rows.append(dict(model=m, H=H, L=L, params=pL, flash_B=flash, fits_flash=ok_flash,
                                 ch_bound=ch, basis="measured" if L == 96 else "bound"))
                print(f"{m:14s} H{H:<4d} L{L:<4d} params {pL:>8,}  flash "
                      f"{'' if flash is None else f'{flash / 1024:7.1f} KB'}  ch<= {ch}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
