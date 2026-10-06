# -*- coding: utf-8 -*-
"""Op count vs measured board latency, frozen models (2026-10-06).

results/complexity.csv predates the 09-23 re-freeze for FEATHer and charges every transform
as the O(N^2) table DFT the board used before 09-29; the board now runs the fast 96-point
FFT.  This recounts the MetroPT3 deployment cell (D = 7) with the CURRENT registry configs
(FEATHer = frozen mix, R = 4) under both transform costings and regresses each against the
board latency of results/board_latency_20260929.csv (fast-FFT rows, direct_dft = 0).

Question for the text: do parameters / op counts predict on-chip latency across these
forecasters?  (MicroNets, MLSys 2021: ops predict latency within one backbone family.)

    python tools/paper/ops_vs_latency.py      -> results/ops_vs_latency_20261006.csv
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "paper"))
os.chdir(REPO)
import measure_complexity as mc  # noqa: E402

BOARD = {"stsf": "SparseTSF", "mixlinear": "MixLinear", "dipe": "DiPE_Linear", "stsf_mlp": "SparseTSF_MLP",
         "fits": "FITS", "dlinear": "DLinear", "cyclenet": "CycleNet", "feather_ci": "FEATHer_CI",
         "feather_mix": "FEATHer"}
OUT = "results/ops_vs_latency_20261006.csv"


def main():
    lat = pd.read_csv("results/board_latency_20260929.csv")
    lat = lat[(lat.direct_dft == 0) & (lat.D == 7) & (lat.status == "PASS")]
    rows = []
    for key, name in BOARD.items():
        for H in (96, 720):
            r = lat[(lat.model == key) & (lat.H == H)]
            if r.empty:
                continue
            extra = {"channel_mode": "ci_full"} if key == "feather_ci" else None
            model, D, _ = mc.build("FEATHer" if name.startswith("FEATHer") else name, "MetroPT3", H, extra)
            assert D == 7
            row = dict(model=key, H=H, latency_ms=float(r.latency_ms.iloc[-1]))
            for tf in ("dft", "fft"):
                params, c, w = mc.measure(model, D, 96, H, transform=tf)
                row.update({"params": params, f"ops_{tf}": w})
            rows.append(row)
            print(f"{key:12s} H{H:<4d} params {params:>7,}  ops(fft) {row['ops_fft']:>11,}  "
                  f"ops(dft) {row['ops_dft']:>11,}  latency {row['latency_ms']:8.1f} ms", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(t)} board cells)")
    for col in ("params", "ops_fft", "ops_dft"):
        for sub, lab in ((t, "all"), (t[t.H == 96], "H96")):
            x, y = sub[col].astype(float), sub.latency_ms
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            rl = np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2
            print(f"  {col:8s} {lab:4s} n={len(sub):2d}  r2 {r2:.3f}   r2(log-log) {rl:.3f}   "
                  f"Spearman {sub[col].rank().corr(sub.latency_ms.rank()):.2f}")
    t["ms_per_Mop_fft"] = t.latency_ms / (t.ops_fft / 1e6)
    print(t[["model", "H", "ms_per_Mop_fft"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
