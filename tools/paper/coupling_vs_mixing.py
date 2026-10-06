# -*- coding: utf-8 -*-
"""Sensor coupling strength vs the gain from channel mixing (2026-10-06).

Pre-declared rule: manuscript/drafts/coupling_plan.md (written before computing).
  C = mean over non-overlapping 96-step TRAIN windows of the mean |same-time correlation|
      over all sensor pairs (per-window, per-sensor z-score = the model's RevIN).
  G = mean over the 4 horizons of (FEATHer-CI MSE / FEATHer-mix MSE - 1) x 100, seed-mean per cell.
  Main rule: Spearman rho(C, G) >= 0.5 and one-sided p < 0.05 over the 14 datasets.

    python tools/paper/coupling_vs_mixing.py      -> results/coupling_vs_mixing_20261006.csv
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)
L = 96
MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange", "Electricity", "Traffic"]
OUT = "results/coupling_vs_mixing_20261006.csv"


def train_matrix(name):
    if name in MFG:                                   # same as tools/dev/sensor_lag.py
        df = pd.read_csv(f"data/{name}/data.csv")
        x = df.drop(columns=[c for c in df.columns if c.lower() in ("date", "unit")]).values.astype(np.float64)
        x = x[: int(len(x) * 0.7)]
        return x[np.isfinite(x).all(1)]
    from utils.data_factory import data_provider
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _ = data_provider("data/", name, "M", 32, L, 0, 96, "train")
    return np.asarray(ds.data_x, dtype=np.float64)


def coupling(x):
    n = len(x) // L
    w = x[: n * L].reshape(n, L, -1)
    sd = w.std(1, keepdims=True)
    w = w[(sd > 1e-6).all(axis=(1, 2))]
    z = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    N, _, D = z.shape
    acc = 0.0
    for k in range(N):                                # per window: D x D correlation
        c = (z[k].T @ z[k]) / L
        acc += (np.abs(c).sum() - D) / (D * (D - 1))
    return acc / N, D, N


def mixing_gain(raw):
    r = raw[raw.model.isin(["FEATHer_CI", "FEATHer_mix"]) & (raw.seq_len == 96)]
    cell = r.groupby(["data", "pred_len", "model"]).MSE.mean().unstack("model")
    g = (cell["FEATHer_CI"] / cell["FEATHer_mix"] - 1) * 100
    return g.groupby(level="data").mean()


G = pd.concat([mixing_gain(pd.read_csv("results/mfg_raw.csv")),
               mixing_gain(pd.read_csv("results/ltsf_raw.csv"))])
rows = []
for name in MFG + LTSF:
    C, D, N = coupling(train_matrix(name))
    rows.append(dict(data=name, group="mfg" if name in MFG else "ltsf", D=D, windows=N,
                     coupling=C, mixing_gain_pct=float(G[name])))
    print(f"{name:13s} D={D:4d} windows={N:5d}  coupling {C:.3f}  mixing gain {G[name]:+6.2f} %", flush=True)
t = pd.DataFrame(rows)
t.to_csv(OUT, index=False)

rho, p2 = spearmanr(t.coupling, t.mixing_gain_pct)
p1 = p2 / 2 if rho > 0 else 1 - p2 / 2
rd, pd2 = spearmanr(np.log(t.D), t.mixing_gain_pct)
m = t[t.group == "mfg"]
rm, pm = spearmanr(m.coupling, m.mixing_gain_pct)
print(f"\nMAIN  Spearman rho(coupling, gain) over 14 = {rho:+.2f}, one-sided p = {p1:.4f}  -> "
      f"{'SUPPORTED' if rho >= 0.5 and p1 < 0.05 else 'NOT supported'} (rule: rho >= 0.5, p < 0.05)")
print(f"compare rho(log D, gain) = {rd:+.2f} (two-sided p {pd2:.3f});  mfg-only rho(coupling, gain) = {rm:+.2f} (n = 6)")
print("wrote", OUT)
