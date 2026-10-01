# -*- coding: utf-8 -*-
"""Do sensors relate with a TIME LAG inside a 96-step window?  (2026-09-21)

The candidate's mixer combines sensors at the SAME time step (x_norm[t, :]).
If one sensor leads another ("current rises, temperature follows 10 min later"),
a same-time mixer only sees it through later per-sensor filters.  Measured on the
TRAIN region: non-overlapping 96-step windows, per-window/per-sensor z-score (the
model's RevIN), cross-correlation c_ij(k) = mean_t z_i(t) z_j(t+k), k in [-K, K],
averaged over windows.  Per sensor pair: the lag with the largest |c| vs lag 0.

    python tools/dev/sensor_lag.py
"""
import sys
import numpy as np
import pandas as pd

L, K = 96, 24
DATA = ["Steel", "EWELD", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3"]


def windows(name):
    df = pd.read_csv(f"data/{name}/data.csv")
    x = df.drop(columns=[c for c in df.columns if c.lower() in ("date", "unit")]).values.astype(np.float64)
    x = x[: int(len(x) * 0.7)]
    x = x[np.isfinite(x).all(1)]
    n = len(x) // L
    w = x[: n * L].reshape(n, L, -1)
    sd = w.std(1, keepdims=True)
    ok = (sd > 1e-6).all(axis=(1, 2))
    w = w[ok]
    return (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)


rows = []
for name in (sys.argv[1:] or DATA):
    z = windows(name)                                   # (N, L, D)
    N, _, D = z.shape
    lags = np.arange(-K, K + 1)
    C = np.zeros((len(lags), D, D))
    for a, k in enumerate(lags):
        if k >= 0:
            C[a] = np.einsum("ntd,nte->de", z[:, :L - k], z[:, k:]) / (N * (L - k))
        else:
            C[a] = np.einsum("ntd,nte->de", z[:, -k:], z[:, :L + k]) / (N * (L + k))
    iu = [(i, j) for i in range(D) for j in range(i + 1, D)]
    c0 = np.array([abs(C[K, i, j]) for i, j in iu])
    best = np.array([np.abs(C[:, i, j]).max() for i, j in iu])
    kbest = np.array([lags[np.abs(C[:, i, j]).argmax()] for i, j in iu])
    strong = best >= 0.3                                # pairs that are related at all
    lagged = strong & (np.abs(kbest) >= 3) & (best - c0 >= 0.1)
    rows.append(dict(data=name, D=D, windows=N, pairs=len(iu),
                     related=int(strong.sum()),
                     lagged=int(lagged.sum()),
                     med_abs_lag_related=float(np.median(np.abs(kbest[strong]))) if strong.any() else np.nan,
                     mean_gain_related=float((best - c0)[strong].mean()) if strong.any() else np.nan,
                     mean_c0_related=float(c0[strong].mean()) if strong.any() else np.nan))
    top = np.argsort(-(best - c0))[:3]
    print(f"{name:13s} D={D:2d} windows={N:4d} pairs={len(iu):3d} | related(|c|>=.3) {strong.sum():3d} | "
          f"lagged(|lag|>=3 & +0.1) {lagged.sum():3d} | median |lag| {rows[-1]['med_abs_lag_related']:.0f} | "
          f"mean |c| lag0 {rows[-1]['mean_c0_related']:.2f} -> best {rows[-1]['mean_c0_related'] + rows[-1]['mean_gain_related']:.2f}")
    for t in top:
        i, j = iu[t]
        print(f"{'':15s} top pair ({i},{j}): lag {kbest[t]:+d}  |c| {c0[t]:.2f} -> {best[t]:.2f}")
pd.DataFrame(rows).to_csv("results/_sensor_lag.csv", index=False)
print("wrote results/_sensor_lag.csv")
