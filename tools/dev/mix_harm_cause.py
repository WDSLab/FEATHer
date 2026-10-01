# -*- coding: utf-8 -*-
"""Which sensors does mixing hurt, and why?  (2026-09-22)

Per sensor (train region, 96-step windows):
  related   max |corr| with any other sensor (lag 0 and best lag within +-24),
            computed on per-window z-scores (the model's RevIN)
  flat      share of windows whose std, in globally standardised units, is < 0.1
            -> RevIN divides by that small std and blows up the wiggle
  ci_mse    the sensor's own error without mixing (how easy it already is)
against the measured mixing effect per sensor (results/_mix_per_sensor.csv,
mix vs mixer-off, negative = mixing helps).

    python tools/dev/mix_harm_cause.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

L, K = 96, 24
ps = pd.read_csv("results/_mix_per_sensor.csv")
rows = []
for _, r in ps.iterrows():
    name = r["data"]
    gain = np.array([float(v) for v in r.per_sensor.split()])
    ci = np.array([float(v) for v in r.mse_ci.split()])
    df = pd.read_csv(f"data/{name}/data.csv")
    x = df.drop(columns=[c for c in df.columns if c.lower() in ("date", "unit")]).values.astype(np.float64)
    ntr = int(len(x) * 0.7)
    tr = x[:ntr]
    tr = tr[np.isfinite(tr).all(1)]
    g = (tr - tr.mean(0)) / (tr.std(0) + 1e-12)                  # global standardisation (train)
    n = len(g) // L
    w = g[: n * L].reshape(n, L, -1)
    sd = w.std(1)                                                # (N, D) in global units
    flat = (sd < 0.1).mean(0)
    ok = (sd > 1e-6).all(1)
    z = (w[ok] - w[ok].mean(1, keepdims=True)) / w[ok].std(1, keepdims=True)
    N, _, D = z.shape
    best = np.zeros((D, D)); c0 = np.zeros((D, D))
    for k in range(-K, K + 1):
        if k >= 0:
            C = np.einsum("ntd,nte->de", z[:, :L - k], z[:, k:]) / (N * (L - k))
        else:
            C = np.einsum("ntd,nte->de", z[:, -k:], z[:, :L + k]) / (N * (L + k))
        best = np.maximum(best, np.abs(C))
        if k == 0:
            c0 = np.abs(C)
    np.fill_diagonal(best, 0); np.fill_diagonal(c0, 0)
    for d in range(D):
        rows.append(dict(data=name, sensor=d, gain=gain[d], ci_mse=ci[d], flat=flat[d],
                         related0=c0[d].max(), related_best=best[d].max(),
                         related_top3=np.sort(best[d])[-3:].mean()))
t = pd.DataFrame(rows)
t.to_csv("results/_mix_harm_cause.csv", index=False)

feats = ["related0", "related_best", "related_top3", "flat", "ci_mse"]
print("Spearman rho between the mixing effect (positive = mixing HURTS) and each feature")
for scope, g in [("all 83 sensors", t), ("BlastFurnace", t[t.data == "BlastFurnace"]),
                 ("WindSCADA", t[t.data == "WindSCADA"]), ("EWELD", t[t.data == "EWELD"])]:
    line = "  ".join(f"{f} {spearmanr(g.gain, g[f])[0]:+.2f} (p={spearmanr(g.gain, g[f])[1]:.2f})" for f in feats)
    print(f"{scope:15s} n={len(g):2d}  {line}")

print("\nhurt (>= +1 %) vs helped (<= -1 %) sensors, feature means")
for scope, g in [("all", t), ("BlastFurnace", t[t.data == "BlastFurnace"])]:
    h, p = g[g.gain >= 1], g[g.gain <= -1]
    print(f"{scope:13s} hurt n={len(h):2d}: " + " ".join(f"{f} {h[f].mean():.3f}" for f in feats))
    print(f"{'':13s} help n={len(p):2d}: " + " ".join(f"{f} {p[f].mean():.3f}" for f in feats))
print("\nBF / WindSCADA sensors sorted by mixing effect")
for name in ["BlastFurnace", "WindSCADA"]:
    g = t[t.data == name].sort_values("gain", ascending=False)
    print(name); print(g[["sensor", "gain", "related_best", "related_top3", "flat", "ci_mse"]].round(3).to_string(index=False))
print("wrote results/_mix_harm_cause.csv")
