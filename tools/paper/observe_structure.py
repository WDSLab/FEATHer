"""Data observations that motivate the model structure (2026-10-13) - TRAIN split only (first 70 % of rows),
no validation or test data is read.

    python tools/paper/observe_structure.py      # -> results/analysis/observation_20261013.csv + printout

Per industrial dataset, on the train split standardised with train statistics, rows with every channel observed
(mask.csv) only:
  pca_k80 / k90 / k95   number of principal components needed for 80 / 90 / 95 % of the cross-channel variance
  pca_top4              variance share of the first 4 components (the mixing width R = 4)
  r2_others             mean over channels of R^2 when a channel is regressed on the other channels at the same
                        time step (how much the sensors co-move)
  period, day           the period-type values the protocol uses (FFT <= 48 rule / daily cycle)
  acf_P                 mean over channels of the autocorrelation at lag P (strength of the period the head uses)
  energy_P              mean over channels of the spectral power share at multiples of 1/P (+-1 bin), DC excluded
  slow96 / slow720      mean over channels of the spectral power share with period > 96 / > 720 steps (DC excluded):
                        how much of the variance is slower than the input window / the longest horizon
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import baselines as B   # noqa: E402

MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
OUT = os.path.join(ROOT, "results", "analysis", "observation_20261013.csv")


def train_split(name):
    p = os.path.join(ROOT, "data", name)
    x = pd.read_csv(os.path.join(p, "data.csv")).drop(columns=["date"]).to_numpy(float)
    mp = os.path.join(p, "mask.csv")
    m = pd.read_csv(mp).drop(columns=["date"]).to_numpy(bool) if os.path.exists(mp) else np.ones_like(x, bool)
    n = int(len(x) * 0.7)
    x, m = x[:n], m[:n]
    mu, sd = x.mean(0), x.std(0) + 1e-8
    return (x - mu) / sd, m


def acf(v, lag):
    v = v - v.mean()
    return float((v[:-lag] * v[lag:]).sum() / ((v * v).sum() + 1e-12))


def energy_at(v, P):
    f = np.abs(np.fft.rfft(v - v.mean())) ** 2
    n = len(v)
    tot = f[1:].sum()
    k0 = n / P
    idx = set()
    for h in range(1, int((len(f) - 1) / k0) + 1):
        c = int(round(h * k0))
        idx.update(range(max(1, c - 1), min(len(f), c + 2)))
    return float(f[sorted(idx)].sum() / (tot + 1e-12))


def slow_share(v, steps):
    f = np.abs(np.fft.rfft(v - v.mean())) ** 2
    per = len(v) / np.arange(1, len(f))
    return float(f[1:][per > steps].sum() / (f[1:].sum() + 1e-12))


rows = []
for d in MFG:
    x, m = train_split(d)
    D = x.shape[1]
    full = m.all(1)
    xo = x[full]
    ev = np.linalg.eigvalsh(np.cov(xo.T))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    k = lambda q: int(np.searchsorted(cum, q) + 1)
    r2 = []
    for c in range(D):
        others = np.delete(xo, c, axis=1)
        A = np.c_[others, np.ones(len(xo))]
        coef, *_ = np.linalg.lstsq(A, xo[:, c], rcond=None)
        res = xo[:, c] - A @ coef
        r2.append(1 - res.var() / (xo[:, c].var() + 1e-12))
    P = int(B.get_dataset_overrides("SparseTSF", d).get("sparsetsf_period_len", 24))
    day = int(B.get_dataset_overrides("CycleNet", d)["cyclenet_cycle"])
    rows.append(dict(data=d, D=D, train_rows=len(x), rows_all_observed=int(full.sum()),
                     pca_k80=k(0.80), pca_k90=k(0.90), pca_k95=k(0.95), pca_top4=float(cum[min(3, D - 1)]),
                     r2_others=float(np.mean(r2)), r2_others_min=float(np.min(r2)),
                     period=P, acf_P=float(np.mean([acf(x[:, c], P) for c in range(D)])),
                     energy_P=float(np.mean([energy_at(x[:, c], P) for c in range(D)])),
                     day=day, acf_day=float(np.mean([acf(x[:, c], day) for c in range(D)])),
                     slow96=float(np.mean([slow_share(x[:, c], 96) for c in range(D)])),
                     slow720=float(np.mean([slow_share(x[:, c], 720) for c in range(D)]))))
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
