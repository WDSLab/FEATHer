# -*- coding: utf-8 -*-
"""Cross-channel stability vs mixing gain - the pre-registered one-shot test.

Rule, written and committed BEFORE this was run: docs/prereg_stability_1009.md.

    S = mean over channel pairs (i < j) of |r_train - r_test|   (Pearson, lag 0)
        train = the experiments' train split, test = test split minus the 96-row input window,
        observed values only (industrial masks), pairwise-complete rows; zero-variance pairs dropped.
    G = mean over the 4 horizons of the per-cell % change (CI - mix) / mix x 100
        (FEATHer frozen, L 96, mean of 3 seeds) - the rule of the paper's 5.9 %.
    verdict: Spearman rho(S, G) over 14 datasets <= -0.5 and one-sided p < 0.05.
    secondary: ETT collapsed to 2 units (12 datasets); per-group S ranges (no test).

    python tools/paper/stability_vs_mixing.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.data_factory import data_provider   # noqa: E402

MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange", "Electricity", "Traffic"]
L = 96
OUT = os.path.join(ROOT, "results", "analysis", "stability_vs_mixing_20261009.csv")


def split_values(data):
    """(train, test) arrays as the experiments see them; imputed industrial cells are NaN."""
    tr, _ = data_provider("data/", data, "M", 32, L, L // 2, 96, flag="train")
    te, _ = data_provider("data/", data, "M", 32, L, L // 2, 96, flag="test")
    return np.asarray(tr.data_y, dtype=float), np.asarray(te.data_y, dtype=float)[L:]


def corr_pairwise(a):
    """Pearson correlation of the columns, pairwise-complete (pandas), lag 0."""
    return pd.DataFrame(a).corr(method="pearson", min_periods=10).to_numpy()


def stability(data):
    tr, te = split_values(data)
    ctr, cte = corr_pairwise(tr), corr_pairwise(te)
    iu = np.triu_indices(ctr.shape[0], k=1)
    d = np.abs(ctr[iu] - cte[iu])
    ok = np.isfinite(d)                       # zero-variance / no-overlap pairs drop out
    return float(d[ok].mean()), ctr.shape[0], int(ok.sum())


def mixing_gain(data):
    g = "mfg" if data in MFG else "ltsf"
    r = pd.read_csv(os.path.join(ROOT, "results", f"raw_{g}.csv"))
    r = r[(r.role == "main") & (r.config_id == "frozen") & (r.seq_len == L) & (r.data == data)
          & r.model.isin(["FEATHer_mix", "FEATHer_CI"])]
    m = r.groupby(["model", "pred_len"]).MSE.mean().unstack(0)
    assert len(m) == 4 and m.notna().all().all(), (data, m)
    return float(((m.FEATHer_CI - m.FEATHer_mix) / m.FEATHer_mix * 100).mean())


def verdict(df, label):
    rho, p = spearmanr(df.S, df.G, alternative="less")
    ok = rho <= -0.5 and p < 0.05
    print(f"{label}: n = {len(df)}  Spearman rho = {rho:+.3f}  one-sided p = {p:.4f}  -> "
          f"{'CONSISTENT with the hypothesis' if ok else 'NOT explained by stability'}")
    return rho, p, ok


def main():
    rows = []
    for d in MFG + LTSF:
        s, nch, npairs = stability(d)
        rows.append(dict(data=d, group="mfg" if d in MFG else "ltsf", channels=nch, pairs=npairs,
                         S=s, G=mixing_gain(d)))
        print(f"  {d:<13} D={nch:<4} S={s:.4f}  G={rows[-1]['G']:+.2f} %", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print()
    rho, p, ok = verdict(df, "PRIMARY (14 datasets)")
    e = df.copy()
    e.loc[e.data.isin(["ETTh1", "ETTm1"]), "data"] = "ETT-1"
    e.loc[e.data.isin(["ETTh2", "ETTm2"]), "data"] = "ETT-2"
    e = e.groupby("data", as_index=False)[["S", "G"]].mean()
    verdict(e, "secondary (ETT collapsed, 12)")
    for g in ("mfg", "ltsf"):
        s = df[df.group == g].S
        print(f"  S range {g}: {s.min():.4f} .. {s.max():.4f} (median {s.median():.4f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
