"""Shape check with the stored CORR (per-window, per-channel Pearson correlation between
forecast and truth over the horizon, averaged; higher = the forecast follows the ups and
downs better, regardless of level and amplitude).

Only datasets without masked targets are compared: the baselines were run before the
2026-08-24 CORR-NAN-MASK fix and FEATHer after it, and the fix only changes CORR where
targets are masked (MetroPT3 / N2O / WindSCADA). Steel and EWELD have no masked cell,
BlastFurnace has one.

Decision rule fixed BEFORE looking (user, 2026-09-30): go on to DTW/TDI only if FEATHer-mix
has a consistently higher CORR than most models, DLinear included.

    python tools/dev/corr_check.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REF = "FEATHer_mix"
M = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS", "CycleNet",
     "DLinear", "FEATHer_CI", "FEATHer_mix"]
CLEAN = ["Steel", "EWELD", "BlastFurnace"]

raw = pd.read_csv("results/_archive/pre_raw_20261007/mfg_raw.csv")
raw = raw[raw.model.isin(M)]
cell = raw.groupby(["model", "data", "pred_len"]).CORR.mean().unstack("model")[M]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def report(sub, title):
    print(f"\n=== {title}  ({len(sub)} cells)")
    print("mean CORR:  " + "  ".join(f"{m}={sub[m].mean():.3f}" for m in
                                     sorted(M, key=lambda x: -sub[x].mean())))
    first = sub.idxmax(axis=1).value_counts()
    print("1st places: " + ", ".join(f"{m}={n}" for m, n in first.items()))
    opp = [m for m in M if m != REF]
    ps = [wilcoxon(sub[REF], sub[m], alternative="greater").pvalue for m in opp]
    for m, p, a in zip(opp, ps, holm(np.array(ps))):
        w = int((sub[REF] > sub[m]).sum()); l = int((sub[REF] < sub[m]).sum())
        print(f"   FEATHer-mix vs {m:14s} {w:2d}-{l:<2d}  Holm p={a:.3f}")


report(cell.loc[CLEAN], "clean datasets (Steel, EWELD, BlastFurnace)")
for d in CLEAN:
    s = cell.loc[d]
    print(f"\n{d}: " + "  ".join(f"H{h}: best={s.loc[h].idxmax()} ({s.loc[h].max():.3f}), "
                                  f"FEATHer-mix={s.loc[h, REF]:.3f}, DLinear={s.loc[h, 'DLinear']:.3f}"
                                  for h in s.index))
