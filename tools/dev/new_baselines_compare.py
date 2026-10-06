# -*- coding: utf-8 -*-
"""New baselines (CMoS, TimeBase, PhaseFormer, SAMformer) vs FEATHer-mix and the T6 chip pool,
seed-matched (seed 2025). 2026-10-06. Reads results/cmos_mfg.csv + results/new3_mfg.csv + results/mfg_raw.csv.

    python tools/dev/new_baselines_compare.py
"""
# CMoS + TimeBase + PhaseFormer + SAMformer vs FEATHer-mix and the T6 chip pool (seed-matched).
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
raw = pd.read_csv("results/mfg_raw.csv")
new = pd.concat([pd.read_csv(f) for f in ("results/cmos_mfg.csv", "results/new3_mfg.csv") if os.path.exists(f)])
new = new[new.seq_len == 96]
NEW = [m for m in ("CMoS", "TimeBase", "PhaseFormer", "SAMformer") if m in set(new.model)]
POOL = ["FEATHer_mix", "FEATHer_CI", "SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS"]
DS = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]
allr = pd.concat([raw, new])

# cells where every new model has seed 2025
S = 2025
have = new[new.seed == S].groupby(["data", "pred_len"]).model.nunique()
for m in NEW:
    n = len(new[(new.model == m) & (new.seed == S)])
    print(f"{m:12s} seed {S}: {n}/24 cells")
cells = [(d, h) for (d, h), k in have.items()]

def val(m, d, h, met):
    s = allr[(allr.model == m) & (allr.data == d) & (allr.pred_len == h) & (allr.seed == S)]
    return float(s[met].iloc[0]) if len(s) else np.nan

for met in ("MSE", "MAE"):
    print(f"\n===== {met} (seed {S}) =====")
    for m in NEW:
        c = [(d, h) for (d, h) in cells if not np.isnan(val(m, d, h, met))]
        a = np.array([val(m, d, h, met) for d, h in c]); b = np.array([val("FEATHer_mix", d, h, met) for d, h in c])
        if len(c) == 0:
            continue
        pct = (a / b - 1) * 100
        p = wilcoxon(a, b, alternative="greater").pvalue if len(c) >= 6 else float("nan")
        byH = pd.Series(pct, index=[h for _, h in c]).groupby(level=0).mean().round(1).to_dict()
        byD = pd.Series(pct, index=[d for d, _ in c]).groupby(level=0).mean().round(1).to_dict()
        print(f"{m:12s} vs FEATHer-mix over {len(c)} cells: ours better {int((pct > 0).sum())}/{len(c)}, "
              f"mean {pct.mean():+.2f}% (+ = model worse), one-sided p {p:.3f}\n    by H {byH}\n    by data {byD}")
    # chip pool ranks: T6 pool + chip-fitting new models (SAMformer excluded: fits only H96)
    chip = POOL + [m for m in NEW if m != "SAMformer"]
    cc = [(d, h) for (d, h) in cells if all(not np.isnan(val(m, d, h, met)) for m in chip)]
    M = pd.DataFrame([[val(m, d, h, met) for m in chip] for d, h in cc], columns=chip)
    R = M.rank(axis=1)
    print(f"  chip pool ({len(chip)} models, {len(cc)} cells) mean rank / 1st places:")
    for m in R.mean().sort_values().index:
        print(f"    {m:14s} {R[m].mean():.2f} ({int((R[m] == 1).sum())})")
    # all lightweight incl. DLinear / CycleNet / SAMformer, filter-free
    full = chip + ["DLinear", "CycleNet"] + (["SAMformer"] if "SAMformer" in NEW else [])
    cc = [(d, h) for (d, h) in cells if all(not np.isnan(val(m, d, h, met)) for m in full)]
    M = pd.DataFrame([[val(m, d, h, met) for m in full] for d, h in cc], columns=full)
    R = M.rank(axis=1)
    print(f"  filter-free lightweight ({len(full)} models, {len(cc)} cells):")
    for m in R.mean().sort_values().index[:6]:
        print(f"    {m:14s} {R[m].mean():.2f} ({int((R[m] == 1).sum())})")
