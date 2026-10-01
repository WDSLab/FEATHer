"""Overall comparison on the six manufacturing datasets: can a summary table say
"FEATHer-mix is best", and under which condition?

For each opponent: wins / losses of FEATHer-mix over the (dataset, H) cells
(seed-averaged) and a one-sided Wilcoxon signed-rank p ("FEATHer-mix lower"),
(a) over all 24 cells and (b) only over the cells where the opponent fits the
microcontroller (analysis/deployment/deploy_cells.csv). Also the mean rank of
every model per cell, ranking only the models that fit that cell.

    python tools/dev/overall_rank_check.py
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REF = "FEATHer_mix"
MODELS = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS", "CycleNet",
          "DLinear", "FEATHer_CI", "FEATHer_mix"]

raw = pd.read_csv("results/mfg_raw.csv")
raw = raw[raw.model.isin(MODELS)]
dep = pd.read_csv("analysis/deployment/deploy_cells.csv")
fits = {(r.model, r.data, r.H): r.state == "FITS" for r in dep.itertuples()}

for metric in ("MSE", "MAE"):
    cell = raw.groupby(["model", "data", "pred_len"])[metric].mean().unstack("model")
    print(f"\n===== {metric}  ({len(cell)} cells)")
    ref_fit = [fits.get((REF, d, h), False) for d, h in cell.index]
    print(f"{REF} fits {sum(ref_fit)}/{len(cell)} cells")
    print(f"{'opponent':14s} | {'all cells: W-L   p':22s} | {'opponent fits: n  W-L   p':30s}")
    for m in MODELS:
        if m == REF:
            continue
        diff = cell[REF] - cell[m]                      # < 0 = FEATHer-mix better
        w, l = int((diff < 0).sum()), int((diff > 0).sum())
        p = wilcoxon(cell[REF], cell[m], alternative="less").pvalue
        mask = np.array([fits.get((m, d, h), False) for d, h in cell.index])
        if mask.sum() >= 5:
            dm = diff[mask]
            pw = wilcoxon(cell[REF][mask], cell[m][mask], alternative="less").pvalue
            s2 = f"{mask.sum():2d}  {int((dm < 0).sum()):2d}-{int((dm > 0).sum()):<2d} p={pw:.1e}"
        else:
            s2 = f"{mask.sum():2d}  (too few cells)"
        print(f"{m:14s} | {w:2d}-{l:<2d} p={p:.1e}        | {s2}")

    # mean rank among the models that fit each cell
    ranks = {m: [] for m in MODELS}
    for (d, h), row in cell.iterrows():
        ok = [m for m in MODELS if fits.get((m, d, h), False)]
        r = row[ok].rank(method="average")
        for m in ok:
            ranks[m].append(r[m])
    print("mean rank among fitting models (cells ranked):")
    for m in sorted(MODELS, key=lambda x: np.mean(ranks[x]) if ranks[x] else 99):
        print(f"   {m:14s} {np.mean(ranks[m]):.2f}  (n={len(ranks[m])})" if ranks[m] else f"   {m:14s} -")
