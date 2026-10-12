# -*- coding: utf-8 -*-
"""Verdict for the control experiment "SparseTSF + small channel mixer vs FEATHer-mix"
(2026-10-06, pre-declared rule in manuscript/drafts/_records/control_sparsetsf_mix_plan.md).

    python tools/dev/control_verdict.py

Inputs : results/_archive/pre_raw_20261007/control_sparsetsf_mix.csv (exp_tag ctrl; models SparseTSF_mix, SparseTSF_mixM,
         SparseTSF_MLP_mix, SparseTSF_MLP_mixM), results/_archive/pre_raw_20261007/control_front_none.csv (exp_tag
         ctrl_frontnone; FEATHer with --front none = arm C), results/_archive/pre_raw_20261007/feather_mfg_candR4.csv
         (FEATHer-mix, the reference), results/_archive/pre_raw_20261007/mfg_raw.csv (SparseTSF / SparseTSF-MLP for context).
Metric : per-cell (dataset x H) seed-mean test MSE; "change" = mean over the 24 cells of the
         per-cell % change of the arm against FEATHer-mix (+ = arm worse), one-sided Wilcoxon on
         the paired cell MSEs ("FEATHer-mix lower").  MAE is reported, not judged.
Rule   : FEATHer-mix better than the BEST arm by >= 1 % (mean per-cell change) with p < 0.05
         -> bands/gate stay as "small corrections" (rule 1); otherwise rule 2 (drop them from the
         core claim); an arm better than FEATHer-mix by >= 1 % with p < 0.05 -> rule 3.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DS = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]
HS = [96, 192, 336, 720]
CELLS = [(d, h) for d in DS for h in HS]


def cells(df):
    g = df.groupby(["data", "pred_len"]).agg(MSE=("MSE", "mean"), MAE=("MAE", "mean"),
                                             n=("seed", "nunique"), p=("num_params", "max"))
    return g


_raw = pd.read_csv("results/raw_mfg.csv")      # 10-07: one managed file (role / config_id)
ref = cells(_raw[(_raw.role == "main") & (_raw.model == "FEATHer_mix") & (_raw.config_id == "frozen")
                 & (_raw.seq_len == 96) & _raw.MSE.notna()])
assert len(ref) == 24 and (ref.n == 3).all()

arms = {}
_c = _raw[(_raw.role == "control") & (_raw.seq_len == 96) & _raw.MSE.notna()]
for cfg in sorted(_c.config_id.unique()):           # "ctrl:<arm>@<host setting>"
    label = cfg[5:].replace("FEATHer_frontnone", "FEATHer_frontnone (C)")
    arms[label] = cells(_c[_c.config_id == cfg])
for m, cfg in (("SparseTSF", "cand:ETTh1"), ("SparseTSF_MLP", "cand:Electricity")):   # hosts as run with the arms
    h = _raw[(_raw.role == "main") & (_raw.model == m) & (_raw.config_id == cfg) & (_raw.seq_len == 96)]
    if len(h):
        arms[f"{m} @{cfg} (plain, ref)"] = cells(h)

if not arms:
    raise SystemExit("no control rows yet")

print(f"FEATHer-mix (reference): mean MSE {ref.MSE.mean():.4f}  MAE {ref.MAE.mean():.4f}\n")
rows = []
for name, a in arms.items():
    have = [c for c in CELLS if c in a.index]
    full = len(have) == 24 and (a.loc[have].n == 3).all()
    if not have:
        print(f"{name:28s} no cells yet")
        continue
    r = ref.loc[have]
    x = a.loc[have]
    pct = (x.MSE.values / r.MSE.values - 1) * 100
    pct_mae = (x.MAE.values / r.MAE.values - 1) * 100
    p = wilcoxon(r.MSE.values, x.MSE.values, alternative="less").pvalue if len(have) >= 6 else np.nan
    p_rev = wilcoxon(x.MSE.values, r.MSE.values, alternative="less").pvalue if len(have) >= 6 else np.nan
    byH = {h: round(float(np.mean([v for (d, hh), v in zip(have, pct) if hh == h])), 1) for h in HS
           if any(hh == h for _, hh in have)}
    rows.append(dict(arm=name, cells=len(have), seeds3=bool(full), meanMSE=x.MSE.mean(),
                     dMSE=pct.mean(), wins=int((pct > 0).sum()), p_ours_better=p, p_arm_better=p_rev,
                     dMAE=pct_mae.mean(), params_Steel96=int(a.loc[("Steel", 96)].p) if ("Steel", 96) in a.index else -1))
    print(f"{name:28s} {len(have):2d} cells{'' if full else ' (INCOMPLETE)'}: mean MSE {x.MSE.mean():.4f}, "
          f"vs FEATHer-mix {pct.mean():+.2f}% (ours better in {int((pct > 0).sum())}/{len(have)}, "
          f"p ours-better {p:.3f} / arm-better {p_rev:.3f}); MAE {pct_mae.mean():+.2f}%; by H {byH}")

tab = pd.DataFrame(rows)
cand = tab[tab.seeds3 & ~tab.arm.str.contains("plain")]
print()
if not len(cand):
    print("verdict: not yet (no complete 3-seed arm)")
else:
    best = cand.sort_values("meanMSE").iloc[0]
    print(f"best complete arm: {best.arm}  (FEATHer-mix {best.dMSE:+.2f}% better, p {best.p_ours_better:.3f})")
    if best.dMSE >= 1.0 and best.p_ours_better < 0.05:
        print("==> RULE 1: FEATHer-mix beats 'SparseTSF + mixer' by >= 1 % (significant): bands/gate stay as small corrections.")
    elif best.dMSE <= -1.0 and best.p_arm_better < 0.05:
        print("==> RULE 3: the control beats FEATHer-mix by >= 1 % (significant): simplification is mandatory.")
    else:
        print("==> RULE 2: within 1 % / not significant: bands/gate leave the core claim; "
              "one-sentence model = 'SparseTSF + the cheapest mixing bus, 43 channels per chip'.")
