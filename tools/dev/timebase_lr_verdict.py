# -*- coding: utf-8 -*-
"""TimeBase mfg learning-rate re-selection (2026-10-06), VALIDATION loss only.

The official TimeBase lr (0.1, scripts/etth1.sh) is 4-25 % worse than SparseTSF under our
protocol, so - like DLinear on ETTh2 - ONE lr for all six industrial datasets is chosen by the
mean per-cell validation loss over seed 2025 among {0.1, 0.01, 0.001}.  Test MSE is printed
for information only and plays no part in the choice.

    python tools/dev/timebase_lr_verdict.py

Inputs: results/_archive/pre_raw_20261007/new3_mfg.csv (exp_tag main = lr 0.1), results/_archive/pre_raw_20261007/timebase_lr_mfg.csv
(exp_tag tb_lr1e-2 / tb_lr1e-3).  Output: the exp_tag to put in merge_raw.TIMEBASE_MFG_LR_TAG.
"""
import os

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TAGS = {"main": 0.1, "tb_lr1e-2": 0.01, "tb_lr1e-3": 0.001}
SEED = 2025

frames = []
for p in ("results/_archive/pre_raw_20261007/new3_mfg.csv", "results/_archive/pre_raw_20261007/timebase_lr_mfg.csv"):
    if os.path.exists(p):
        d = pd.read_csv(p, on_bad_lines="skip")
        frames.append(d[(d.model == "TimeBase") & (d.seed == SEED) & d.exp_tag.isin(TAGS)])
d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not len(d):
    raise SystemExit("no TimeBase rows yet")
d = d.sort_values("timestamp").drop_duplicates(["exp_tag", "data", "pred_len"], keep="last")

n = d.groupby("exp_tag").size().reindex(TAGS).fillna(0).astype(int)
print("cells per lr (of 24):", {TAGS[t]: int(n[t]) for t in TAGS})
piv_val = d.pivot(index=["data", "pred_len"], columns="exp_tag", values="val_loss").reindex(columns=list(TAGS))
piv_mse = d.pivot(index=["data", "pred_len"], columns="exp_tag", values="MSE").reindex(columns=list(TAGS))
common = piv_val.dropna()
print(f"cells with all three lrs: {len(common)}/24")
if len(common):
    mv = common.mean()
    print("\nmean VAL loss (decision):")
    for t in TAGS:
        print(f"  lr {TAGS[t]:<6} {mv[t]:.4f}   (best in {int((common.idxmin(axis=1) == t).sum())}/{len(common)} cells)")
    best = mv.idxmin()
    print(f"\n==> choose lr {TAGS[best]} (exp_tag '{best}')"
          + ("" if best == "main" else f"; set merge_raw.TIMEBASE_MFG_LR_TAG = '{best}' and run seeds "
                                      f"2026-27 with --lr {TAGS[best]} --exp_tag {best}"))
    mt = piv_mse.loc[common.index].mean()
    print("\n(for information only) mean TEST MSE on the same cells:",
          {TAGS[t]: round(float(mt[t]), 4) for t in TAGS})
    print("per-dataset mean VAL loss:")
    print(common.groupby(level="data").mean().rename(columns=TAGS).round(4).to_string())
else:
    print("not decidable yet; rows per lr:\n", piv_val.notna().sum().rename(TAGS).to_string())
