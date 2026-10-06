# -*- coding: utf-8 -*-
"""P1 shared-factor dev verdict (2026-10-06) — applies manuscript/drafts/p1_factor_plan.md as written.

Per arm and (data, H) cell: lr chosen on VAL (seed-mean val_loss), then seed-mean test MSE/MAE.
Baseline = frozen model, nf_L96_* rows of results/dev_nearfar.csv (same dev protocol, 10-04).
Main rule: mean over the 12 cells of the % MSE change <= +0.5 % and no dataset worse than +5 %.
Choice: among passing arms, p3 if within 0.3 %p of the best passing arm, else the best.

    python tools/dev/p1_verdict.py
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
DS = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
HS = [96, 720]


def cells(df, prefix):
    out = {}
    if df.empty or "data" not in df.columns:
        return out
    d = df[df.exp_tag.astype(str).str.startswith(prefix)]
    for (data, H), g in d.groupby(["data", "pred_len"]):
        if data not in DS or H not in HS:
            continue
        per_lr = g.groupby("exp_tag").agg(val=("val_loss", "mean"), mse=("MSE", "mean"),
                                         mae=("MAE", "mean"), n=("seed", "nunique"))
        per_lr = per_lr[per_lr.n == 3]
        if len(per_lr):
            best = per_lr.val.idxmin()
            out[(data, H)] = (per_lr.loc[best, "mse"], per_lr.loc[best, "mae"], best.rsplit("_", 1)[-1])
    return out


base = cells(pd.read_csv("results/dev_nearfar.csv"), "nf_L96_")
p1 = pd.read_csv("results/dev_p1.csv") if os.path.exists("results/dev_p1.csv") else pd.DataFrame(columns=["exp_tag"])
res = {}
for arm in ("p1", "p2", "p3"):
    c = cells(p1, f"p1_{arm}_")
    keys = [k for k in base if k in c]
    if not keys:
        continue
    rows = []
    for k in sorted(keys, key=lambda k: (DS.index(k[0]), k[1])):
        bm, ba, _ = base[k]; am, aa, lr = c[k]
        rows.append(dict(data=k[0], H=k[1], base_mse=bm, arm_mse=am, d_mse=(am / bm - 1) * 100,
                         d_mae=(aa / ba - 1) * 100, lr=lr))
    t = pd.DataFrame(rows)
    by_data = t.groupby("data").d_mse.mean()
    mean = t.d_mse.mean()
    ok = len(t) == 12 and mean <= 0.5 and (by_data <= 5.0).all()
    res[arm] = (mean, ok, len(t))
    print(f"\n=== {arm}: {len(t)}/12 cells | mean MSE change {mean:+.2f}% | MAE {t.d_mae.mean():+.2f}% | "
          f"worst dataset {by_data.idxmax()} {by_data.max():+.2f}% | {'PASS' if ok else ('FAIL' if len(t) == 12 else 'incomplete')}")
    print(t.round({"base_mse": 4, "arm_mse": 4, "d_mse": 2, "d_mae": 2}).to_string(index=False))

passing = {a: m for a, (m, ok, n) in res.items() if ok}
if len(res) == 3 and all(n == 12 for _, _, n in res.values()):
    if not passing:
        print("\nVERDICT: no arm passes -> keep the frozen model (plan item 4)")
    else:
        best = min(passing, key=passing.get)
        pick = "p3" if "p3" in passing and passing["p3"] - passing[best] <= 0.3 else best
        print(f"\nVERDICT: passing {sorted(passing)}; best {best} ({passing[best]:+.2f}%); chosen = {pick}")
else:
    print("\n(not all arms complete yet)")
