#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Downstream KPI on Steel: does forecast accuracy buy peak-demand reduction?

MSE is a symmetric average over every interval and every magnitude. The
decision a plant actually makes from a day-ahead load forecast is not "be
close everywhere" but "name the intervals that will set the demand charge" --
which depends only on being on the right side of a ranking, in the tail. A
model can be the most accurate on average and still be the worst at that.

Scoring (all from dumped test predictions; no retraining):

  capture@k   fraction of the day's true top-k intervals the model named.
              Parameter-free apart from k.

  peak relief the decision-level metric. If the plant can curtail the k
              intervals it flags, the demand charge is set by the largest
              UNFLAGGED interval. Perfect foresight leaves the (k+1)-th
              largest; doing nothing leaves the largest. Relief is where a
              model lands between those two, in [0, 1]:

                  relief = (do_nothing - realized) / (do_nothing - oracle)

Baselines are the honest ones for a load series with a daily cycle:
persistence (yesterday's profile) and climatology (mean profile). Steel's
peaks are 15x its median, and both baselines capture < 20% of them, so there
is real headroom -- but that is exactly what this script tests, per model.

    python tools/paper/kpi_steel_peak.py
    python tools/paper/kpi_steel_peak.py --k 4,8,12 --channel Usage_kWh
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

DISPLAY = {"SparseTSF": "SparseTSF-Linear", "DiPE_Linear": "DiPE-Linear",
           "SparseTSF_MLP": "SparseTSF-MLP"}


def topk_mask(a: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k largest entries of each row."""
    idx = np.argsort(-a, axis=1)[:, :k]
    m = np.zeros(a.shape, dtype=bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def score(y_true: np.ndarray, y_flag: np.ndarray, k: int) -> dict:
    """Score a decision rule that flags the k intervals y_flag ranks highest.

    y_true / y_flag are (days, intervals) in physical units.
    """
    flagged = topk_mask(y_flag, k)
    truth = topk_mask(y_true, k)

    capture = (flagged & truth).sum(axis=1) / k

    # Demand charge is set by the largest interval you did NOT curtail.
    realized = np.where(flagged, -np.inf, y_true).max(axis=1)
    do_nothing = y_true.max(axis=1)
    oracle = np.sort(y_true, axis=1)[:, -(k + 1)]   # (k+1)-th largest
    span = do_nothing - oracle
    ok = span > 1e-9                                # flat day -> no decision
    relief = np.where(ok, (do_nothing - realized) / np.where(ok, span, 1.0), np.nan)

    return {"capture": float(capture.mean()),
            "relief": float(np.nanmean(relief)),
            "residual_peak": float(realized.mean()),
            "oracle_peak": float(oracle.mean()),
            "do_nothing_peak": float(do_nothing.mean()),
            "days": int(len(y_true))}


def load_dumps(d: str, channel: str):
    """model -> list of (y_true, y_pred) per seed, in physical units."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        m = re.match(r"(.+)__(.+)__H(\d+)__s(\d+)\.npz", os.path.basename(f))
        if not m:
            continue
        model = m.group(1)
        z = np.load(f, allow_pickle=True)
        cols = [str(c) for c in z["columns"]]
        if channel not in cols:
            raise SystemExit("channel %r not in %s (have %s)" % (channel, f, cols))
        ci = cols.index(channel)
        mean, scale = z["scaler_mean"][ci], z["scaler_scale"][ci]
        yt = z["y_true"][:, :, ci] * scale + mean
        yp = z["y_pred"][:, :, ci] * scale + mean
        out.setdefault(model, []).append((yt, yp))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", default="results/preds_steel")
    p.add_argument("--channel", default="Usage_kWh")
    p.add_argument("--k", default="4,8,12")
    p.add_argument("--mse_csv", default="results/analysis/kpi_steel.csv")
    args = p.parse_args()

    ks = [int(x) for x in args.k.split(",")]
    dumps = load_dumps(args.dumps, args.channel)
    if not dumps:
        raise SystemExit("no dumps in %s -- run the sweep with --dump_preds"
                         % args.dumps)

    # Truth is identical across models; take it from any one.
    y_true = next(iter(dumps.values()))[0][0]
    for mdl, runs in dumps.items():
        for yt, _ in runs:
            if not np.allclose(yt, y_true, atol=1e-3):
                raise SystemExit("targets differ across models -- dumps are "
                                 "not aligned (%s)" % mdl)

    print("Steel day-ahead peak decision  |  channel=%s  days=%d"
          % (args.channel, len(y_true)))
    print("daily peak %.1f  median %.1f  (ratio %.1fx)\n"
          % (y_true.max(1).mean(), np.median(y_true),
             y_true.max(1).mean() / max(np.median(y_true), 1e-9)))

    rows = []
    # naive references
    refs = {"persistence": np.vstack([y_true[:1], y_true[:-1]]),
            "climatology": y_true.mean(0, keepdims=True).repeat(len(y_true), 0)}
    for name, flag in refs.items():
        for k in ks:
            s = score(y_true, flag, k)
            rows.append(dict(model=name, seeds=0, k=k, **s))
    for mdl, runs in dumps.items():
        for k in ks:
            per = [score(y_true, yp, k) for _, yp in runs]
            rows.append(dict(
                model=DISPLAY.get(mdl, mdl), seeds=len(runs), k=k,
                capture=float(np.mean([r["capture"] for r in per])),
                capture_sd=float(np.std([r["capture"] for r in per])),
                relief=float(np.mean([r["relief"] for r in per])),
                relief_sd=float(np.std([r["relief"] for r in per])),
                residual_peak=float(np.mean([r["residual_peak"] for r in per])),
                oracle_peak=per[0]["oracle_peak"],
                do_nothing_peak=per[0]["do_nothing_peak"],
                days=per[0]["days"]))

    t = pd.DataFrame(rows)
    for k in ks:
        sub = t[t.k == k].sort_values("relief", ascending=False)
        print("--- k=%d flagged intervals/day (%.1f%% of the day) ---"
              % (k, k / y_true.shape[1] * 100))
        cols = ["model", "capture", "relief", "residual_peak", "seeds"]
        print(sub[cols].to_string(index=False,
                                  formatters={"capture": "{:.3f}".format,
                                              "relief": "{:.3f}".format,
                                              "residual_peak": "{:.1f}".format}))
        print("   oracle peak %.1f | do-nothing peak %.1f\n"
              % (sub.oracle_peak.dropna().iloc[0],
                 sub.do_nothing_peak.dropna().iloc[0]))

    # The actual question: does MSE rank models the way the decision does?
    if os.path.exists(args.mse_csv):
        mse = pd.read_csv(args.mse_csv)
        mse = mse[mse.pred_len == 96].groupby("model").MSE.mean()
        mse.index = [DISPLAY.get(i, i) for i in mse.index]
        for k in ks:
            sub = t[(t.k == k) & (t.seeds > 0)].set_index("model")
            common = sub.index.intersection(mse.index)
            if len(common) < 3:
                continue
            a, b = mse[common], sub.loc[common, "relief"]
            print("k=%d  Spearman(MSE rank, peak-relief rank) = %+.3f  (n=%d)"
                  % (k, pd.Series(a).corr(pd.Series(b), method="spearman"),
                     len(common)))
            cmp = pd.DataFrame({"MSE": a, "MSE_rank": a.rank(),
                                "relief": b, "relief_rank": b.rank(ascending=False)})
            print(cmp.sort_values("MSE").round(4).to_string(), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
