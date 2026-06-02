# -*- coding: utf-8 -*-
"""
Wilcoxon signed-rank test for pairwise model comparison.

For each (data, pred_len) cell we have 5 seed-paired observations per
model. Two model variants compared on the *same set of seeds* form a
paired sample; aggregate across all (data, pred_len) cells gives the
test budget. Reports two outputs:

  (1) FEATHer vs each baseline — one p-value per opponent (one-sided,
      "FEATHer is lower / better").
  (2) Full N x N pairwise matrix of p-values.

Directly addresses:
  R1 #5, R2 #11, R8 #3  — "no statistical significance / mean +/- std only"
  R2 #17, #18           — "0.293 vs 0.291 is within run-to-run variance"

Usage:
    python tools/paper/wilcoxon.py
    python tools/paper/wilcoxon.py --exp_tag main --metric MSE
    python tools/paper/wilcoxon.py --reference FEATHer --output wilcoxon.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

LOWER_IS_BETTER = {"MSE", "MAE", "RMSE"}


def paired_observations(df, metric, model_a, model_b):
    """Return matched arrays of metric values across all common
    (data, pred_len, seed) cells. Same seeds, same horizons."""
    key = ["data", "pred_len", "seed"]
    a = df[df["model"] == model_a][key + [metric]].rename(columns={metric: "a"})
    b = df[df["model"] == model_b][key + [metric]].rename(columns={metric: "b"})
    m = pd.merge(a, b, on=key, how="inner")
    return m["a"].to_numpy(), m["b"].to_numpy()


def wilcoxon_pair(df, metric, m_a, m_b):
    """One-sided Wilcoxon: H1 = m_a is better than m_b.

    For lower-is-better metrics, 'better' = smaller, so we test whether
    (a - b) < 0. Returns (p_value, n_pairs, median_diff). NaN p when too
    few pairs or all differences are zero.
    """
    a, b = paired_observations(df, metric, m_a, m_b)
    if len(a) < 2:
        return np.nan, len(a), np.nan
    diff = a - b
    if metric not in LOWER_IS_BETTER:
        diff = -diff  # bigger-is-better -> flip so 'better' is still negative
    if np.all(diff == 0):
        return 1.0, len(a), 0.0
    try:
        _, p = wilcoxon(diff, alternative="less", zero_method="wilcox")
    except ValueError:
        return np.nan, len(a), float(np.median(diff))
    return float(p), len(a), float(np.median(diff))


def reference_vs_all(df, metric, ref):
    models = sorted(df["model"].unique())
    rows = []
    for m in models:
        if m == ref:
            continue
        p, n, md = wilcoxon_pair(df, metric, ref, m)
        rows.append({
            "reference": ref, "opponent": m, "n_pairs": n,
            "median_diff": md, "p_value": p,
            "significant_05": (p is not None and not np.isnan(p) and p < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value")


def full_matrix(df, metric):
    models = sorted(df["model"].unique())
    mat = pd.DataFrame(index=models, columns=models, dtype=float)
    for a in models:
        for b in models:
            if a == b:
                mat.loc[a, b] = np.nan
                continue
            p, _, _ = wilcoxon_pair(df, metric, a, b)
            mat.loc[a, b] = p
    return mat


def main():
    p = argparse.ArgumentParser(description="Pairwise Wilcoxon signed-rank test")
    p.add_argument("--csv", default="results/fcst_results.csv")
    p.add_argument("--exp_tag", default="main")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2"])
    p.add_argument("--reference", default="FEATHer",
                   help="Reference model for the one-vs-all table.")
    p.add_argument("--output", default=None,
                   help="Write reference-vs-all CSV; matrix prints to stderr.")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[empty] {args.csv} does not exist yet", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if args.exp_tag:
        df = df[df["exp_tag"] == args.exp_tag]
    if df.empty:
        print(f"No rows for exp_tag={args.exp_tag}", file=sys.stderr)
        sys.exit(1)

    ref_tbl = reference_vs_all(df, args.metric, args.reference)
    print(f"\n=== {args.reference} vs all (one-sided Wilcoxon, metric={args.metric}) ===")
    print(ref_tbl.to_string(index=False))

    print(f"\n=== Full pairwise matrix (p-values, metric={args.metric}) ===")
    mat = full_matrix(df, args.metric)
    print(mat.round(4).to_string())

    if args.output:
        ref_tbl.to_csv(args.output, index=False)
        print(f"\nWrote reference-vs-all to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
