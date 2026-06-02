# -*- coding: utf-8 -*-
"""
Progress dashboard for forecasting experiments.

Default mode (fcst_results.csv) prints a coverage matrix per
(exp_tag, model, dataset, pred_len) — how many seeds are done.

`--robust` mode reads results/robust_results.csv and additionally
breaks coverage down by (fault_type, severity).

Usage:
    python tools/audit/check_progress.py
    python tools/audit/check_progress.py --exp_tag main
    python tools/audit/check_progress.py --model FEATHer
    python tools/audit/check_progress.py --csv results/fcst_results.csv
    python tools/audit/check_progress.py --robust
    python tools/audit/check_progress.py --robust --model FEATHer
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load(csv_path):
    if not os.path.exists(csv_path):
        print(f"[empty] {csv_path} does not exist yet")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[empty] {csv_path}")
    return df


def report_forecast(df, csv_path):
    group_cols = ["exp_tag", "model", "data", "pred_len", "seq_len"]
    grp = (df.groupby(group_cols)["seed"]
             .agg(['nunique', lambda s: sorted(s.unique().tolist())])
             .reset_index())
    grp.columns = group_cols + ["n_seeds", "seeds"]
    grp = grp.sort_values(["exp_tag", "model", "data", "pred_len"])

    print("\n" + "=" * 90)
    print(f"  Coverage - {csv_path}")
    print("=" * 90)
    for _, r in grp.iterrows():
        seeds_str = ",".join(str(s) for s in r.seeds)
        print(f"  [{r.exp_tag:>12}] {r.model:>15} | {r.data:>12} | "
              f"H={r.pred_len:<4} L={r.seq_len} | "
              f"n={r.n_seeds:<2} | seeds: {seeds_str}")

    print("\n" + "-" * 90)
    print("  Per-model totals (rows present)")
    print("-" * 90)
    totals = (df.groupby(["exp_tag", "model"]).size()
                .reset_index(name="rows")
                .sort_values(["exp_tag", "model"]))
    for _, r in totals.iterrows():
        print(f"  [{r.exp_tag:>12}] {r.model:>15} | {r.rows} rows")

    print("\n" + "-" * 90)
    print("  Mean +/- std preview (over seeds, all metrics)")
    print("-" * 90)
    metrics_cols = ["MSE", "MAE", "RMSE", "CORR", "R2"]
    summary = (df.groupby(group_cols)[metrics_cols]
                 .agg(['mean', 'std']).round(4).reset_index())
    head = summary.head(20)
    print(head.to_string(index=False))
    if len(summary) > 20:
        print(f"  ... ({len(summary) - 20} more rows)")


def report_robust(df, csv_path):
    # Robustness CSV adds fault_type/severity columns. Group by checkpoint
    # cell first to show overall completeness, then break down by fault.
    cell_cols = ["exp_tag", "train_exp_tag", "model", "data",
                 "pred_len", "seq_len", "seed"]
    cell_counts = df.groupby(cell_cols).size().reset_index(name="rows")
    cell_counts = cell_counts.sort_values(
        ["exp_tag", "model", "data", "pred_len", "seed"]
    )

    print("\n" + "=" * 100)
    print(f"  Robustness coverage - {csv_path}")
    print(f"  (expected 20 rows/cell: 1 clean + 5 gauss + 5 miss + 4 impulse + 5 quant)")
    print("=" * 100)
    for _, r in cell_counts.iterrows():
        flag = "OK " if r.rows >= 20 else "..."
        print(f"  {flag} [{r.exp_tag:>10}/{r.train_exp_tag:>6}] "
              f"{r.model:>15} | {r.data:>12} | H={r.pred_len:<4} "
              f"L={r.seq_len} | seed={r.seed} | rows={r.rows}")

    print("\n" + "-" * 100)
    print("  Per-(model, fault_type) totals")
    print("-" * 100)
    fault_totals = (df.groupby(["exp_tag", "model", "fault_type"]).size()
                      .reset_index(name="rows")
                      .sort_values(["exp_tag", "model", "fault_type"]))
    for _, r in fault_totals.iterrows():
        print(f"  [{r.exp_tag:>10}] {r.model:>15} | {r.fault_type:>10} | {r.rows} rows")

    print("\n" + "-" * 100)
    print("  MSE mean +/- std by (model, data, pred_len, fault_type, severity)")
    print("-" * 100)
    summary = (df.groupby(["model", "data", "pred_len", "fault_type", "severity"])["MSE"]
                 .agg(['mean', 'std', 'count']).round(4).reset_index())
    head = summary.head(30)
    print(head.to_string(index=False))
    if len(summary) > 30:
        print(f"  ... ({len(summary) - 30} more rows)")


def main():
    p = argparse.ArgumentParser(description="Forecast / robustness progress audit")
    p.add_argument("--csv",     type=str, default=None,
                   help="Path to results CSV. Defaults to "
                        "results/robust_results.csv when --robust is set, "
                        "results/fcst_results.csv otherwise.")
    p.add_argument("--exp_tag", type=str, default=None)
    p.add_argument("--model",   type=str, default=None)
    p.add_argument("--data",    type=str, default=None)
    p.add_argument("--robust",  action="store_true",
                   help="Audit robust_results.csv (fault_type / severity breakdown).")
    args = p.parse_args()

    if args.csv is None:
        args.csv = ("results/robust_results.csv" if args.robust
                    else "results/fcst_results.csv")

    df = load(args.csv)
    if df.empty:
        return

    if args.exp_tag:
        df = df[df.exp_tag == args.exp_tag]
    if args.model:
        df = df[df.model == args.model]
    if args.data:
        df = df[df.data == args.data]

    if df.empty:
        print("No rows match the filter.")
        return

    if args.robust:
        report_robust(df, args.csv)
    else:
        report_forecast(df, args.csv)


if __name__ == "__main__":
    main()
