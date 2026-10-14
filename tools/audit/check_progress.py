# -*- coding: utf-8 -*-
"""
Progress dashboard for forecasting experiments.

Prints a coverage matrix per (exp_tag, model, dataset, pred_len) — how many
seeds are done — plus an epoch-budget check (default file results/raw_mfg.csv).

Usage:
    python tools/audit/check_progress.py
    python tools/audit/check_progress.py --exp_tag main
    python tools/audit/check_progress.py --model FEATHer_mix
    python tools/audit/check_progress.py --csv results/raw_ltsf.csv
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


def report_epoch_budget(df, num_epochs, patience):
    """Flag runs whose best-val epoch landed inside the final patience
    window. Those runs ended at the epoch cap, NOT by early stopping —
    val was still improving near the end, so the cell is potentially
    undertrained and may deserve a re-run with a larger budget.
    """
    if "best_epoch" not in df.columns:
        return
    cap_threshold = num_epochs - patience  # 0-indexed best_epoch
    flagged = df[df["best_epoch"] >= cap_threshold]

    print("\n" + "-" * 90)
    print(f"  Epoch-budget audit (cap-hit = best_epoch >= {cap_threshold} "
          f"with budget {num_epochs}, patience {patience})")
    print("-" * 90)
    if flagged.empty:
        print("  No cap-hit runs — the epoch budget was sufficient everywhere.")
        return

    stats = (df.assign(cap_hit=df["best_epoch"] >= cap_threshold)
               .groupby(["exp_tag", "model"])
               .agg(runs=("best_epoch", "size"), cap_hits=("cap_hit", "sum"),
                    max_best_epoch=("best_epoch", "max"))
               .reset_index())
    stats = stats[stats.cap_hits > 0].sort_values(
        "cap_hits", ascending=False)
    for _, r in stats.iterrows():
        share = 100.0 * r.cap_hits / r.runs
        marker = " <-- CHECK" if share > 20 else ""
        print(f"  [{r.exp_tag:>12}] {r.model:>15} | "
              f"cap-hit {int(r.cap_hits)}/{int(r.runs)} runs ({share:.0f}%) | "
              f"max best_epoch {int(r.max_best_epoch)}{marker}")
    print("  Models marked CHECK were still improving near the cap on >20% "
          "of runs;\n  consider re-running those cells with --num_epochs 100.")


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


def main():
    p = argparse.ArgumentParser(description="Forecast progress audit")
    p.add_argument("--csv",     type=str, default="results/raw_mfg.csv", help="results CSV")
    p.add_argument("--exp_tag", type=str, default=None)
    p.add_argument("--model",   type=str, default=None)
    p.add_argument("--data",    type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=50,
                   help="Epoch budget used in the sweep (for the cap-hit audit).")
    p.add_argument("--patience",   type=int, default=10,
                   help="Early-stopping patience used in the sweep.")
    args = p.parse_args()

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

    report_forecast(df, args.csv)
    report_epoch_budget(df, args.num_epochs, args.patience)


if __name__ == "__main__":
    main()
