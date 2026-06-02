# -*- coding: utf-8 -*-
"""
Robustness sweep summary — heatmaps and FEATHer-vs-best D_w tables.

Reads results/robust_results.csv (schema: exp_tag, train_exp_tag, model,
data, pred_len, seq_len, seed, fault_type, severity, MSE, MAE, RMSE,
CORR, R2, MSE_clean, Dw, timestamp). Produces:

  (1) Per-(fault_type, model) MSE x severity heatmap (one PNG per data,
      horizon) — saved under tools/paper/figures/.
  (2) D_w summary table: relative degradation (MSE_corrupt - MSE_clean)
      / MSE_clean, aggregated across (data, pred_len, seed). FEATHer vs
      best baseline per fault axis.

Directly addresses:
  R6 #7a  — "no noise robustness experiments"  (IoT-J selling point)
  R8 #7   — partial; energy/RAM/bit-width still need on-device measurement

Usage:
    python tools/paper/robust_summary.py
    python tools/paper/robust_summary.py --metric MSE --no_heatmap
    python tools/paper/robust_summary.py --output_dir tools/paper/figures
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

LOWER_IS_BETTER = {"MSE", "MAE", "RMSE"}


def dw_table(df, metric):
    """For each (model, fault_type), aggregate per-row D_w =
    (metric_corrupt - metric_clean) / metric_clean across all cells
    and severities. Lower D_w = more robust."""
    work = df[df["fault_type"] != "clean"].copy()
    if metric in work.columns and "MSE_clean" in work.columns and metric == "MSE":
        # Worker already populates Dw using MSE.
        work["dw_row"] = work["Dw"]
    else:
        # Recompute Dw for non-MSE metrics by joining against the clean row.
        clean = (df[df["fault_type"] == "clean"]
                 .groupby(["model", "data", "pred_len", "seed"])[metric]
                 .first()
                 .rename("clean_val")
                 .reset_index())
        work = work.merge(clean, on=["model", "data", "pred_len", "seed"], how="left")
        if metric in LOWER_IS_BETTER:
            work["dw_row"] = (work[metric] - work["clean_val"]) / work["clean_val"]
        else:
            work["dw_row"] = (work["clean_val"] - work[metric]) / work["clean_val"]

    agg = (work.groupby(["model", "fault_type"])["dw_row"]
                 .agg(["mean", "std", "count"])
                 .reset_index())
    agg.columns = ["model", "fault_type", "Dw_mean", "Dw_std", "n"]
    return agg


def best_vs_feather(dw):
    """For each fault_type, pick the baseline with lowest Dw_mean and
    show how FEATHer compares (lower Dw = more robust)."""
    rows = []
    for ft, sub in dw.groupby("fault_type"):
        sub = sub.sort_values("Dw_mean")
        best_row = sub.iloc[0]
        feather = sub[sub["model"] == "FEATHer"]
        if feather.empty:
            continue
        f = feather.iloc[0]
        rows.append({
            "fault_type": ft,
            "best_model": best_row["model"],
            "best_Dw": best_row["Dw_mean"],
            "FEATHer_Dw": f["Dw_mean"],
            "FEATHer_rank": (sub["Dw_mean"].rank().loc[f.name]
                             if f.name in sub.index else np.nan),
            "n_models": len(sub),
        })
    return pd.DataFrame(rows).sort_values("fault_type")


def heatmap(df, metric, data, pred_len, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — install for heatmaps", file=sys.stderr)
        return

    sub = df[(df["data"] == data) & (df["pred_len"] == pred_len) & (df["fault_type"] != "clean")]
    if sub.empty:
        return

    fault_types = sorted(sub["fault_type"].unique())
    fig, axes = plt.subplots(1, len(fault_types), figsize=(4 * len(fault_types), 6),
                              squeeze=False)
    for ax, ft in zip(axes[0], fault_types):
        sub_ft = sub[sub["fault_type"] == ft]
        pivot = sub_ft.groupby(["model", "severity"])[metric].mean().unstack(fill_value=np.nan)
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{ft} ({metric})")
        ax.set_xlabel("severity")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{data}  H={int(pred_len)}", y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"robust_{data}_H{int(pred_len)}_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Robustness summary tables + heatmaps")
    p.add_argument("--csv", default="results/robust_results.csv")
    p.add_argument("--exp_tag", default="robust")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2"])
    p.add_argument("--no_heatmap", action="store_true",
                   help="Skip PNG heatmap generation.")
    p.add_argument("--output_dir", default="tools/paper/figures",
                   help="Where to save heatmaps.")
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

    print(f"\n=== D_w per (model, fault_type), metric={args.metric} ===")
    dw = dw_table(df, args.metric)
    print(dw.round(4).to_string(index=False))

    print(f"\n=== FEATHer vs best baseline per fault axis ===")
    summary = best_vs_feather(dw)
    print(summary.round(4).to_string(index=False))

    if not args.no_heatmap:
        for (data, h), _ in df.groupby(["data", "pred_len"]):
            heatmap(df, args.metric, data, h, args.output_dir)


if __name__ == "__main__":
    main()
