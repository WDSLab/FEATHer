# -*- coding: utf-8 -*-
"""Accuracy-vs-parameters Pareto plot (edge tier + ceiling reference).

The paper's core figure: at the sub-1K / edge budget, where does each model sit
on the accuracy-per-parameter frontier? Edge-tier models are the main
comparison; ceiling models appear ONLY as background points (the "unconstrained
server-tier" — what's achievable if you drop the deployment constraint), which
is exactly what lets us claim "X% of the accuracy at 1/N the parameters".

Reads results/fcst_results.csv. Per model: mean metric over the chosen
datasets/horizons/seeds, and median num_params. Draws a scatter (log-x params),
colours edge vs ceiling, highlights FEATHer, and traces the lower-left Pareto
frontier. Saves a PNG (+ prints the table).

Usage:
    python tools/paper/pareto_plot.py --data Steel,WindSCADA,EWELD
    python tools/paper/pareto_plot.py --metric MSE --pred_len 96
    python tools/paper/pareto_plot.py --csv results/fcst_results.csv --output_dir tools/paper/figures
"""

import argparse
import os

import numpy as np
import pandas as pd

EDGE = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FEATHer_mix",
        "FEATHer_CI", "FITS", "CycleNet", "DLinear"]
CEILING = ["PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet",
           "iTransformer", "MDMLP_EIA"]
MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]


def frontier(points):
    """Lower-left Pareto frontier: sort by params asc, keep a point only if its
    metric is strictly better (lower) than every cheaper kept point."""
    pts = sorted(points, key=lambda p: p[0])          # by params
    keep, best = [], float("inf")
    for params, metric, name in pts:
        if metric < best:
            keep.append((params, metric, name))
            best = metric
    return keep


def main():
    p = argparse.ArgumentParser(description="Accuracy-vs-params Pareto plot")
    p.add_argument("--csv", default="results/mfg_raw.csv")
    p.add_argument("--exp_tag", default="main")
    p.add_argument("--metric", default="MSE", choices=["MSE", "MAE", "RMSE"])
    p.add_argument("--data", default=",".join(MFG),
                   help="comma list of datasets to average over")
    p.add_argument("--pred_len", default="", help="comma list of horizons (default all)")
    p.add_argument("--output_dir", default="tools/paper/figures")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[missing] {args.csv}")
        return
    df = pd.read_csv(args.csv)
    df = df[df.exp_tag == args.exp_tag]
    df = df[df.data.isin(args.data.split(","))]
    if args.pred_len:
        df = df[df.pred_len.isin([int(x) for x in args.pred_len.split(",")])]
    if df.empty:
        print("No rows match the filter.")
        return

    agg = (df.groupby("model")
             .agg(metric=(args.metric, "mean"), params=("num_params", "median"),
                  n=("seed", "size"))
             .reset_index())
    agg = agg.sort_values("params")

    print(f"=== {args.metric} vs params ({args.data}) ===")
    print(agg.round(4).to_string(index=False))

    pts = [(r.params, r.metric, r.model) for r in agg.itertuples()]
    fr = frontier(pts)
    fr_names = {n for _, _, n in fr}
    print("\nPareto frontier (lowest metric per param budget):",
          " -> ".join(n for _, _, n in fr))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — table only", flush=True)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in agg.itertuples():
        is_edge = r.model in EDGE
        colour = "#1f77b4" if is_edge else "#bbbbbb"
        ours = r.model.startswith("FEATHer")
        marker = "*" if ours else ("o" if is_edge else "s")
        size = 320 if ours else (90 if is_edge else 60)
        ax.scatter(r.params, r.metric, c=colour, marker=marker, s=size,
                   edgecolors="k" if r.model in fr_names else "none",
                   linewidths=1.2, zorder=3)
        ax.annotate(r.model, (r.params, r.metric), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    # frontier line
    fx = [pp for pp, _, _ in fr]
    fy = [mm for _, mm, _ in fr]
    ax.plot(fx, fy, "--", color="#1f77b4", lw=1.0, zorder=2, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel(f"{args.metric} (mean over {args.data})")
    ax.set_title("Accuracy vs parameters — edge tier (blue) vs ceiling (grey)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"pareto_{args.metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
