# -*- coding: utf-8 -*-
"""
Main forecast result table (12 models x 8 datasets x 4 horizons).

Reads results/fcst_results.csv, aggregates the 5 seeds per cell into
mean +/- std, and emits one of:

  --format md     GitHub-flavored markdown table
  --format latex  LaTeX booktabs table (one row per (data, horizon))
  --format csv    Flat (data, pred_len, model, MSE_mean, MSE_std, ...) CSV

Winner per (data, horizon) row is bolded (md/latex) or flagged with
'<-' in the CSV's `is_best` column.

Directly addresses reviewer comments:
  R1 #5, R2 #11, R8 #3   — "mean +/- std, not just means"
  R8 #1                  — "broadly SOTA overstated" (table makes wins visible)

Usage:
    python tools/paper/main_table.py
    python tools/paper/main_table.py --exp_tag main --metric MSE --format latex
    python tools/paper/main_table.py --metric MAE --format md --output table.md
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Canonical column order — matches paper convention (ultra-light first).
MODEL_ORDER = [
    "SparseTSF", "DiPE_Linear", "FEATHer", "FITS",
    "DLinear", "PatchTST", "TimeMixer", "TQNet",
    "LMS_AutoTSF", "TimesNet", "iTransformer", "MDMLP_EIA",
]

DATASET_ORDER = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Exchange", "Electricity", "Traffic",
]

LOWER_IS_BETTER = {"MSE", "MAE", "RMSE"}


def aggregate(df, metric):
    """(data, pred_len, model) -> (mean, std, n_seeds)."""
    g = df.groupby(["data", "pred_len", "model"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["data", "pred_len", "model", "mean", "std", "n"]
    return out


def pick_winners(agg, metric):
    """Per (data, pred_len), mark the best-mean model."""
    if metric in LOWER_IS_BETTER:
        idx = agg.groupby(["data", "pred_len"])["mean"].idxmin()
    else:
        idx = agg.groupby(["data", "pred_len"])["mean"].idxmax()
    agg = agg.copy()
    agg["is_best"] = False
    agg.loc[idx, "is_best"] = True
    return agg


def format_cell(mean, std, n, is_best, decimals=3):
    if pd.isna(mean):
        return "-"
    if pd.isna(std) or n < 2:
        s = f"{mean:.{decimals}f}"
    else:
        s = f"{mean:.{decimals}f}±{std:.{decimals}f}"
    return s


def render_markdown(agg, metric):
    lines = []
    models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    models += sorted(set(agg["model"].unique()) - set(MODEL_ORDER))
    datasets = [d for d in DATASET_ORDER if d in agg["data"].unique()]
    datasets += sorted(set(agg["data"].unique()) - set(DATASET_ORDER))

    header = ["Data", "H"] + models
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for d in datasets:
        sub = agg[agg["data"] == d]
        for h in sorted(sub["pred_len"].unique()):
            row = [d, str(int(h))]
            for m in models:
                cell = sub[(sub["pred_len"] == h) & (sub["model"] == m)]
                if cell.empty:
                    row.append("-")
                else:
                    r = cell.iloc[0]
                    s = format_cell(r["mean"], r["std"], r["n"], r["is_best"])
                    if r["is_best"]:
                        s = f"**{s}**"
                    row.append(s)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_latex(agg, metric):
    models = [m for m in MODEL_ORDER if m in agg["model"].unique()]
    models += sorted(set(agg["model"].unique()) - set(MODEL_ORDER))
    datasets = [d for d in DATASET_ORDER if d in agg["data"].unique()]
    datasets += sorted(set(agg["data"].unique()) - set(DATASET_ORDER))

    lines = []
    lines.append(r"\begin{tabular}{ll" + "c" * len(models) + "}")
    lines.append(r"\toprule")
    lines.append("Data & H & " + " & ".join(m.replace("_", r"\_") for m in models) + r" \\")
    lines.append(r"\midrule")
    for d in datasets:
        sub = agg[agg["data"] == d]
        for h in sorted(sub["pred_len"].unique()):
            cells = []
            for m in models:
                row_match = sub[(sub["pred_len"] == h) & (sub["model"] == m)]
                if row_match.empty:
                    cells.append("-")
                else:
                    r = row_match.iloc[0]
                    s = format_cell(r["mean"], r["std"], r["n"], r["is_best"])
                    s = s.replace("±", r"$\pm$")
                    if r["is_best"]:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
            lines.append(f"{d} & {int(h)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_csv(agg, metric):
    return agg.to_csv(index=False)


def main():
    p = argparse.ArgumentParser(description="Main forecast result table generator")
    p.add_argument("--csv", default="results/fcst_results.csv")
    p.add_argument("--exp_tag", default="main",
                   help="Filter rows by exp_tag (default: 'main').")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2"])
    p.add_argument("--format", default="md", choices=["md", "latex", "csv"])
    p.add_argument("--output", default=None,
                   help="Write to file; default stdout.")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[empty] {args.csv} does not exist yet", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if df.empty:
        print(f"[empty] {args.csv}", file=sys.stderr)
        sys.exit(1)

    if args.exp_tag:
        df = df[df["exp_tag"] == args.exp_tag]
    if df.empty:
        print(f"No rows for exp_tag={args.exp_tag}", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(df, args.metric)
    agg = pick_winners(agg, args.metric)

    renderers = {"md": render_markdown, "latex": render_latex, "csv": render_csv}
    out = renderers[args.format](agg, args.metric)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.output} ({args.format}, metric={args.metric})", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
