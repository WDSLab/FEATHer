# -*- coding: utf-8 -*-
"""
FEATHer ablation aggregation across 5 axes x 4 horizons x 5 seeds.

Reads rows from results/fcst_results.csv where `model` starts with
`FEATHer_<axis>_` (i.e. one of the 30 ablation variants registered by
baselines/__init__.py:_ABLATION_BUILDERS) and emits one summary table
per axis:

  ms          15 variants (P, H, M, L, PH, ..., PHML)
  gate         4 variants (none, uniform, softmax, fft)
  dtk          4 variants (none, mlp, shallow, full)
  head         4 variants (linear, mlp, conv, spk)
  complexity   3 variants (half, full, double)

Each table shows MSE/MAE mean +/- std across 5 seeds for each of the 4
horizons, with the best variant per (axis, horizon) bolded.

Directly addresses:
  R8 #6  — "ablations only on H=96 or H=720, not all horizons"
  R5 #5  — asymmetric design (pooling for low-freq vs conv for high-freq):
           bolded multiscale rows make the comparison explicit.

Usage:
    python tools/paper/ablation_table.py
    python tools/paper/ablation_table.py --data ETTh1
    python tools/paper/ablation_table.py --axis ms --format latex
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

AXIS_ORDER = ["ms", "gate", "dtk", "head", "complexity"]

AXIS_VARIANTS = {
    "ms": ["P", "H", "M", "L",
           "PH", "PM", "PL", "HM", "HL", "ML",
           "PHM", "PHL", "PML", "HML",
           "PHML"],
    "gate": ["none", "uniform", "softmax", "fft"],
    "dtk":  ["none", "mlp", "shallow", "full"],
    "head": ["linear", "mlp", "conv", "spk"],
    "complexity": ["half", "full", "double"],
}

LOWER_IS_BETTER = {"MSE", "MAE", "RMSE"}

_ABLATION_RE = re.compile(r"^FEATHer_([a-z]+)_(.+)$")


def split_ablation_name(name):
    m = _ABLATION_RE.match(name)
    return m.groups() if m else (None, None)


def filter_ablation(df):
    df = df.copy()
    parts = df["model"].apply(split_ablation_name)
    df["axis"] = parts.apply(lambda t: t[0])
    df["variant"] = parts.apply(lambda t: t[1])
    return df[df["axis"].isin(AXIS_ORDER)]


def format_cell(mean, std, n, is_best, decimals=3):
    if pd.isna(mean):
        return "-"
    if pd.isna(std) or n < 2:
        s = f"{mean:.{decimals}f}"
    else:
        s = f"{mean:.{decimals}f}±{std:.{decimals}f}"
    return s


def per_axis_table(df, axis, metric, fmt):
    sub = df[df["axis"] == axis]
    if sub.empty:
        return f"(no rows for axis={axis})"

    variants = [v for v in AXIS_VARIANTS[axis] if v in sub["variant"].unique()]
    horizons = sorted(sub["pred_len"].unique())

    agg = (sub.groupby(["variant", "pred_len"])[metric]
              .agg(["mean", "std", "count"])
              .reset_index())
    agg.columns = ["variant", "pred_len", "mean", "std", "n"]

    if metric in LOWER_IS_BETTER:
        winners = agg.loc[agg.groupby("pred_len")["mean"].idxmin()]
    else:
        winners = agg.loc[agg.groupby("pred_len")["mean"].idxmax()]
    winner_set = set(zip(winners["variant"], winners["pred_len"]))

    if fmt == "md":
        head = ["Variant"] + [f"H={int(h)}" for h in horizons]
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] * len(head)) + "|"]
        for v in variants:
            row = [v]
            for h in horizons:
                c = agg[(agg["variant"] == v) & (agg["pred_len"] == h)]
                if c.empty:
                    row.append("-")
                else:
                    r = c.iloc[0]
                    s = format_cell(r["mean"], r["std"], r["n"], (v, h) in winner_set)
                    if (v, h) in winner_set:
                        s = f"**{s}**"
                    row.append(s)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    if fmt == "latex":
        lines = [r"\begin{tabular}{l" + "c" * len(horizons) + "}",
                 r"\toprule",
                 "Variant & " + " & ".join(f"H={int(h)}" for h in horizons) + r" \\",
                 r"\midrule"]
        for v in variants:
            cells = []
            for h in horizons:
                c = agg[(agg["variant"] == v) & (agg["pred_len"] == h)]
                if c.empty:
                    cells.append("-")
                else:
                    r = c.iloc[0]
                    s = format_cell(r["mean"], r["std"], r["n"], (v, h) in winner_set)
                    s = s.replace("±", r"$\pm$")
                    if (v, h) in winner_set:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
            lines.append(f"{v} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # csv
    out = agg.copy()
    out["is_best"] = list(zip(out["variant"], out["pred_len"]))
    out["is_best"] = out["is_best"].apply(lambda k: k in winner_set)
    return out.to_csv(index=False)


def main():
    p = argparse.ArgumentParser(description="FEATHer ablation aggregation table")
    p.add_argument("--csv", default="results/fcst_results.csv")
    p.add_argument("--exp_tag", default=None,
                   help="Filter rows by exp_tag prefix (e.g. 'ablation_'). "
                        "Default: all rows whose model matches the ablation pattern.")
    p.add_argument("--data", default=None, help="Single dataset (default: all).")
    p.add_argument("--axis", default=None,
                   choices=AXIS_ORDER, help="Single axis (default: all 5).")
    p.add_argument("--metric", default="MSE",
                   choices=["MSE", "MAE", "RMSE", "CORR", "R2"])
    p.add_argument("--format", default="md", choices=["md", "latex", "csv"])
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[empty] {args.csv} does not exist yet", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.csv)
    if args.exp_tag:
        df = df[df["exp_tag"].str.startswith(args.exp_tag)]
    if args.data:
        df = df[df["data"] == args.data]
    df = filter_ablation(df)
    if df.empty:
        print("No ablation rows match the filter.", file=sys.stderr)
        sys.exit(1)

    axes = [args.axis] if args.axis else AXIS_ORDER
    for ax in axes:
        print(f"\n## Axis: {ax}  (metric={args.metric}"
              + (f", data={args.data}" if args.data else ", all datasets")
              + ")\n")
        print(per_axis_table(df, ax, args.metric, args.format))


if __name__ == "__main__":
    main()
