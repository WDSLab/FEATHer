"""Post-hoc R2 from stored MSE — no re-training required.

Rows written before 2026-10-15 hold a per-window R2 that normalised each
(window, channel) series by its WITHIN-WINDOW variance along the time axis.
On near-constant channels that denominator collapses and the score explodes
to ~-1e12, which makes those stored values unusable.  Workers now write the
global R2 below (utils.metrics.TestScores); this script recomputes it for
any CSV, old or new, from the stored MSE.

The standard forecasting definition instead compares the error against the
variance of the target over the whole evaluation set:

    R2 = 1 - MSE / Var(y_test)

Both terms live in the same normalised space the workers score in (--inverse
is off everywhere), and MSE is already stored per run, so the only missing
piece is Var(y_test) per (dataset, pred_len).  This script computes that once
per cell by streaming the test loader, then joins it onto the results CSV.

Variance is accumulated over exactly the cells the metrics use: finite
(observed) targets only, pooled across windows / horizon steps / channels —
matching the worker's global nansum/finite-count estimator.

Usage
-----
    python tools/paper/compute_r2.py --csv results/_archive/pre_raw_20261007/mfg_raw.csv --out results/analysis/mfg_r2.csv
    python tools/paper/compute_r2.py --csv results/_archive/pre_raw_20261007/ltsf_raw.csv --variance_only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils import data_factory  # noqa: E402


def target_variance(data_name: str, seq_len: int, pred_len: int, root_path: str,
                    batch_size: int = 32, features: str = "M") -> tuple[float, int]:
    """Variance of the test targets over observed cells, pooled globally."""
    _, loader = data_factory.data_provider(
        root_path, data_name, features, batch_size,
        seq_len, seq_len // 2, pred_len, "test",
    )
    total = 0.0
    total_sq = 0.0
    count = 0
    for _, batch_y, _, _ in loader:
        y = batch_y.numpy()[:, -pred_len:, :]
        y = y[np.isfinite(y)]
        total += float(y.sum())
        total_sq += float((y.astype(np.float64) ** 2).sum())
        count += y.size
    if count == 0:
        return float("nan"), 0
    mean = total / count
    return total_sq / count - mean * mean, count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="results CSV holding the MSE column")
    ap.add_argument("--out", default=None, help="where to write the augmented CSV")
    ap.add_argument("--root_path", default="data/",
                    help="dataset root; needs the trailing separator the loader concatenates on")
    ap.add_argument("--data", default=None, help="comma-separated subset")
    ap.add_argument("--variance_only", action="store_true",
                    help="print the per-cell variances and stop")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.data:
        df = df[df.data.isin(args.data.split(","))]

    cells = df[["data", "seq_len", "pred_len"]].drop_duplicates().sort_values(
        ["data", "pred_len"]).itertuples(index=False)

    variances = {}
    for data_name, seq_len, pred_len in cells:
        var, n = target_variance(data_name, int(seq_len), int(pred_len), args.root_path)
        variances[(data_name, int(seq_len), int(pred_len))] = var
        print(f"{data_name:14s} L={seq_len} H={pred_len:4d}  "
              f"Var={var:.6f}  cells={n:,}", flush=True)

    if args.variance_only:
        return 0

    key = list(zip(df.data, df.seq_len, df.pred_len))
    df["target_var"] = [variances.get(k, np.nan) for k in key]
    df["R2_global"] = 1.0 - df.MSE / df.target_var

    out = args.out or args.csv.replace(".csv", "_r2.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")

    piv = df.groupby(["model", "data"]).R2_global.mean().unstack()
    piv["OVERALL"] = piv.mean(axis=1)
    print("\nR2_global by model (mean over horizons/seeds):")
    print(piv.round(4).sort_values("OVERALL", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
