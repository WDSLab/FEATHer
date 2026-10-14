# -*- coding: utf-8 -*-
"""
Surgical row removal from a results CSV, so that run.py re-runs exactly the
removed cells (it only runs cells missing from the results).

    python tools/audit/purge_rows.py --csv results/raw_ltsf.csv --data Weather --apply

Dry-run by default — prints what WOULD be deleted; add --apply to write.
A timestamped .bak copy of the CSV is made before any modification.

Works on any results CSV because it only filters on the columns you pass.
Filters combine with AND across flags and OR within a comma-separated list, e.g.
    --model FEATHer_CI --data Weather   -> FEATHer_CI rows on Weather only
    --data Weather,ETTh1                -> all models on Weather or ETTh1
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import pandas as pd


FILTER_COLUMNS = {
    # flag name -> (csv column, cast)
    "model":    ("model",    str),
    "data":     ("data",     str),
    "exp_tag":  ("exp_tag",  str),
    "train_exp_tag": ("train_exp_tag", str),
    "pred_len": ("pred_len", int),
    "seed":     ("seed",     int),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Delete matching rows from a results CSV (dry-run by default)")
    p.add_argument("--csv", required=True, help="Path to the results CSV.")
    p.add_argument("--model",    type=str, default=None,
                   help="Comma-separated model name(s) to purge.")
    p.add_argument("--data",     type=str, default=None,
                   help="Comma-separated dataset name(s) to purge.")
    p.add_argument("--exp_tag",  type=str, default=None,
                   help="Comma-separated exp_tag(s) to purge.")
    p.add_argument("--train_exp_tag", type=str, default=None,
                   help="Comma-separated train_exp_tag(s) to purge (robust CSV).")
    p.add_argument("--pred_len", type=str, default=None,
                   help="Comma-separated pred_len value(s) to purge.")
    p.add_argument("--seed",     type=str, default=None,
                   help="Comma-separated seed value(s) to purge.")
    p.add_argument("--apply", action="store_true",
                   help="Actually rewrite the CSV (default: dry-run report only).")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[ERROR] CSV not found: {args.csv}")

    active = {}
    for flag, (col, cast) in FILTER_COLUMNS.items():
        raw = getattr(args, flag)
        if raw is None:
            continue
        values = [v.strip() for v in str(raw).split(",") if v.strip()]
        if not values:
            continue
        active[col] = {cast(v) for v in values}

    if not active:
        sys.exit("[ERROR] Refusing to run without at least one filter "
                 "(--model/--data/--exp_tag/--pred_len/--seed) — that would "
                 "delete the entire CSV.")

    df = pd.read_csv(args.csv)
    n_before = len(df)

    mask = pd.Series(True, index=df.index)
    for col, values in active.items():
        if col not in df.columns:
            sys.exit(f"[ERROR] Column '{col}' not present in {args.csv} "
                     f"(columns: {list(df.columns)})")
        if all(isinstance(v, int) for v in values):
            colvals = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            mask &= colvals.isin(values)
        else:
            mask &= df[col].astype(str).isin({str(v) for v in values})

    to_drop = df[mask]
    kept = df[~mask]

    print(f"CSV: {args.csv}")
    print(f"Filters: " + "  ".join(f"{c} in {sorted(map(str, v))}"
                                   for c, v in active.items()))
    print(f"Rows: {n_before} total | {len(to_drop)} match (would be DELETED) "
          f"| {len(kept)} kept")

    if to_drop.empty:
        print("Nothing to purge.")
        return

    # Breakdown so the operator can sanity-check scope before --apply.
    group_cols = [c for c in ("exp_tag", "model", "data", "pred_len")
                  if c in df.columns]
    if group_cols:
        print("\nBreakdown of rows to delete:")
        summary = to_drop.groupby(group_cols).size().reset_index(name="rows")
        for r in summary.itertuples(index=False):
            desc = " | ".join(f"{c}={getattr(r, c)}" for c in group_cols)
            print(f"  {desc}  ->  {r.rows} rows")

    if not args.apply:
        print("\n[DRY-RUN] No changes written. Re-run with --apply to delete.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.csv}.bak_{stamp}"
    shutil.copy2(args.csv, backup)
    kept.to_csv(args.csv, index=False)
    print(f"\n[APPLIED] {len(to_drop)} rows deleted "
          f"({n_before} -> {len(kept)}).")
    print(f"Backup: {backup}")
    print("run.py --check will now show these cells as "
          "missing and refill them on the next run.")


if __name__ == "__main__":
    main()
