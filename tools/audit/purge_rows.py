# -*- coding: utf-8 -*-
"""
Surgical row removal from a results CSV, so the orchestrators' resume logic
re-runs exactly the invalidated cells.

Written for the 2026-07-14 audit fixes (see
manuscript/notes/code_audit_2026-07-14.md):

  * Weather split fix (P1-2) — every Weather row in every CSV was produced
    under the legacy hard-coded split and is invalid:
        python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
        python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
  * Spectral-separation loss inversion fix (P1-1) — FEATHer-only training
    runs (baselines never use the spec loss). Frozen OFAT rows in
    hp_search.csv are deliberately KEPT (lambda axis was flat); purge the
    main/lr streams:
        python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
        python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply

Dry-run by default — prints what WOULD be deleted; add --apply to write.
A timestamped .bak copy of the CSV is made before any modification.

Generic: works on any results CSV (fcst / lr / hp / robust) because it only
filters on the columns you pass. Filters combine with AND across flags and
OR within a comma-separated list, e.g.
    --model FEATHer --data Weather      -> FEATHer rows on Weather only
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
    print("The orchestrators' resume (--check) will now show these cells as "
          "missing and refill them on the next run.")


if __name__ == "__main__":
    main()
