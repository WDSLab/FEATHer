# -*- coding: utf-8 -*-
"""
Duplicate / overlap audit for the append-only results CSVs.

`results/fcst_results.csv` is append-only: the orchestrator resumes by
reading the CSV into a set of resume keys and skipping completed cells, but
it never rewrites the file. So if the SAME cell was actually trained twice
(e.g. two overlapping launches before either finished writing, a stray run
under a different `--exp_tag`, or a wrong `--group` that steered FEATHer at
the manufacturing datasets), the CSV can physically hold duplicate rows.

Resume is unaffected (it dedups on read), but `tools/paper/main_table.py`
averages over seed rows per (model, data, pred_len) - duplicate seed rows
would be double-counted and bias the mean/std. This tool surfaces those
duplicates before they reach a table, and `--fix` rewrites a deduped CSV
(keeping the LATEST timestamp per key) after backing up the original.

Resume key (matches run_forecast.py load_done_set):
    exp_tag, model, data, pred_len, seq_len, seed

Usage:
    python tools/audit/check_duplicates.py
    python tools/audit/check_duplicates.py --csv results/fcst_results.csv
    python tools/audit/check_duplicates.py --exp_tag main
    python tools/audit/check_duplicates.py --fix          # dedup in place (backs up first)
"""

import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RESUME_KEY = ["exp_tag", "model", "data", "pred_len", "seq_len", "seed"]

# FEATHer belongs on the LTSF-8 (generalization) datasets under --group ltsf.
# If it shows up on a manufacturing / PdM dataset it was almost certainly
# launched without --group ltsf (default group is mfg) - flag it.
LTSF_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2",
                 "Weather", "Electricity", "Traffic", "Exchange"}


def load(csv_path):
    if not os.path.exists(csv_path):
        print(f"[missing] {csv_path} does not exist")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[empty] {csv_path}")
    return df


def report_distributions(df):
    print("\n" + "-" * 90)
    print("  Row distribution (sanity check - spot wrong exp_tag / stray models)")
    print("-" * 90)
    print("  [exp_tag]")
    for tag, n in df["exp_tag"].value_counts().items():
        print(f"    {tag:>16} | {n} rows")
    print("  [model]")
    for m, n in df["model"].value_counts().items():
        print(f"    {m:>16} | {n} rows")


def report_group_anomalies(df):
    """FEATHer rows that landed on a non-LTSF dataset = likely missing
    --group ltsf (default group is mfg)."""
    feat = df[df["model"] == "FEATHer"]
    if feat.empty:
        return
    stray = feat[~feat["data"].isin(LTSF_DATASETS)]
    if stray.empty:
        return
    print("\n" + "-" * 90)
    print("  [!] FEATHer rows on NON-LTSF datasets (missing --group ltsf?)")
    print("-" * 90)
    for data, n in stray["data"].value_counts().items():
        print(f"    FEATHer x {data:>12} | {n} rows  <-- check the launch command")


def report_duplicates(df, csv_path):
    key_present = [c for c in RESUME_KEY if c in df.columns]
    missing_cols = set(RESUME_KEY) - set(key_present)
    if missing_cols:
        print(f"[warn] CSV missing resume-key columns: {sorted(missing_cols)}; "
              f"using {key_present}")

    n_total = len(df)
    n_unique = df[key_present].drop_duplicates().shape[0]
    dup_mask = df.duplicated(key_present, keep=False)
    dups = df[dup_mask].sort_values(key_present + (["timestamp"]
                                    if "timestamp" in df.columns else []))

    print("\n" + "=" * 90)
    print(f"  Duplicate audit - {csv_path}")
    print("=" * 90)
    print(f"  total rows     : {n_total}")
    print(f"  unique keys    : {n_unique}")
    print(f"  duplicate rows : {n_total - n_unique} extra "
          f"({dup_mask.sum()} rows across {df[dup_mask][key_present].drop_duplicates().shape[0]} keys)")

    if dups.empty:
        print("\n  OK - no duplicate resume keys.")
        return dups, key_present

    print("\n  [duplicate keys - count per key, worst first]")
    grp = (dups.groupby(key_present).size()
               .reset_index(name="copies")
               .sort_values("copies", ascending=False))
    for _, r in grp.head(40).iterrows():
        print(f"    x{int(r.copies)}  [{r.exp_tag}] {r.model} | {r.data} | "
              f"H={r.pred_len} L={r.seq_len} | seed={r.seed}")
    if len(grp) > 40:
        print(f"    ... ({len(grp) - 40} more duplicated keys)")

    # Show whether the duplicates AGREE (same MSE) or DIVERGE (different runs).
    if "MSE" in dups.columns:
        spread = (dups.groupby(key_present)["MSE"]
                      .agg(lambda s: float(s.max() - s.min())))
        n_diverge = int((spread > 1e-9).sum())
        print(f"\n  of {len(grp)} duplicated keys, {n_diverge} have DIFFERENT "
              f"MSE across copies (genuine re-runs), "
              f"{len(grp) - n_diverge} are identical re-appends.")
    return dups, key_present


def do_fix(df, key_present, csv_path):
    """Keep the latest row per resume key (by timestamp if present, else the
    last-appended row) and rewrite the CSV, backing up the original first."""
    if "timestamp" in df.columns:
        ordered = df.sort_values("timestamp")
    else:
        ordered = df  # preserve append order; last wins
    deduped = ordered.drop_duplicates(key_present, keep="last")

    n_removed = len(df) - len(deduped)
    if n_removed == 0:
        print("\n  Nothing to fix - no duplicates.")
        return

    backup = csv_path + ".bak"
    shutil.copy2(csv_path, backup)
    # Preserve original column order.
    deduped = deduped[df.columns]
    deduped.to_csv(csv_path, index=False)
    print(f"\n  FIXED: removed {n_removed} duplicate rows "
          f"({len(df)} -> {len(deduped)}).")
    print(f"  backup written to {backup}")


def main():
    p = argparse.ArgumentParser(description="Duplicate/overlap audit for results CSVs")
    p.add_argument("--csv",     type=str, default="results/fcst_results.csv")
    p.add_argument("--exp_tag", type=str, default=None,
                   help="Restrict the audit to one exp_tag.")
    p.add_argument("--model",   type=str, default=None)
    p.add_argument("--fix",     action="store_true",
                   help="Rewrite the CSV keeping the latest row per resume key "
                        "(backs up to <csv>.bak first).")
    args = p.parse_args()

    df = load(args.csv)
    if df.empty:
        return

    if args.exp_tag:
        df = df[df.exp_tag == args.exp_tag]
    if args.model:
        df = df[df.model == args.model]
    if df.empty:
        print("No rows match the filter.")
        return

    report_distributions(df)
    report_group_anomalies(df)
    dups, key_present = report_duplicates(df, args.csv)

    if args.fix:
        if args.exp_tag or args.model:
            print("\n  [!] --fix refuses to run with --exp_tag/--model filters "
                  "(it would drop the filtered-out rows). Re-run --fix without "
                  "filters.")
            return
        do_fix(df, key_present, args.csv)


if __name__ == "__main__":
    main()
