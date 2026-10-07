"""Component-ablation verdict table.

Pairs the `abl_<arm>` rows written by `run_forecast.py --ablation_arm` against
the shipping FEATHer row in the main sweep, which is the baseline and is never
re-run. Comparison is per dataset on the datasets an arm actually covers, so a
partially-finished sweep still reports honestly.

A 24-cell mean carries +-0.3~0.5% seed noise (2026-08-16 calibration), so only
a >=2% change is called signal; those arms are the ones worth promoting to 3
seeds.

    python tools/paper/ablation_summary.py
    python tools/paper/ablation_summary.py --per_dataset
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")))
from protocol import ABLATION_ARMS as _ARMS  # noqa: E402  (10-07: arm table moved out of the old orchestrator)

RAW = "results/raw_mfg.csv"   # ablation rows: role "ablation"; baseline: FEATHer-mix main rows (config "frozen")
_TIER = {"norevin": "standard", "noresid": "standard", "r2": "variant", "r8": "variant",
         "readd_dtk": "readd", "readd_inproj": "readd"}
ABLATION_ARMS = {a: (_TIER.get(a, "design"), desc, None) for a, (desc, _f, _k) in _ARMS.items()}

SIGNAL_PCT = 2.0


def _paired(arm_rows, base_rows, metric):
    """Per-dataset means of the arm and the baseline over the (dataset, horizon) cells the arm has."""
    a = arm_rows.groupby(["data", "pred_len"])[metric].mean()
    b = base_rows.groupby(["data", "pred_len"])[metric].mean()
    cells = a.index.intersection(b.index)
    return a[cells].groupby(level="data").mean(), b[cells].groupby(level="data").mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RAW)
    ap.add_argument("--metric", default="MSE")
    ap.add_argument("--per_dataset", action="store_true",
                    help="also break each arm down by dataset")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    abl = raw[raw.role == "ablation"]
    if abl.empty:
        print(f"no ablation rows in {args.csv} - run `python run_plan.py --stage ablation` first")
        return 1
    base = raw[(raw.role == "main") & (raw.model == "FEATHer_mix") & (raw.config_id == "frozen")]
    if base.empty:
        print(f"no FEATHer-mix main rows in {args.csv}")
        return 1

    base_by_ds = base.groupby("data")[args.metric].mean()
    print(f"baseline: FEATHer-mix (frozen), "
          f"{base.seed.nunique()} seed(s), {len(base_by_ds)} datasets, "
          f"{args.metric} {base_by_ds.mean():.4f}\n")

    tag2arm = {f"abl_{a}": a for a in ABLATION_ARMS}
    rows = []
    for tag, grp in abl.groupby("exp_tag"):
        arm = tag2arm.get(tag)
        if arm is None:
            continue
        # compare on the arm's own (dataset, horizon) cells only (most arms ran at H = 96 only);
        # change = mean over datasets of the per-dataset % change (the paper's rule)
        a_ds, b_ds = _paired(grp, base, args.metric)
        common = a_ds.index
        if len(common) == 0:
            continue
        a_mean, b_mean = a_ds.mean(), b_ds.mean()
        delta = ((a_ds / b_ds - 1) * 100).mean()
        tier, removed, _ = ABLATION_ARMS[arm]
        rows.append({
            "tier": tier,
            "arm": arm,
            "removed": removed,
            "cells": len(grp),
            "ds": len(common),
            "ablated": round(a_mean, 4),
            "baseline": round(b_mean, 4),
            "delta_%": round(delta, 2),
            "verdict": "MATTERS" if abs(delta) >= SIGNAL_PCT else "flat (noise)",
        })

    if not rows:
        print("no arm rows to compare yet")
        return 1

    df = pd.DataFrame(rows)
    # Report the two tiers separately: FEATHer's own frequency machinery is the
    # claim under test, the borrowed capacity component is context. Mixing them
    # in one ranked list invites reading a standard FFN bottleneck as one of
    # our contributions.
    for tier, label in [("design", "FEATHer design components (the method's claim)"),
                        ("standard", "standard borrowed capacity (NOT a contribution)")]:
        sub = df[df.tier == tier].sort_values("delta_%", ascending=False)
        if sub.empty:
            continue
        print(f"\n--- {label} ---")
        print(sub.drop(columns="tier").to_string(index=False))

    print(f"\n+delta_% = removing it made the model WORSE = the component helps.")
    print(f"|delta| >= {SIGNAL_PCT}% counts as signal; below that is seed noise.")
    print("nospk swaps in a much LARGER head (~938 -> ~10k params at Steel/H96), "
          "so read its delta next to that cost: a near-tie favours SPK.")

    missing = [a for a in ABLATION_ARMS if a not in df.arm.values]
    if missing:
        print(f"\nnot run yet: {missing}")

    if args.per_dataset:
        for r in df.itertuples():
            a_ds, b_ds = _paired(abl[abl.exp_tag == f"abl_{r.arm}"], base, args.metric)
            per = pd.DataFrame({
                "ablated": a_ds.round(4),
                "baseline": b_ds.round(4),
                "delta_%": ((a_ds / b_ds - 1) * 100).round(2),
            })
            print(f"\n--- {r.arm} ({r.removed}) ---")
            print(per.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
