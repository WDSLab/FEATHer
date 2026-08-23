#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Component ablation for FEATHer-N on the manufacturing benchmark.

Ablates the model that is actually IN the main table — FEATHer-N as frozen in
`_DATASET_OVERRIDES` (channel_mode=mix, residual MLP rank 16, d_state 16,
kernel 7, num_bands 4, per-dataset FFT period) — by switching exactly one
component off per arm and re-training.

This deliberately does NOT use the `FEATHer_<axis>_<variant>` registry entries.
Those builders (baselines/__init__.py) predate the residual MLP and the channel
mode, so they construct the OLD architecture: their numbers would not be
comparable to the main-table baseline.

Baseline row = the existing main sweep (`results/mfg_raw.csv`, exp_tag `main`).
It is NOT re-run; `--summary` reads it for the deltas.

Arms (one component each):
    noresid   residual MLP off          expect: matters (Steel -7%, WindSCADA -8%)
    ci        channel mixing -> independent   expect: matters (~5% on mfg)
    norevin   RevIN off                 expect: matters
    gateuni   FFT gate -> uniform       expect: ~0 (linear collapse)
    band2     4 bands -> 2 (the floor)  expect: ~0 (linear collapse)
    d4        d_state 16 -> 4           expect: small

Verdict threshold: seed noise on a 24-cell mean is +-0.3~0.5%, so only a >=2%
change counts as signal (2026-08-16 calibration). Arms that clear it get
promoted to 3 seeds.

Usage
-----
    python run_ablation_mfg.py --check
    python run_ablation_mfg.py --ngpu 2 --jobs_per_gpu 3
    python run_ablation_mfg.py --arms gateuni,band1        # subset
    python run_ablation_mfg.py --summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys

RESULTS_CSV = "results/ablation_mfg.csv"
BASELINE_CSV = "results/mfg_raw.csv"
BASELINE_TAG = "main"

# arm -> (exp_tag, extra flags for run_forecast.py)
ARMS: dict[str, tuple[str, list[str]]] = {
    "noresid": ("abl_noresid", ["--no_residual_mlp"]),
    "ci":      ("abl_ci",      ["--channel_mode", "ci_full"]),
    "norevin": ("abl_norevin", ["--no_revin"]),
    "gateuni": ("abl_gateuni", ["--gate_uniform"]),
    # num_bands is asserted to be in {2,3,4} by the model, so 2 is the floor —
    # a single-band arm would need a code change, and 4->2 already halves the
    # decomposition.
    "band2":   ("abl_band2",   ["--num_bands", "2"]),
    "d4":      ("abl_d4",      ["--d_state", "4"]),
}


def build_cmd(arm: str, args) -> list[str]:
    exp_tag, extra = ARMS[arm]
    cmd = [
        sys.executable, "run_forecast.py",
        "--model", "FEATHer",
        "--group", "mfg",
        "--exp_tag", exp_tag,
        "--results_csv", args.results_csv,
        "--num_seeds", str(args.num_seeds),
        "--ngpu", str(args.ngpu),
        "--jobs_per_gpu", str(args.jobs_per_gpu),
        "--gpu", str(args.gpu),
    ]
    if args.data:
        cmd += ["--data", args.data]
    if args.check:
        cmd += ["--check"]
    return cmd + extra


def summary(args) -> int:
    import pandas as pd

    try:
        abl = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print(f"no ablation results yet at {args.results_csv}")
        return 1
    base = pd.read_csv(args.baseline_csv)
    base = base[(base.model == "FEATHer") & (base.exp_tag == BASELINE_TAG)]
    if base.empty:
        print(f"no baseline FEATHer rows in {args.baseline_csv}")
        return 1

    tag2arm = {t: a for a, (t, _) in ARMS.items()}
    # dataset-mean of cell-means, matching how the main table aggregates
    base_by_ds = base.groupby("data").MSE.mean()
    base_overall = base_by_ds.mean()
    print(f"baseline (FEATHer-N, exp_tag={BASELINE_TAG}): OVERALL {base_overall:.4f}"
          f"   [{base.seed.nunique()} seed(s)]\n")

    rows = []
    for tag, grp in abl.groupby("exp_tag"):
        arm = tag2arm.get(tag, tag)
        by_ds = grp.groupby("data").MSE.mean()
        common = by_ds.index.intersection(base_by_ds.index)
        if len(common) == 0:
            continue
        # compare on the datasets this arm actually covers
        a_mean = by_ds[common].mean()
        b_mean = base_by_ds[common].mean()
        rows.append({
            "arm": arm,
            "cells": len(grp),
            "datasets": len(common),
            "ablated": round(a_mean, 4),
            "baseline": round(b_mean, 4),
            "delta_%": round((a_mean - b_mean) / b_mean * 100, 2),
        })

    if not rows:
        print("no comparable rows yet")
        return 1
    df = pd.DataFrame(rows).sort_values("delta_%", ascending=False)
    df["verdict"] = ["MATTERS" if abs(x) >= 2.0 else "flat (within noise)"
                     for x in df["delta_%"]]
    print(df.to_string(index=False))
    print("\n(+delta_% = removing the component made it WORSE = the component helps)")
    print("threshold: >=2% counts as signal; below that is seed noise.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", type=str, default=None,
                   help=f"comma-separated subset of {','.join(ARMS)}")
    p.add_argument("--data", type=str, default=None, help="restrict datasets")
    p.add_argument("--num_seeds", type=int, default=1,
                   help="1 for the screening pass; promote winners to 3")
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--baseline_csv", type=str, default=BASELINE_CSV)
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true", help="dry-run job counts")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.summary:
        return summary(args)

    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        p.error(f"unknown arm(s): {unknown}. choose from {list(ARMS)}")

    print(f"ablation arms: {arms}")
    print(f"results -> {args.results_csv}   (baseline stays in {args.baseline_csv})\n")

    failed = []
    for arm in arms:
        cmd = build_cmd(arm, args)
        print("=" * 70)
        print(f"ARM {arm}  ({ARMS[arm][0]})")
        print(" ".join(cmd))
        print("=" * 70, flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append(arm)
            print(f"[WARN] arm {arm} exited {rc}", flush=True)

    if failed:
        print(f"\nFAILED arms: {failed}")
        return 1
    print("\nall arms done. run --summary for the verdict table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
