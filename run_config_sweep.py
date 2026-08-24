#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pick the smallest FEATHer-N config that keeps the accuracy.

The shipping mfg config (ds16_k7_B4 + residual MLP, 938 params at the
deployment cell) is not on the accuracy-vs-size Pareto front. In the 27-config
mfg grid, 18 configs sit within the 1.2% per-cell seed noise of the best, and
the shipping one is among the largest of them — ds8_k7_B4 matches its MSE
exactly at 770 parameters, and ds8_k3_B4 is within 0.09% at 738.

The grid cannot settle it on its own because every one of its 27 configs has
the residual MLP ON, and the residual is what pins SRAM: the DTK latent buffer
is sized L * max(d_state, mlp_rank), so with mlp_rank=16 fixed, lowering
d_state frees no RAM at all. Dropping the residual is what turns d_state into
a memory lever — ds8_k3_B4 goes 18.7 -> 15.7 KB, ds4_k3_B3 to 14.2 KB — and
RAM is the binding constraint on this board (18.6 of 20 KB at H=96, and the
reason H=192 will not fit).

What the residual is worth at a SMALL config is unmeasured. The one data point
we have is the 08-05 ceiling scan at d8/k7/B3, where it bought 2.16%; whether
that holds when d_state is halved is exactly the open question, since a
smaller latent could either need the nonlinearity more or be too small to use
it.

So: six arms, each a (config, residual) pair, one seed, isolated CSV. The
winner gets promoted to 3 seeds and becomes the reported config.

    python run_config_sweep.py --check
    python run_config_sweep.py --ngpu 1 --gpu 0 --jobs_per_gpu 3
    python run_config_sweep.py --summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys

RESULTS_CSV = "results/config_sweep.csv"

# arm -> (exp_tag, d_state, kernel, bands, residual, params@D7/H96, RAM KB)
ARMS = {
    "current":   ("cfg_current",   16, 7, 4, True,  938, 18.7),
    "d8B4":      ("cfg_d8B4",       8, 3, 4, True,  738, 18.7),
    "d8B4_nr":   ("cfg_d8B4_nr",    8, 3, 4, False, 491, 15.7),
    "d8B4k7_nr": ("cfg_d8B4k7_nr",  8, 7, 4, False, 523, 15.7),
    "d4B3":      ("cfg_d4B3",       4, 3, 3, True,  636, 18.7),
    "d4B3_nr":   ("cfg_d4B3_nr",    4, 3, 3, False, 389, 14.2),
}


def build_cmd(arm: str, args) -> list[str]:
    tag, d, k, b, resid, _, _ = ARMS[arm]
    cmd = [
        sys.executable, "run_forecast.py",
        "--model", "FEATHer",
        "--group", "mfg",
        "--exp_tag", tag,
        "--results_csv", args.results_csv,
        "--num_seeds", str(args.num_seeds),
        "--d_state", str(d),
        "--kernel_size", str(k),
        "--num_bands", str(b),
        "--ngpu", str(args.ngpu),
        "--jobs_per_gpu", str(args.jobs_per_gpu),
        "--gpu", str(args.gpu),
    ]
    # channel_mode stays mix (FEATHer-N); the residual is the toggle.
    cmd += ["--channel_mode", "mix"]
    cmd += ["--use_residual_mlp", "--mlp_rank", "16"] if resid else ["--no_residual_mlp"]
    if args.data:
        cmd += ["--data", args.data]
    if args.check:
        cmd += ["--check"]
    return cmd


def summary(args) -> int:
    import pandas as pd
    try:
        df = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print("no results yet at %s" % args.results_csv)
        return 1

    tag2arm = {t: a for a, (t, *_rest) in ARMS.items()}
    rows = []
    for tag, grp in df.groupby("exp_tag"):
        arm = tag2arm.get(tag)
        if arm is None:
            continue
        _, d, k, b, resid, p, ram = ARMS[arm]
        rows.append({
            "arm": arm,
            "config": "ds%d_k%d_B%d" % (d, k, b),
            "resid": "on" if resid else "OFF",
            "params": p,
            "RAM_KB": ram,
            "MSE": grp.groupby("data").MSE.mean().mean(),
            "cells": len(grp),
            "datasets": grp.data.nunique(),
        })
    if not rows:
        print("no arm rows yet")
        return 1

    t = pd.DataFrame(rows).sort_values("MSE")
    best = t.MSE.min()
    t["vs_best_%"] = ((t.MSE - best) / best * 100).round(2)
    # 1.2% is the per-cell seed noise; a 24-cell mean is tighter, but keep the
    # conservative bar so we do not chase noise into a smaller model.
    t["verdict"] = ["tied" if v <= 1.2 else "worse" for v in t["vs_best_%"]]
    print(t.to_string(index=False))

    tied = t[t.verdict == "tied"].sort_values("params")
    if len(tied):
        w = tied.iloc[0]
        print("\nsmallest arm statistically tied with the best:")
        print("  %s (%s, residual %s) — %d params, %.1f KB RAM, "
              "%.4f MSE (%+.2f%% vs best)"
              % (w.arm, w.config, w.resid, w.params, w.RAM_KB, w.MSE, w["vs_best_%"]))
        print("\nPromote it to 3 seeds before reporting; these are 1-seed runs.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=None, help="subset of %s" % ",".join(ARMS))
    p.add_argument("--data", default=None)
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--results_csv", default=RESULTS_CSV)
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    if args.summary:
        return summary(args)

    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        p.error("unknown arm(s) %s; choose from %s" % (unknown, list(ARMS)))

    print("config sweep -> %s\n" % args.results_csv)
    failed = []
    for arm in arms:
        tag, d, k, b, resid, params, ram = ARMS[arm]
        cmd = build_cmd(arm, args)
        print("=" * 72)
        print("ARM %-11s ds%d_k%d_B%d  residual %-3s  %d params  %.1f KB RAM"
              % (arm, d, k, b, "on" if resid else "OFF", params, ram))
        print(" ".join(cmd))
        print("=" * 72, flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append(arm)
            print("[WARN] arm %s exited %d" % (arm, rc), flush=True)

    if failed:
        print("\nFAILED arms: %s" % failed)
        return 1
    print("\nall arms done; run --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
