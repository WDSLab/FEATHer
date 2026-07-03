# -*- coding: utf-8 -*-
"""
Per-(method, dataset) LEARNING-RATE search for the manufacturing main table.

Why this exists: the manufacturing datasets (Steel/GasTurbine/WindSCADA +
C-MAPSS, JMS pivot) are NOT in any baseline's paper, so there are no published
per-dataset hyperparameters to copy. Per the protocol decision (2026-06-25,
Option B), we keep each method's architecture fixed at its paper default and
select ONLY the learning rate per (method, dataset) on the validation split —
lr is the dominant axis and this keeps the comparison apples-to-apples and
the compute tractable. FEATHer is treated identically (canonical architecture,
lr searched) for a symmetric fair-comparison story.

How architecture stays at "paper default": we call the worker directly with
just --lr (no --model_overrides), so each wrapper's getattr(args, key,
default) falls back to its baked-in paper config. (The orchestrator's
_DATASET_OVERRIDES merge is bypassed on purpose here.)

Selection: ONLY the worker-written `val_loss` (best-val-epoch loss on val).
Test metrics land in the CSV too but never drive selection.

CAVEAT (flagged for later): a few baselines tie an architecture HP to the
data's sampling period (SparseTSF `period_len`, TQNet `cycle`). Those stay at
the ETTh1/hourly default here; if a manufacturing dataset's periodicity
differs materially, set that HP per-dataset in _DATASET_OVERRIDES by hand.

Usage:
    python run_lr_search.py --check        # show pending runs
    python run_lr_search.py                # run all missing (resumable)
    python run_lr_search.py --summary      # per-(method,dataset) best lr + paste lines

Multi-GPU: `--ngpu 2` runs a CF-JEPA-style dynamic queue (one worker
subprocess per GPU; a free GPU grabs the next job) — preferred, auto
load-balancing:
    python run_lr_search.py --ngpu 2
Manual disjoint --exclude/--model splits across separate processes still
work (one per --gpu); sharing one results CSV is fine either way (one
append per finished run).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import list_models  # noqa: E402
from utils.dispatch_queue import run_on_gpus  # noqa: E402


# Learning-rate grid spanning the range baselines use across their papers
# (1e-4 .. 5e-2). Five points, log-spaced.
LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

MODELS = list_models()                                   # all 12
# TEP dropped 2026-06-26. Short-horizon PdM group (2026-07-02): CMAPSS FD001 +
# CMAPSS3 (FD003) + PMSM (rebuilt: 69 sessions, 30-s mean, unit-aware loader —
# dropped from the long-horizon main table, re-enters here).
LR_DATASETS = ["Steel", "GasTurbine", "WindSCADA", "CMAPSS", "CMAPSS3", "PMSM"]
LR_PREDS = [96, 720]                                     # representative short/long
LR_PREDS_SHORT = [24, 96]                                # short-horizon PdM group
SHORT_DATASETS = ("CMAPSS", "CMAPSS3", "PMSM")
LR_SEEDS = [2025, 2026]


def _lr_preds(dataset):
    # Unit-segmented PdM trajectories are too short for 720 (see
    # Dataset_CMAPSS / CMAPSS_PREDS); use their own representative short/long.
    return LR_PREDS_SHORT if dataset in SHORT_DATASETS else LR_PREDS

DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/lr_search.csv"
WORKER = "scripts/benchmarks/run_forecast.py"


def loss_for(model):
    # FEATHer trains on L1 (spectral-separation friendly); baselines on MSE,
    # matching each method's official default.
    return "l1" if model == "FEATHer" else "mse"


def select_models(args):
    """Apply --model / --exclude to the registry list (run_forecast.py
    convention). Both take comma-separated names; unknown names are fatal
    so a typo can't silently shrink the sweep."""
    models = list(MODELS)
    if args.model:
        want = [m.strip() for m in args.model.split(",") if m.strip()]
        unknown = sorted(set(want) - set(MODELS))
        if unknown:
            sys.exit(f"Unknown --model name(s) {unknown}; known: {MODELS}")
        models = [m for m in MODELS if m in want]
    if args.exclude:
        drop = {m.strip() for m in args.exclude.split(",") if m.strip()}
        unknown = sorted(drop - set(MODELS))
        if unknown:
            sys.exit(f"Unknown --exclude name(s) {unknown}; known: {MODELS}")
        models = [m for m in models if m not in drop]
    return models


def enumerate_runs(args, models):
    """(model, lr, data, pred_len, seq_len, seed) for the full grid."""
    runs = []
    for m in models:
        for lr in LR_GRID:
            for d in LR_DATASETS:
                for h in _lr_preds(d):
                    for s in LR_SEEDS:
                        runs.append((m, lr, d, h, args.seq_len, s))
    return runs


def _exp_tag(lr):
    # One exp_tag per lr so (exp_tag, model, data, pred_len, seq_len, seed)
    # stays a unique resume key in the shared CSV schema.
    return f"lr_{lr:g}"


def load_done_set(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    return set(
        (str(r.exp_tag), str(r.model), str(r.data), int(r.pred_len),
         int(r.seq_len), int(r.seed))
        for r in df[["exp_tag", "model", "data", "pred_len", "seq_len", "seed"]]
            .itertuples(index=False)
    )


def dispatch(missing, args):
    # Group by (model, lr, data, pred_len) — worker iterates seeds internally.
    groups = {}
    for m, lr, d, h, sl, s in missing:
        groups.setdefault((m, lr, d, h, sl), []).append(s)

    jobs = []
    for (m, lr, d, h, sl), seeds in sorted(groups.items()):
        label = f"{m} | {d} | H={h} | lr={lr:g} | seeds={sorted(seeds)}"
        cmd = [
            sys.executable, WORKER,
            "--model", m,
            "--data", d,
            "--pred_len", str(h),
            "--seq_len", str(sl),
            "--seeds", ",".join(str(s) for s in sorted(seeds)),
            "--exp_tag", _exp_tag(lr),
            "--results_csv", args.results_csv,
            "--num_epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--loss", loss_for(m),
            "--lr", str(lr),
        ]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; their runs "
              "stay missing — re-run to retry.")


def summarize(args):
    if not os.path.exists(args.results_csv):
        print(f"No results yet: {args.results_csv}")
        return
    df = pd.read_csv(args.results_csv)
    if df.empty:
        print("No rows in the CSV.")
        return

    # Mean over seeds per (model, data, lr, horizon) cell.
    df["lr_val"] = df["exp_tag"].str.replace("lr_", "", regex=False).astype(float)
    cell = (df.groupby(["model", "data", "lr_val", "pred_len"])
              .agg(val_loss=("val_loss", "mean"), MSE=("MSE", "mean"),
                   n_seeds=("seed", "nunique"))
              .reset_index())

    print("\n=== Per-(method, dataset) lr selection (val_loss; "
          "test MSE for reference only) ===")
    override_lines = []
    for m in MODELS:
        for d in LR_DATASETS:
            cd = cell[(cell["model"] == m) & (cell["data"] == d)]
            if cd.empty:
                continue
            # Rank lr within each horizon by val_loss, average across horizons.
            cd = cd.copy()
            cd["rank"] = cd.groupby("pred_len")["val_loss"].rank()
            agg = (cd.groupby("lr_val")
                     .agg(mean_rank=("rank", "mean"), mean_val=("val_loss", "mean"),
                          mean_mse=("MSE", "mean"), cells=("rank", "size")))
            n_cells = cd["pred_len"].nunique()
            best_lr = agg["mean_rank"].idxmin()
            incomplete = agg.loc[best_lr, "cells"] < n_cells
            flag = "  (incomplete)" if incomplete else ""
            print(f"  {m:>12} | {d:<10} -> lr={best_lr:g}  "
                  f"val={agg.loc[best_lr,'mean_val']:.4f}  "
                  f"testMSE={agg.loc[best_lr,'mean_mse']:.4f}{flag}")
            override_lines.append(f'    ("{m}", "{d}"): {{"lr": {best_lr:g}}},')

    print("\n=== Paste into baselines/__init__.py _DATASET_OVERRIDES ===")
    print("    # ---- per-(method, dataset) lr, val-selected on manufacturing "
          "datasets (run_lr_search.py --summary) ----")
    print("\n".join(override_lines))
    print("\nNote: architecture HPs stay at each method's paper default; only "
          "lr is dataset-selected. Frequency-tied HPs (SparseTSF period_len, "
          "TQNet cycle) may still want a manual per-dataset value.")


def main():
    p = argparse.ArgumentParser(description="Per-(method, dataset) lr search "
                                            "(manufacturing main table)")
    p.add_argument("--seq_len",     type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--num_epochs",  type=int, default=50)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index (with --ngpu N: the FIRST of N indices)")
    p.add_argument("--ngpu",        type=int, default=1,
                   help="Number of GPUs; N>1 = dynamic job queue across "
                        "GPUs gpu..gpu+N-1 (CF-JEPA-style load balancing)")
    p.add_argument("--check",   action="store_true", help="show pending runs")
    p.add_argument("--summary", action="store_true", help="rank finished runs")
    p.add_argument("--model",   type=str, default="",
                   help="comma-separated subset of models to run")
    p.add_argument("--exclude", type=str, default="",
                   help="comma-separated models to skip (for disjoint GPU splits)")
    args = p.parse_args()

    if args.summary:
        summarize(args)
        return

    models = select_models(args)
    runs = enumerate_runs(args, models)
    done = load_done_set(args.results_csv)
    missing = [r for r in runs
               if (_exp_tag(r[1]), r[0], r[2], r[3], r[4], r[5]) not in done]

    print(f"\nlr search: {len(models)}/{len(MODELS)} models x {len(LR_GRID)} lr x "
          f"{len(LR_DATASETS)} datasets x ~{len(LR_PREDS)} horizons x "
          f"{len(LR_SEEDS)} seeds (short-horizon PdM sets use {LR_PREDS_SHORT})")
    print(f"Total: {len(runs)} runs | Done: {len(runs) - len(missing)} "
          f"| Missing: {len(missing)}")

    if args.check:
        for m, lr, d, h, sl, s in missing[:60]:
            print(f"  [{_exp_tag(lr):>8}] {m:>12} | {d:>10} | H={h:<4} L={sl} | seed {s}")
        if len(missing) > 60:
            print(f"  ... (+{len(missing) - 60} more)")
        return

    if not missing:
        print("All complete — run with --summary to rank lr.")
        return

    dispatch(missing, args)
    print("\nDone. Run `python run_lr_search.py --summary` to rank lr.")


if __name__ == "__main__":
    main()
