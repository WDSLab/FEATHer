# -*- coding: utf-8 -*-
"""
Forecasting orchestrator.

User-facing wrapper. Reads `results/fcst_results.csv`, figures out which
(exp_tag, model, dataset, pred_len, seed) combos are still missing, and
dispatches the missing ones to the worker (`scripts/benchmarks/run_forecast.py`)
one (model, dataset, pred_len) at a time. Re-running this script picks up
where it left off — completed seeds are skipped.

Usage:
    python run_forecast.py --check                       # show status
    python run_forecast.py                               # run all missing
    python run_forecast.py --model FEATHer               # one model
    python run_forecast.py --data ETTh1 --pred_len 96    # one combo
    python run_forecast.py --num_seeds 5 --exp_tag main  # main 5-seed sweep
    python run_forecast.py --exclude TimesNet,S_Mamba    # skip heavy models
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (list_models, list_ablation_models,
                       get_method_defaults, get_dataset_overrides)


# -----------------------------------------------------------------------------
# Defaults — adjust here if the main sweep changes
# -----------------------------------------------------------------------------

DEFAULT_MODELS = list_models()  # all registered

# Long-term horizon datasets
LTSF_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
                 "Traffic", "Exchange"]
LTSF_PREDS = [96, 192, 336, 720]

# Short-horizon datasets
SHORT_DATASETS = ["SML", "Volatility"]
SHORT_PREDS = [24, 48, 96, 192]

# Spatio-temporal
ST_DATASETS = ["PEMS03", "PEMS04", "PEMS08", "PEMS_BAY", "METR"]
ST_PREDS = [12, 24, 48, 96]

DEFAULT_BASE_SEED = 2025
DEFAULT_NUM_SEEDS = 5
DEFAULT_EXP_TAG = "main"
DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/fcst_results.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pred_lens_for(dataset):
    if dataset in SHORT_DATASETS:
        return SHORT_PREDS
    if dataset in ST_DATASETS:
        return ST_PREDS
    return LTSF_PREDS


def load_done_set(csv_path):
    """Return set of (exp_tag, model, data, pred_len, seq_len, seed) tuples
    already present in the results CSV."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    need = ["exp_tag", "model", "data", "pred_len", "seq_len", "seed"]
    for col in need:
        if col not in df.columns:
            return set()
    return set(
        (str(r.exp_tag), str(r.model), str(r.data),
         int(r.pred_len), int(r.seq_len), int(r.seed))
        for r in df[need].itertuples(index=False)
    )


def enumerate_combos(args):
    """Generate (exp_tag, model, data, pred_len, seq_len, seed) tuples for
    the requested sweep."""

    if args.ablation_axis:
        # FEATHer ablation sweep — one axis (ms/gate/dtk/head/complexity)
        # or "all" 30 variants. Combine with --data / --pred_len to scope.
        if args.ablation_axis == "all":
            models = list_ablation_models()
        else:
            prefix = f"FEATHer_{args.ablation_axis}_"
            models = [m for m in list_ablation_models()
                      if m.startswith(prefix)]
            if not models:
                raise SystemExit(
                    f"Unknown ablation axis '{args.ablation_axis}' "
                    f"(use ms, gate, dtk, head, complexity, or all)")
    elif args.model:
        models = [args.model]
    elif args.exclude:
        excl = {s.strip() for s in args.exclude.split(",") if s.strip()}
        models = [m for m in DEFAULT_MODELS if m not in excl]
    else:
        models = DEFAULT_MODELS

    if args.data:
        datasets = [args.data]
    else:
        datasets = LTSF_DATASETS  # default sweep — short/ST opt-in only

    if args.pred_len > 0:
        pred_lens_override = [args.pred_len]
    else:
        pred_lens_override = None

    seeds = [args.base_seed + i for i in range(args.num_seeds)]

    combos = []
    for m in models:
        for d in datasets:
            preds = pred_lens_override or pred_lens_for(d)
            for h in preds:
                for s in seeds:
                    combos.append((args.exp_tag, m, d, h, args.seq_len, s))
    return combos


def group_missing(missing):
    """Group missing combos by (model, data, pred_len) → [seeds]. The
    worker takes one (model, data, pred_len) per call and iterates seeds
    internally."""
    groups = {}
    for exp_tag, m, d, h, sl, sd in missing:
        key = (exp_tag, m, d, h, sl)
        groups.setdefault(key, []).append(sd)
    return groups


def print_status(missing, total, csv_path=RESULTS_CSV):
    print("\n" + "=" * 70)
    print(f"  Forecast experiment status")
    print(f"  CSV: {csv_path}")
    print(f"  Total requested: {total}  |  Done: {total - len(missing)}  "
          f"|  Missing: {len(missing)}")
    print("=" * 70)
    if not missing:
        print("  All complete!")
        return

    groups = group_missing(missing)
    for (exp_tag, m, d, h, sl), seeds in sorted(groups.items()):
        seeds_str = ",".join(str(s) for s in sorted(seeds))
        print(f"  [{exp_tag:>12}] {m:>15} | {d:>12} | H={h:<4} L={sl} | "
              f"seeds: {seeds_str}")


def dispatch(missing, args):
    groups = group_missing(missing)
    method_order = {m: i for i, m in enumerate(DEFAULT_MODELS)}
    keys_sorted = sorted(groups.keys(),
                         key=lambda k: (method_order.get(k[1], 999), k[2], k[3]))

    for (exp_tag, m, d, h, sl) in keys_sorted:
        seeds = sorted(groups[(exp_tag, m, d, h, sl)])

        # Merge precedence: paper-default (method) → per-dataset paper
        # override → CLI flag. lr and loss are top-level worker flags;
        # everything else goes through --model_overrides.
        method_def  = get_method_defaults(m)
        dataset_ovr = get_dataset_overrides(m, d)
        merged = {**method_def, **dataset_ovr}

        lr_to_use   = args.lr   if args.lr   is not None else merged.get("lr",   1e-3)
        loss_to_use = args.loss if args.loss is not None else merged.get("loss", "mse")

        hp_overrides = {k: v for k, v in merged.items() if k not in ("lr", "loss")}
        overrides_str = ";".join(f"{k}={v}" for k, v in sorted(hp_overrides.items()))

        print(f"\n>>> {m} | {d} | H={h} | seeds={seeds} | lr={lr_to_use} | "
              f"loss={loss_to_use}"
              + (f" | overrides={overrides_str}" if overrides_str else ""))

        cmd = [
            sys.executable,
            "scripts/benchmarks/run_forecast.py",
            "--model", m,
            "--data", d,
            "--pred_len", str(h),
            "--seq_len", str(sl),
            "--seeds", ",".join(str(s) for s in seeds),
            "--exp_tag", exp_tag,
            "--results_csv", args.results_csv,
            "--num_epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--lr", str(lr_to_use),
            "--loss", loss_to_use,
            "--gpu", str(args.gpu),
        ]
        if args.save_model:
            cmd += ["--save_model"]
        if overrides_str:
            cmd += ["--model_overrides", overrides_str]
        # FEATHer hyperparameters — harmless to pass to other models
        cmd += [
            "--d_state", str(args.d_state),
            "--kernel_size", str(args.kernel_size),
            "--period", str(args.period),
            "--num_bands", str(args.num_bands),
            "--lambda_spec", str(args.lambda_spec),
        ]
        print("  CMD:", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  [WARN] worker returned non-zero exit ({ret.returncode}); "
                  f"continuing")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Forecast experiment orchestrator")

    # What to run
    p.add_argument("--model",   type=str, default=None,
                   help="Single model; omit to run all registered.")
    p.add_argument("--ablation_axis", type=str, default=None,
                   help="FEATHer ablation sweep: ms, gate, dtk, head, "
                        "complexity, or all (30 variants).")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated models to skip.")
    p.add_argument("--data",    type=str, default=None,
                   help="Single dataset; omit for the LTSF sweep.")
    p.add_argument("--pred_len", type=int, default=0,
                   help="Single horizon; 0 = use defaults per dataset.")
    p.add_argument("--seq_len",  type=int, default=DEFAULT_SEQ_LEN)

    # Seeds + grouping
    p.add_argument("--num_seeds",  type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--base_seed",  type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--exp_tag",    type=str, default=DEFAULT_EXP_TAG)

    # Per-worker overrides (all forwarded)
    p.add_argument("--num_epochs", type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early-stopping patience on val loss (epochs).")
    p.add_argument("--batch_size", type=int,   default=32)
    # `--lr` / `--loss` default to None so we can detect "user did not set
    # this" and fall back to each method's paper default
    # (baselines.get_method_defaults). Pass a value explicitly to override
    # across the whole sweep (ablations).
    p.add_argument("--lr",         type=float, default=None,
                   help="Override LR for all models. Defaults to each method's "
                        "paper-recommended value (see baselines.get_method_defaults).")
    p.add_argument("--loss",       type=str,   default=None, choices=["l1", "mse"],
                   help="Override loss. Defaults to each method's paper "
                        "loss (MSE for baselines, L1 for FEATHer).")
    p.add_argument("--d_state",     type=int,   default=8)
    p.add_argument("--kernel_size", type=int,   default=7)
    p.add_argument("--period",      type=int,   default=12)
    p.add_argument("--num_bands",   type=int,   default=3)
    p.add_argument("--lambda_spec", type=float, default=0.01)

    # Output / system
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--save_model",  action="store_true",
                   help="Save checkpoints (forwarded to worker).")

    # Mode
    p.add_argument("--check", action="store_true",
                   help="Print status without running anything")

    args = p.parse_args()

    combos = enumerate_combos(args)
    done = load_done_set(args.results_csv)
    missing = [c for c in combos if c not in done]

    print_status(missing, total=len(combos), csv_path=args.results_csv)

    if args.check:
        return

    if not missing:
        print("\nNothing to run.")
        return

    print(f"\n>>> Dispatching {len(missing)} missing runs "
          f"({len(group_missing(missing))} worker invocations)...")
    dispatch(missing, args)

    print("\n>>> Final status:")
    done_final = load_done_set(args.results_csv)
    missing_final = [c for c in combos if c not in done_final]
    print_status(missing_final, total=len(combos), csv_path=args.results_csv)


if __name__ == "__main__":
    main()
