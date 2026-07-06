# -*- coding: utf-8 -*-
"""
Robustness orchestrator.

Top-level wrapper that figures out which (train_exp_tag, model, dataset,
pred_len, seed) checkpoints already have a complete robustness sweep in
`results/robust_results.csv`, then dispatches the missing combos to the
worker. Mirrors `run_forecast.py` so the same mental model applies.

A combo is "done" when the CSV contains a `clean` row plus one row for
every (fault_type, severity) pair in `utils.noise.SEVERITY_SWEEPS` —
total 1 (clean) + 5 (gauss) + 5 (miss) + 4 (impulse) + 5 (quant) = 20 rows.

Usage:
    python run_robustness.py --check                              # status only
    python run_robustness.py                                       # run all missing
    python run_robustness.py --model FEATHer                       # one model
    python run_robustness.py --data ETTh1 --pred_len 96            # one combo
    python run_robustness.py --fault_types gauss,miss              # subset of faults
    python run_robustness.py --train_exp_tag main --exp_tag robust # explicit tags
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import list_models
from utils import noise as noise_mod
from utils.dispatch_queue import run_on_gpus
from scripts.benchmarks.run_forecast import checkpoint_path


# Defaults — mirror run_forecast.py
DEFAULT_MODELS = list_models()

# IoT-J robustness is reported on 4 representative datasets, not the full
# 8-dataset main sweep — keeps the experiment tractable while covering the
# canonical (ETTh1), real-sensor (Weather), smart-grid (Electricity), and
# smart-home (SML) regimes.
ROBUSTNESS_DATASETS = ["ETTh1", "Weather", "Electricity", "SML"]
LTSF_PREDS = [96, 192, 336, 720]
SHORT_PREDS = [24, 48, 96, 192]      # SML
SHORT_DATASETS = {"SML", "Volatility"}

DEFAULT_BASE_SEED = 2025
DEFAULT_NUM_SEEDS = 5
DEFAULT_TRAIN_EXP_TAG = "main"
DEFAULT_EXP_TAG = "robust"
DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/robust_results.csv"


def pred_lens_for(dataset):
    return SHORT_PREDS if dataset in SHORT_DATASETS else LTSF_PREDS


def expected_rows(fault_types):
    """Total (fault_type, severity) row count expected per checkpoint —
    plus 1 for the 'clean' baseline row."""
    n = 1  # clean
    for ft in fault_types:
        n += len(noise_mod.SEVERITY_SWEEPS[ft])
    return n


def load_robust_set(csv_path, fault_types):
    """Return set of (exp_tag, train_exp_tag, model, data, pred_len, seq_len,
    seed) tuples that already have a *complete* sweep in the CSV."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()

    need = expected_rows(fault_types)
    keep_faults = set(fault_types) | {"clean"}

    df_keep = df[df["fault_type"].isin(keep_faults)]
    grp_cols = ["exp_tag", "train_exp_tag", "model", "data",
                "pred_len", "seq_len", "seed"]
    counts = df_keep.groupby(grp_cols).size().reset_index(name="n")
    done = counts[counts["n"] >= need]

    return set(
        (str(r.exp_tag), str(r.train_exp_tag), str(r.model), str(r.data),
         int(r.pred_len), int(r.seq_len), int(r.seed))
        for r in done.itertuples(index=False)
    )


def enumerate_combos(args, fault_types):
    if args.model:
        models = [args.model]
    elif args.exclude:
        excl = {s.strip() for s in args.exclude.split(",") if s.strip()}
        models = [m for m in DEFAULT_MODELS if m not in excl]
    else:
        models = DEFAULT_MODELS

    if args.data:
        datasets = [args.data]
    else:
        datasets = ROBUSTNESS_DATASETS

    pred_lens_override = [args.pred_len] if args.pred_len > 0 else None
    seeds = [args.base_seed + i for i in range(args.num_seeds)]

    combos = []
    for m in models:
        for d in datasets:
            for h in (pred_lens_override or pred_lens_for(d)):
                for s in seeds:
                    combos.append((args.exp_tag, args.train_exp_tag, m, d,
                                   h, args.seq_len, s))
    return combos


def checkpoints_present(combos):
    have, miss = [], []
    for combo in combos:
        _, train_tag, m, d, h, L, s = combo
        ckpt = checkpoint_path(train_tag, m, d, h, L, s)
        (have if os.path.exists(ckpt) else miss).append(combo)
    return have, miss


def print_status(combos, done, missing_ckpts, csv_path, fault_types):
    n_total = len(combos)
    n_done = sum(1 for c in combos if c in done)
    n_no_ckpt = len(missing_ckpts)
    n_missing = n_total - n_done

    print("\n" + "=" * 76)
    print(f"  Robustness experiment status")
    print(f"  CSV: {csv_path}")
    print(f"  Faults: {','.join(fault_types)}  (expected rows/ckpt = "
          f"{expected_rows(fault_types)})")
    print(f"  Total requested: {n_total}  |  Done: {n_done}  "
          f"|  Missing: {n_missing}  |  No-checkpoint: {n_no_ckpt}")
    print("=" * 76)

    if n_no_ckpt:
        print(f"\n  [WARN] {n_no_ckpt} combos lack a trained checkpoint. "
              f"Run `python run_forecast.py --save_model "
              f"--exp_tag <train_exp_tag> ...` first for these:")
        for c in missing_ckpts[:8]:
            _, train_tag, m, d, h, L, s = c
            print(f"      {m:>15} | {d:>12} | H={h:<4} L={L} | seed={s} "
                  f"(train_exp_tag={train_tag})")
        if len(missing_ckpts) > 8:
            print(f"      ... ({len(missing_ckpts) - 8} more)")

    if n_missing - n_no_ckpt > 0:
        runnable = [c for c in combos if c not in done and c not in missing_ckpts]
        print(f"\n  Runnable now ({len(runnable)} combos):")
        for c in runnable[:12]:
            _, train_tag, m, d, h, L, s = c
            print(f"      {m:>15} | {d:>12} | H={h:<4} L={L} | seed={s}")
        if len(runnable) > 12:
            print(f"      ... ({len(runnable) - 12} more)")


def dispatch(combos, done, args, fault_types):
    method_order = {m: i for i, m in enumerate(DEFAULT_MODELS)}

    runnable = [c for c in combos if c not in done]
    runnable.sort(key=lambda c: (method_order.get(c[2], 999), c[3], c[4], c[6]))

    jobs = []
    for combo in runnable:
        _, train_tag, m, d, h, L, s = combo
        ckpt = checkpoint_path(train_tag, m, d, h, L, s)
        if not os.path.exists(ckpt):
            continue  # surfaced by print_status already

        label = f"{m} | {d} | H={h} | seed={s}"
        cmd = [
            sys.executable,
            "scripts/benchmarks/run_robustness.py",
            "--model", m,
            "--data", d,
            "--pred_len", str(h),
            "--seq_len", str(L),
            "--seed", str(s),
            "--train_exp_tag", train_tag,
            "--exp_tag", args.exp_tag,
            "--results_csv", args.results_csv,
            "--batch_size", str(args.batch_size),
            # FEATHer hyperparameters — needed to reconstruct architecture
            "--d_state", str(args.d_state),
            "--kernel_size", str(args.kernel_size),
            "--period", str(args.period),
            "--num_bands", str(args.num_bands),
        ]
        if args.fault_types:
            cmd += ["--fault_types", args.fault_types]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} job(s) exited non-zero; their rows "
              "stay missing — re-run to retry.")


def main():
    p = argparse.ArgumentParser(description="Robustness experiment orchestrator")
    p.add_argument("--model",   type=str, default=None)
    p.add_argument("--exclude", type=str, default=None)
    p.add_argument("--data",    type=str, default=None)
    p.add_argument("--pred_len", type=int, default=0)
    p.add_argument("--seq_len",  type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--num_seeds", type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--base_seed", type=int, default=DEFAULT_BASE_SEED)

    p.add_argument("--train_exp_tag", type=str, default=DEFAULT_TRAIN_EXP_TAG,
                   help="Tag the training run used (controls checkpoint lookup).")
    p.add_argument("--exp_tag",       type=str, default=DEFAULT_EXP_TAG,
                   help="Tag to attach to robustness CSV rows.")
    p.add_argument("--fault_types",   type=str, default="",
                   help="Comma-separated subset (empty = all four).")

    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index (with --ngpu N: the FIRST of N indices)")
    p.add_argument("--ngpu",        type=int, default=1,
                   help="Number of GPUs; N>1 = dynamic job queue across "
                        "GPUs gpu..gpu+N-1 (CF-JEPA-style load balancing)")
    p.add_argument("--jobs_per_gpu", type=int, default=1,
                   help="Concurrent workers per GPU (oversubscription for "
                        "tiny host-bound models; OMP threads auto-pinned)")

    # FEATHer architecture HPs (forwarded; harmless for others)
    p.add_argument("--d_state",     type=int, default=8)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--period",      type=int, default=12)
    p.add_argument("--num_bands",   type=int, default=3)

    p.add_argument("--check", action="store_true",
                   help="Print status without dispatching")
    args = p.parse_args()

    fault_types = ([s.strip() for s in args.fault_types.split(",") if s.strip()]
                   if args.fault_types else noise_mod.FAULT_TYPES)

    combos = enumerate_combos(args, fault_types)
    done = load_robust_set(args.results_csv, fault_types)
    _, missing_ckpts = checkpoints_present(combos)

    print_status(combos, done, missing_ckpts, args.results_csv, fault_types)

    if args.check:
        return

    if all(c in done for c in combos):
        print("\nNothing to run.")
        return

    dispatch(combos, done, args, fault_types)

    print("\n>>> Final status:")
    done_final = load_robust_set(args.results_csv, fault_types)
    _, missing_ckpts_final = checkpoints_present(combos)
    print_status(combos, done_final, missing_ckpts_final,
                 args.results_csv, fault_types)


if __name__ == "__main__":
    main()
