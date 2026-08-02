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
    python run_forecast.py --ngpu 2                      # dynamic 2-GPU queue

Multi-GPU: `--ngpu N` runs a CF-JEPA-style dynamic queue — one worker
subprocess per GPU (indices --gpu .. --gpu+N-1); a free GPU grabs the next
(model, dataset, pred_len) job, so fast models never leave a GPU idle.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (list_models, list_ablation_models,
                       get_method_defaults, get_dataset_overrides)
from utils.dispatch_queue import run_on_gpus


# -----------------------------------------------------------------------------
# Defaults — adjust here if the main sweep changes
# -----------------------------------------------------------------------------

DEFAULT_MODELS = list_models()  # all registered

# Manufacturing datasets — MAIN table (JMS pivot, 2026-06-25).
# 3 real continuous sensor streams, all D<=14 (sub-1K holds across the table).
# TEP dropped 2026-06-26 (anomaly-detection benchmark, not forecasting; D=50
# broke sub-1K). C-MAPSS replaces it but as a SEPARATE short-horizon section
# (CMAPSS_DATASETS) — its run-to-failure trajectories are too short for the
# [96..720] long horizons (see Dataset_CMAPSS / CMAPSS_PREDS).
# PMSM dropped 2026-07-02: at native 2Hz the [96..720] horizons span only
# 48s-6min of slow rotor-thermal drift — copy-last persistence already scores
# MSE 0.004 (target) / 0.117 (all-channel mean) at H=96, so the cells are
# trivially easy; a multi-session+downsample rebuild would only support short
# horizons anyway. Data + prep recipe kept on disk like TEP.
# BlastFurnace added 2026-07-13 (Mendeley 6d7jbc7tb5, Data in Brief 2025):
# 27 hourly process channels + Si (hot-metal silicon = product-quality target,
# asof-merged). D=28 -> above sub-1K, so it is the manufacturing STRESS/breadth
# tier (like Weather/Electricity in LTSF); the sub-1K claim stays on the D<=14
# core (Steel/WindSCADA/MetroPT3/N2O/EWELD). Forecasting-native, gapless, no
# LTSF precedent -> "first to cast as edge-PdM LTSF" holds.
# MetroPT3 added 2026-07-13 (UCI #791, metro-train APU compressor): 7 analog
# channels @ 1-min mean, D=7 -> sub-1K CORE. Persistence-audited non-trivial
# (0.80-1.14 std). Native anomaly-detection set cast as forecasting (TIME
# precedent). Adds an air-compressor PdM domain to the manufacturing lineup.
# N2O added 2026-07-13 (Avedore WWTP, Mendeley xmbxhscgpr): 14 continuous
# process channels @ 5-min mean, D=14 -> sub-1K core. Persistence-audited
# non-trivial (1.10-1.53 std). Forecasting-native. Adds a continuous
# industrial-process (bio-chemical) domain; frame as "industrial process
# monitoring" (it is a utility process, not discrete manufacturing).
# Manufacturing MAIN table — FINAL 6 (locked 2026-07-14). All real, continuous,
# non-trivial, domain-diverse: Steel (energy), WindSCADA (wind, forecasting
# precedent), BlastFurnace (iron process, gapless), N2O (WWTP process),
# MetroPT3 (compressor machine-sensor), EWELD (factory load — 6-yr ANNUAL cycle
# showcases the multi-band head + labeled extreme-weather windows = real
# distribution-shift robustness). GasTurbine fully REMOVED 2026-07-14 (data +
# loader + prep deleted — data-verified discontinuity: ~15% hours missing/year,
# synthetic index mixes non-adjacent real times = broken sliding window; rejected
# as BROKEN, not deferred-shelf). Vetted+rejected likewise: TEP/Machine-health/
# AI4I (synthetic), Gas-Sensor (mediocre lab). CMAPSS/CMAPSS3/PMSM = PdM shelf
# (good data, deferred section — kept).
MFG_DATASETS = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
MFG_PREDS = [96, 192, 336, 720]

# 2026-07-27: UNIFIED dispatch-order priority across mfg AND LTSF — cheap
# datasets run first, expensive ones last, so the fast datasets finish first
# and the slow tail doesn't block partial results. Same rule for every group
# (mfg / ltsf / all) so the two sweeps behave identically. Tier ~ dispatch
# cost ≈ rows × channels (a rough wall-clock proxy). Lower = runs earlier.
# CMAPSS shelf datasets aren't listed → default tier (5); they only run in the
# isolated `--group cmapss`, where they simply tie and keep (model, name) order.
_DATA_DISPATCH_ORDER = {
    # ---- cheap (small rows × few channels) ----
    "Exchange":     0,   # 7.6k × 8
    "ETTh1":        1,   # 17k  × 7
    "ETTh2":        1,   # 17k  × 7
    "Steel":        2,   # 35k  × 7
    "ETTm1":        3,   # 70k  × 7
    "ETTm2":        3,   # 70k  × 7
    "WindSCADA":    4,   # 52k  × 15
    "BlastFurnace": 4,   # 30k  × 29
    "Weather":      5,   # 53k  × 21
    "EWELD":        6,   # 105k × 15
    # ---- expensive (big rows, or huge channel counts) ----
    "MetroPT3":     7,   # 306k × 8
    "N2O":          7,   # 210k × 15
    "Electricity":  8,   # 26k  × 321  (channel-heavy)
    "Traffic":      9,   # 17k  × 862  (channel-heaviest)
}

# 2026-07-31: MODEL dispatch priority — the lightweight edge-tier (sub-1K to
# ~18K, our main comparison table) runs BEFORE the heavier ceiling models, so
# the main table fills first and the slow ceiling (Pareto reference) doesn't
# block partial results. Ordered by param count (H=96, D=7). Lower = earlier.
_MODEL_DISPATCH_ORDER = {name: i for i, name in enumerate([
    # edge tier (main table) — lightest first
    "SparseTSF", "MixLinear", "DiPE_Linear", "FEATHer", "FITS", "CycleNet", "DLinear",
    # ceiling tier (Pareto reference) — heaviest last
    "PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet", "iTransformer", "MDMLP_EIA",
])}

# ARCHIVED 2026-07-13 — short-horizon PdM section DROPPED from the paper.
# Rationale: it was the one experiment block that attached to neither the
# long-term "frequency forecaster" identity (it is short-horizon, where the
# multi-scale/FFT inductive bias is weakest) nor the robustness param-
# justification (its anchors are Steel/GT/Wind/ETTh1/Weather), AND it created a
# grid-asymmetry attack surface ("why short horizons only here?"). Dropping it
# unifies every dataset to [96..720] and keeps the identity sharp. The loader
# (Dataset_CMAPSS), prep recipes, and lr search (run_lr_search) are KEPT as a
# shelf so a reviewer-driven revision can restore the section immediately.
# Still reachable via `--group cmapss`; just out of the default + `all`.
#   CMAPSS = C-MAPSS FD001 · CMAPSS3 = FD003 · PMSM = Paderborn motor (69-sess)
CMAPSS_DATASETS = ["CMAPSS", "CMAPSS3", "PMSM"]   # shelf (archived from paper)
CMAPSS_PREDS = [24, 48, 96]

# Long-term horizon datasets — now the GENERALIZATION section (demoted from
# main when the paper pivoted to manufacturing).
LTSF_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
                 "Traffic", "Exchange"]
LTSF_PREDS = [96, 192, 336, 720]

DEFAULT_BASE_SEED = 2025
DEFAULT_NUM_SEEDS = 5
DEFAULT_EXP_TAG = "main"
DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/fcst_results.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pred_lens_for(dataset):
    if dataset in MFG_DATASETS:
        return MFG_PREDS
    if dataset in CMAPSS_DATASETS:
        return CMAPSS_PREDS
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
    elif args.group == "ltsf":
        datasets = LTSF_DATASETS      # generalization section
    elif args.group == "cmapss":
        datasets = CMAPSS_DATASETS    # separate short-horizon PdM section
    elif args.group == "all":
        # CMAPSS archived from the paper (2026-07-13); use --group cmapss to reach it.
        datasets = MFG_DATASETS + LTSF_DATASETS
    else:
        datasets = MFG_DATASETS       # default: manufacturing MAIN table

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
    # Sort order: small datasets first (_DATA_DISPATCH_ORDER), then within a
    # data tier the lightweight edge models before the heavy ceiling models
    # (_MODEL_DISPATCH_ORDER) so the main table fills first and the slow
    # ceiling×big-data jobs land last without blocking partial results.
    keys_sorted = sorted(groups.keys(),
                         key=lambda k: (_DATA_DISPATCH_ORDER.get(k[2], 5),
                                        _MODEL_DISPATCH_ORDER.get(k[1], 999), k[2], k[3]))

    jobs = []
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

        # AUDIT-FIX (2026-07-14, P2-4): the worker applies --model_overrides
        # AFTER parse_args, so an override key silently beats an explicit CLI
        # arch flag (e.g. `--d_state 4` on ETTh1 used to train d16 while the
        # CSV recorded nothing). Explicitly-set FEATHer arch flags (default
        # None = unset) are therefore DROPPED from the override dict and
        # forwarded as worker CLI flags instead — CLI wins, sweeps unchanged.
        explicit_arch = {k: v for k, v in {
            "d_state":     args.d_state,
            "kernel_size": args.kernel_size,
            "period":      args.period,
            "num_bands":   args.num_bands,
            "lambda_spec": args.lambda_spec,
        }.items() if v is not None}

        hp_overrides = {k: v for k, v in merged.items()
                        if k not in ("lr", "loss") and k not in explicit_arch}
        overrides_str = ";".join(f"{k}={v}" for k, v in sorted(hp_overrides.items()))

        label = (f"{m} | {d} | H={h} | seeds={seeds} | lr={lr_to_use} | "
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
        ]
        if args.train_frac is not None:
            cmd += ["--train_frac", str(args.train_frac),
                    "--train_frac_mode", args.train_frac_mode]
        no_save = {x.strip() for x in (args.no_save_data or "").split(",") if x.strip()}
        if args.save_model and d not in no_save:
            cmd += ["--save_model"]
        if overrides_str:
            cmd += ["--model_overrides", overrides_str]
        # FEATHer arch flags: forward ONLY the explicitly-set ones (P2-4).
        # Unset flags fall back to the worker's own defaults (d8/k7/p12/B3/
        # λ0.01 — identical values), still overridable per dataset above.
        for k, v in explicit_arch.items():
            cmd += [f"--{k}", str(v)]
        # Capacity A/B: forward only when enabled (canonical stays byte-identical).
        if args.use_residual_mlp:
            cmd += ["--use_residual_mlp"]
            if args.mlp_rank is not None:
                cmd += ["--mlp_rank", str(args.mlp_rank)]
        if args.no_revin:
            cmd += ["--no_revin"]
        if args.gate_uniform:
            cmd += ["--gate_uniform"]
        if args.robust_revin:
            cmd += ["--robust_revin"]
        if args.revin_mode:
            cmd += ["--revin_mode", args.revin_mode]
        if args.winsor_bound is not None:
            cmd += ["--winsor_bound", str(args.winsor_bound)]
        if args.center_trim is not None:
            cmd += ["--center_trim", str(args.center_trim)]
        if args.channel_mode:
            cmd += ["--channel_mode", args.channel_mode]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; their runs "
              "stay missing — re-run to retry.")


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
                   help="Single dataset; omit to sweep a --group.")
    p.add_argument("--group",   type=str, default="mfg",
                   choices=["mfg", "ltsf", "cmapss", "all"],
                   help="Dataset group when --data is omitted: mfg (6 "
                        "manufacturing, MAIN table [96..720]; default), cmapss "
                        "(C-MAPSS short-horizon PdM section [24,48,96]), ltsf "
                        "(8 LTSF generalization), or all.")
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
    # Data-efficiency / cold-start: train on only `train_frac` of the train
    # split (mode=recent -> most recent, the 'new line, limited history' case).
    # Forwarded to the worker. Use a distinct --exp_tag per fraction.
    p.add_argument("--train_frac", type=float, default=None,
                   help="fraction (0,1] of the train split (data-efficiency sweep)")
    p.add_argument("--train_frac_mode", default="recent", choices=["recent", "first"])
    # FEATHer arch flags default to None (= unset) so dispatch can tell an
    # explicit user value from the default: explicit values BEAT
    # _DATASET_OVERRIDES (audit P2-4); unset falls back to per-dataset
    # override, then the worker default (d8/k7/p12/B3/lambda 0.01).
    p.add_argument("--d_state",     type=int,   default=None)
    p.add_argument("--kernel_size", type=int,   default=None)
    p.add_argument("--period",      type=int,   default=None)
    p.add_argument("--num_bands",   type=int,   default=None)
    p.add_argument("--lambda_spec", type=float, default=None)
    # Capacity A/B (2026-07-31): forwarded to the worker only when set, so the
    # canonical sweep is byte-identical when omitted.
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--mlp_rank",    type=int,   default=None)
    # Robustness diagnostics (2026-07-31): forwarded to worker only when set.
    p.add_argument("--no_revin",     action="store_true")
    p.add_argument("--gate_uniform", action="store_true")
    p.add_argument("--robust_revin", action="store_true")
    p.add_argument("--revin_mode", type=str, default=None,
                   choices=["standard", "mad", "winsor", "center"])
    p.add_argument("--winsor_bound", type=float, default=None)
    p.add_argument("--center_trim",  type=float, default=None)
    p.add_argument("--channel_mode", type=str, default=None,
                   choices=["mix", "ci_full"])

    # Output / system
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index (with --ngpu N: the FIRST of N indices)")
    p.add_argument("--ngpu",        type=int, default=1,
                   help="Number of GPUs; N>1 = dynamic job queue across "
                        "GPUs gpu..gpu+N-1 (CF-JEPA-style load balancing)")
    p.add_argument("--jobs_per_gpu", type=int, default=1,
                   help="Concurrent workers per GPU (oversubscription for "
                        "tiny host-bound models; OMP threads auto-pinned "
                        "to cpu_count/total_workers)")
    p.add_argument("--save_model",  action="store_true",
                   help="Save checkpoints (forwarded to worker).")
    p.add_argument("--no_save_data", type=str, default="",
                   help="Comma-separated datasets to train+score normally "
                        "but skip --save_model for. Default '' saves every "
                        "dataset (~42GB). Pass 'Traffic' to skip its unused "
                        "checkpoints (~24GB of TimesNet weights; Traffic is "
                        "not a robustness dataset).")

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
