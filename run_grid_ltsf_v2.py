"""FEATHer LTSF v2 arch grid — MLP + MSE aware, kernel>=3 (identity-preserving).

Why v2 (vs _archive/run_grid_search.py):
  The frozen per-dataset LTSF configs in _DATASET_OVERRIDES were OFAT-tuned under
  (loss=L1, NO residual-MLP). The shipping FEATHer now trains under (loss=MSE,
  residual-MLP ON). That double mismatch means the frozen arch is likely
  suboptimal for the current setup. This grid re-selects the arch PER DATASET
  under the correct (MSE + MLP) conditions, on the validation loss.

Key differences from the archived grid:
  - FIXED loss = "mse"  (was l1)
  - every run passes --use_residual_mlp --mlp_rank 16  (MLP on, like shipping v2)
  - kernel_size grid = {3,5,7,9}  (drops kernel=1: a pointwise DTK does NO
    temporal mixing => FEATHer collapses to a near-linear model and "wins" by
    abandoning its own inductive bias. kernel>=3 keeps the DTK genuinely
    convolutional so a grid win is a REAL FEATHer win.)
  - d_state {4,8,16} and period {6,8,12,24}: dropped the degenerate-tiny corners
    (d_state=2, period in {2,4}) for the same identity-preserving reason.
  - isolated CSV (results/grid_ltsf_v2.csv) + tag prefix "gridv2_" so it never
    collides with the old grid_* rows.

Selection = best VALIDATION loss per dataset (test read once, reference only).
Standard cheap-search pattern: run with --seeds 2025 (1 seed) to SELECT, then
re-run the winning per-dataset configs with 3 seeds to finalize.

Usage:
    python run_grid_ltsf_v2.py --check --data ETTh1
    python run_grid_ltsf_v2.py --data ETTh1 --seeds 2025 --ngpu 2 --jobs_per_gpu 3
    python run_grid_ltsf_v2.py --summary --data ETTh1
Windows/PowerShell: run --summary with $env:PYTHONUTF8=1 (em-dash vs cp949).
"""

import argparse
import os
import sys
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dispatch_queue import run_on_gpus  # noqa: E402

WORKER = os.path.join("scripts", "benchmarks", "run_forecast.py")

# -----------------------------------------------------------------------------
# Identity-preserving grid. kernel>=3 (no pointwise collapse); d_state/period
# trimmed of degenerate-tiny corners. lambda_spec + lr held fixed to isolate
# arch interactions under the shipping (MSE + MLP) regime.
# -----------------------------------------------------------------------------
GRID = {
    "d_state":     [4, 8, 16],
    "kernel_size": [3, 5, 7, 9],          # kernel>=3: DTK stays convolutional
    "period":      [6, 8, 12, 24],        # divide gcd(96,192,336,720)=48
    "num_bands":   [2, 3, 4],
}
FIXED = {"lambda_spec": 0.01, "lr": 1e-3, "loss": "mse"}   # <-- mse, not l1
MLP = {"use_residual_mlp": True, "mlp_rank": 16}           # <-- MLP on, like v2

BASE_CONFIG = {"d_state": 8, "kernel_size": 7, "period": 12, "num_bands": 3}

GRID_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
# Electricity/Traffic excluded by default: 321/862 channels blow FEATHer past
# sub-1K (D^2 in_proj) and are not its regime — scope them in explicitly with
# --data if wanted, but expect them to stay weak.
GRID_PREDS = [96, 192, 336, 720]
GRID_SEEDS = [2025, 2026, 2027]


def _tag(cfg):
    return "gridv2_ds{d_state}_k{kernel_size}_p{period}_B{num_bands}".format(**cfg)


def enumerate_configs():
    keys = list(GRID)
    for values in product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, values))
        yield _tag(cfg), cfg


def enumerate_runs(args):
    runs = []
    for exp_tag, cfg in enumerate_configs():
        for d in args.datasets:
            for h in args.preds:
                if args.seq_len % cfg["period"] or h % cfg["period"]:
                    continue  # SPK: period must divide seq_len AND pred_len
                for s in args.seed_list:
                    runs.append((exp_tag, cfg, d, h, args.seq_len, s))
    return runs


def load_done_set(csv_path):
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    return set(
        (str(r.exp_tag), str(r.data), int(r.pred_len), int(r.seq_len), int(r.seed))
        for r in df[["exp_tag", "data", "pred_len", "seq_len", "seed"]]
            .itertuples(index=False)
    )


def dispatch(missing, args):
    groups = {}
    for exp_tag, cfg, d, h, sl, s in missing:
        groups.setdefault((exp_tag, d, h, sl), (cfg, []))[1].append(s)

    jobs = []
    for (exp_tag, d, h, sl), (cfg, seeds) in sorted(groups.items()):
        label = f"{exp_tag} | {d} | H={h} | seeds={sorted(seeds)}"
        cmd = [
            sys.executable, WORKER,
            "--model", "FEATHer",
            "--data", d,
            "--pred_len", str(h),
            "--seq_len", str(sl),
            "--seeds", ",".join(str(s) for s in sorted(seeds)),
            "--exp_tag", exp_tag,
            "--results_csv", args.results_csv,
            "--num_epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--loss", FIXED["loss"],
            "--lr", str(FIXED["lr"]),
            "--d_state", str(cfg["d_state"]),
            "--kernel_size", str(cfg["kernel_size"]),
            "--period", str(cfg["period"]),
            "--num_bands", str(cfg["num_bands"]),
            "--lambda_spec", str(FIXED["lambda_spec"]),
            "--mlp_rank", str(MLP["mlp_rank"]),
        ]
        if MLP["use_residual_mlp"]:
            cmd.append("--use_residual_mlp")
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; their runs "
              "stay missing — re-run to retry.")


def _print_ranking(df, d, preds, seed_list, sort_by, label, base_tag):
    sub = df[(df.data == d) & (df.pred_len.isin(preds))]
    if sub.empty:
        return
    g = sub.groupby("exp_tag").agg(
        val=("val_loss", "mean"), mse=("MSE", "mean"),
        n=("seed", "count")).reset_index()
    full = len(preds) * len(seed_list)
    g["complete"] = g["n"] >= full
    gc = g[g["complete"]]
    n_prog = int((~g["complete"]).sum())
    sort_col = "mse" if sort_by == "MSE" else "val"
    if sort_by == "MSE":
        print(f"\n--- {label} ---  [!] sorted by TEST MSE = ORACLE / test-peek — "
              f"VIEW ONLY, never select on this.  ({n_prog} config(s) in progress)")
    else:
        print(f"\n--- {label} ---  (best-VAL ranking, complete configs only; "
              f"test MSE reference; {n_prog} config(s) in progress)")
    if gc.empty:
        print(f"  (no config has all {full} run(s) yet — still early)")
        return
    for _, r in gc.sort_values(sort_col).head(8).iterrows():
        star = "  <== base(canonical)" if r.exp_tag == base_tag else ""
        print(f"  {r.exp_tag:<28} val={r.val:.4f}  testMSE={r.mse:.4f}{star}")
    best = gc.sort_values("val").head(1)
    base = gc[gc.exp_tag == base_tag]
    if not best.empty and not base.empty:
        b, bs = best.iloc[0], base.iloc[0]
        delta = bs["val"] - b["val"]
        verdict = ("grid BEATS base" if delta > 0 else "base ties/wins")
        print(f"  => best={b.exp_tag} val={b['val']:.4f} vs "
              f"base val={bs['val']:.4f}  (Δval={delta:+.4f}, {verdict})")


def summarize(args):
    if not os.path.exists(args.results_csv):
        print(f"No results yet in {args.results_csv}.")
        return
    df = pd.read_csv(args.results_csv)
    df = df[df.exp_tag.str.startswith("gridv2_")]
    if df.empty:
        print("No gridv2_* rows yet.")
        return
    base_tag = _tag(BASE_CONFIG)
    for d in args.datasets:
        if args.per_horizon:
            for h in args.preds:
                _print_ranking(df, d, [h], args.seed_list, args.sort_by,
                               f"{d} H={h}", base_tag)
        else:
            _print_ranking(df, d, args.preds, args.seed_list, args.sort_by,
                           d, base_tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="",
                   help="comma-list to scope datasets (default: 6 low-channel LTSF)")
    p.add_argument("--pred_len", type=str, default="",
                   help="comma-list to scope horizons (default: 96,192,336,720)")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in GRID_SEEDS),
                   help="comma-list of seeds (use --seeds 2025 for fast 1-seed "
                        "selection grid; finalize winners with 3 seeds after)")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_csv", type=str, default="results/grid_ltsf_v2.csv")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--sort_by", choices=["val", "MSE"], default="val",
                   help="--summary display sort: val (selection metric, default) "
                        "or MSE (TEST/oracle — view only, never select on it)")
    p.add_argument("--per_horizon", action="store_true",
                   help="--summary: separate ranking per horizon in one command")
    args = p.parse_args()

    args.datasets = (args.data.split(",") if args.data else GRID_DATASETS)
    args.preds = ([int(x) for x in args.pred_len.split(",")]
                  if args.pred_len else GRID_PREDS)
    args.seed_list = [int(x) for x in args.seeds.split(",")]

    if args.summary:
        summarize(args)
        return

    runs = enumerate_runs(args)
    n_cfg = len(list(enumerate_configs()))
    done = load_done_set(args.results_csv)
    missing = [r for r in runs
               if (r[0], r[2], r[3], r[4], r[5]) not in done]
    print(f"Grid v2 (MSE+MLP, kernel>=3): {n_cfg} configs x "
          f"{len(args.datasets)} datasets x {len(args.preds)} horizons x "
          f"{len(args.seed_list)} seeds")
    print(f"Total runs: {len(runs)} | Done: {len(runs) - len(missing)} "
          f"| Missing: {len(missing)}")
    if args.check:
        return
    if not missing:
        print("Nothing to run.")
        return
    dispatch(missing, args)


if __name__ == "__main__":
    main()
