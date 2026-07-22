"""FEATHer joint (full-grid) arch HP search — the interaction-aware upgrade
over the OFAT search in run_hp_search.py.

OFAT varies one axis at a time around a base config, so it never trains
combinations like (d_state=16 AND kernel=9 AND period=8) that may only pay off
jointly. This orchestrator sweeps the FULL Cartesian product of the arch axes
and selects the best config PER DATASET on the validation loss (test read once,
for reference only — no test-set selection).

Same infra as run_hp_search.py: resumable (skips CSV rows already present),
`--check`, `--summary`, dispatch via utils.dispatch_queue.run_on_gpus.

Usage:
    python run_grid_search.py --check --data ETTh1 --pred_len 96
    python run_grid_search.py --data ETTh1 --pred_len 96 --jobs_per_gpu 3
    python run_grid_search.py --summary --data ETTh1
Windows/PowerShell note: run --summary with $env:PYTHONUTF8=1 (em-dash vs cp949).
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
# Joint grid — full product of the arch axes. lambda_spec (OFAT-flat) and lr
# are held FIXED here so the grid isolates arch interactions; add them to GRID
# to widen the search (grid size = product of all list lengths).
# -----------------------------------------------------------------------------
GRID = {
    "d_state":     [4, 8, 16],
    "kernel_size": [3, 5, 7, 9],
    "period":      [6, 8, 12, 24],   # must divide seq_len AND pred_len
    "num_bands":   [2, 3, 4],
}
FIXED = {"lambda_spec": 0.01, "lr": 1e-3, "loss": "l1"}

BASE_CONFIG = {"d_state": 8, "kernel_size": 7, "period": 12, "num_bands": 3}

GRID_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2",
                 "Weather", "Exchange", "Electricity", "Traffic"]
GRID_PREDS = [96, 720]
GRID_SEEDS = [2025, 2026, 2027]


def _tag(cfg):
    return "grid_ds{d_state}_k{kernel_size}_p{period}_B{num_bands}".format(**cfg)


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
                # SPK constraint: period must divide both lengths.
                if args.seq_len % cfg["period"] or h % cfg["period"]:
                    continue
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
        ]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; their runs "
              "stay missing — re-run to retry.")


def summarize(args):
    if not os.path.exists(args.results_csv):
        print(f"No results yet in {args.results_csv}.")
        return
    df = pd.read_csv(args.results_csv)
    df = df[df.exp_tag.str.startswith("grid_")]
    if df.empty:
        print("No grid_* rows yet.")
        return
    base_tag = _tag(BASE_CONFIG)
    for d in args.datasets:
        sub = df[df.data == d]
        if sub.empty:
            continue
        # Per-config mean over horizons+seeds, plus completeness.
        g = sub.groupby("exp_tag").agg(
            val=("val_loss", "mean"), mse=("MSE", "mean"),
            n=("seed", "count")).reset_index()
        full = len(args.preds) * len(args.seed_list)
        g["complete"] = g["n"] >= full
        g = g.sort_values("val")
        print(f"\n--- {d} ---  (best-val ranking; test MSE reference only)")
        for _, r in g.head(8).iterrows():
            flag = "" if r.complete else f"  (incomplete {int(r.n)}/{full})"
            star = "  <== base" if r.exp_tag == base_tag else ""
            print(f"  {r.exp_tag:<26} val={r.val:.4f}  testMSE={r.mse:.4f}{flag}{star}")
        best = g[g.complete].head(1)
        base = g[g.exp_tag == base_tag]
        if not best.empty and not base.empty:
            b, bs = best.iloc[0], base.iloc[0]
            delta = bs["val"] - b["val"]
            verdict = ("grid BEATS base" if delta > 0 else "base ties/wins")
            print(f"  => best={b.exp_tag} val={b['val']:.4f} vs "
                  f"base val={bs['val']:.4f}  (Δval={delta:+.4f}, {verdict})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="",
                   help="comma-list to scope datasets (default: all 8 LTSF)")
    p.add_argument("--pred_len", type=str, default="",
                   help="comma-list to scope horizons (default: 96,720)")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in GRID_SEEDS),
                   help="comma-list of seeds (default 3; use e.g. --seeds 2025 "
                        "for a fast 1-seed selection grid)")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_csv", type=str, default="results/grid_search.csv")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--summary", action="store_true")
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
    print(f"Grid: {n_cfg} configs x {len(args.datasets)} datasets x "
          f"{len(args.preds)} horizons x {len(args.seed_list)} seeds")
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
