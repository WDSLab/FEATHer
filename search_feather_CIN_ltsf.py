"""FEATHer-CIN (channel_mode=ci_full + residual MLP) LTSF arch grid search.

Purpose: per-dataset best config for the LTSF headline model FEATHer-CIN.
Unlike mfg (one canonical, symmetric protocol), LTSF ALLOWS per-dataset tuning
(baselines tune per-dataset too), so the val-selected per-dataset config is the
REPORTED number, not just a ceiling. Trains with MSE (LTSF headline = MSE;
CIN's MSE-trained test-MSE < MAE-trained). period = per-dataset FFT
(get_dataset_overrides). CIN params are channel-independent -> sub-1K at any D,
so the ceiling is deployable throughout (no sub-1K caveat).

Grid (around canonical): d_state{4,8,16} x kernel{3,5,7} x num_bands{2,3,4} = 27,
small-config-first. k=1 excluded (pointwise DTK = arch-degenerate). mlp_rank
fixed 16, lambda_spec 0.01, lr 1e-3. channel_mode=ci_full + --use_residual_mlp
== the CIN cell.

    python search_feather_CIN_ltsf.py --check
    python search_feather_CIN_ltsf.py --ngpu 2 --jobs_per_gpu 4      # 27 x 6 LTSF x 4H x 1seed
    python search_feather_CIN_ltsf.py --data ETTh1,ETTm1            # subset first
    python search_feather_CIN_ltsf.py --summary                     # per-dataset val-selected + ceiling

Isolated CSV: results/feather_CIN_grid.csv (CINgrid_* tags).
"""

import argparse
import os
import sys
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dispatch_queue import run_on_gpus          # noqa: E402
from baselines import get_dataset_overrides           # noqa: E402

WORKER = os.path.join("scripts", "benchmarks", "run_forecast.py")

GRID = {
    "d_state":     [4, 8, 16],
    "kernel_size": [3, 5, 7],       # k=1 excluded (arch-degenerate)
    "num_bands":   [2, 3, 4],
}
FIXED = {"lambda_spec": 0.01, "lr": 1e-3, "mlp_rank": 16}
BASE_CONFIG = {"d_state": 8, "kernel_size": 7, "num_bands": 3}

LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
SEARCH_PREDS = [96, 192, 336, 720]
SEARCH_SEEDS = [2025]                # 1 seed (selection tool)


def _tag(cfg):
    return "CINgrid_ds{d_state}_k{kernel_size}_B{num_bands}".format(**cfg)


def _period(dataset):
    return int(get_dataset_overrides("FEATHer", dataset).get("period", 12))


def enumerate_configs():
    keys = list(GRID)
    cfgs = [dict(zip(keys, v)) for v in product(*(GRID[k] for k in keys))]
    cfgs.sort(key=lambda c: (c["d_state"], c["num_bands"], c["kernel_size"]))
    for cfg in cfgs:
        yield _tag(cfg), cfg


def enumerate_runs(args):
    runs = []
    for exp_tag, cfg in enumerate_configs():
        for d in args.datasets:
            p = _period(d)
            for h in args.preds:
                if args.seq_len % p or h % p:     # SPK: period | seq_len AND pred_len
                    continue
                for s in args.seed_list:
                    runs.append((exp_tag, cfg, d, h, p, args.seq_len, s))
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
    for exp_tag, cfg, d, h, p, sl, s in missing:
        groups.setdefault((exp_tag, d, h, p, sl), (cfg, []))[1].append(s)

    jobs = []
    for (exp_tag, d, h, p, sl), (cfg, seeds) in sorted(groups.items()):
        label = f"{exp_tag} | {d} | H={h} | p={p} | seeds={sorted(seeds)}"
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
            "--loss", args.loss,
            "--lr", str(FIXED["lr"]),
            "--d_state", str(cfg["d_state"]),
            "--kernel_size", str(cfg["kernel_size"]),
            "--period", str(p),
            "--num_bands", str(cfg["num_bands"]),
            "--lambda_spec", str(FIXED["lambda_spec"]),
            # === FEATHer-CIN cell ===
            "--channel_mode", "ci_full",
            "--use_residual_mlp",
            "--mlp_rank", str(FIXED["mlp_rank"]),
        ]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; re-run to retry.")


def summarize(args):
    if not os.path.exists(args.results_csv):
        print(f"No results yet in {args.results_csv}.")
        return
    df = pd.read_csv(args.results_csv)
    df = df[df.exp_tag.str.startswith("CINgrid_")]
    if df.empty:
        print("No CINgrid_* rows yet.")
        return
    base_tag = _tag(BASE_CONFIG)
    full = len(args.preds) * len(args.seed_list)

    print("\n########## per-dataset best (LTSF = val-selected is REPORTED) ##########")
    for d in args.datasets:
        sub = df[(df.data == d) & (df.pred_len.isin(args.preds))]
        if sub.empty:
            continue
        g = sub.groupby("exp_tag").agg(
            val=("val_loss", "mean"), mse=("MSE", "mean"),
            par=("num_params", "max"), n=("MSE", "size")).reset_index()
        gc = g[g["n"] >= full].copy()
        n_prog = int((g["n"] < full).sum())
        print(f"\n--- {d}  ({len(gc)} complete / {n_prog} in progress) ---")
        if gc.empty:
            print(f"  (no config complete over all {full} run(s) yet)")
            continue
        valpick = gc.sort_values("val").iloc[0]      # REPORTED
        ceil = gc.sort_values("mse").iloc[0]          # diagnostic
        brow = gc[gc.exp_tag == base_tag]
        print(f"  val-selected(REPORTED): {valpick.exp_tag.replace('CINgrid_',''):<12} "
              f"testMSE={valpick.mse:.4f}  par={int(valpick.par)}")
        print(f"  ceiling(testMSE)      : {ceil.exp_tag.replace('CINgrid_',''):<12} "
              f"testMSE={ceil.mse:.4f}  par={int(ceil.par)}")
        if not brow.empty:
            print(f"  base d8/k7/B3         : testMSE={brow.iloc[0].mse:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="")
    p.add_argument("--pred_len", type=str, default="")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEARCH_SEEDS))
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--loss", type=str, default="mse", choices=["l1", "mse"])
    p.add_argument("--results_csv", type=str, default="results/feather_CIN_grid.csv")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    args.datasets = args.data.split(",") if args.data else LTSF
    args.preds = ([int(x) for x in args.pred_len.split(",")]
                  if args.pred_len else SEARCH_PREDS)
    args.seed_list = [int(x) for x in args.seeds.split(",")]

    if args.summary:
        summarize(args)
        return

    runs = enumerate_runs(args)
    done = load_done_set(args.results_csv)
    missing = [r for r in runs if (r[0], r[2], r[3], r[5], r[6]) not in done]
    n_cfg = sum(1 for _ in enumerate_configs())
    print(f"grid: {n_cfg} configs x {len(args.datasets)} datasets x "
          f"{len(args.preds)} H x {len(args.seed_list)} seed")
    print(f"total {len(runs)} runs | done {len(runs)-len(missing)} | missing {len(missing)}")
    print(f"CSV: {args.results_csv}")
    if args.check:
        return
    dispatch(missing, args)


if __name__ == "__main__":
    main()
