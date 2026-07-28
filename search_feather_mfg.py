"""FEATHer mfg canonical arch search (LOCAL, throwaway — NOT the committed
run_grid_search.py which is LTSF/full-360/period-sweep).

Focused arch grid on a few FAST mfg datasets to (a) see the per-dataset best
config [diagnostic + one-config-gap appendix] and (b) pick a single CANONICAL
(d_state,kernel,num_bands) by mean-rank across datasets on VAL — applied
uniformly to all mfg (symmetric with baselines' fixed defaults). period is NOT
searched: it is the per-dataset FFT value (get_dataset_overrides), same rule as
all period-models. lr/lambda/loss fixed (mfg loss-flat wrt lr, OFAT-flat lambda).

k=1 excluded on purpose (pointwise DTK = arch-degenerate; the LTSF grid showed
k=1 wins only by killing the temporal kernel — not a canonical we want).

    python search_feather_mfg.py --check
    python search_feather_mfg.py --ngpu 1 --jobs_per_gpu 4
    python search_feather_mfg.py --summary          # per-dataset best + mean-rank canonical
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

# Focused arch grid around canonical d8/k7/B3. period handled per-dataset (FFT).
GRID = {
    "d_state":     [4, 8, 16],
    "kernel_size": [3, 5, 7],      # k=1 excluded (arch-degenerate)
    "num_bands":   [2, 3, 4],
}
FIXED = {"lambda_spec": 0.01, "lr": 1e-3, "loss": "l1"}
BASE_CONFIG = {"d_state": 8, "kernel_size": 7, "num_bands": 3}

SEARCH_DATASETS = ["Steel", "WindSCADA", "EWELD"]   # fast + robustness-relevant
SEARCH_PREDS = [96, 192, 336, 720]
SEARCH_SEEDS = [2025]                                # 1 seed (selection tool)


def _tag(cfg):
    return "search_ds{d_state}_k{kernel_size}_B{num_bands}".format(**cfg)


def _period(dataset):
    """Per-dataset FFT period (from _DATASET_OVERRIDES); fallback 12."""
    return int(get_dataset_overrides("FEATHer", dataset).get("period", 12))


def enumerate_configs():
    keys = list(GRID)
    for values in product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, values))
        yield _tag(cfg), cfg


def enumerate_runs(args):
    runs = []
    for exp_tag, cfg in enumerate_configs():
        for d in args.datasets:
            p = _period(d)
            for h in args.preds:
                # SPK constraint: period must divide seq_len AND pred_len.
                if args.seq_len % p or h % p:
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
            "--loss", FIXED["loss"],
            "--lr", str(FIXED["lr"]),
            "--d_state", str(cfg["d_state"]),
            "--kernel_size", str(cfg["kernel_size"]),
            "--period", str(p),
            "--num_bands", str(cfg["num_bands"]),
            "--lambda_spec", str(FIXED["lambda_spec"]),
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
    df = df[df.exp_tag.str.startswith("search_")]
    if df.empty:
        print("No search_* rows yet.")
        return
    base_tag = _tag(BASE_CONFIG)
    full = len(args.preds) * len(args.seed_list)

    # ---- per-dataset ranking (diagnostic view) ----
    rank_frames = []
    for d in args.datasets:
        sub = df[(df.data == d) & (df.pred_len.isin(args.preds))]
        if sub.empty:
            continue
        g = sub.groupby("exp_tag").agg(
            val=("val_loss", "mean"), mse=("MSE", "mean"),
            par=("num_params", "max"), n=("seed", "count")).reset_index()
        gc = g[g["n"] >= full].copy()
        n_prog = int((g["n"] < full).sum())
        print(f"\n--- {d}  (best-VAL, complete configs; {n_prog} in progress) ---")
        if gc.empty:
            print(f"  (no config has all {full} run(s) yet)")
            continue
        gc = gc.sort_values("val").reset_index(drop=True)
        gc["rank"] = gc.index + 1
        for _, r in gc.head(6).iterrows():
            star = " <== base" if r.exp_tag == base_tag else ""
            print(f"  {r.exp_tag:<22} val={r.val:.4f}  testMSE={r.mse:.4f}  "
                  f"par={int(r.par)}{star}")
        brow = gc[gc.exp_tag == base_tag]
        if not brow.empty:
            print(f"  base {base_tag}: val-rank {int(brow.iloc[0]['rank'])}/{len(gc)}")
        rank_frames.append(gc[["exp_tag", "rank", "val", "mse"]].assign(data=d))

    # ---- mean-rank canonical (across datasets complete in ALL) ----
    if not rank_frames:
        return
    allr = pd.concat(rank_frames)
    n_ds = allr["data"].nunique()
    piv = allr.pivot_table(index="exp_tag", values="rank", aggfunc="mean")
    cnt = allr.groupby("exp_tag")["data"].nunique()
    full_cfgs = cnt[cnt == n_ds].index
    piv = piv.loc[piv.index.isin(full_cfgs)].sort_values("rank")
    print(f"\n=== CANONICAL by mean val-rank across {n_ds} datasets "
          f"(configs complete in all) ===")
    for tag, r in piv.head(8).iterrows():
        star = " <== base" if tag == base_tag else ""
        print(f"  {tag:<22} mean-rank={r['rank']:.2f}{star}")
    if base_tag in piv.index:
        print(f"  base {base_tag}: mean-rank={piv.loc[base_tag, 'rank']:.2f}")
    best = piv.head(1)
    if not best.empty:
        bt = best.index[0]
        verdict = "SAME as base" if bt == base_tag else f"BEATS base -> candidate {bt}"
        print(f"  => canonical candidate = {bt}  ({verdict})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="")
    p.add_argument("--pred_len", type=str, default="")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEARCH_SEEDS))
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--results_csv", type=str, default="results/feather_mfg_search.csv")
    p.add_argument("--ngpu", type=int, default=1)
    p.add_argument("--jobs_per_gpu", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    args.datasets = args.data.split(",") if args.data else SEARCH_DATASETS
    args.preds = ([int(x) for x in args.pred_len.split(",")]
                  if args.pred_len else SEARCH_PREDS)
    args.seed_list = [int(x) for x in args.seeds.split(",")]

    if args.summary:
        summarize(args)
        return

    runs = enumerate_runs(args)
    done = load_done_set(args.results_csv)
    missing = [r for r in runs if (r[0], r[3], r[5], r[6]) not in done]
    n_cfg = sum(1 for _ in enumerate_configs())
    print("=" * 66)
    print(f"  FEATHer mfg arch search  ({n_cfg} configs, k=1 excluded)")
    print(f"  datasets: {args.datasets}  |  H: {args.preds}  |  seeds: {args.seed_list}")
    print(f"  periods (FFT): " + ", ".join(f"{d}={_period(d)}" for d in args.datasets))
    print(f"  CSV: {args.results_csv}")
    print(f"  total: {len(runs)}  |  done: {len(runs)-len(missing)}  |  missing: {len(missing)}")
    print("=" * 66)
    if args.check:
        return
    dispatch(missing, args)


if __name__ == "__main__":
    main()
