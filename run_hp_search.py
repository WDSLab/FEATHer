# -*- coding: utf-8 -*-
"""
FEATHer hyperparameter search orchestrator (OFAT — one factor at a time).

Purpose: pick FEATHer's PER-DATASET configuration for the main accuracy
table (one best config per dataset, mirroring how each baseline uses its
original-paper per-dataset hyperparameters) and, as a by-product, produce
the per-axis HP sensitivity analysis reviewers asked for (R2 #23).

Protocol:
  - Selection uses ONLY the `val_loss` column the worker writes (loss of
    the best-val-epoch model on the validation split). Test metrics land
    in the CSV too — they feed the sensitivity figure, never selection.
  - OFAT around the canonical config: one axis varies at a time, all
    other axes held at their defaults. The base config runs once as
    exp_tag="hp_base"; each variant as exp_tag="hp_<axis>_<value>".
  - Scope: all 8 main-table datasets x 2 horizons x 2 seeds. FEATHer is
    tiny (~453 params on ETT), so even the 8-dataset search is cheap on
    the small-D sets; Traffic/Electricity carry most of the cost.
  - `--summary` recommends one config PER DATASET (per-axis val winners
    within each dataset), ready to paste into baselines._DATASET_OVERRIDES.

Usage:
    python run_hp_search.py --check       # show pending runs
    python run_hp_search.py               # run all missing (resumable)
    python run_hp_search.py --summary     # rank configs by val_loss
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# -----------------------------------------------------------------------------
# Search space — OFAT around the canonical config
# -----------------------------------------------------------------------------

BASE_CONFIG = {
    "d_state":     8,
    "kernel_size": 7,
    "period":      12,
    "num_bands":   3,
    "lambda_spec": 0.01,
    "lr":          1e-3,
}

AXES = {
    "d_state":     [4, 8, 16],
    "kernel_size": [3, 5, 7, 9],
    "period":      [6, 8, 12, 24],   # must divide seq_len AND pred_len
    "num_bands":   [2, 3, 4],
    "lambda_spec": [0.0, 0.001, 0.01, 0.1],
    "lr":          [5e-4, 1e-3, 5e-3],
}

# All 8 main-table datasets — FEATHer is tuned per-dataset (like every
# baseline), so the search must cover every dataset the main table reports.
# SML stays out (robustness-only; not in the main accuracy table).
HP_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2",
               "Weather", "Exchange", "Electricity", "Traffic"]
HP_PREDS = [96, 720]
HP_SEEDS = [2025, 2026]

DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/hp_search.csv"
WORKER = "scripts/benchmarks/run_forecast.py"


def enumerate_configs():
    """Return [(exp_tag, config_dict)] — base first, then OFAT variants.

    Axis values equal to the base default are skipped (they would
    duplicate hp_base).
    """
    configs = [("hp_base", dict(BASE_CONFIG))]
    for axis, values in AXES.items():
        for v in values:
            if v == BASE_CONFIG[axis]:
                continue
            cfg = dict(BASE_CONFIG)
            cfg[axis] = v
            configs.append((f"hp_{axis}_{v}", cfg))
    return configs


def enumerate_runs(args):
    """(exp_tag, config, data, pred_len, seq_len, seed) for the sweep."""
    runs = []
    for exp_tag, cfg in enumerate_configs():
        for d in HP_DATASETS:
            for h in HP_PREDS:
                # SPK constraint: period must divide both lengths.
                if args.seq_len % cfg["period"] or h % cfg["period"]:
                    print(f"  [skip] {exp_tag} on H={h}: period "
                          f"{cfg['period']} does not divide both lengths")
                    continue
                for s in HP_SEEDS:
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
    # Group by (exp_tag, data, pred_len) — worker iterates seeds internally.
    groups = {}
    for exp_tag, cfg, d, h, sl, s in missing:
        groups.setdefault((exp_tag, d, h, sl), (cfg, []))[1].append(s)

    for (exp_tag, d, h, sl), (cfg, seeds) in sorted(groups.items()):
        print(f"\n>>> {exp_tag} | {d} | H={h} | seeds={sorted(seeds)} | {cfg}")
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
            "--loss", "l1",
            "--lr", str(cfg["lr"]),
            "--d_state", str(cfg["d_state"]),
            "--kernel_size", str(cfg["kernel_size"]),
            "--period", str(cfg["period"]),
            "--num_bands", str(cfg["num_bands"]),
            "--lambda_spec", str(cfg["lambda_spec"]),
            "--gpu", str(args.gpu),
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"  [WARN] worker returned {ret.returncode}; continuing")


# -----------------------------------------------------------------------------
# Summary — rank configs by val_loss, recommend the single config
# -----------------------------------------------------------------------------

def _recommend_for_dataset(cell_d, axis_quiet=False):
    """OFAT per-axis val winner for one dataset.

    cell_d: rows for a single dataset, columns
    [exp_tag, pred_len, val_loss, MSE]. Returns (recommended_cfg, lines)
    where lines is the printable per-axis breakdown.
    """
    # Rank configs within each horizon by val_loss, average across horizons
    # — scale-free so a high-loss horizon doesn't dominate selection.
    cell_d = cell_d.copy()
    cell_d["rank"] = cell_d.groupby("pred_len")["val_loss"].rank()
    agg = (cell_d.groupby("exp_tag")
                 .agg(mean_rank=("rank", "mean"),
                      mean_val=("val_loss", "mean"),
                      mean_mse=("MSE", "mean"),
                      cells=("rank", "size")))
    n_cells = cell_d["pred_len"].nunique()

    recommended = dict(BASE_CONFIG)
    lines = []
    for axis, values in AXES.items():
        best_rank, best_v = None, BASE_CONFIG[axis]
        for v in values:
            tag = "hp_base" if v == BASE_CONFIG[axis] else f"hp_{axis}_{v}"
            if tag not in agg.index:
                if not axis_quiet:
                    lines.append(f"    {axis:>11}={v!s:<6} (no results)")
                continue
            r = agg.loc[tag]
            flag = " (incomplete)" if r["cells"] < n_cells else ""
            if not axis_quiet:
                lines.append(f"    {axis:>11}={v!s:<6} "
                             f"rank={r['mean_rank']:.2f} val={r['mean_val']:.4f} "
                             f"testMSE={r['mean_mse']:.4f}{flag}")
            if best_rank is None or r["mean_rank"] < best_rank:
                best_rank, best_v = r["mean_rank"], v
        recommended[axis] = best_v
    return recommended, lines


def _override_line(data, cfg):
    """Format a recommended config as a baselines._DATASET_OVERRIDES line."""
    # lr goes top-level in the override dict; the rest are arch HPs the
    # orchestrator forwards via --model_overrides.
    keys = ["d_state", "kernel_size", "period", "num_bands", "lambda_spec", "lr"]
    body = ", ".join(f'"{k}": {cfg[k]}' for k in keys)
    return f'    ("FEATHer", "{data}"): {{{body}}},'


def summarize(args):
    if not os.path.exists(args.results_csv):
        print(f"No results yet: {args.results_csv}")
        return
    df = pd.read_csv(args.results_csv)
    df = df[df["model"] == "FEATHer"]
    if df.empty:
        print("No FEATHer rows in the CSV.")
        return

    # Mean over seeds per (config, dataset, horizon) cell.
    cell = (df.groupby(["exp_tag", "data", "pred_len"])
              .agg(val_loss=("val_loss", "mean"), MSE=("MSE", "mean"),
                   n_seeds=("seed", "nunique"))
              .reset_index())

    print("\n=== Per-dataset HP selection (OFAT, val_loss; "
          "test MSE for reference only) ===")
    override_lines = []
    for data in HP_DATASETS:
        cell_d = cell[cell["data"] == data]
        if cell_d.empty:
            print(f"\n--- {data}: (no results yet) ---")
            continue
        cfg, lines = _recommend_for_dataset(cell_d)
        print(f"\n--- {data} ---")
        print("\n".join(lines))
        diff = {k: v for k, v in cfg.items() if v != BASE_CONFIG[k]}
        tag = "  ".join(f"{k}={v}" for k, v in cfg.items())
        delta = ("  [differs from base: "
                 + ", ".join(f"{k}={v}" for k, v in diff.items()) + "]") \
                if diff else "  [== base config]"
        print(f"  recommended: {tag}{delta}")
        override_lines.append(_override_line(data, cfg))

    print("\n=== Paste into baselines/__init__.py _DATASET_OVERRIDES ===")
    print("    # ---- FEATHer (this work) — per-dataset val winners "
          "from run_hp_search.py --summary ----")
    print("\n".join(override_lines))
    print("\nNote: OFAT picks per-axis winners independently per dataset; "
          "before freezing a dataset whose recommendation differs from "
          "base, confirm the combined config beats base on that dataset's "
          "val_loss (axes can interact).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FEATHer OFAT HP search")
    p.add_argument("--seq_len",     type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--num_epochs",  type=int, default=50)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--check",   action="store_true", help="show pending runs")
    p.add_argument("--summary", action="store_true", help="rank finished runs")
    args = p.parse_args()

    if args.summary:
        summarize(args)
        return

    runs = enumerate_runs(args)
    done = load_done_set(args.results_csv)
    missing = [r for r in runs
               if (r[0], r[2], r[3], r[4], r[5]) not in done]

    n_configs = len(enumerate_configs())
    print(f"\nHP search: {n_configs} configs x {len(HP_DATASETS)} datasets "
          f"x {len(HP_PREDS)} horizons x {len(HP_SEEDS)} seeds")
    print(f"Total: {len(runs)} runs | Done: {len(runs) - len(missing)} "
          f"| Missing: {len(missing)}")

    if args.check:
        for exp_tag, _, d, h, sl, s in missing:
            print(f"  [{exp_tag:>22}] {d:>8} | H={h:<4} L={sl} | seed {s}")
        return

    if not missing:
        print("All complete — run with --summary to rank configs.")
        return

    dispatch(missing, args)
    print("\nDone. Run `python run_hp_search.py --summary` to rank configs.")


if __name__ == "__main__":
    main()
