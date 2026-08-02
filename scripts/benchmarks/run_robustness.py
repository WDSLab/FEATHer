# -*- coding: utf-8 -*-
"""
Robustness worker.

Loads a checkpoint trained by `scripts/benchmarks/run_forecast.py
--save_model`, runs the SensorFault-Bench-style four-axis corruption sweep
on test data, and appends one row per (fault_type, severity) to
`results/robust_results.csv`.

Each invocation handles a single (model, data, pred_len, seed). The
top-level `run_robustness.py` orchestrator dispatches missing combos
similarly to `run_forecast.py`.

Usage:
    python scripts/benchmarks/run_robustness.py \
        --model FEATHer --data ETTh1 --pred_len 96 \
        --seed 2025 --train_exp_tag main --exp_tag robust
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_model, list_models, list_ablation_models
from utils import data_factory
from utils import metrics
from utils import noise as noise_mod
from utils.seed import set_seed

# AUDIT-FIX (2026-07-14, cfd90c5, P0-1): reuse the trained-checkpoint path
# convention + the override parser, so the model is reconstructed with
# EXACTLY the architecture the checkpoint was trained with (the orchestrator
# forwards the same _DATASET_OVERRIDES merge that run_forecast.py used at
# training time). Before this fix the worker rebuilt bare paper defaults and
# the strict load_state_dict failed on every per-dataset architecture.
from scripts.benchmarks.run_forecast import checkpoint_path, _apply_overrides

warnings.filterwarnings("ignore")


CSV_FIELDS = [
    "exp_tag", "train_exp_tag", "model", "data", "pred_len", "seq_len",
    "seed", "fault_type", "severity",
    "MSE", "MAE", "RMSE", "CORR", "R2",
    "MSE_clean", "Dw",
    "timestamp",
]


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df = pd.DataFrame([row], columns=CSV_FIELDS)
    # AUDIT-FIX (2026-07-14, cfd90c5, P2-2) — atomic create-with-header (see
    # run_forecast.append_result_row): avoids a duplicate mid-file header
    # when concurrent workers race on a fresh CSV.
    try:
        with open(csv_path, "x", newline="") as f:
            df.to_csv(f, index=False)
    except FileExistsError:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def load_done_conditions(csv_path, args):
    """(fault_type, severity) pairs already in the CSV for THIS worker's key.

    AUDIT-FIX (2026-07-14, cfd90c5, P0-3): lets a retried worker skip
    conditions it already wrote instead of re-appending the full 20-row
    sweep (which would duplicate rows and corrupt count/paired analyses
    downstream)."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        # AUDIT-FIX (2026-07-19, #9): a CORRUPT (but present) results CSV must
        # NOT be read as "nothing done" — that silently re-runs every
        # condition and appends duplicate rows onto the corruption. Fail so a
        # human repairs the CSV first.
        raise RuntimeError(
            f"could not parse existing results CSV {csv_path}: {e}") from e
    if df.empty:
        return set()
    sel = df[
        (df["exp_tag"].astype(str) == str(args.exp_tag))
        & (df["train_exp_tag"].astype(str) == str(args.train_exp_tag))
        & (df["model"].astype(str) == str(args.model))
        & (df["data"].astype(str) == str(args.data))
        # to_numeric(coerce): a torn row from a concurrent append (jobs_per_gpu
        # > 1) leaves NaN in a numeric column; astype(int) on the whole column
        # would raise IntCastingNaNError. Coercing makes such rows compare False
        # (excluded from the resume set) instead of crashing the worker.
        & (pd.to_numeric(df["pred_len"], errors="coerce") == int(args.pred_len))
        & (pd.to_numeric(df["seq_len"], errors="coerce") == int(args.seq_len))
        & (pd.to_numeric(df["seed"], errors="coerce") == int(args.seed))
    ]
    return set(zip(sel["fault_type"].astype(str), sel["severity"].astype(float)))


def evaluate_on_test(model, test_loader, n_features, pred_len, device,
                     fault_type=None, severity=None):
    """Run inference on the test loader, optionally corrupting each batch.

    Returns:
        np.array of [MSE, MAE, RMSE, CORR, R2]. MSE/MAE are global
        observed-cell means; CORR/R2 are batch-weighted pair averages.
    """
    model.eval()
    # AUDIT-FIX (2026-07-19, #3): global observed-cell MSE/MAE (matches the
    # run_forecast test loop and the persistence/classical estimator; byte-
    # identical on unmasked data, correct on the mask.csv sets incl. WindSCADA).
    se_sum = ae_sum = 0.0
    n_obs = 0
    cr_sum = np.zeros(2, dtype=np.float64)   # [CORR, R2]
    cr_cnt = np.zeros(2, dtype=np.float64)
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            x_in = batch_x[:, :, :n_features]
            if fault_type is not None:
                x_in = noise_mod.apply_corruption(x_in, fault_type, severity)
            output = model(x_in)
            y_true = batch_y[:, :, :n_features][:, -pred_len:].cpu().numpy()
            y_pred = output[:, -pred_len:].cpu().numpy()
            # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #2): a corruption that
            # drives the model to non-finite predictions is a REAL failure
            # of that condition. The nan-aware metric would silently drop
            # those cells and report a deceptively good score, so mark the
            # whole condition failed (NaN) — never a partial-score win, and
            # (with the zero-count guard below) never a fake 0.0 in mCE/AUC.
            obs = np.isfinite(y_true)
            if not np.isfinite(y_pred[obs]).all():
                return np.full(5, np.nan, dtype=np.float64)
            diff = y_pred - y_true
            fin = np.isfinite(diff)
            se_sum += float(np.sum(diff[fin] ** 2))
            ae_sum += float(np.sum(np.abs(diff[fin])))
            n_obs += int(fin.sum())
            bs = y_true.shape[0]
            corr, r2 = metrics.corr_mean(y_pred, y_true), metrics.r2_mean(y_pred, y_true)
            for i, val in enumerate((corr, r2)):
                if np.isfinite(val):
                    cr_sum[i] += val * bs
                    cr_cnt[i] += bs
    # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #1): no observed cell -> NaN, not
    # 0.0. A condition with no finite cell must record NaN (visibly failed),
    # never a fake perfect-robustness 0.0 that would win mCE/AUC/Dw.
    if n_obs == 0:
        return np.full(5, np.nan, dtype=np.float64)
    mse = se_sum / n_obs
    out = np.array([
        mse, ae_sum / n_obs, float(np.sqrt(mse)),
        cr_sum[0] / max(cr_cnt[0], 1.0),
        cr_sum[1] / max(cr_cnt[1], 1.0),
    ], dtype=np.float64)
    return out


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # See run_forecast.py: make args.gpu the default cuda device so upstream
        # hardcoded `.cuda()`/`.to('cuda')` lands on THIS gpu under --ngpu N.
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    print(f"Model: {args.model}  |  data: {args.data}  |  pred_len: {args.pred_len}  "
          f"|  seed: {args.seed}  |  train_exp_tag: {args.train_exp_tag}")

    ckpt = checkpoint_path(args.train_exp_tag, args.model, args.data,
                           args.pred_len, args.seq_len, args.seed)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Re-run training with `python run_forecast.py "
            f"--save_model --exp_tag {args.train_exp_tag} "
            f"--model {args.model} --data {args.data} --pred_len {args.pred_len}` "
            f"first."
        )

    # Reseed for the corruption RNG; model weights are loaded so the seed
    # only affects which noise realizations we draw.
    set_seed(args.seed, deterministic=args.deterministic)

    # Load data
    df, _, _ = data_factory.data_select(args.data, args.root_path)
    n_features = df.shape[1] - 1
    _, test_loader = data_factory.data_provider(
        args.root_path, args.data, args.features, args.batch_size,
        args.seq_len, args.label_len, args.pred_len, "test",
        starting_percent=args.starting_percent, percent=args.percent,
    )

    # Build model + load checkpoint
    model = get_model(args.model, args, n_features=n_features,
                      seq_len=args.seq_len, pred_len=args.pred_len)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.float().to(device)

    # Per-condition resume: skip (fault, severity) rows this key already has,
    # so a retried worker never appends duplicates.
    done_conditions = load_done_conditions(args.results_csv, args)
    if done_conditions:
        print(f"  resume: {len(done_conditions)} condition(s) already in CSV — skipping those")

    # Clean baseline — always re-evaluated (deterministic: no corruption
    # involved) because Dw needs mse_clean; only WRITTEN if missing.
    clean_metric = evaluate_on_test(model, test_loader, n_features,
                                    args.pred_len, device,
                                    fault_type=None, severity=None)
    mse_clean = float(clean_metric[0])
    # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #1): a non-finite clean baseline
    # means the loaded checkpoint is degenerate — every Dw (= MSE/mse_clean)
    # would be garbage. Fail loudly (exit non-zero) instead of writing a CSV
    # full of NaN/inf Dw rows.
    if not np.isfinite(mse_clean):
        raise RuntimeError(
            f"non-finite clean MSE ({mse_clean}) for checkpoint {ckpt} — "
            f"degenerate model; refusing to run corruption sweep")
    print(f"  clean: MSE={mse_clean:.4f}  MAE={clean_metric[1]:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write(fault_type, severity, m):
        row = {
            "exp_tag":       args.exp_tag,
            "train_exp_tag": args.train_exp_tag,
            "model":         args.model,
            "data":          args.data,
            "pred_len":      args.pred_len,
            "seq_len":       args.seq_len,
            "seed":          args.seed,
            "fault_type":    fault_type,
            "severity":      severity,
            "MSE":           round(float(m[0]), 6),
            "MAE":           round(float(m[1]), 6),
            "RMSE":          round(float(m[2]), 6),
            "CORR":          round(float(m[3]), 6),
            "R2":            round(float(m[4]), 6),
            "MSE_clean":     round(mse_clean, 6),
            "Dw":            round(float(m[0]) / max(mse_clean, 1e-12), 6),
            "timestamp":     timestamp,
        }
        append_row(args.results_csv, row)

    # 1) Bake the clean baseline into the CSV under fault_type="clean"
    #    severity=0 so plotting code can include it without special-casing.
    if ("clean", 0.0) not in done_conditions:
        write("clean", 0.0, clean_metric)

    # 2) Restrict to a fault subset if requested
    fault_types = args.fault_types.split(",") if args.fault_types else noise_mod.FAULT_TYPES

    for ft in fault_types:
        if ft not in noise_mod.SEVERITY_SWEEPS:
            print(f"  [skip] unknown fault type: {ft}")
            continue
        for sev in noise_mod.SEVERITY_SWEEPS[ft]:
            if (ft, float(sev)) in done_conditions:
                print(f"  [skip] {ft}={sev} already in CSV")
                continue
            # AUDIT-FIX (2026-07-14, cfd90c5, P0-2): reseed per (fault,
            # severity) so every process draws the SAME corruption
            # realization (matched noise across models/re-runs). NOT hash():
            # string hashing is PYTHONHASHSEED-salted per process.
            set_seed(noise_mod.corruption_seed(args.seed, ft, sev),
                     deterministic=args.deterministic)
            m = evaluate_on_test(model, test_loader, n_features,
                                 args.pred_len, device,
                                 fault_type=ft, severity=sev)
            write(ft, sev, m)
            print(f"  {ft:>8}={sev!s:<6} | MSE={m[0]:.4f}  Dw={m[0]/max(mse_clean,1e-12):.3f}")

    print(f"\n  -> appended → {args.results_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Robustness worker")

    # What to evaluate
    p.add_argument("--model", type=str, required=True,
                   choices=list_models() + list_ablation_models())
    p.add_argument("--data",  type=str, required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--seq_len",  type=int, default=96)
    p.add_argument("--label_len", type=int, default=96)
    p.add_argument("--seed", type=int, required=True)

    # FEATHer hyperparameters (need to reconstruct the same architecture)
    p.add_argument("--d_state",     type=int, default=8)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--period",      type=int, default=12)
    p.add_argument("--num_bands",   type=int, default=3)
    p.add_argument("--use_topk_gate", action="store_true")
    p.add_argument("--topk",        type=int, default=2)
    # Capacity A/B (2026-07-31): must match the trained checkpoint's arch so
    # load_state_dict finds nl_down/nl_up. OFF by default (canonical unchanged).
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--mlp_rank",    type=int, default=16)
    # Diagnostics: must match training so the loaded model's forward matches
    # (RevIN/gate have no/unchanged params, but change the computation).
    p.add_argument("--no_revin",    dest="use_norm", action="store_false", default=True)
    p.add_argument("--gate_uniform", dest="use_gate", action="store_false", default=True)
    p.add_argument("--robust_revin", dest="use_robust_norm", action="store_true", default=False)
    p.add_argument("--revin_mode", type=str, default=None,
                   choices=["standard", "mad", "winsor", "center"])
    p.add_argument("--winsor_bound", type=float, default=None)
    p.add_argument("--center_trim",  type=float, default=None)

    # Per-(model, dataset) HP injection — MUST mirror the training-time merge
    # or per-dataset architectures (FEATHer period/d_state, sparsetsf_period_len,
    # tqnet_cycle, patchtst_d_model, ...) won't match the checkpoint and the
    # strict load_state_dict fails. Same format as run_forecast.py.
    p.add_argument("--model_overrides", type=str, default="",
                   help='Semicolon-separated key=value HP overrides '
                        '(e.g. "period=24;sparsetsf_period_len=48").')

    # Data
    p.add_argument("--root_path", type=str, default="data/")
    p.add_argument("--features",  type=str, default="M")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--starting_percent", type=int, default=0)
    p.add_argument("--percent",          type=int, default=100)

    # Robustness control
    p.add_argument("--train_exp_tag", type=str, default="main",
                   help="exp_tag used at training time — controls checkpoint lookup.")
    p.add_argument("--fault_types",   type=str, default="",
                   help="Comma-separated subset (e.g. 'gauss,miss'); empty = all four.")
    p.add_argument("--exp_tag",       type=str, default="robust",
                   help="Tag attached to robustness rows.")
    p.add_argument("--results_csv",   type=str, default="results/robust_results.csv")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--gpu", type=int, default=0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _apply_overrides(args)
    run(args)
