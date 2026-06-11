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

# Reuse the trained-checkpoint path convention.
from scripts.benchmarks.run_forecast import checkpoint_path

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
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def evaluate_on_test(model, test_loader, n_features, pred_len, device,
                     fault_type=None, severity=None):
    """Run inference on the test loader, optionally corrupting each batch.

    Returns:
        np.array of [MSE, MAE, RMSE, CORR, R2] (sample-count-weighted mean).
    """
    model.eval()
    sum_vec = np.zeros(5, dtype=np.float64)
    cnt = 0
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
            bs = y_true.shape[0]
            m = np.array(metrics.metric(y_pred, y_true), dtype=np.float64)
            sum_vec += m * bs
            cnt += bs
    return sum_vec / cnt


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
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

    # Clean baseline
    clean_metric = evaluate_on_test(model, test_loader, n_features,
                                    args.pred_len, device,
                                    fault_type=None, severity=None)
    mse_clean = float(clean_metric[0])
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
    write("clean", 0.0, clean_metric)

    # 2) Restrict to a fault subset if requested
    fault_types = args.fault_types.split(",") if args.fault_types else noise_mod.FAULT_TYPES

    for ft in fault_types:
        if ft not in noise_mod.SEVERITY_SWEEPS:
            print(f"  [skip] unknown fault type: {ft}")
            continue
        for sev in noise_mod.SEVERITY_SWEEPS[ft]:
            # Reseed per (fault, severity) so corruption is reproducible.
            set_seed(args.seed * 1000 + hash((ft, sev)) % 1000,
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
    run(args)
