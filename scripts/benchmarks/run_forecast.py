# -*- coding: utf-8 -*-
"""
Forecasting worker.

Runs one (model, dataset, pred_len) configuration across a given list of
seeds. Each completed run appends one row to `results/fcst_results.csv`.

This is the *worker* — call it directly when iterating manually, or let the
top-level `run_forecast.py` orchestrator dispatch missing combos
automatically.

Usage:
    python scripts/benchmarks/run_forecast.py \
        --model FEATHer --data ETTh1 --pred_len 96 \
        --seeds "2025,2026,2027,2028,2029" --exp_tag main
"""

import os
import sys
import argparse
import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_model, list_models
from utils import data_factory
from utils import metrics
from utils import losses
from utils.seed import set_seed, parse_seed_list

warnings.filterwarnings("ignore")


# =============================================================================
# CSV append
# =============================================================================

CSV_FIELDS = [
    "exp_tag", "model", "data", "pred_len", "seq_len", "seed",
    "MSE", "MAE", "RMSE", "CORR", "R2",
    "val_loss", "best_epoch", "num_params", "timestamp",
]


def append_result_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_new = pd.DataFrame([row], columns=CSV_FIELDS)
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def checkpoint_path(exp_tag, model_name, data_name, pred_len, seq_len, seed):
    """Deterministic location for the trained-model state dict.

    Designed so the robustness worker can locate any (exp_tag, model, data,
    pred_len, seq_len, seed) checkpoint without consulting the CSV.
    """
    return os.path.join(
        "results", "checkpoints", exp_tag,
        f"{model_name}_{data_name}_H{pred_len}_L{seq_len}_s{seed}.pth",
    )


# =============================================================================
# Per-run training
# =============================================================================

def train_single(args, data_name, pred_len, device, seed):
    """Train one (model, data, pred_len, seed) configuration."""

    set_seed(seed, deterministic=args.deterministic)

    now = datetime.now()
    num_metrics = 5  # MSE, MAE, RMSE, CORR, R2

    print(f"\n{'-'*60}")
    print(f"Model: {args.model} | Data: {data_name} | Pred_len: {pred_len} "
          f"| Seq_len: {args.seq_len} | Seed: {seed}")
    print(f"{'-'*60}")

    df, freq, embed = data_factory.data_select(data_name, args.root_path)
    n_features = df.shape[1] - 1

    train_data, train_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, "train",
        starting_percent=args.starting_percent, percent=args.percent,
    )
    val_data, val_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, "val",
        starting_percent=args.starting_percent, percent=args.percent,
    )
    test_data, test_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, "test",
        starting_percent=args.starting_percent, percent=args.percent,
    )

    model = get_model(
        args.model, args,
        n_features=n_features, seq_len=args.seq_len, pred_len=pred_len,
    ).float().to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.4f}M)")

    if args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # AMP is disabled across the board for cross-model fairness: some
    # baselines (FITS) carry complex parameters that AMP cannot unscale,
    # others (TimesNet) call cuFFT which only supports power-of-two sizes
    # in half precision. Keeping fp32 everywhere also makes seed-to-seed
    # variance comparable across models.

    # FEATHer and its 30 ablation variants all expose return_components=True
    # and benefit from the spectral-separation auxiliary loss. Other
    # baselines have no band structure to disentangle.
    use_spec_loss = args.model.startswith("FEATHer") and (args.lambda_spec > 0)

    # Validation-based model selection — same protocol as every baseline's
    # official implementation: val loss picks the epoch (with early
    # stopping), test is evaluated exactly once on the selected model.
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(args.num_epochs):
        # ---------- Train ----------
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_spec_loss:
                output, H = model(batch_x[:, :, :n_features], return_components=True)
            else:
                output = model(batch_x[:, :, :n_features])
                H = None

            output = output[:, -pred_len:]
            target = batch_y[:, :, :n_features][:, -pred_len:]

            main_loss = criterion(output, target)
            if use_spec_loss:
                loss = main_loss + args.lambda_spec * losses.spectral_separation_loss_scales(H)
            else:
                loss = main_loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        scheduler.step()

        # ---------- Validate ----------
        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, _, _ in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                output = model(batch_x[:, :, :n_features])[:, -pred_len:]
                target = batch_y[:, :, :n_features][:, -pred_len:]
                bs = batch_x.size(0)
                val_sum += criterion(output, target).item() * bs
                val_cnt += bs
        val_loss = val_sum / max(1, val_cnt)

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        print(f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
              f"lr {scheduler.get_last_lr()[0]:.6f} | "
              f"train {avg_loss:.4f} | val {val_loss:.4f} | "
              f"best val {best_val:.4f} (ep {best_epoch+1}) | "
              f"patience {patience_left}/{args.patience}")

        if patience_left <= 0:
            print(f"  Early stop at epoch {epoch+1} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    # ---------- Test (once, on the best-val model) ----------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    sum_vector = np.zeros(num_metrics, dtype=np.float64)
    cnt = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x[:, :, :n_features])
            y_true = batch_y[:, :, :n_features][:, -pred_len:].cpu().numpy()
            y_pred = output[:, -pred_len:].cpu().numpy()
            bs = y_true.shape[0]
            m = np.array(metrics.metric(y_pred, y_true), dtype=np.float64)
            sum_vector += m * bs
            cnt += bs
    test_metric = sum_vector / max(1, cnt)

    # ---------- Persist ----------
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    row = {
        "exp_tag":   args.exp_tag,
        "model":     args.model,
        "data":      data_name,
        "pred_len":  pred_len,
        "seq_len":   args.seq_len,
        "seed":      seed,
        "MSE":       round(float(test_metric[0]), 6),
        "MAE":       round(float(test_metric[1]), 6),
        "RMSE":      round(float(test_metric[2]), 6),
        "CORR":      round(float(test_metric[3]), 6),
        "R2":        round(float(test_metric[4]), 6),
        "val_loss":  round(float(best_val), 6),
        "best_epoch": int(best_epoch),
        "num_params": int(num_params),
        "timestamp":  timestamp,
    }
    append_result_row(args.results_csv, row)

    # Optional: deterministic checkpoint for downstream robustness eval.
    # The model currently holds the best-val state, so the saved checkpoint
    # is exactly the model whose test metrics were just reported.
    if args.save_model:
        ckpt = checkpoint_path(args.exp_tag, args.model, data_name,
                               pred_len, args.seq_len, seed)
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  -> checkpoint saved → {ckpt}")

    print(f"  -> Test (best-val ep {best_epoch+1}): MSE={test_metric[0]:.4f} "
          f"MAE={test_metric[1]:.4f} saved → {args.results_csv}")


# =============================================================================
# Main
# =============================================================================

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}  |  data: {args.data}  |  pred_len: {args.pred_len}  "
          f"|  exp_tag: {args.exp_tag}")

    seeds = parse_seed_list(args.seeds, args.num_seeds, args.base_seed)
    print(f"Seeds ({len(seeds)}): {seeds}")

    if args.pred_len <= 0:
        raise ValueError("--pred_len must be > 0 in worker mode (orchestrator picks it)")

    for seed in seeds:
        try:
            train_single(args, args.data, args.pred_len, device, seed)
        except Exception as e:
            print(f"[ERROR] {args.model} / {args.data} / {args.pred_len} / seed={seed}: {e}")
            import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting worker (one model × dataset × pred_len)")

    # Data
    p.add_argument("--root_path", type=str, default="data/")
    p.add_argument("--data",      type=str, required=True)
    p.add_argument("--features",  type=str, default="M")

    # Model
    p.add_argument("--model", type=str, default="FEATHer", choices=list_models())
    p.add_argument("--seq_len",   type=int, default=96)
    p.add_argument("--pred_len",  type=int, required=True)
    p.add_argument("--label_len", type=int, default=96)

    # FEATHer-specific (other baselines ignore irrelevant fields via wrapper)
    p.add_argument("--d_state",       type=int, default=8)
    p.add_argument("--kernel_size",   type=int, default=7)
    p.add_argument("--period",        type=int, default=12)
    p.add_argument("--num_bands",     type=int, default=3)
    p.add_argument("--use_topk_gate", action="store_true")
    p.add_argument("--topk",          type=int, default=2)

    # Training
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--num_epochs",  type=int,   default=50)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early-stopping patience on val loss (epochs).")
    p.add_argument("--loss",        type=str,   default="l1", choices=["l1", "mse"])
    p.add_argument("--lambda_spec", type=float, default=0.01)

    # Seed control
    p.add_argument("--seeds",         type=str, default="")
    p.add_argument("--num_seeds",     type=int, default=5)
    p.add_argument("--base_seed",     type=int, default=2025)
    p.add_argument("--deterministic", action="store_true", default=True)

    # Experiment grouping
    p.add_argument("--exp_tag",    type=str, default="default")
    p.add_argument("--results_csv", type=str, default="results/fcst_results.csv")

    # Generic per-(model, dataset) HP injection — orchestrator passes
    # "key=val;key2=val2" pairs derived from get_dataset_overrides(); each
    # entry is parsed (int/float/bool/str) and set on `args` BEFORE the
    # model is constructed, so wrapper.py's getattr(args, "key", default)
    # picks up the dataset-specific value.
    p.add_argument("--model_overrides", type=str, default="",
                   help='Semicolon-separated key=value HP overrides '
                        '(e.g. "patchtst_d_model=16;patchtst_n_heads=4").')

    # Data slicing (legacy)
    p.add_argument("--starting_percent", type=int, default=0)
    p.add_argument("--percent",          type=int, default=100)

    # System
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_model", action="store_true",
                   help="Save state dict to results/checkpoints/<exp_tag>/...pth "
                        "for downstream robustness evaluation.")

    return p.parse_args()


def _apply_overrides(args):
    """Parse --model_overrides string and assign each key=value to `args`.

    Values are coerced to int → float → bool → str in that order.
    """
    if not args.model_overrides:
        return
    for kv in args.model_overrides.split(";"):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        # Type coercion
        parsed = v
        if v.lower() in ("true", "false"):
            parsed = (v.lower() == "true")
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    parsed = v  # keep string
        setattr(args, k, parsed)


if __name__ == "__main__":
    args = parse_args()
    _apply_overrides(args)
    train(args)
