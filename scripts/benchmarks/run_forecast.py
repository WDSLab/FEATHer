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

from baselines import get_model, list_models, list_ablation_models
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
    # AUDIT-FIX (2026-07-14, cfd90c5, P2-2) — atomic create-with-header:
    # under --ngpu/--jobs_per_gpu several workers can race on a FRESH csv; a
    # non-atomic exists()-then-write can leave a second header mid-file and
    # break resume parsing. open(mode="x") is O_CREAT|O_EXCL — exactly one
    # worker wins the header.
    try:
        with open(csv_path, "x", newline="") as f:
            df_new.to_csv(f, index=False)
    except FileExistsError:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)


def checkpoint_path(exp_tag, model_name, data_name, pred_len, seq_len, seed):
    """Deterministic location for the trained-model state dict.

    Designed so the robustness worker can locate any (exp_tag, model, data,
    pred_len, seq_len, seed) checkpoint without consulting the CSV.

    Layout (2026-07-31): per-model subfolder under the exp_tag, so a sweep's
    ~1000 checkpoints are grouped by model instead of one flat directory:
        results/checkpoints/<exp_tag>/<model>/<data>_H<h>_L<L>_s<seed>.pth
    (os.makedirs on dirname at save time creates the model subfolder.)
    """
    return os.path.join(
        "results", "checkpoints", exp_tag, model_name,
        f"{data_name}_H{pred_len}_L{seq_len}_s{seed}.pth",
    )


# =============================================================================
# Per-run training
# =============================================================================

def _masked_loss(output, target, loss_name):
    """L1/MSE over OBSERVED target cells only. MASK-FIX (2026-07-15).

    The loader NaN-outs imputed TARGET cells for the gap-filled sensor sets
    (observed-only scoring). Plain nn.L1Loss/MSELoss returns NaN for any batch
    that contains a NaN, which the training loop then skips — silently dropping
    nearly every partially-masked batch. Reduce over the finite cells instead.
    For datasets with no masked cells the mask is all-True -> identical to the
    plain mean (byte-identical results). Returns NaN if a batch has zero
    observed cells (caller skips it, exactly like a diverged batch)."""
    # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD): index the tensors BEFORE the
    # arithmetic. Computing (output - target) on the FULL tensors first lets
    # the NaN target cells enter the autograd graph; even though err[mask]
    # excludes them from the forward value, the backward pass propagates
    # `0 * NaN = NaN` from those cells into EVERY gradient, so the model
    # diverges to NaN weights at the first optimizer step. This only fired on
    # the mask.csv sets (N2O/WindSCADA/MetroPT3), whose targets carry NaN, and
    # the divergence was then silently reported as a perfect 0.0 (NaN batches
    # skipped -> sum/max(1,cnt) == 0). Selecting first keeps NaN out of the graph.
    mask = torch.isfinite(target)
    diff = output[mask] - target[mask]
    err = diff.abs() if loss_name == "l1" else diff ** 2
    if err.numel() == 0:
        return torch.tensor(float("nan"), device=output.device)
    return err.mean()


def train_single(args, data_name, pred_len, device, seed):
    """Train one (model, data, pred_len, seed) configuration."""

    set_seed(seed, deterministic=args.deterministic)

    now = datetime.now()

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

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

            # MASK-FIX (2026-07-15): observed-only loss (see _masked_loss).
            main_loss = _masked_loss(output, target, args.loss)
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
                # MASK-FIX (2026-07-15): observed-only val loss; skip any batch
                # whose target window is fully imputed (no observed cell).
                vl = _masked_loss(output, target, args.loss).item()
                if np.isfinite(vl):
                    val_sum += vl * bs
                    val_cnt += bs
        # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD): if NOT ONE val batch had a
        # finite observed loss (fully diverged, or every window fully imputed),
        # val_cnt == 0 must yield NaN — not 0.0. The old `sum / max(1, cnt)`
        # returned 0.0, which reads as a perfect score, gets accepted as
        # best_val, and lets the P2-5 "no finite val_loss" guard pass on a
        # degenerate (NaN-weight) model. NaN keeps best_state None -> P2-5 fails.
        val_loss = val_sum / val_cnt if val_cnt > 0 else float("nan")

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
    # AUDIT-FIX (2026-07-14, P2-5): if val_loss was never finite, there is no
    # best-val model — evaluating the last-epoch weights would write a
    # normal-looking row (val_loss=inf, best_epoch=0) and --save_model would
    # checkpoint a degenerate state. Fail the seed instead: no row is
    # written, and the worker exits non-zero via the P2-1 path so the
    # orchestrator surfaces the group.
    if best_state is None:
        raise RuntimeError(
            f"no finite val_loss in {args.num_epochs} epoch(s) — degenerate "
            f"run (diverged/NaN); refusing to report last-epoch metrics")
    model.load_state_dict(best_state)
    model.eval()
    # AUDIT-FIX (2026-07-19, #3): MSE/MAE are a GLOBAL observed-cell mean
    # (nansum of squared/abs error over finite cells / finite-cell count),
    # NOT a batch-size-weighted mean of per-batch nanmeans. On unmasked data
    # the two are algebraically identical (every cell observed); on the
    # mask.csv sets they diverge — up to ~13% on MetroPT3 — because batch-size
    # weighting ignores that batches carry different observed-cell counts.
    # This matches the persistence/classical baselines' estimator so the main
    # table compares like with like. CORR/R2 stay batch-weighted (they are
    # per-(B,F) pair averages with no simple global closed form).
    se_sum = ae_sum = 0.0
    n_obs = 0
    cr_sum = np.zeros(2, dtype=np.float64)   # [CORR, R2]
    cr_cnt = np.zeros(2, dtype=np.float64)
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x[:, :, :n_features])
            y_true = batch_y[:, :, :n_features][:, -pred_len:].cpu().numpy()
            y_pred = output[:, -pred_len:].cpu().numpy()
            # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #2): the nan-aware metric
            # drops NaN cells — intended for imputed TARGET cells, but it would
            # ALSO silently drop a non-finite PREDICTION on an observed cell,
            # improving a partially-diverged model's score. A best-val model
            # must be finite; treat any non-finite pred on an observed cell as
            # a degenerate run (P2-5 path).
            obs = np.isfinite(y_true)
            if not np.isfinite(y_pred[obs]).all():
                raise RuntimeError(
                    "non-finite prediction on observed test cells — "
                    "degenerate best-val model; refusing to report metrics")
            diff = y_pred - y_true               # NaN where target imputed
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
    # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #1): no observed cell across all
    # batches (fully diverged / fully masked) must NOT become `0/max(1,0)==0`
    # — a fake perfect score that slips past the P2-5 guard (the val loop was
    # fixed in d56aa05; this is its test-loop twin).
    if n_obs == 0:
        raise RuntimeError(
            "no observed test cells across all batches — degenerate run")
    mse = se_sum / n_obs
    mae = ae_sum / n_obs
    # RMSE = sqrt(global MSE) (P1-3): identity holds by construction now.
    test_metric = np.array([
        mse, mae, float(np.sqrt(mse)),
        cr_sum[0] / max(cr_cnt[0], 1.0),
        cr_sum[1] / max(cr_cnt[1], 1.0),
    ], dtype=np.float64)

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
    if torch.cuda.is_available():
        # Make args.gpu the DEFAULT cuda device for this worker process, so any
        # hardcoded `.cuda()` / `.to('cuda')` inside upstream baselines (e.g.
        # MDMLP_EIA/layers/ema.py, dema.py) resolves to THIS gpu, not cuda:0.
        # Without this, --ngpu 2 workers on cuda:1 crash with a cuda:1/cuda:0
        # device mismatch. See CLAUDE.md "Known issues".
        torch.cuda.set_device(args.gpu)
    print(f"Using device: {device}")
    print(f"Model: {args.model}  |  data: {args.data}  |  pred_len: {args.pred_len}  "
          f"|  exp_tag: {args.exp_tag}")

    seeds = parse_seed_list(args.seeds, args.num_seeds, args.base_seed)
    print(f"Seeds ({len(seeds)}): {seeds}")

    if args.pred_len <= 0:
        raise ValueError("--pred_len must be > 0 in worker mode (orchestrator picks it)")

    n_failed = 0
    for seed in seeds:
        try:
            train_single(args, args.data, args.pred_len, device, seed)
        except Exception as e:
            n_failed += 1
            print(f"[ERROR] {args.model} / {args.data} / {args.pred_len} / seed={seed}: {e}")
            import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()

    if n_failed:
        # AUDIT-FIX (2026-07-14, cfd90c5, P2-1): a failed run writes NO csv
        # row, so without a non-zero exit the orchestrator would silently
        # treat it as "not yet run" forever (this is exactly how the
        # missing-reformer_pytorch incident hid 4 models). Finish the
        # remaining seeds, then flag the group.
        print(f"[FATAL] {n_failed}/{len(seeds)} seed run(s) failed — "
              f"exiting non-zero so the orchestrator reports this group.")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting worker (one model × dataset × pred_len)")

    # Data
    p.add_argument("--root_path", type=str, default="data/")
    p.add_argument("--data",      type=str, required=True)
    p.add_argument("--features",  type=str, default="M")

    # Model
    p.add_argument("--model", type=str, default="FEATHer",
                   choices=list_models() + list_ablation_models())
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
    # Capacity A/B (2026-07-31): nonlinear residual MLP (channel-axis low-rank
    # bottleneck) before the head. OFF by default -> canonical byte-identical.
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--mlp_rank",      type=int, default=16)
    # Robustness diagnostics (2026-07-31): isolate the fragility source.
    p.add_argument("--no_revin",    dest="use_norm", action="store_false", default=True)
    p.add_argument("--gate_uniform", dest="use_gate", action="store_false", default=True)
    # robust-RevIN (2026-08-01): median/MAD statistics instead of mean/std.
    p.add_argument("--robust_revin", dest="use_robust_norm", action="store_true", default=False)
    # Instance-norm variant (2026-08-02): winsorized full RevIN vs center-only.
    p.add_argument("--revin_mode", type=str, default=None,
                   choices=["standard", "mad", "winsor", "center"])

    # Training
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--num_epochs",  type=int,   default=50)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early-stopping patience on val loss (epochs).")
    p.add_argument("--loss",        type=str,   default="l1", choices=["l1", "mse"])
    p.add_argument("--lambda_spec", type=float, default=0.01)
    # Probe flags (2026-07-22, backward-compatible: defaults reproduce the
    # legacy Adam/no-wd behavior byte-identically). Added to test the TPAMI
    # optimizer recipe (AdamW + weight_decay 1e-4 + lr 1e-2).
    p.add_argument("--optimizer",    type=str,   default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight_decay", type=float, default=0.0)

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

    # Data slicing (legacy raw knobs — subsample the TRAIN split only)
    p.add_argument("--starting_percent", type=int, default=0)
    p.add_argument("--percent",          type=int, default=100)
    # Data-efficiency / cold-start convenience: use only `train_frac` of the
    # train split. mode=recent keeps the MOST RECENT fraction ("new line with
    # limited history"); mode=first keeps the earliest. Maps onto the raw
    # starting_percent/percent knobs above. val/test are unaffected (the loader
    # applies the slice only to set_type==0). Use a distinct --exp_tag per
    # fraction so resume rows don't collide (train_frac is not in the key).
    p.add_argument("--train_frac", type=float, default=None,
                   help="fraction (0,1] of the train split to use (data-efficiency)")
    p.add_argument("--train_frac_mode", default="recent", choices=["recent", "first"])

    # System
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_model", action="store_true",
                   help="Save state dict to results/checkpoints/<exp_tag>/...pth "
                        "for downstream robustness evaluation.")

    args = p.parse_args()
    if args.train_frac is not None:
        pct = max(1, min(100, int(round(args.train_frac * 100))))
        args.percent = pct
        args.starting_percent = (100 - pct) if args.train_frac_mode == "recent" else 0
    return args


def _apply_overrides(args):
    """Parse --model_overrides string and assign each key=value to `args`.

    Values are coerced to int → float → bool → str in that order.
    """
    if not args.model_overrides:
        return
    # AUDIT-FIX (2026-07-19, #10): overrides carry per-(model,dataset) ARCH
    # hyperparameters only. The unified-protocol axes must never be set this
    # way — a stray key in _DATASET_OVERRIDES would silently break cross-model
    # fairness (different batch/epochs/seq_len). Refuse loudly instead.
    PROTOCOL_AXES = {"batch_size", "num_epochs", "patience", "seq_len",
                     "label_len"}
    for kv in args.model_overrides.split(";"):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in PROTOCOL_AXES:
            raise ValueError(
                f"--model_overrides may not set protocol axis '{k}' "
                f"(unified across all models); got '{kv}'")
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
