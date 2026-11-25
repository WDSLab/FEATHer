# -*- coding: utf-8 -*-
"""Forecasting worker: trains one (model, dataset, pred_len) for a list of seeds and appends one results row per seed.

Normally called by run.py, which resolves every setting (tools/results/protocol.py, configs/*.yaml).  Manual use:

    python scripts/train.py --model FEATHer --data ETTh1 --pred_len 96 --seeds 2025,2026,2027
        --stage manual --results_csv results/_incoming/manual.csv

Protocol: Adam + cosine schedule, gradient clip 1.0, fp32 (no AMP), early stopping on the
validation loss, test evaluated once on the best-validation weights, loss and metrics on observed target cells only.
"""
import argparse
import gc
import os
import socket
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from baselines import get_model, list_models
from utils import data_factory, losses, metrics
from utils.seed import set_seed

# tools/results/protocol.py RESULT_FIELDS (stage / tier / trial / arm / settings come from run.py)
CSV_FIELDS = ["stage", "tier", "trial", "arm", "model", "data", "H", "seed", "MSE", "MAE", "val_loss", "best_epoch",
              "params", "host", "time", "settings"]
PROTOCOL_AXES = {"batch_size", "num_epochs", "patience", "seq_len", "label_len"}   # never per-model overrides


def append_result_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if os.path.exists(csv_path):
        header = open(csv_path, newline="").readline().strip().split(",")
        if header != CSV_FIELDS:
            raise SystemExit(f"{csv_path} has an older column layout; write to a new file (run.py does this itself)")
    df_new = pd.DataFrame([row], columns=CSV_FIELDS)
    # open(mode="x") is atomic: of several workers starting a fresh file, exactly one writes the header
    try:
        with open(csv_path, "x", newline="") as f:
            df_new.to_csv(f, index=False)
    except FileExistsError:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)


def checkpoint_path(args, model_name, data_name, pred_len, seq_len, seed):
    """results/checkpoints/<stage>__<tier>__<trial>__<arm>/<model>/<data>_H<h>_L<L>_s<seed>.pth ("-" parts left out;
    ':' is not legal in Windows paths)."""
    folder = "__".join(x for x in (args.stage, args.tier, args.trial, args.arm) if x != "-").replace(":", "-")
    return os.path.join("results", "checkpoints", folder, model_name, f"{data_name}_H{pred_len}_L{seq_len}_s{seed}.pth")


def masked_loss(output, target, loss_name):
    """L1 / MSE over the observed target cells (imputed cells are NaN); NaN if the batch has none.
    Index before subtracting: otherwise the NaN targets enter the autograd graph and 0 * NaN reaches every gradient."""
    mask = torch.isfinite(target)
    diff = output[mask] - target[mask]
    err = diff.abs() if loss_name == "l1" else diff ** 2
    return err.mean() if err.numel() else torch.tensor(float("nan"), device=output.device)


def train_single(args, seed, device, on_epoch=None, run_test=True):
    """Train one seed, test the best-validation weights once, append the results row.
    Tuning (tools/results/tune.py): on_epoch(epoch, best_val, best_epoch, params) after each epoch (it may raise to stop
    the run, e.g. optuna.TrialPruned); run_test=False returns dict(val_loss, best_epoch, params) without testing or
    writing a row (val_loss NaN when no epoch gave a finite validation loss)."""
    set_seed(seed)
    now = datetime.now()
    data_name, pred_len = args.data, args.pred_len
    result_name = args.result_model_name or args.model
    print(f"\n{'-' * 60}\nModel: {args.model} | Data: {data_name} | Pred_len: {pred_len} "
          f"| Seq_len: {args.seq_len} | Seed: {seed}\n{'-' * 60}")

    df, _, _ = data_factory.data_select(data_name, args.root_path)
    n_features = df.shape[1] - 1
    loaders = {flag: data_factory.data_provider(args.root_path, data_name, args.features, args.batch_size,
                                                args.seq_len, args.label_len, pred_len, flag)[1]
               for flag in ("train", "val", "test")}

    def batches(flag):
        """(input window, observed-or-NaN target) per batch, on the device."""
        for batch_x, batch_y, _, _ in loaders[flag]:
            yield (batch_x.float().to(device)[:, :, :n_features],
                   batch_y.float().to(device)[:, :, :n_features][:, -pred_len:])

    model = get_model(args.model, args, n_features=n_features, seq_len=args.seq_len, pred_len=pred_len).float().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.4f}M)")

    # No AMP for any model: FITS has complex weights AMP cannot unscale.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    use_spec_loss = args.model.startswith("FEATHer") and args.lambda_spec > 0   # FEATHer's band-separation term
    # a baseline's own training recipe, exposed by its wrapper (e.g. TimeBase's orthogonality loss)
    epoch_hook = getattr(model, "_epoch_start", None)
    train_loss_hook = getattr(model, "_train_loss", None)
    test_hook = getattr(model, "_test_start", None)

    best_val, best_state, best_epoch, patience_left = float("inf"), None, 0, args.patience
    for epoch in range(args.num_epochs):
        if epoch_hook is not None:
            epoch_hook(epoch)
        model.train()
        total_loss = 0.0
        for x, target in batches("train"):
            optimizer.zero_grad(set_to_none=True)
            if use_spec_loss:
                output, bands = model(x, return_components=True)
            else:
                output = model(x)
            output = output[:, -pred_len:]
            loss = masked_loss(output, target, args.loss)
            if train_loss_hook is not None:
                loss = train_loss_hook(output, target, loss)
            if use_spec_loss:
                loss = loss + args.lambda_spec * losses.spectral_separation_loss_scales(bands)
            aux = getattr(model, "_aux_loss", None)
            if aux is not None:
                loss = loss + aux
            if torch.isnan(loss) or torch.isinf(loss):       # e.g. a batch without observed targets
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for x, target in batches("val"):
                # one mean over all observed cells of the split, like the test MSE: a per-batch mean
                # weighted by batch size over-weights batches with few observed cells on the gap-filled sets
                mask = torch.isfinite(target)
                diff = model(x)[:, -pred_len:][mask] - target[mask]
                val_sum += float((diff.abs() if args.loss == "l1" else diff ** 2).sum())
                val_cnt += int(mask.sum())
        # NaN (not 0.0) when no cell counted or the forecast is not finite, so a dead run never becomes "best"
        val_loss = val_sum / val_cnt if val_cnt else float("nan")
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val, best_epoch, patience_left = val_loss, epoch, args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        print(f"  Epoch {epoch + 1:3d}/{args.num_epochs} | lr {scheduler.get_last_lr()[0]:.6f} | "
              f"train {total_loss / max(1, len(loaders['train'])):.4f} | val {val_loss:.4f} | "
              f"best val {best_val:.4f} (ep {best_epoch + 1}) | patience {patience_left}/{args.patience}")
        if on_epoch is not None:
            on_epoch(epoch, best_val, best_epoch, num_params)
        if patience_left <= 0:
            print(f"  Early stop at epoch {epoch + 1} (no val improvement for {args.patience} epochs)")
            break

    if not run_test:
        return dict(val_loss=float(best_val) if best_state is not None else float("nan"),
                    best_epoch=int(best_epoch) if best_state is not None else -1, params=int(num_params))
    row = {"stage": args.stage, "tier": args.tier, "trial": args.trial, "arm": args.arm, "model": result_name,
           "data": data_name, "H": pred_len, "seed": seed, "params": int(num_params),
           "host": socket.gethostname().split(".")[0], "settings": args.settings}
    if best_state is None:
        # never report last-epoch weights; write a NaN row so run.py counts the cell as done and the verdict tools see
        # a non-finite candidate, and exit non-zero so the failure is reported
        append_result_row(args.results_csv, {**row, "MSE": np.nan, "MAE": np.nan, "val_loss": np.nan, "best_epoch": -1,
                                             "time": datetime.now().strftime("%Y%m%d_%H%M%S")})
        raise RuntimeError(f"no finite val_loss in {args.num_epochs} epoch(s) - degenerate run (diverged/NaN)")

    model.load_state_dict(best_state)
    model.eval()
    if test_hook is not None:
        test_hook()
    scores = metrics.TestScores()
    with torch.no_grad():
        for x, target in batches("test"):
            scores.update(model(x)[:, -pred_len:].cpu().numpy(), target.cpu().numpy())
    test_metric = scores.result()
    append_result_row(args.results_csv, {
        **row, **{k: round(float(v), 6) for k, v in zip(("MSE", "MAE"), test_metric)},
        "val_loss": round(float(best_val), 6), "best_epoch": int(best_epoch), "time": now.strftime("%Y%m%d_%H%M%S")})

    if args.save_model:      # the best-val weights, i.e. the model whose metrics were just written (board export / replay)
        ckpt = checkpoint_path(args, result_name, data_name, pred_len, args.seq_len, seed)
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  -> checkpoint saved -> {ckpt}")
    print(f"  -> Test (best-val ep {best_epoch + 1}): MSE={test_metric[0]:.4f} "
          f"MAE={test_metric[1]:.4f} saved -> {args.results_csv}")


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # make args.gpu the default device: some upstream baselines call .cuda() without an index
        torch.cuda.set_device(args.gpu)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Using device: {device} | {args.model} | {args.data} | H {args.pred_len} | {args.stage} {args.tier} {args.trial} {args.arm} | seeds {seeds}")
    n_failed = 0
    for seed in seeds:
        try:
            train_single(args, seed, device)
        except Exception as e:
            n_failed += 1
            print(f"[ERROR] {args.model} / {args.data} / {args.pred_len} / seed={seed}: {e}")
            traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
    if n_failed:   # finish the other seeds, then exit non-zero so run.py reports the failure
        print(f"[FATAL] {n_failed}/{len(seeds)} seed run(s) failed")
        sys.exit(1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Forecasting worker (one model x dataset x pred_len)")
    # data and run
    p.add_argument("--root_path", default="data/")
    p.add_argument("--data", required=True)
    p.add_argument("--features", default="M")
    p.add_argument("--model", default="FEATHer", choices=list_models())
    p.add_argument("--result_model_name", default=None,
                   help="name written to the results row (run.py: FEATHer_core / FEATHer_CI); default = --model")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--label_len", type=int, default=96)
    p.add_argument("--seeds", default="2025,2026,2027")
    p.add_argument("--stage", default="manual", help="stored fields of the row (run.py, tools/results/protocol.py)")
    p.add_argument("--tier", default="-")
    p.add_argument("--trial", default="-")
    p.add_argument("--arm", default="-")
    p.add_argument("--settings", default="-", help="the searched values, \"lr=... key=value ...\" (run.py)")
    p.add_argument("--results_csv", default="results/_incoming/train.csv")
    p.add_argument("--model_overrides", default="",
                   help='per-model hyperparameters "key=val;key2=val2" (run.py, from configs/<model>.yaml)')
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_model", action="store_true", help="save the best-val state dict (board export, replay)")
    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience on the validation loss")
    p.add_argument("--loss", default="mse", choices=["l1", "mse"])
    p.add_argument("--lambda_spec", type=float, default=0.01, help="FEATHer band-separation loss weight")
    # FEATHer switches (frozen values come from configs/FEATHer.yaml; these are the size / ablation options)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--num_bands", type=int, default=3)
    p.add_argument("--point_kernel", type=int, default=None, help="POINT band conv width (odd)")
    p.add_argument("--gate_mode", default=None, choices=["banded", "static", "none"])
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--mlp_rank", type=int, default=16)
    p.add_argument("--channel_mode", default=None, choices=["mix", "ci_full"])
    p.add_argument("--no_revin", dest="use_norm", action="store_false", default=True)
    p.add_argument("--head_mode", default=None, choices=["spk", "linear"])
    p.add_argument("--head_seg", type=int, default=0, help="size: SPK segment length (0 = period)")
    p.add_argument("--agg_shared", type=int, default=0, help="size: one SPK smoothing kernel for all channels")
    return p.parse_args(argv)


def _apply_overrides(args):
    """Set each --model_overrides key=value on args (bool, int, float, else str)."""
    for kv in (s.strip() for s in args.model_overrides.split(";")):
        if "=" not in kv:
            continue
        k, v = (t.strip() for t in kv.split("=", 1))
        if k in PROTOCOL_AXES:   # unified across models; a per-model value would break the comparison
            raise ValueError(f"--model_overrides may not set protocol axis '{k}'; got '{kv}'")
        setattr(args, k, _parse_value(v))


def _parse_value(v):
    """"True" / "false" -> bool, then int, then float, else the string."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


if __name__ == "__main__":
    args = parse_args()
    _apply_overrides(args)
    train(args)
