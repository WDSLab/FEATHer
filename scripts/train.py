# -*- coding: utf-8 -*-
"""Forecasting worker: trains one (model, dataset, pred_len) for a list of seeds and appends one results row per seed.

Normally called by run.py, which resolves every setting (tools/results/protocol.py, configs/*.yaml).  Manual use:

    python scripts/train.py --model FEATHer --data ETTh1 --pred_len 96 --seeds 2025,2026,2027
        --exp_tag "manual|test" --results_csv results/_incoming/manual.csv

Protocol (docs/EXPERIMENT_PROTOCOL.md): Adam + cosine schedule, gradient clip 1.0, fp32 (no AMP), early stopping on the
validation loss, test evaluated once on the best-validation weights, loss and metrics on observed target cells only.
"""
import argparse
import gc
import os
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

CSV_FIELDS = [
    "exp_tag", "model", "data", "pred_len", "seq_len", "seed",
    "MSE", "MAE", "RMSE", "CORR", "R2",
    "val_loss", "best_epoch", "num_params", "timestamp",
    "hparams",      # every hyperparameter the run used (JSON from run.py); run.py re-runs a cell whose value changed
]
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


def checkpoint_path(exp_tag, model_name, data_name, pred_len, seq_len, seed):
    """results/checkpoints/<exp_tag>/<model>/<data>_H<h>_L<L>_s<seed>.pth ('|' and ':' are not legal in Windows paths)."""
    folder = exp_tag.replace("|", "__").replace(":", "-")
    return os.path.join("results", "checkpoints", folder, model_name, f"{data_name}_H{pred_len}_L{seq_len}_s{seed}.pth")


def masked_loss(output, target, loss_name):
    """L1 / MSE over the observed target cells (imputed cells are NaN); NaN if the batch has none.
    Index before subtracting: otherwise the NaN targets enter the autograd graph and 0 * NaN reaches every gradient."""
    mask = torch.isfinite(target)
    diff = output[mask] - target[mask]
    err = diff.abs() if loss_name == "l1" else diff ** 2
    return err.mean() if err.numel() else torch.tensor(float("nan"), device=output.device)


def train_single(args, seed, device):
    """Train one seed, test the best-validation weights once, append the results row."""
    set_seed(seed)
    now = datetime.now()
    data_name, pred_len = args.data, args.pred_len
    result_name = args.result_model_name or args.model
    print(f"\n{'-' * 60}\nModel: {args.model} | Data: {data_name} | Pred_len: {pred_len} "
          f"| Seq_len: {args.seq_len} | Seed: {seed}\n{'-' * 60}")

    df, _, _ = data_factory.data_select(data_name, args.root_path)
    n_features = df.shape[1] - 1
    args.label_len = min(args.label_len, args.seq_len)     # label_len > seq_len would give empty target windows
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

    # SAMformer trains with sharpness-aware minimisation (part of its method); every other model uses Adam.
    # No AMP for any model: FITS has complex weights AMP cannot unscale, TimesNet's fp16 cuFFT needs powers of two.
    use_sam = args.model == "SAMformer"
    if use_sam:
        from baselines.SAMformer.wrapper import make_sam
        optimizer = make_sam(model.parameters(), lr=args.lr, rho=float(getattr(args, "samformer_rho", 0.5)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    use_spec_loss = args.model.startswith("FEATHer") and args.lambda_spec > 0   # FEATHer's band-separation term
    # a baseline's own training recipe, exposed by its wrapper (DiPE: temperature schedule + loss mix; TimeBase: aux loss)
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
            if use_sam:                                      # perturb to w + e(w), recompute the loss there, update w
                optimizer.first_step(zero_grad=True)
                loss2 = masked_loss(model(x)[:, -pred_len:], target, args.loss)
                if torch.isfinite(loss2):
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.second_step(zero_grad=True)
                else:                                        # restore w without stepping
                    with torch.no_grad():
                        for g in optimizer.param_groups:
                            for prm in g["params"]:
                                if "e_w" in optimizer.state[prm]:
                                    prm.sub_(optimizer.state[prm]["e_w"])
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for x, target in batches("val"):
                vl = masked_loss(model(x)[:, -pred_len:], target, args.loss).item()
                if np.isfinite(vl):                          # skip a batch whose targets are all imputed
                    val_sum += vl * x.size(0)
                    val_cnt += x.size(0)
        # NaN (not 0.0) when no batch counted, so a dead run never becomes "best"
        val_loss = val_sum / val_cnt if val_cnt else float("nan")
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val, best_epoch, patience_left = val_loss, epoch, args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        print(f"  Epoch {epoch + 1:3d}/{args.num_epochs} | lr {scheduler.get_last_lr()[0]:.6f} | "
              f"train {total_loss / max(1, len(loaders['train'])):.4f} | val {val_loss:.4f} | "
              f"best val {best_val:.4f} (ep {best_epoch + 1}) | patience {patience_left}/{args.patience}")
        if patience_left <= 0:
            print(f"  Early stop at epoch {epoch + 1} (no val improvement for {args.patience} epochs)")
            break

    row = {"exp_tag": args.exp_tag, "model": result_name, "data": data_name, "pred_len": pred_len,
           "seq_len": args.seq_len, "seed": seed, "num_params": int(num_params), "hparams": args.hparams}
    if best_state is None:
        # never report last-epoch weights; write a NaN row so run.py counts the cell as done and the verdict tools see
        # a non-finite candidate, and exit non-zero so the failure is reported
        append_result_row(args.results_csv, {**row, "MSE": np.nan, "MAE": np.nan, "RMSE": np.nan, "CORR": np.nan,
                                             "R2": np.nan, "val_loss": np.nan, "best_epoch": -1,
                                             "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")})
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
        **row, **{k: round(float(v), 6) for k, v in zip(("MSE", "MAE", "RMSE", "CORR", "R2"), test_metric)},
        "val_loss": round(float(best_val), 6), "best_epoch": int(best_epoch), "timestamp": now.strftime("%Y%m%d_%H%M%S")})

    if args.save_model:      # the best-val weights, i.e. the model whose metrics were just written (board export / replay)
        ckpt = checkpoint_path(args.exp_tag, result_name, data_name, pred_len, args.seq_len, seed)
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
    print(f"Using device: {device} | {args.model} | {args.data} | H {args.pred_len} | {args.exp_tag} | seeds {seeds}")
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


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting worker (one model x dataset x pred_len)")
    # data and run
    p.add_argument("--root_path", default="data/")
    p.add_argument("--data", required=True)
    p.add_argument("--features", default="M")
    p.add_argument("--model", default="FEATHer", choices=list_models())
    p.add_argument("--result_model_name", default=None,
                   help="name written to the results row (run.py: FEATHer_mix / FEATHer_CI); default = --model")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--label_len", type=int, default=96)
    p.add_argument("--seeds", default="2025,2026,2027")
    p.add_argument("--exp_tag", default="default")
    p.add_argument("--results_csv", default="results/_incoming/train.csv")
    p.add_argument("--hparams", default="", help="JSON of the run's hyperparameters, stored as is (run.py)")
    p.add_argument("--model_overrides", default="",
                   help='per-model hyperparameters "key=val;key2=val2" (run.py, from configs/<model>.yaml)')
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_model", action="store_true", help="save the best-val state dict (board export, replay)")
    # training (docs/EXPERIMENT_PROTOCOL.md)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience on the validation loss")
    p.add_argument("--loss", default="mse", choices=["l1", "mse"])
    p.add_argument("--lambda_spec", type=float, default=0.01, help="FEATHer band-separation loss weight")
    # FEATHer switches (frozen values come from configs/FEATHer.yaml; these are the ablation / control / size options)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--num_bands", type=int, default=3)
    p.add_argument("--point_kernel", type=int, default=None, help="POINT band conv width (odd)")
    p.add_argument("--gate_mode", default=None, choices=["banded", "static", "none"])
    p.add_argument("--gate_uniform", dest="use_gate", action="store_false", default=True, help="band weights 1/S")
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--mlp_rank", type=int, default=16)
    p.add_argument("--channel_mode", default=None, choices=["mix", "ci_full"])
    p.add_argument("--no_revin", dest="use_norm", action="store_false", default=True)
    p.add_argument("--head_mode", default=None, choices=["spk", "linear"])
    p.add_argument("--front", default=None, help="bands (default) | none (control: no bands, no gate)")
    p.add_argument("--mix_sep", type=int, default=0, help="control: residual MLP per channel (1->R->1)")
    p.add_argument("--head_seg", type=int, default=0, help="size: SPK segment length (0 = period)")
    p.add_argument("--agg_shared", type=int, default=0, help="size: one SPK smoothing kernel for all channels")
    return p.parse_args()


def _apply_overrides(args):
    """Set each --model_overrides key=value on args (bool, int, float, else str)."""
    for kv in (s.strip() for s in args.model_overrides.split(";")):
        if "=" not in kv:
            continue
        k, v = (t.strip() for t in kv.split("=", 1))
        if k in PROTOCOL_AXES:   # unified across models; a per-model value would break the comparison
            raise ValueError(f"--model_overrides may not set protocol axis '{k}'; got '{kv}'")
        if v.lower() in ("true", "false"):
            parsed = v.lower() == "true"
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    parsed = v
        setattr(args, k, parsed)


if __name__ == "__main__":
    args = parse_args()
    _apply_overrides(args)
    train(args)
