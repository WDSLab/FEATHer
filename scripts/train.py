# -*- coding: utf-8 -*-
"""
Forecasting worker.

Runs one (model, dataset, pred_len) configuration across a given list of
seeds. Each completed run appends one row to --results_csv (run.py passes results/_incoming/<host>_<stage>.csv).

Normally called by the entry point `run.py`, which resolves the setting of every cell
(tools/results/protocol.py) and passes exp_tag "<role>|<config_id>". Manual use:

Usage:
    python scripts/train.py --model FEATHer --data ETTh1 --pred_len 96 --seeds 2025,2026,2027
        --exp_tag "manual|test" --results_csv results/_incoming/manual.csv
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (scripts/train.py)

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
    # AUDIT-FIX (2026-07-14, cfd90c5, P2-2) ??atomic create-with-header:
    # under --ngpu/--jobs_per_gpu several workers can race on a FRESH csv; a
    # non-atomic exists()-then-write can leave a second header mid-file and
    # break resume parsing. open(mode="x") is O_CREAT|O_EXCL ??exactly one
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
    # run.py exp_tags look like "main|cand:ETTh1"; '|' and ':' are not legal in Windows paths
    # (older tags such as "candR4" contain neither, so their checkpoint paths are unchanged).
    folder = exp_tag.replace("|", "__").replace(":", "-")
    return os.path.join(
        "results", "checkpoints", folder, model_name,
        f"{data_name}_H{pred_len}_L{seq_len}_s{seed}.pth",
    )


def resolve_gate_bounds(args, train_data, n_features):
    """--gate_bounds tercile -> explicit slice edges, from the TRAIN split only.

    (2026-09-15) Equal-width gate slices put 80-95% of the energy in the lowest
    slice on every mfg dataset, so the gate's input barely varies. Tercile edges
    put ~1/S of the mean TRAIN spectrum in each slice. Computed on
    non-overlapping, instance-normalized train windows (the normalization the
    model applies), so val/test never influence it, and deterministic, so a
    checkpoint is rebuilt with the same edges (tools/dev/gate_swap.py calls it).
    """
    mode = getattr(args, "gate_bounds", None)
    if mode not in ("tercile", "tercile_sensor"):
        return
    L, S = args.seq_len, args.num_bands
    X = np.asarray(train_data.data_x, dtype=np.float64)[:, :n_features]
    n = (len(X) // L) * L
    W = X[:n].reshape(-1, L, X.shape[1])
    W = W[np.isfinite(W).all(axis=(1, 2))]
    W = (W - W.mean(1, keepdims=True)) / np.sqrt(W.var(1, keepdims=True) + 1e-5)
    P = (np.abs(np.fft.rfft(W, axis=1)) ** 2).mean(axis=0)      # (nf, D)
    nf = L // 2 + 1

    def edges_of(spectrum):
        c = np.cumsum(spectrum) / spectrum.sum()
        out = []
        for k in range(1, S):
            e = int(np.searchsorted(c, k / S)) + 1
            lo = out[-1] + 1 if out else 1
            out.append(min(max(e, lo), nf - (S - k)))
        return ",".join(str(e) for e in out)

    if mode == "tercile_sensor":
        args.gate_bounds = ";".join(edges_of(P[:, d]) for d in range(P.shape[1]))
    else:
        args.gate_bounds = edges_of(P.mean(1))
    print(f"[gate_bounds] {mode} edges from the train split: {args.gate_bounds}")


# =============================================================================
# Per-run training
# =============================================================================

def _masked_loss(output, target, loss_name):
    """L1/MSE over OBSERVED target cells only. MASK-FIX (2026-07-15).

    The loader NaN-outs imputed TARGET cells for the gap-filled sensor sets
    (observed-only scoring). Plain nn.L1Loss/MSELoss returns NaN for any batch
    that contains a NaN, which the training loop then skips ??silently dropping
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

    # Lookback fix (2026-08-03): label_len must be <= seq_len, else __getitem__'s
    # r_begin = index + seq_len - label_len goes negative for early indices,
    # producing empty [0, D] target windows that crash the DataLoader collate.
    # Default label_len=96 breaks the short-lookback sweep (seq_len=48). No-op at
    # seq_len>=96 (byte-identical main/L192/L336). Edge models ignore label_len.
    args.label_len = min(args.label_len, args.seq_len)

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

    resolve_gate_bounds(args, train_data, n_features)

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

    # 2026-10-06: SAMformer is trained with sharpness-aware minimisation (part of
    # the method, ICML'24); every other model keeps the shared Adam optimizer.
    use_sam = args.model == "SAMformer"
    if use_sam:
        from baselines.SAMformer.wrapper import make_sam
        optimizer = make_sam(model.parameters(), lr=args.lr,
                             rho=float(getattr(args, "samformer_rho", 0.5)))
    elif args.optimizer == "adamw":
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

    # Validation-based model selection ??same protocol as every baseline's
    # official implementation: val loss picks the epoch (with early
    # stopping), test is evaluated exactly once on the selected model.
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = args.patience

    # --oracle_test (diagnostic, opt-in): min-over-epochs TEST MSE, the legacy
    # FASTer selection (no val, best test epoch). Ceiling probe ONLY ??test
    # peeking, never reportable. Recorded as a second "<exp_tag>_oracle" row.
    oracle_mse = float("inf")
    oracle_mae = float("nan")
    oracle_epoch = -1

    # 2026-10-07: optional per-model hooks for a baseline's own training recipe (DiPE: router
    # temperature schedule + time/frequency loss mix). Models without them are untouched.
    _epoch_hook = getattr(model, "_epoch_start", None)
    _train_loss = getattr(model, "_train_loss", None)
    _test_hook = getattr(model, "_test_start", None)

    for epoch in range(args.num_epochs):
        if _epoch_hook is not None:
            _epoch_hook(epoch)
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
            if _train_loss is not None:
                main_loss = _train_loss(output, target, main_loss)
            if use_spec_loss:
                loss = main_loss + args.lambda_spec * losses.spectral_separation_loss_scales(H)
            else:
                loss = main_loss
            # 2026-10-06: a baseline's own auxiliary training term (TimeBase's
            # orthogonality loss) is exposed by its wrapper as `_aux_loss`.
            aux = getattr(model, "_aux_loss", None)
            if aux is not None:
                loss = loss + aux

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_sam:
                # SAM: perturb to w + e(w), recompute the loss there, update w.
                optimizer.first_step(zero_grad=True)
                out2 = model(batch_x[:, :, :n_features])[:, -pred_len:]
                loss2 = _masked_loss(out2, target, args.loss)
                if torch.isfinite(loss2):
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.second_step(zero_grad=True)
                else:   # restore w without stepping
                    with torch.no_grad():
                        for g in optimizer.param_groups:
                            for prm in g["params"]:
                                if "e_w" in optimizer.state[prm]:
                                    prm.sub_(optimizer.state[prm]["e_w"])
                    optimizer.zero_grad(set_to_none=True)
            else:
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
        # val_cnt == 0 must yield NaN ??not 0.0. The old `sum / max(1, cnt)`
        # returned 0.0, which reads as a perfect score, gets accepted as
        # best_val, and lets the P2-5 "no finite val_loss" guard pass on a
        # degenerate (NaN-weight) model. NaN keeps best_state None -> P2-5 fails.
        val_loss = val_sum / val_cnt if val_cnt > 0 else float("nan")

        # ---------- Oracle test probe (diagnostic, opt-in) ----------
        # NOTE: with early stopping active the probe only sees epochs up to the
        # stop; pass --patience == --num_epochs for the full legacy-style sweep.
        if args.oracle_test:
            o_se = o_ae = 0.0
            o_n = 0
            with torch.no_grad():
                for batch_x, batch_y, _, _ in test_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    output = model(batch_x[:, :, :n_features])
                    y_true = batch_y[:, :, :n_features][:, -pred_len:].cpu().numpy()
                    y_pred = output[:, -pred_len:].cpu().numpy()
                    diff = y_pred - y_true          # NaN where target imputed
                    fin = np.isfinite(diff)
                    o_se += float(np.sum(diff[fin] ** 2))
                    o_ae += float(np.sum(np.abs(diff[fin])))
                    o_n += int(fin.sum())
            if o_n > 0:
                o_mse = o_se / o_n
                if np.isfinite(o_mse) and o_mse < oracle_mse:
                    oracle_mse = o_mse
                    oracle_mae = o_ae / o_n
                    oracle_epoch = epoch

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
    # best-val model ??evaluating the last-epoch weights would write a
    # normal-looking row (val_loss=inf, best_epoch=0) and --save_model would
    # checkpoint a degenerate state. Fail the seed instead: no row is
    # written, and the worker exits non-zero via the P2-1 path so the
    # orchestrator surfaces the group.
    if best_state is None:
        # 10-07: record the failure as a row (metrics NaN, best_epoch -1) so run.py counts the cell as
        # done and select_verdict sees a non-finite candidate (protocol rule 6) instead of retrying it
        # forever; the worker still exits non-zero so the failure is reported.
        append_result_row(args.results_csv, {
            "exp_tag": args.exp_tag, "model": getattr(args, "result_model_name", None) or args.model,
            "data": data_name, "pred_len": pred_len, "seq_len": args.seq_len, "seed": seed,
            "MSE": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "CORR": float("nan"),
            "R2": float("nan"), "val_loss": float("nan"), "best_epoch": -1,
            "num_params": int(sum(p.numel() for p in model.parameters())),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")})
        raise RuntimeError(
            f"no finite val_loss in {args.num_epochs} epoch(s) ??degenerate "
            f"run (diverged/NaN); refusing to report last-epoch metrics")
    model.load_state_dict(best_state)
    model.eval()
    if _test_hook is not None:
        _test_hook()
    # AUDIT-FIX (2026-07-19, #3): MSE/MAE are a GLOBAL observed-cell mean
    # (nansum of squared/abs error over finite cells / finite-cell count),
    # NOT a batch-size-weighted mean of per-batch nanmeans. On unmasked data
    # the two are algebraically identical (every cell observed); on the
    # mask.csv sets they diverge ??up to ~13% on MetroPT3 ??because batch-size
    # weighting ignores that batches carry different observed-cell counts.
    # This matches the persistence/classical baselines' estimator so the main
    # table compares like with like. CORR/R2 stay batch-weighted (they are
    # per-(B,F) pair averages with no simple global closed form).
    se_sum = ae_sum = 0.0
    n_obs = 0
    cr_sum = np.zeros(2, dtype=np.float64)   # [CORR, R2]
    cr_cnt = np.zeros(2, dtype=np.float64)
    # Downstream-KPI dump. The test loader is shuffle=False / drop_last=False
    # (data_factory), so a running counter IS the window index within the test
    # split, which is what maps a window back to its wall-clock timestamp.
    dump_pred, dump_true, dump_widx = [], [], []
    win0 = 0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output = model(batch_x[:, :, :n_features])
            y_true = batch_y[:, :, :n_features][:, -pred_len:].cpu().numpy()
            y_pred = output[:, -pred_len:].cpu().numpy()
            # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #2): the nan-aware metric
            # drops NaN cells ??intended for imputed TARGET cells, but it would
            # ALSO silently drop a non-finite PREDICTION on an observed cell,
            # improving a partially-diverged model's score. A best-val model
            # must be finite; treat any non-finite pred on an observed cell as
            # a degenerate run (P2-5 path).
            obs = np.isfinite(y_true)
            if not np.isfinite(y_pred[obs]).all():
                raise RuntimeError(
                    "non-finite prediction on observed test cells ??"
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
            if args.dump_preds:
                # Select strided windows by their GLOBAL index so the choice is
                # independent of batch_size (a batch boundary must not shift
                # which days get kept).
                gidx = np.arange(win0, win0 + bs)
                keep = np.nonzero(gidx % max(1, args.dump_stride) == 0)[0]
                if len(keep):
                    dump_pred.append(y_pred[keep].astype(np.float32))
                    dump_true.append(y_true[keep].astype(np.float32))
                    dump_widx.append(gidx[keep])
            win0 += bs
    # AUDIT-FIX (2026-07-19, MASK-NAN-GRAD #1): no observed cell across all
    # batches (fully diverged / fully masked) must NOT become `0/max(1,0)==0`
    # ??a fake perfect score that slips past the P2-5 guard (the val loop was
    # fixed in d56aa05; this is its test-loop twin).
    if n_obs == 0:
        raise RuntimeError(
            "no observed test cells across all batches ??degenerate run")
    mse = se_sum / n_obs
    mae = ae_sum / n_obs
    # RMSE = sqrt(global MSE) (P1-3): identity holds by construction now.
    test_metric = np.array([
        mse, mae, float(np.sqrt(mse)),
        cr_sum[0] / max(cr_cnt[0], 1.0),
        cr_sum[1] / max(cr_cnt[1], 1.0),
    ], dtype=np.float64)

    # ---------- Dump test predictions for downstream-KPI analysis ----------
    if args.dump_preds and dump_pred:
        os.makedirs(args.dump_preds, exist_ok=True)
        yp = np.concatenate(dump_pred, axis=0)
        yt = np.concatenate(dump_true, axis=0)
        widx = np.concatenate(dump_widx, axis=0)
        # Physical units require the train-fit StandardScaler. Store its
        # parameters rather than inverting here, so the archive keeps the same
        # normalized values that produced the reported MSE and the inversion
        # stays auditable. Column order is data_x's, i.e. df_raw minus 'date'.
        sc = getattr(test_data, "scaler", None)
        mean = np.asarray(getattr(sc, "mean_", np.zeros(n_features)), dtype=np.float64)
        scale = np.asarray(getattr(sc, "scale_", np.ones(n_features)), dtype=np.float64)
        # Window w covers raw rows [border1+w+seq_len, border1+w+seq_len+pred_len).
        border1 = int(getattr(test_data, "border1", 0))
        stamps = test_data.df_raw["date"].to_numpy()[
            border1 + args.seq_len + widx][:len(widx)]
        out = os.path.join(
            args.dump_preds,
            "%s__%s__H%d__s%d.npz" % (args.model, data_name, pred_len, seed))
        np.savez_compressed(
            out, y_pred=yp, y_true=yt, window_index=widx,
            start_time=stamps.astype("datetime64[s]").astype(np.int64),
            scaler_mean=mean[:n_features], scaler_scale=scale[:n_features],
            columns=np.array([c for c in test_data.df_raw.columns if c != "date"],
                             dtype=object)[:n_features],
            seq_len=args.seq_len, pred_len=pred_len, stride=args.dump_stride,
            border1=border1, MSE=mse, MAE=mae)
        print("[dump_preds] %s  windows=%d  shape=%s" % (out, len(widx), yp.shape))

    # ---------- Persist ----------
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    row = {
        "exp_tag":   args.exp_tag,
        "model":     getattr(args, "result_model_name", None) or args.model,
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

    # Diagnostic oracle row (same schema, "<exp_tag>_oracle") ??min-over-epochs
    # test MSE, the legacy FASTer-style ceiling. CORR/R2 not tracked per epoch.
    if args.oracle_test and np.isfinite(oracle_mse):
        orow = dict(row)
        orow.update({
            "exp_tag":    args.exp_tag + "_oracle",
            "MSE":        round(float(oracle_mse), 6),
            "MAE":        round(float(oracle_mae), 6),
            "RMSE":       round(float(np.sqrt(oracle_mse)), 6),
            "CORR":       float("nan"),
            "R2":         float("nan"),
            "best_epoch": int(oracle_epoch),
        })
        append_result_row(args.results_csv, orow)
        print(f"  -> ORACLE (min-over-epochs test, ep {oracle_epoch+1}): "
              f"MSE={oracle_mse:.4f} [diagnostic only ??test peeking]")

    # Optional: deterministic checkpoint for downstream robustness eval.
    # The model currently holds the best-val state, so the saved checkpoint
    # is exactly the model whose test metrics were just reported.
    if args.save_model:
        # result name (FEATHer_mix / FEATHer_CI), not the registry name "FEATHer": both variants share
        # exp_tag "main|frozen" and would otherwise overwrite each other's checkpoint
        ckpt = checkpoint_path(args.exp_tag, getattr(args, "result_model_name", None) or args.model,
                               data_name, pred_len, args.seq_len, seed)
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"  -> checkpoint saved ??{ckpt}")

    print(f"  -> Test (best-val ep {best_epoch+1}): MSE={test_metric[0]:.4f} "
          f"MAE={test_metric[1]:.4f} saved ??{args.results_csv}")


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
        print(f"[FATAL] {n_failed}/{len(seeds)} seed run(s) failed ??"
              f"exiting non-zero so the orchestrator reports this group.")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting worker (one model 횞 dataset 횞 pred_len)")

    # Data
    p.add_argument("--root_path", type=str, default="data/")
    p.add_argument("--data",      type=str, required=True)
    p.add_argument("--features",  type=str, default="M")

    # Model
    p.add_argument("--result_model_name", type=str, default=None,
                   help="name written to the results row (run.py: FEATHer_mix / FEATHer_CI); default = --model")
    p.add_argument("--model", type=str, default="FEATHer",
                   choices=list_models())
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
    # SPK head nonlinearity (2026-08-26): 0 = canonical linear cross-period
    # backbone; >0 inserts one GELU hidden layer of that width.
    p.add_argument("--spk_hidden",    type=int, default=0)

    # Per-band nonlinearity (2026-08-27): breaks the band-bank collapse.

    p.add_argument("--gate_zero_init", action="store_true",


                   help="Zero-init the global gate conv so all bands start at 1/S.")


    p.add_argument("--band_mode", type=str, default=None,


                   choices=["conv", "diff", "fft", "fftoct"])


    p.add_argument("--band_act", type=str, default=None,

                   choices=["none", "gelu", "se"])
    # Robustness diagnostics (2026-07-31): isolate the fragility source.
    p.add_argument("--no_revin",    dest="use_norm", action="store_false", default=True)
    p.add_argument("--gate_uniform", dest="use_gate", action="store_false", default=True)
    # robust-RevIN (2026-08-01): median/MAD statistics instead of mean/std.
    p.add_argument("--robust_revin", dest="use_robust_norm", action="store_true", default=False)
    # Instance-norm variant (2026-08-02): winsorized full RevIN vs center-only.
    p.add_argument("--revin_mode", type=str, default=None,
                   choices=["standard", "mad", "winsor", "center",
                            "last", "last_std", "mean", "learn_level"])
    p.add_argument("--winsor_bound", type=float, default=None)
    p.add_argument("--center_trim",  type=float, default=None)
    p.add_argument("--channel_mode", type=str, default=None,
                   choices=["mix", "ci_full"])
    # FFT-gate variant (2026-08-14): "global" = canonical (shape-blind, see
    # FFTFrequencyGate docstring); "banded" = per-band energy-ratio gate.
    p.add_argument("--gate_mode", type=str, default=None,
                   choices=["global", "banded", "banded_sensor", "static", "static_rng", "none", "none_rng", "softmax"])
    p.add_argument("--gate_bounds", type=str, default=None,
                   help="banded gate slice edges: linear = equal width (canonical), "
                        "octave = LOW slice matched to the AvgPool4 branch, then doubling; "
                        "tercile = equal-energy edges from the TRAIN split, tercile_sensor = "
                        "one edge pair per sensor; or explicit 'e1,e2' / 'e1,e2;e1,e2;...'.")
    # DTK latent nonlinearity (2026-08-16): "none" = canonical (linear DTK,
    # which makes the multi-band block collapse to one filter).
    p.add_argument("--dtk_act", type=str, default=None,
                   choices=["none", "gelu", "relu", "tanh"])
    # Forecasting head (2026-08-24): "spk" = canonical period-aware sparse
    # kernel; "linear" = dense time projection, the SPK ablation control.
    # NB the control is much LARGER (Steel/H96 ~166 -> ~9.3k params), so read
    # its accuracy delta together with the parameter count.
    p.add_argument("--head_mode", type=str, default=None,
                   choices=["spk", "linear", "mlp", "conv"])
    # DTK presence (2026-08-24): "none" removes the shared temporal kernel
    # outright (bands pass through), vs --kernel_size 1 which only drops the
    # temporal mixing.
    p.add_argument("--dtk_mode", type=str, default=None,
                   choices=["full", "none", "mlp"])
    # Redundancy controls (2026-08-25). The architecture carries two mechanisms
    # on each axis ??time: band convs vs the DTK conv; channel: in_proj vs the
    # residual MLP ??so removing either one alone leaves the other to cover for
    # it and reads as "no contribution". These two flags make it possible to
    # keep exactly ONE mechanism per axis.
    p.add_argument("--point_kernel", type=int, default=None,
                   help="Width of the POINT branch conv (odd; default 1 = a "
                        "per-channel scalar, i.e. no temporal mixing).")
    p.add_argument("--band_kernels", type=str, default=None,
                   help="밴드 conv 커널 사다리 (num_bands-1 개). 예: '3,7,15'")
    p.add_argument("--band_dilations", type=str, default=None,
                   help="밴드 conv 팽창률 (num_bands-1 개). 예: '1,4,15'")
    p.add_argument("--no_in_proj", action="store_true",
                   help="Drop the linear channel mixer, keep the residual MLP.")
    # Mixer cost-reduction screen (2026-09-14). Unset = shipping model.
    p.add_argument("--mix_mode", type=str, default=None, choices=["hidden", "input", "input_raw"],
                   help="hidden = residual MLP on the gated bands (canonical); "
                        "input = on the time-pooled normalized input.")
    p.add_argument("--mix_pool", type=int, default=None,
                   help="Time pooling factor for --mix_mode input (divides seq_len).")
    p.add_argument("--mlp_tie", action="store_true",
                   help="Tie the residual MLP: W_up = W_down^T (own bias).")
    p.add_argument("--band_zero_sum", action="store_true",
                   help="POINT/MID conv taps forced to sum to 0 (DC gain 0).")
    p.add_argument("--gate_mix", type=str, default=None, choices=["none", "fft", "static"],
                   help="G2: spectrum sets the per-sensor strength of the mixer correction.")
    p.add_argument("--head2", type=str, default=None, choices=["none", "fft", "static"],
                   help="G4-lite: second SPK readout at 2x/0.5x period, blended by the spectrum.")
    p.add_argument("--mix_listen", action="store_true", help="A: per-sensor scale on the mixer correction.")
    p.add_argument("--mix_lag", type=str, default=None, help="B: 'k,dil' temporal conv on the mixer summary.")
    p.add_argument("--mix_zero_init", action="store_true", help="C: zero-init the mixer up-projection.")
    p.add_argument("--far_coarse", type=int, default=0, help="Near/far head (idea A): coarse factor r=4; 0 = off.")
    p.add_argument("--near_len", type=int, default=96, help="Near/far head: steps of the full-resolution detail head.")
    p.add_argument("--factor_head", type=int, default=0, help="P1 (10-06): shared-factor forecasting; 0 = off.")
    p.add_argument("--factor_src", type=str, default=None, help="P1 factor source: hidden (default) | low (P2).")
    p.add_argument("--front", type=str, default=None, help="bands (default) | none (P3: no bands, no FFT gate).")
    p.add_argument("--mix_sep", type=int, default=0,
                   help="10-11 mixing-only control: residual MLP per channel (1->R->1), no cross-channel term; 0 = off.")
    p.add_argument("--mix_stats", type=int, default=0,
                   help="10-12 redesign candidate: the residual mixer also reads window mean/log-std; 0 = off.")

    # Training
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--num_epochs",  type=int,   default=50)
    p.add_argument("--patience",    type=int,   default=10,
                   help="Early-stopping patience on val loss (epochs).")
    p.add_argument("--loss",        type=str,   default="mse", choices=["l1", "mse"])   # protocol: MSE
    p.add_argument("--lambda_spec", type=float, default=0.01)
    # Probe flags (2026-07-22, backward-compatible: defaults reproduce the
    # legacy Adam/no-wd behavior byte-identically). Added to test the TPAMI
    # optimizer recipe (AdamW + weight_decay 1e-4 + lr 1e-2).
    p.add_argument("--optimizer",    type=str,   default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    # Oracle probe (2026-08-09): min-over-epochs TEST MSE ??the legacy FASTer
    # selection. DIAGNOSTIC ceiling only (test peeking, never reportable).
    # Writes an extra "<exp_tag>_oracle" row. Use --patience == --num_epochs
    # to disable early stop for the full legacy-style epoch sweep.
    p.add_argument("--oracle_test", action="store_true")

    # Seed control
    p.add_argument("--seeds",         type=str, default="")
    p.add_argument("--num_seeds",     type=int, default=3)   # protocol: 3 seeds
    p.add_argument("--base_seed",     type=int, default=2025)
    p.add_argument("--deterministic", action="store_true", default=True)

    # Experiment grouping
    p.add_argument("--exp_tag",    type=str, default="default")
    p.add_argument("--results_csv", type=str, default="results/_incoming/train.csv")

    # Generic per-(model, dataset) HP injection ??orchestrator passes
    # "key=val;key2=val2" pairs derived from get_dataset_overrides(); each
    # entry is parsed (int/float/bool/str) and set on `args` BEFORE the
    # model is constructed, so wrapper.py's getattr(args, "key", default)
    # picks up the dataset-specific value.
    p.add_argument("--model_overrides", type=str, default="",
                   help='Semicolon-separated key=value HP overrides '
                        '(e.g. "patchtst_d_model=16;patchtst_n_heads=4").')

    # Data slicing (legacy raw knobs ??subsample the TRAIN split only)
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
    # Downstream-KPI support: the test loop scores predictions and then throws
    # them away, so any decision-level metric (peak capture, alarm lead time)
    # is impossible after the fact. Dump the test predictions alongside the
    # scaler so post-processing can invert back to physical units.
    p.add_argument("--dump_preds", type=str, default=None,
                   help="Directory to write test predictions (.npz) for "
                        "downstream KPI analysis. Predictions are stored in "
                        "NORMALIZED space exactly as scored, with the "
                        "StandardScaler mean/scale needed to invert them.")
    p.add_argument("--dump_stride", type=int, default=1,
                   help="Keep every Nth test window in --dump_preds. Test "
                        "windows are stride-1 and thus ~99%% overlapping; a "
                        "day-ahead decision needs only non-overlapping ones "
                        "(e.g. 96 for EWELD's 15-min grid = one per day).")

    args = p.parse_args()
    if args.train_frac is not None:
        pct = max(1, min(100, int(round(args.train_frac * 100))))
        args.percent = pct
        args.starting_percent = (100 - pct) if args.train_frac_mode == "recent" else 0
    return args


def _apply_overrides(args):
    """Parse --model_overrides string and assign each key=value to `args`.

    Values are coerced to int ??float ??bool ??str in that order.
    """
    if not args.model_overrides:
        return
    # AUDIT-FIX (2026-07-19, #10): overrides carry per-(model,dataset) ARCH
    # hyperparameters only. The unified-protocol axes must never be set this
    # way ??a stray key in _DATASET_OVERRIDES would silently break cross-model
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




