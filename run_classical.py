# -*- coding: utf-8 -*-
"""Classical (non-neural) forecasting harness — practitioner-tier baselines.

The gradient-boosted-tree and statistical methods shop-floor engineers
actually deploy do NOT fit the neural worker (no epochs / optimizer / nn.Module),
so they live here instead of the `baselines/` registry. Currently: **XGBoost**
(lightgbm is not in the `feather` env; xgboost 2.x is). Writes rows to the SAME
`results/fcst_results.csv` schema as the neural worker, so the paper tables
(main_table.py etc.) pick them up automatically under model name "XGBoost".

DESIGN (reasonable defaults, 2026-07-15):
  * channel-independent — one model per channel (D models), the per-sensor way
    practitioners deploy trees;
  * DIRECT multi-horizon via *horizon-as-feature*: features = [seq_len lags of
    the channel] + [horizon step h]; target = value at t+h (scalar). One model
    per channel predicts any h -> no D×H model explosion, no recursive error
    accumulation;
  * MASK-aware (observed-only): training rows whose target cell is imputed
    (NaN, from the loader mask) are dropped; test metrics use the shared
    nan-aware utils.metrics -> consistent with the neural models;
  * compute bound: per-channel training (window,h) rows subsampled to a cap;
    early stopping on a held-out tail of the train windows.

Not param-budget-constrained: this is the "practitioner tier" (accurate but
not MCU-deployable) contrasted with the sub-1K edge tier. `num_params` stores
the total tree-node count across the D boosters (a real model-size proxy, not
comparable to NN parameter counts — flagged at table time).

Usage:
    python run_classical.py --check
    python run_classical.py --model XGBoost --exp_tag main
    python run_classical.py --data Steel --pred_len 96
"""

import os
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import data_factory
from utils import metrics
from utils.seed import set_seed, parse_seed_list

warnings.filterwarnings("ignore")

CSV_FIELDS = [
    "exp_tag", "model", "data", "pred_len", "seq_len", "seed",
    "MSE", "MAE", "RMSE", "CORR", "R2",
    "val_loss", "best_epoch", "num_params", "timestamp",
]

# Same default group/horizons as the neural sweep (mfg main table).
MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange", "Electricity", "Traffic"]
GROUPS = {"mfg": MFG, "ltsf": LTSF, "all": MFG + LTSF}
PRED_LENS = [96, 192, 336, 720]

TRAIN_ROW_CAP = 300_000       # per-channel (window,h) subsample ceiling (fit)
VAL_ROW_CAP = 100_000         # per-channel val ceiling (early-stopping eval)


def _resolve_device(pref):
    """'auto' -> 'cuda' if a GPU is visible else 'cpu'. XGBoost 2.x hist runs
    5-10x faster on GPU, which the horizon-as-feature harness needs at scale."""
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_new = pd.DataFrame([row], columns=CSV_FIELDS)
    try:
        with open(csv_path, "x", newline="") as f:
            df_new.to_csv(f, index=False)
    except FileExistsError:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)


def _windows(arr, seq_len, pred_len):
    """(T, D) split array -> lag windows (N, seq_len, D) and target windows
    (N, pred_len, D), N = T - seq_len - pred_len + 1 (mirrors the loader)."""
    swv = np.lib.stride_tricks.sliding_window_view
    n = len(arr) - seq_len - pred_len + 1
    lags = swv(arr, seq_len, axis=0)[:n]                 # (N, D, seq_len)
    lags = np.transpose(lags, (0, 2, 1))                 # (N, seq_len, D)
    tgt = swv(arr[seq_len:], pred_len, axis=0)[:n]       # (N, D, pred_len)
    tgt = np.transpose(tgt, (0, 2, 1))                   # (N, pred_len, D)
    return lags, tgt


def run_one(args, data_name, pred_len, seed):
    import xgboost as xgb
    set_seed(seed, deterministic=True)
    rng = np.random.default_rng(seed)

    _, tr_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.seq_len, pred_len, "train")
    _, te_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.seq_len, pred_len, "test")
    tr, te = tr_loader.dataset, te_loader.dataset
    seq_len = args.seq_len

    # AUDIT-FIX (2026-07-19, #8): early-stop on the PROTOCOL validation split
    # (the same middle-10% window the 12 NN models select on), NOT the last
    # 10% of the train region — so every model, NN or tree, is selected on
    # identical held-out data. XGBoost fits on the FULL train split.
    _, va_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.seq_len, pred_len, "val")
    va = va_loader.dataset

    lags_tr, y_tr = _windows(np.asarray(tr.data_x, float), seq_len, pred_len)
    lags_va, _ = _windows(np.asarray(va.data_x, float), seq_len, pred_len)
    lags_te, _ = _windows(np.asarray(te.data_x, float), seq_len, pred_len)
    # targets from data_y (imputed cells are NaN -> observed-only)
    _, ty_tr = _windows(np.asarray(tr.data_y, float), seq_len, pred_len)
    _, ty_va = _windows(np.asarray(va.data_y, float), seq_len, pred_len)
    _, ty_te = _windows(np.asarray(te.data_y, float), seq_len, pred_len)

    N_tr, _, D = lags_tr.shape
    N_te = lags_te.shape[0]
    h_col = np.arange(pred_len, dtype=float)[None, :]     # (1, H)

    preds = np.full((N_te, pred_len, D), np.nan)
    total_nodes = 0
    val_sse = val_cnt = 0.0

    def _rows(L, Y, cap):
        """SAMPLE-FIRST (window,h) rows: pick observed (finite-target) flat
        indices, cap them, THEN gather only those feature rows. Avoids
        materialising the full n_windows*H expansion (multi-GB on big sets)."""
        yflat = Y.reshape(-1)                              # (n*H,)
        idx = np.flatnonzero(np.isfinite(yflat))          # observed cells only
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        win = idx // pred_len
        hh = (idx % pred_len).astype(float)
        X = np.concatenate([L[win], hh[:, None]], axis=1)  # (len(idx), seq_len+1)
        return X, yflat[idx]

    for c in range(D):
        # features = [seq_len lags of channel c, horizon step h]; target = t+h.
        # Fit on the FULL train split; early-stop on the protocol val split.
        Xf, yf = _rows(lags_tr[:, :, c], ty_tr[:, :, c], TRAIN_ROW_CAP)
        Xv, yv = _rows(lags_va[:, :, c], ty_va[:, :, c], VAL_ROW_CAP)

        use_es = len(yv) > 0
        model = xgb.XGBRegressor(
            n_estimators=getattr(args, "xgb_n_estimators", 400),
            max_depth=getattr(args, "xgb_max_depth", 6),
            learning_rate=getattr(args, "xgb_lr", 0.05),
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device=args.device,
            random_state=seed, n_jobs=4,
            early_stopping_rounds=30 if use_es else None,
        )
        if use_es:
            model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        else:
            model.fit(Xf, yf, verbose=False)

        # ---- predict test ----
        Lt = lags_te[:, :, c]
        Xt = np.concatenate([np.repeat(Lt, pred_len, axis=0),
                             np.tile(h_col.T, (N_te, 1))], axis=1)
        pc = model.predict(Xt).reshape(N_te, pred_len)
        preds[:, :, c] = pc

        total_nodes += model.get_booster().trees_to_dataframe().shape[0]
        if len(yv) > 0:
            val_sse += float(np.sum((model.predict(Xv) - yv) ** 2))
            val_cnt += len(yv)

    m = metrics.metric(preds, ty_te)                       # nan-aware
    val_mse = val_sse / max(1.0, val_cnt)
    return dict(MSE=m[0], MAE=m[1], RMSE=float(np.sqrt(m[0])), CORR=m[3],
                R2=m[4], val_loss=val_mse, num_params=total_nodes)


def load_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    if df.empty:
        return set()
    return set(zip(df["exp_tag"].astype(str), df["model"].astype(str),
                   df["data"].astype(str), df["pred_len"].astype(int),
                   df["seed"].astype(int)))


def main():
    p = argparse.ArgumentParser(description="Classical (XGBoost) forecasting harness")
    p.add_argument("--root_path", default="data/")
    p.add_argument("--features", default="M")
    p.add_argument("--model", default="XGBoost", choices=["XGBoost"])
    p.add_argument("--group", default="mfg", choices=list(GROUPS))
    p.add_argument("--data", default=None, help="single dataset (overrides --group)")
    p.add_argument("--pred_len", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seeds", default="")
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--base_seed", type=int, default=2025)
    p.add_argument("--exp_tag", default="main")
    p.add_argument("--results_csv", default="results/fcst_results.csv")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu (XGBoost)")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    args.device = _resolve_device(args.device)

    datasets = [args.data] if args.data else GROUPS[args.group]
    plens = [args.pred_len] if args.pred_len else PRED_LENS
    seeds = parse_seed_list(args.seeds, args.num_seeds, args.base_seed)
    done = load_done(args.results_csv)

    jobs = [(d, h, s) for d in datasets for h in plens for s in seeds]
    todo = [(d, h, s) for (d, h, s) in jobs
            if (args.exp_tag, args.model, d, h, s) not in done]
    print(f"{args.model}: {len(jobs)} cells | done {len(jobs)-len(todo)} | TODO {len(todo)}")
    if args.check:
        for d in datasets:
            miss = [h for h in plens for s in seeds
                    if (args.exp_tag, args.model, d, h, s) not in done]
            print(f"  {d:14s} missing {len(miss)}/{len(plens)*len(seeds)}")
        return

    n_failed = 0
    for (d, h, s) in todo:
        t0 = datetime.now()
        print(f"\n[XGB] {d} H{h} seed{s} ...", flush=True)
        try:
            r = run_one(args, d, h, s)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {d}/{h}/{s}: {e}")
            n_failed += 1
            continue
        row = {"exp_tag": args.exp_tag, "model": args.model, "data": d,
               "pred_len": h, "seq_len": args.seq_len, "seed": s,
               "MSE": round(r["MSE"], 6), "MAE": round(r["MAE"], 6),
               "RMSE": round(r["RMSE"], 6), "CORR": round(r["CORR"], 6),
               "R2": round(r["R2"], 6), "val_loss": round(r["val_loss"], 6),
               "best_epoch": 0, "num_params": int(r["num_params"]),
               "timestamp": t0.strftime("%Y%m%d_%H%M%S")}
        append_row(args.results_csv, row)
        print(f"  -> MSE={r['MSE']:.4f} MAE={r['MAE']:.4f} "
              f"nodes={r['num_params']} ({(datetime.now()-t0).seconds}s)")

    # AUDIT-FIX (2026-07-19, #7): exit non-zero if any cell failed, so a
    # silent-skip (bad row never written) surfaces instead of looking "not
    # yet run" — same P2-1 contract as the NN worker.
    if n_failed:
        print(f"\n[FAIL] {n_failed}/{len(todo)} cell(s) errored — see traceback(s) above")
        sys.exit(1)


if __name__ == "__main__":
    main()
