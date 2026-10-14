# -*- coding: utf-8 -*-
"""One dead sensor at inference: can channel mixing cover for it?

Exactly ONE sensor d fails and is scored on its own:

    fault k = 48 (primary)  the last 48 of the 96 input steps are stuck at the last
                            good value (sensor dies mid-window, logger holds the value)
    fault k = 96            the whole window is stuck at its first value

Trained checkpoints only, no retraining: FEATHer-mix (candR4) vs its mixer-off twin
FEATHer-CI (candR4ci) = the same model with / without mixing, 6 mfg data x 4 H x 3
seeds, every sensor in turn.  Each checkpoint must reproduce its CSV test MSE first
(catches a wrong model rebuild).  Scoring = observed cells only (NaN targets skipped).

PRE-DECLARED RULE (written before any run):
  per dataset, mean over H, seeds and failed sensors, failed-sensor MSE:
     gap_clean = mix/ci - 1 on that sensor, clean input
     gap_fault = mix/ci - 1 on that sensor, fault k=48
  "mixing covers a dead sensor" iff gap_fault <= gap_clean - 2 percentage points on
  >= 5 of 6 datasets, AND the healthy sensors of the mix model lose <= 2 % on average
  (corruption must not spread).  Otherwise the idea is dropped.  k=96 is reported only.

    python tools/dev/sensor_dropout.py            -> results/dev/dev_sensor_dropout.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools", "dev"))
os.chdir(REPO)
from worker import load_worker  # noqa: E402
from baselines import get_method_defaults, get_dataset_overrides  # noqa: E402

DATA = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
HS = [96, 192, 336, 720]
SEEDS = [2025, 2026, 2027]
ARMS = {"mix": ("candR4", "results/_archive/pre_raw_20261007/feather_mfg_candR4.csv", []),
        "ci": ("candR4ci", "results/_archive/pre_raw_20261007/feather_mfg_candR4_ci.csv", ["--channel_mode", "ci_full"])}
KS = [48, 96]
OUT = "results/dev/dev_sensor_dropout.csv"


def build_args(worker, arm, data, H, seed):
    tag, csv, extra = ARMS[arm]
    merged = {**get_method_defaults("FEATHer"), **get_dataset_overrides("FEATHer", data)}
    lr = merged.pop("lr", 1e-3)
    merged.pop("loss", None)
    if arm == "ci":
        merged.pop("channel_mode", None)
    ovr = ";".join(f"{k}={v}" for k, v in sorted(merged.items()))
    argv = ["--model", "FEATHer", "--data", data, "--pred_len", str(H), "--seq_len", "96",
            "--seeds", str(seed), "--exp_tag", tag, "--results_csv", csv,
            "--lr", str(lr), "--loss", "mse", "--model_overrides", ovr] + extra
    sys.argv = ["train.py"] + argv
    args = worker.parse_args()
    worker._apply_overrides(args)
    args.label_len = min(args.label_len, args.seq_len)
    return args


@torch.no_grad()
def predict(model, X, device, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(device)).float().cpu())
    return torch.cat(out)


def sse(pred, Y):
    d = (pred - Y).numpy().astype(np.float64)   # float32 sums drift ~0.1 % over 6e5 cells
    f = np.isfinite(d)
    return np.where(f, d ** 2, 0).sum((0, 1)), f.sum((0, 1))   # per sensor


def main():
    worker = load_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = set()
    if os.path.exists(OUT):
        o = pd.read_csv(OUT)
        done = set(zip(o.arm, o.data, o.pred_len, o.seed))
    for data in DATA:
        for H in HS:
            a0 = build_args(worker, "mix", data, H, SEEDS[0])
            df, _, _ = worker.data_factory.data_select(data, a0.root_path)
            nf = df.shape[1] - 1
            test, _ = worker.data_factory.data_provider(
                a0.root_path, data, a0.features, a0.batch_size, a0.seq_len, a0.label_len, H, "test")
            xs, ys = [], []
            for i in range(len(test)):
                bx, by, _, _ = test[i]
                xs.append(torch.as_tensor(bx)[:, :nf]); ys.append(torch.as_tensor(by)[-H:, :nf])
            X = torch.stack(xs).float(); Y = torch.stack(ys).float()
            L = X.shape[1]
            for arm in ARMS:
                for seed in SEEDS:
                    if (arm, data, H, seed) in done:
                        continue
                    args = build_args(worker, arm, data, H, seed)
                    ckpt = worker.checkpoint_path(args.exp_tag, "FEATHer", data, H, 96, seed)
                    torch.manual_seed(seed)
                    m = worker.get_model("FEATHer", args, n_features=nf, seq_len=96, pred_len=H).float().to(device)
                    m.load_state_dict(torch.load(ckpt, map_location=device))
                    m.eval()
                    s0, n0 = sse(predict(m, X, device), Y)
                    ref = pd.read_csv(ARMS[arm][1])
                    ref = ref[(ref.exp_tag == ARMS[arm][0]) & (ref.data == data) & (ref.pred_len == H) & (ref.seed == seed)]
                    mse_clean_all = s0.sum() / n0.sum()
                    rel = abs(mse_clean_all / ref.MSE.iloc[0] - 1)
                    if rel > 1e-5:
                        raise SystemExit(f"rebuild mismatch {arm} {data} H{H} s{seed}: "
                                         f"{mse_clean_all:.6f} vs CSV {ref.MSE.iloc[0]:.6f}")
                    rows = []
                    for k in KS:
                        for d in range(nf):
                            Xf = X.clone()
                            hold = Xf[:, L - k - 1, d] if k < L else Xf[:, 0, d]
                            Xf[:, L - k:, d] = hold.unsqueeze(1)
                            s1, n1 = sse(predict(m, Xf, device), Y)
                            hm = np.ones(nf, bool); hm[d] = False
                            rows.append(dict(
                                arm=arm, data=data, pred_len=H, seed=seed, k=k, sensor=d, D=nf,
                                mse_failed_clean=s0[d] / max(n0[d], 1), mse_failed_fault=s1[d] / max(n1[d], 1),
                                mse_healthy_clean=s0[hm].sum() / max(n0[hm].sum(), 1),
                                mse_healthy_fault=s1[hm].sum() / max(n1[hm].sum(), 1),
                                rebuild_rel=rel))
                    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                    print(f"{arm:3s} {data:12s} H{H:<3d} s{seed} D={nf} clean {mse_clean_all:.4f} "
                          f"(CSV {ref.MSE.iloc[0]:.4f})", flush=True)
    print("SENSOR DROPOUT DONE")


if __name__ == "__main__":
    main()
