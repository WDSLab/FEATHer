# -*- coding: utf-8 -*-
"""Which sensors does the mixer help, and which does it hurt?  (2026-09-21)

Loads the candidate (mix) and its mixer-off twin (ci) checkpoints for the same
seeds, scores test MSE PER SENSOR, and reports how many sensors gain / lose from
mixing (threshold 1 %), plus per-seed MSE side by side (bad-basin check).

    python tools/dev/mix_per_sensor.py
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
from gate_check import argv_for  # noqa: E402
from gate_swap import load_worker  # noqa: E402

ARMS = {"mix": "mx_in1_r4_gsensor", "ci": "mx_in1_r4_gsensor_ci"}
DATA = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    ["Steel", "EWELD", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3"]


def setup(worker, arm, data):
    argvs, _ = argv_for(worker, "mix0914", arm, data)
    best = None
    for argv in argvs:
        sys.argv = ["run_forecast.py"] + argv
        cand = worker.parse_args()
        ref = pd.read_csv(cand.results_csv, on_bad_lines="skip")
        r = ref[(ref.exp_tag == cand.exp_tag) & (ref.data == data) & (ref.pred_len == int(cand.pred_len))]
        if not r.empty and (best is None or r.val_loss.median() < best[0]):
            best = (r.val_loss.median(), argv)
    sys.argv = ["run_forecast.py"] + best[1]
    args = worker.parse_args()
    worker._apply_overrides(args)
    args.label_len = min(args.label_len, args.seq_len)
    ref = pd.read_csv(args.results_csv, on_bad_lines="skip")
    r = ref[(ref.exp_tag == args.exp_tag) & (ref.data == data) & (ref.pred_len == int(args.pred_len))]
    return args, r.set_index("seed").MSE


@torch.no_grad()
def per_sensor(worker, args, data, seed, test, nf, device):
    from torch.utils.data import DataLoader
    H = int(args.pred_len)
    ckpt = worker.checkpoint_path(args.exp_tag, args.model, data, H, args.seq_len, seed)
    if not os.path.exists(ckpt):
        return None
    torch.manual_seed(seed)
    m = worker.get_model(args.model, args, n_features=nf, seq_len=args.seq_len, pred_len=H).float().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    se, n = np.zeros(nf), np.zeros(nf)
    for bx, by, _, _ in DataLoader(test, batch_size=512, shuffle=False, num_workers=0):
        y = m(bx.float().to(device)[:, :, :nf])[:, -H:].cpu().numpy()
        t = by.float()[:, -H:, :nf].numpy()
        d = y - t
        f = np.isfinite(d)
        se += np.where(f, d ** 2, 0).sum((0, 1))
        n += f.sum((0, 1))
    return se, n


def main():
    worker = load_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for data in DATA:
        cfg = {k: setup(worker, v, data) for k, v in ARMS.items()}
        args = cfg["mix"][0]
        df, _, _ = worker.data_factory.data_select(data, args.root_path)
        nf = df.shape[1] - 1
        test, _ = worker.data_factory.data_provider(args.root_path, data, args.features, args.batch_size,
                                                    args.seq_len, args.label_len, int(args.pred_len), "test",
                                                    starting_percent=args.starting_percent, percent=args.percent)
        seeds = sorted(set(cfg["mix"][1].index) & set(cfg["ci"][1].index))
        # per-seed MSE side by side (from the CSV, all common seeds)
        pm, pc = cfg["mix"][1].reindex(seeds), cfg["ci"][1].reindex(seeds)
        print(f"\n{data}  lr mix {cfg['mix'][0].lr} / ci {cfg['ci'][0].lr}  seeds {len(seeds)}")
        print("  per-seed MSE mix: " + " ".join(f"{v:.3f}" for v in pm.values))
        print("  per-seed MSE ci : " + " ".join(f"{v:.3f}" for v in pc.values))
        acc = {k: [np.zeros(nf), np.zeros(nf)] for k in ARMS}
        used = []
        for s in seeds:
            res = {k: per_sensor(worker, cfg[k][0], data, s, test, nf, device) for k in ARMS}
            if any(v is None for v in res.values()):
                continue
            used.append(s)
            for k in ARMS:
                acc[k][0] += res[k][0]; acc[k][1] += res[k][1]
        if not used:
            print("  (no seed with both checkpoints)")
            continue
        mse = {k: acc[k][0] / acc[k][1] for k in ARMS}
        gain = 100 * (mse["mix"] / mse["ci"] - 1)           # negative = mixing helps
        help_, hurt = int((gain <= -1).sum()), int((gain >= 1).sum())
        print(f"  per-sensor (ckpt seeds {used}): mix vs ci % = " + " ".join(f"{g:+.1f}" for g in gain))
        print(f"  sensors helped (<=-1%) {help_}/{nf}, hurt (>=+1%) {hurt}/{nf}, overall "
              f"{100 * (mse['mix'].mean() / mse['ci'].mean() - 1):+.2f}%")
        out.append(dict(data=data, D=nf, seeds_ckpt=len(used), helped=help_, hurt=hurt,
                        overall=100 * (acc['mix'][0].sum() / acc['mix'][1].sum() / (acc['ci'][0].sum() / acc['ci'][1].sum()) - 1),
                        per_sensor=" ".join(f"{g:+.1f}" for g in gain),
                        mse_mix=" ".join(f"{v:.5f}" for v in mse["mix"]),
                        mse_ci=" ".join(f"{v:.5f}" for v in mse["ci"]),
                        seed_mix=" ".join(f"{v:.4f}" for v in pm.values),
                        seed_ci=" ".join(f"{v:.4f}" for v in pc.values)))
    pd.DataFrame(out).to_csv("results/_mix_per_sensor.csv", index=False)
    print("\nwrote results/_mix_per_sensor.csv")


if __name__ == "__main__":
    main()
