# -*- coding: utf-8 -*-
"""Gate-swap test: does a trained FEATHer actually USE its input-dependent gate?

The banded FFT gate outputs 3 branch weights per input window (per window AND
per sensor for gate_mode=banded_sensor).  This script keeps the trained model
fixed and only replaces the gate output:

    own          the window's own gate (reference; reproduces the reported MSE)
    trainmean    the mean gate over the TRAIN split (per sensor for a per-sensor
                 gate), the same for every window
    shuffle      the gate of a different, randomly chosen window of the same split
    othersensor  per-sensor gate only: every sensor gets its neighbour's gate
                 from the same window (tests whether sensors need their own)
    uniform      1/3, 1/3, 1/3  (tests whether the gate VALUE matters at all)
    dropS        the window's own gate with branch S zeroed (no renormalisation)

If trainmean and shuffle score like own, the model is not exploiting the
per-window adaptivity; uniform separates that from "the values do not matter".
A weight is not a contribution (branch outputs differ in scale), so the RMS of
each gated branch, g_s * band_s, and of the mixer correction is reported too.

Decisions should be read on VAL; TEST is printed alongside.  Checkpoints and the
exact worker arguments come from a tools/dev/run_dev.py --save_model run.  When
the run swept several learning rates, the one with the lowest median VAL loss is
used, as in dev_summary.py.

    python tools/dev/gate_swap.py --log_prefix "gswap0915_{data}" --arm mx_in1_r4
    python tools/dev/gate_swap.py --log_prefix mix0914 --arm mx_in1_r4_gsensor
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

MODES = ["own", "trainmean", "shuffle", "othersensor", "uniform", "drop0", "drop1", "drop2"]
BRANCH = ["POINT(k3)", "MID(k5)", "LOW(pool)"]


def load_worker():
    path = os.path.join(REPO, "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("fcst_worker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_argv(log_prefix, arm, data):
    """Worker argument lists for this (arm, data), one per learning rate, read
    from the dev logs.  A log can hold several datasets, so match --data."""
    pat = os.path.join(REPO, "results", "_archive", "_devlogs", f"{log_prefix}_{arm}_*_h*.log")
    found = []
    # The glob also matches arms that merely START with this name
    # (mx_in1_r4_gsensor_* catches _oct/_ter/_ters) - keep "<arm>_<lr>_h<H>.log" only.
    exact = re.compile(re.escape(f"{log_prefix}_{arm}_") + r"[0-9.e+-]+_h\d+\.log$")
    for log in sorted(glob.glob(pat)):
        if not exact.match(os.path.basename(log)):
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                # dev logs written before 10-07 name the worker scripts/benchmarks/run_forecast.py
                wk = next((w for w in ("scripts/train.py", "scripts/benchmarks/run_forecast.py") if w in line), None)
                if "CMD:" not in line or wk is None:
                    continue
                toks = line.split("CMD:", 1)[1].split()
                argv = toks[toks.index(wk) + 1:]
                if argv[argv.index("--data") + 1] == data:
                    found.append(argv)
                    break
    if not found:
        raise SystemExit(f"no worker CMD for {data} in {pat}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_prefix", default="gswap0915_{data}",
                    help="run_dev --prefix; '{data}' is replaced by the dataset")
    ap.add_argument("--arm", default="mx_in1_r4")
    ap.add_argument("--data", default="Steel,EWELD,WindSCADA,BlastFurnace")
    ap.add_argument("--out", default="results/gate_swap0915.csv")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    worker = load_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, stats = [], []

    for data in [s.strip() for s in a.data.split(",")]:
        # pick the learning rate on median VAL loss
        best = None
        for argv in worker_argv(a.log_prefix.format(data=data), a.arm, data):
            sys.argv = ["run_forecast.py"] + argv
            cand = worker.parse_args()
            ref = pd.read_csv(cand.results_csv, on_bad_lines="skip")
            r = ref[(ref.exp_tag == cand.exp_tag) & (ref.data == data)]
            if r.empty:
                continue
            v = r.val_loss.median()
            if best is None or v < best[0]:
                best = (v, argv, r)
        if best is None:
            print(f"[skip] {data}: no finished rows")
            continue
        _, argv, ref = best
        sys.argv = ["run_forecast.py"] + argv
        args = worker.parse_args()
        worker._apply_overrides(args)
        args.label_len = min(args.label_len, args.seq_len)
        H = int(args.pred_len)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        print(f"\n{data}: exp_tag {args.exp_tag} (lr {args.lr}, gate_mode {args.gate_mode})")

        df, _, _ = worker.data_factory.data_select(data, args.root_path)
        nf = df.shape[1] - 1
        sets = {}
        for split in ("train", "val", "test"):
            ds, _ = worker.data_factory.data_provider(
                args.root_path, data, args.features, args.batch_size,
                args.seq_len, args.label_len, H, split,
                starting_percent=args.starting_percent, percent=args.percent)
            sets[split] = ds

        # --gate_bounds tercile is resolved from the train split, exactly as in training
        if hasattr(worker, "resolve_gate_bounds"):
            worker.resolve_gate_bounds(args, sets["train"], nf)

        for seed in seeds:
            ckpt = worker.checkpoint_path(args.exp_tag, args.model, data, H,
                                          args.seq_len, seed)
            if not os.path.exists(ckpt):
                print(f"[skip] {data} s{seed}: no checkpoint {ckpt}")
                continue
            torch.manual_seed(seed)
            model = worker.get_model(args.model, args, n_features=nf,
                                     seq_len=args.seq_len, pred_len=H).float().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            st = dict(mode="own", idx=None, record=None, G=None, perm=None,
                      gbar=None, last=None, contrib=None)

            def gate_hook(mod, inp, out):
                if st["record"] is not None:
                    st["record"].append(out.detach().cpu())
                m = st["mode"]
                if m == "own":
                    new = out
                elif m == "trainmean":
                    new = st["gbar"].to(out).expand_as(out)
                elif m == "shuffle":
                    new = st["G"][st["perm"][st["idx"]]].to(out)
                elif m == "othersensor":
                    new = out.roll(1, dims=1) if out.dim() == 3 else out
                elif m == "uniform":
                    new = torch.full_like(out, 1.0 / out.shape[-1])
                else:                                   # dropS
                    new = out.clone()
                    new[..., int(m[4:])] = 0.0
                st["last"] = new.detach()
                return new

            model.freq_gate.register_forward_hook(gate_hook)
            orig_synth = model._synthesize

            def synth(x_norm, bands_list, *args_, **kw):
                y = orig_synth(x_norm, bands_list, *args_, **kw)
                c = st["contrib"]
                if c is not None:
                    g = st["last"]
                    for s, b in enumerate(bands_list):
                        gs = g[..., s]
                        gs = gs.view(-1, 1, 1) if gs.dim() == 1 else gs.unsqueeze(1)
                        c[s] += float(((gs * b) ** 2).sum())
                    c["n"] += bands_list[0].numel()
                    if getattr(model, "use_residual_mlp", False) and model.mix_mode == "input":  # (input_raw would need the raw window)
                        c["mix"] += float((model._mixer(x_norm) ** 2).sum())
                return y

            model._synthesize = synth

            def run(split, record=False, contrib=False):
                loader = DataLoader(sets[split], batch_size=a.batch, shuffle=False,
                                    drop_last=False, num_workers=0)
                st["record"] = [] if record else None
                st["contrib"] = ({0: 0.0, 1: 0.0, 2: 0.0, "n": 0, "mix": 0.0}
                                 if contrib else None)
                se, n, start = 0.0, 0, 0
                with torch.no_grad():
                    for bx, by, _, _ in loader:
                        B = bx.shape[0]
                        st["idx"] = torch.arange(start, start + B)
                        start += B
                        out = model(bx.float().to(device)[:, :, :nf])
                        yt = by.float()[:, :, :nf][:, -H:].numpy()
                        yp = out[:, -H:].cpu().numpy()
                        d = yp - yt
                        fin = np.isfinite(d)
                        se += float((d[fin] ** 2).sum())
                        n += int(fin.sum())
                rec = torch.cat(st["record"]) if record else None
                st["record"] = None
                return se / n, rec

            st["mode"] = "own"
            _, Gtr = run("train", record=True)
            st["gbar"] = Gtr.mean(0)                    # (S,) or (D, S)
            per_sensor = Gtr.dim() == 3

            for split in ("val", "test"):
                st["mode"] = "own"
                own, G = run(split, record=True, contrib=True)
                c = st["contrib"]
                st["G"] = G
                gen = torch.Generator().manual_seed(seed)
                st["perm"] = torch.randperm(G.shape[0], generator=gen)
                for mode in MODES:
                    if mode == "othersensor" and not per_sensor:
                        continue
                    if mode == "own":
                        mse = own
                    else:
                        st["mode"] = mode
                        mse, _ = run(split)
                    rows.append(dict(data=data, seed=seed, split=split, mode=mode,
                                     MSE=mse, delta_pct=100.0 * (mse / own - 1.0)))
                st["mode"] = "own"
                row = dict(data=data, seed=seed, split=split, per_sensor=per_sensor,
                           rms_mix=(c["mix"] / c["n"]) ** 0.5)
                for k in range(3):
                    gk = G[..., k]
                    row[f"g_trainmean_{k}"] = float(st["gbar"][..., k].mean())
                    # spread across WINDOWS (per sensor, then averaged for a per-sensor gate)
                    row[f"g_std_{k}"] = float(gk.std(0).mean())
                    row[f"g_range_{k}"] = float((gk.max(0).values - gk.min(0).values).mean())
                    # spread across SENSORS of the window-averaged gate
                    row[f"g_sensorstd_{k}"] = float(gk.mean(0).std()) if per_sensor else 0.0
                    row[f"rms_{k}"] = (c[k] / c["n"]) ** 0.5
                stats.append(row)
                if split == "test":
                    r = ref[ref.seed == seed]
                    rep = float(r.MSE.iloc[0]) if len(r) else float("nan")
                    print(f"{data} s{seed}: own test MSE {own:.6f} vs worker CSV {rep:.6f}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    sdf = pd.DataFrame(stats)
    sdf.to_csv(a.out.replace(".csv", "_gatestats.csv"), index=False)

    pd.set_option("display.width", 220)
    for split in ("val", "test"):
        s = out[out.split == split]
        print(f"\n=== {split.upper()}: MSE change vs own gate, % (median over seeds; [min, max]) ===")
        med = s.groupby(["mode", "data"]).delta_pct.median().unstack("data")
        lo = s.groupby(["mode", "data"]).delta_pct.min().unstack("data")
        hi = s.groupby(["mode", "data"]).delta_pct.max().unstack("data")
        order = [m for m in MODES if m in med.index]
        fmt = med.reindex(order).copy().astype(object)
        for m in order:
            for d in med.columns:
                fmt.loc[m, d] = ("-" if pd.isna(med.loc[m, d]) else
                                 f"{med.loc[m, d]:+.2f} [{lo.loc[m, d]:+.2f},{hi.loc[m, d]:+.2f}]")
        print(fmt.to_string())
    v = sdf[sdf.split == "val"].groupby("data").median(numeric_only=True)
    print("\n=== VAL gate: train-mean weight (spread across windows; across sensors) ===")
    for d, r in v.iterrows():
        print(f"{d:13s} " + "  ".join(
            f"{BRANCH[k]} {r[f'g_trainmean_{k}']:.3f} (win std {r[f'g_std_{k}']:.4f},"
            f" range {r[f'g_range_{k}']:.3f}; sensor std {r[f'g_sensorstd_{k}']:.3f})"
            for k in range(3)))
    print("\n=== VAL contribution: RMS of gated branch output and of the mixer correction ===")
    for d, r in v.iterrows():
        print(f"{d:13s} " + "  ".join(f"{BRANCH[k]} {r[f'rms_{k}']:.4f}" for k in range(3))
              + f"  mixer {r['rms_mix']:.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
