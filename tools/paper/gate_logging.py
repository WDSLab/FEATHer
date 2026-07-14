# -*- coding: utf-8 -*-
"""
FFT gate-under-corruption logging — interpretability evidence for FEATHer.

FEATHer's FFTFrequencyGate emits, per input sample, a softmax weight over the
frequency bands (POINT / HIGH / MID / LOW, depending on num_bands). The paper
claims this gate *adaptively re-weights* bands by spectral content. This script
tests that claim UNDER CORRUPTION: it loads a trained FEATHer checkpoint,
registers a forward hook on the gate, and dumps the mean per-band gate weights
on clean test data and under each SensorFault-Bench axis (gauss / miss /
impulse / quant) at every severity. If the gate re-allocates weight away from
the corrupted band (e.g. Gaussian high-frequency noise -> weight shifts toward
the LOW band), that is a MECHANISTIC explanation for FEATHer's graceful
degradation — not black-box capacity, but the gate routing around the fault.

Directly addresses:
  reject_mail R2 #3 — "how does frequency-aware gating dynamically adjust the
                       contributions of different time scales?"
  reject_mail R3 #3 — methodology narrative: justify each design choice.
Feeds the component-ablation / robustness story
  (manuscript/notes/robustness_justification_evidence.md).

The hook captures the gate output BEFORE the optional top-k sparsification
(default use_topk_gate=False, so for the paper config the captured weights are
exactly the applied ones).

Checkpoints come from the same --save_model runs the robustness sweep uses
(mfg main sweep). Until those exist this script can be exercised for its
plumbing but the numbers are only meaningful on a trained model.

Usage:
    python tools/paper/gate_logging.py --model FEATHer --data Steel --pred_len 96 \
        --seed 2025 --train_exp_tag main
    python tools/paper/gate_logging.py --data Steel --pred_len 96 --seed 2025 \
        --no_plot --out results/gate_weights.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_model
from baselines.FEATHer.FEATHer import FFTFrequencyGate
from utils import data_factory
from utils import noise as noise_mod
from utils.seed import set_seed
from scripts.benchmarks.run_forecast import checkpoint_path

# Band identity depends on num_bands (mirror of FEATHer.forward's bands_list):
#   2 -> [POINT, LOW]; 3 -> [POINT, MID, LOW]; 4 -> [POINT, HIGH, MID, LOW]
_BAND_NAMES = {
    2: ["POINT", "LOW"],
    3: ["POINT", "MID", "LOW"],
    4: ["POINT", "HIGH", "MID", "LOW"],
}
_ALL_BANDS = ["POINT", "HIGH", "MID", "LOW"]

# Normalized-severity helper shared with robust_summary.py (quant bit-axis
# reversed so "more severe" always increases).
_QUANT_CLEAN_BITS = 32.0


def normalized_severity(fault_type, severity):
    if fault_type == "clean":
        return 0.0
    if fault_type == "quant":
        inv = 1.0 / max(float(severity), 1e-9)
        inv_clean = 1.0 / _QUANT_CLEAN_BITS
        inv_worst = 1.0 / min(noise_mod.SEVERITY_SWEEPS["quant"])
        return (inv - inv_clean) / ((inv_worst - inv_clean) or 1.0)
    worst = max(noise_mod.SEVERITY_SWEEPS[fault_type])
    return float(severity) / (worst or 1.0)


def find_gate(model):
    """Locate the FFTFrequencyGate submodule regardless of any wrapper."""
    for m in model.modules():
        if isinstance(m, FFTFrequencyGate):
            return m
    raise RuntimeError("No FFTFrequencyGate found in model — is this FEATHer?")


def mean_gate_weights(model, gate_mod, loader, n_features, device,
                      fault_type=None, severity=None):
    """Sample-weighted mean of the gate's per-band softmax weights over the
    test loader, optionally corrupting each input batch first.

    Returns np.array of shape (num_bands,).
    """
    captured = {}

    def hook(_mod, _inp, out):
        captured["g"] = out.detach().float().cpu()

    handle = gate_mod.register_forward_hook(hook)
    model.eval()
    band_sum = None
    cnt = 0
    try:
        with torch.no_grad():
            for batch_x, _batch_y, _, _ in loader:
                batch_x = batch_x.float().to(device)
                x_in = batch_x[:, :, :n_features]
                if fault_type is not None:
                    x_in = noise_mod.apply_corruption(x_in, fault_type, severity)
                model(x_in)
                g = captured["g"]                       # (B, num_bands)
                bs = g.shape[0]
                s = g.sum(dim=0).numpy()
                band_sum = s if band_sum is None else band_sum + s
                cnt += bs
    finally:
        handle.remove()
    return band_sum / max(cnt, 1)


def _row(args, fault_type, severity, weights, band_names):
    row = {
        "exp_tag": args.exp_tag, "model": args.model, "data": args.data,
        "pred_len": args.pred_len, "seq_len": args.seq_len, "seed": args.seed,
        "fault_type": fault_type, "severity": severity,
        "sev_norm": round(normalized_severity(fault_type, severity), 6),
        "num_bands": len(band_names),
    }
    named = dict(zip(band_names, weights))
    for b in _ALL_BANDS:
        row[f"w_{b}"] = round(float(named[b]), 6) if b in named else np.nan
    return row


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ckpt = checkpoint_path(args.train_exp_tag, args.model, args.data,
                           args.pred_len, args.seq_len, args.seed)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nTrain it first with "
            f"`python run_forecast.py --save_model --exp_tag {args.train_exp_tag} "
            f"--model {args.model} --data {args.data} --pred_len {args.pred_len}`.")

    set_seed(args.seed, deterministic=True)
    df, _, _ = data_factory.data_select(args.data, args.root_path)
    n_features = df.shape[1] - 1
    _, test_loader = data_factory.data_provider(
        args.root_path, args.data, args.features, args.batch_size,
        args.seq_len, args.label_len, args.pred_len, "test",
        starting_percent=args.starting_percent, percent=args.percent)

    model = get_model(args.model, args, n_features=n_features,
                      seq_len=args.seq_len, pred_len=args.pred_len)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.float().to(device)
    gate_mod = find_gate(model)

    rows = []

    def log(ft, sev):
        w = mean_gate_weights(model, gate_mod, test_loader, n_features, device,
                              fault_type=(None if ft == "clean" else ft),
                              severity=(None if ft == "clean" else sev))
        band_names = _BAND_NAMES[len(w)]
        rows.append(_row(args, ft, sev, w, band_names))
        print(f"  {ft:>8}={sev!s:<6} | " +
              "  ".join(f"{b}={v:.3f}" for b, v in zip(band_names, w)))

    log("clean", 0.0)
    fault_types = args.fault_types.split(",") if args.fault_types else noise_mod.FAULT_TYPES
    for ft in fault_types:
        for sev in noise_mod.SEVERITY_SWEEPS[ft]:
            set_seed(args.seed * 1000 + hash((ft, sev)) % 1000, deterministic=True)
            log(ft, sev)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = not os.path.exists(args.out)
    out.to_csv(args.out, mode="a", header=header, index=False)
    print(f"\n  -> appended {len(out)} rows -> {args.out}")

    print_shift_table(out)
    if not args.no_plot:
        plot_reallocation(out, args.output_dir, args.data, args.pred_len, args.seed)


def print_shift_table(df):
    """Clean vs worst-severity band weight, per fault axis — the re-allocation
    delta that shows whether the gate routes around corruption."""
    clean = df[df["fault_type"] == "clean"]
    if clean.empty:
        return
    base = {b: clean.iloc[0][f"w_{b}"] for b in _ALL_BANDS}
    print("\n=== gate re-allocation: clean -> worst severity (delta per band) ===")
    for ft in [f for f in df["fault_type"].unique() if f != "clean"]:
        sub = df[df["fault_type"] == ft]
        worst = sub.loc[sub["sev_norm"].idxmax()]
        deltas = []
        for b in _ALL_BANDS:
            if pd.isna(base[b]) or pd.isna(worst[f"w_{b}"]):
                continue
            deltas.append(f"{b} {worst[f'w_{b}'] - base[b]:+.3f}")
        print(f"  {ft:>8}: " + "  ".join(deltas))


def plot_reallocation(df, output_dir, data, pred_len, seed):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — install for the plot", file=sys.stderr)
        return
    faults = [f for f in df["fault_type"].unique() if f != "clean"]
    present = [b for b in _ALL_BANDS if not df[f"w_{b}"].isna().all()]
    fig, axes = plt.subplots(1, len(faults), figsize=(4 * len(faults), 4), squeeze=False)
    for ax, ft in zip(axes[0], faults):
        clean = df[df["fault_type"] == "clean"].iloc[0]
        sub = df[df["fault_type"] == ft].sort_values("sev_norm")
        for b in present:
            xs = np.concatenate([[0.0], sub["sev_norm"].to_numpy()])
            ys = np.concatenate([[clean[f"w_{b}"]], sub[f"w_{b}"].to_numpy()])
            ax.plot(xs, ys, "-o", label=b)
        ax.set_title(ft)
        ax.set_xlabel("normalized severity")
        ax.set_ylabel("mean gate weight")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(f"FFT gate re-allocation under corruption — {data} H={int(pred_len)} s{seed}",
                 y=1.03)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"gate_{data}_H{int(pred_len)}_s{seed}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description="FFT gate-under-corruption logging (FEATHer)")
    p.add_argument("--model", default="FEATHer")
    p.add_argument("--data", required=True)
    p.add_argument("--pred_len", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=96)
    p.add_argument("--seed", type=int, required=True)
    # FEATHer arch (must match the trained checkpoint)
    p.add_argument("--d_state", type=int, default=8)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--period", type=int, default=12)
    p.add_argument("--num_bands", type=int, default=3)
    p.add_argument("--use_topk_gate", action="store_true")
    p.add_argument("--topk", type=int, default=2)
    # data
    p.add_argument("--root_path", default="data/")
    p.add_argument("--features", default="M")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--starting_percent", type=int, default=0)
    p.add_argument("--percent", type=int, default=100)
    # control
    p.add_argument("--train_exp_tag", default="main")
    p.add_argument("--exp_tag", default="gate")
    p.add_argument("--fault_types", default="",
                   help="Comma subset (e.g. 'gauss,quant'); empty = all four.")
    p.add_argument("--out", default="results/gate_weights.csv")
    p.add_argument("--output_dir", default="tools/paper/figures")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
