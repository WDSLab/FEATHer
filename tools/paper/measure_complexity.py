"""Measure Parameters + MACs per (model, dataset, horizon) — FFT-aware.

MACs = matmul/conv MACs (torch FlopCounterMode, FLOPs/2) + FFT MACs (analytic).
Standard auto-counters (thop, FlopCounterMode) MISS FFT and pure element-wise
frequency ops, so DiPE reads 0 and FITS/FEATHer under-count. We add FFT cost
at the aten level (_fft_r2c / _fft_c2r / _fft_c2c are captured via a dispatch
mode) using the convention:

    MACs(FFT length N) = N * log2(N)   per transform, per (batch,channel) row.

This is a stated approximation (radix-2 butterfly count) — MACs are convention-
dependent and NOT comparable across papers; report the convention in a footnote.
Element-wise complex scaling (e.g. DiPE's per-frequency gain) is NOT a MAC and is
excluded, consistent with standard MAC accounting (matmul/conv/FFT only).

Params are cross-checked against the logged num_params (=> build args correct).

Usage:
    python tools/paper/measure_complexity.py --models SparseTSF,MixLinear,DiPE_Linear,FITS --data ETTh1 --pred_len 96
    python tools/paper/measure_complexity.py --out results/complexity.csv
"""
import argparse
import math
import warnings
import sys
import os
from types import SimpleNamespace

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from utils import data_factory  # noqa: E402
from baselines import (get_model, get_method_defaults,  # noqa: E402
                       get_dataset_overrides)

MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
        "Electricity", "Traffic"]

_FFT_OPS = ("_fft_r2c", "_fft_c2r", "_fft_c2c")


class FFTMacCounter(TorchDispatchMode):
    """Accumulate analytic FFT MACs = num_transforms * N * log2(N)."""
    def __init__(self):
        self.fft_macs = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if any(op in name for op in _FFT_OPS):
            x = args[0]
            dims = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else [-1]
            try:
                N = 1
                for d in dims:
                    N *= x.shape[d]
                if N > 1:
                    n_transforms = x.numel() // N
                    self.fft_macs += int(n_transforms * N * math.log2(N))
            except Exception:
                pass
        return func(*args, **kwargs)


def base_args():
    return SimpleNamespace(
        root_path="data/", features="M", seq_len=96, label_len=96,
        d_state=8, kernel_size=7, period=12, num_bands=3,
        use_topk_gate=False, topk=2, use_residual_mlp=False, mlp_rank=16,
        use_norm=True, use_gate=True, use_robust_norm=False,
        revin_mode=None, winsor_bound=None, center_trim=None, channel_mode=None,
        lr=0.001, loss="l1", lambda_spec=0.01, optimizer="adam", weight_decay=0.0,
        batch_size=32, num_epochs=50, patience=10)


def build(model_name, data_name, pred_len, extra=None):
    df, _, _ = data_factory.data_select(data_name, "data/")
    D = df.shape[1] - 1
    args = base_args()
    args.model = model_name
    for k, v in (get_method_defaults(model_name) or {}).items():
        setattr(args, k, v)
    for k, v in (get_dataset_overrides(model_name, data_name) or {}).items():
        setattr(args, k, v)
    for k, v in (extra or {}).items():
        setattr(args, k, v)
    model = get_model(model_name, args, n_features=D, seq_len=args.seq_len,
                      pred_len=pred_len).float().eval()
    return model, D, args


def measure(model, D, seq_len, pred_len):
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, seq_len, D)
    # matmul/conv MACs
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(x)
    mm_macs = fc.get_total_flops() // 2
    # FFT MACs
    fftc = FFTMacCounter()
    with fftc, torch.no_grad():
        model(x)
    fft_macs = fftc.fft_macs
    return params, int(mm_macs), int(fft_macs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="SparseTSF,MixLinear,DiPE_Linear,FITS,DLinear,CycleNet")
    p.add_argument("--data", default=",".join(MFG + LTSF))
    p.add_argument("--pred_len", default="96,192,336,720")
    p.add_argument("--out", default="")
    a = p.parse_args()
    models = a.models.split(",")
    datas = a.data.split(",")
    preds = [int(x) for x in a.pred_len.split(",")]

    rows = []
    for m in models:
        for d in datas:
            for h in preds:
                try:
                    model, D, args = build(m, d, h)
                    params, mm, fft = measure(model, D, args.seq_len, h)
                    total = mm + fft
                    rows.append((m, d, h, D, params, mm, fft, total))
                    print(f"{m:<13}{d:<13}H{h:<4} D={D:<4} params={params:<8} "
                          f"MACs={total:<10} (mm={mm} fft={fft})", flush=True)
                except Exception as e:
                    print(f"{m:<13}{d:<13}H{h:<4} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
    if a.out:
        import csv
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "data", "pred_len", "D", "params",
                        "macs_matmul", "macs_fft", "macs_total"])
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
