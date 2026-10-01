# -*- coding: utf-8 -*-
"""Deployment-cost accounting: count every float operation the chip performs.

WHY NOT MACs. The usual counters (thop, fvcore, FlopCounterMode) count matmul
and convolution and nothing else. For a paper about running on an MCU that is
the wrong quantity, because the ops they skip — normalisation, gating,
activations, and above all complex arithmetic — are exactly the ones that
dominate on a Cortex-M3 with no FPU. Measured against the board, MAC counts
predict nothing: per MAC, FITS costs 14.6 us and DiPE 17.4 us against
FEATHer's 1.9, a 9x spread that only exists because most of their real work is
invisible to a MAC counter.

WHAT THIS COUNTS. Every aten op that touches floats, at the dispatch level,
bucketed by what it costs in software floating point:

    arith   add / sub / mul, and the mul+add pairs inside matmul and conv
    div     division and reciprocal
    sqrt    sqrt / rsqrt
    trans   exp, log, tanh, erf (so: GELU, softmax, and friends)

Complex arithmetic is expanded into the real operations it compiles down to:
one complex multiply is 4 real multiplies and 2 real adds, one complex add is
2 real adds.

TRANSFORMS ARE COUNTED AS THE BOARD IMPLEMENTS THEM, NOT AS PyTorch DOES.
deployment/board/fea_kernels.c computes a direct O(N^2) DFT against a flash
twiddle table (`fea_rfft96`, nested f x t loops) rather than a radix-2 FFT —
at N=96 a table-driven DFT beats pulling in an FFT library, and it keeps the
golden vectors exact. Charging N*log2(N) here would understate every
frequency-domain model by ~15x and is the single largest error in the
MAC-style accounting: it is why FITS and DiPE looked cheap on paper and ran
slowest on the chip. Set --transform fft to see the difference.

The buckets are then weighted by their soft-float cost relative to a multiply,
using figures in the range gcc's libgcc routines take on an M3:

    WEIGHTS = arith 1, div 3, sqrt 4, trans 15

`ops_weighted` is therefore a cycle proxy, not a physical count — it is only
meaningful because it is validated. `--validate` regresses it against the four
board-measured latencies; report that fit alongside any use of this column.

Both the raw per-bucket counts and the weighted total are written, so a reader
who disagrees with the weights can re-weight from the same measurements.

Usage:
    python tools/paper/measure_complexity.py --out results/complexity.csv
    python tools/paper/measure_complexity.py --validate
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
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from utils import data_factory  # noqa: E402
from baselines import (get_model, get_method_defaults,  # noqa: E402
                       get_dataset_overrides)

MFG = ["Steel", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
        "Electricity", "Traffic"]

# Relative soft-float cost per operation, normalised to a multiply. libgcc's
# __aeabi_fdiv runs a few times __aeabi_fmul, and expf/erff are library calls
# an order of magnitude beyond that. Deliberately coarse — the point is the
# ordering, and --validate reports how well it holds.
WEIGHTS = {"arith": 1.0, "div": 3.0, "sqrt": 4.0, "trans": 15.0}

# Board measurements — 2026-08-24 session, all seven models re-measured in one
# sitting at one build setting (NUCLEO-F103RB, 72 MHz, no FPU, -O2 soft-float,
# D=7, seq_len=96, pred_len=96). Four of them reproduced the 07-27 numbers to
# the microsecond across a different session and rebuild, which is the evidence
# that the timing harness itself is sound; FEATHer's own figure moved 349 -> 655
# ms because the model did (453p pre-08 config -> the shipping 938p FEATHer-N).
#
# !! These two dicts are a MATCHED PAIR and are used ONLY by --validate, which
#    checks that the weighted-op model predicts measured latency (r2=0.87).
#    FEATHer's entry is the 938p architecture that was actually flashed, paired
#    with the latency that architecture actually produced -- internally
#    consistent, so the fit stays valid.  It is NOT the shipping model.
#    !! The FROZEN 452p FEATHer HAS NO BOARD LATENCY.  It needs hardware and has
#    not been measured.  Never copy 655.1 ms into a table describing the 452p
#    model: roughly half the weighted ops were removed with the DTK, so the real
#    number is expected near ~350 ms -- expected, not measured.  Leave the cell
#    empty until the board runs.
BOARD_MS = {
    "SparseTSF": 47.4,
    "MixLinear": 79.7,
    "SparseTSF_MLP": 430.6,
    "FEATHer": 655.1,
    "FITS": 796.2,
    "DiPE_Linear": 1312.1,
}
BOARD_CFG = {
    # pre-freeze 938p config, stated in full so it cannot drift when
    # _DATASET_OVERRIDES changes (build() with data_name=None applies only
    # _METHOD_DEFAULTS + this dict)
    "FEATHer": dict(d_state=16, period=24, num_bands=4, use_residual_mlp=True,
                    mlp_rank=16, channel_mode="mix", lambda_spec=0.01),
    "SparseTSF": dict(sparsetsf_period_len=24, sparsetsf_model_type="linear"),
    "SparseTSF_MLP": dict(sparsetsf_period_len=24, sparsetsf_d_model=128),
    "MixLinear": dict(mixlinear_period_len=24),
    "FITS": dict(fits_individual=False, fits_cut_freq=0),
    "DiPE_Linear": dict(),
}

_MATMUL = ("mm", "bmm", "addmm", "baddbmm", "matmul", "linear", "convolution",
           "conv1d", "conv2d", "einsum")
_FFT = ("_fft_r2c", "_fft_c2r", "_fft_c2c")
_ARITH = ("add", "sub", "mul", "neg", "rsub")
_DIV = ("div", "reciprocal", "true_divide")
_SQRT = ("sqrt", "rsqrt")
_TRANS = ("exp", "log", "tanh", "erf", "sigmoid", "gelu", "softmax", "pow")
_REDUCE = ("sum", "mean", "var", "std", "cumsum", "norm")


class OpCounter(TorchDispatchMode):
    """Bucket every float-touching aten op by soft-float cost class.

    transform="dft" (default) charges transforms as the board's O(N^2)
    table-driven DFT; "fft" charges the textbook O(N log N) radix-2 cost.
    """

    def __init__(self, transform="dft"):
        self.c = {k: 0 for k in WEIGHTS}
        self.transform = transform

    @staticmethod
    def _numel(t):
        return t.numel() if isinstance(t, torch.Tensor) else 0

    @staticmethod
    def _is_complex(*ts):
        return any(isinstance(t, torch.Tensor) and t.is_complex() for t in ts)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func).split(".")[-2] if "." in str(func) else str(func)
        full = str(func)
        out = func(*args, **kwargs)

        try:
            self._account(name, full, args, out)
        except Exception:
            pass
        return out

    def _account(self, name, full, args, out):
        first = args[0] if args else None

        # ---- transform
        if any(op in full for op in _FFT):
            x = args[0]
            dims = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else [-1]
            N = 1
            for d in dims:
                N *= x.shape[d]
            if N > 1:
                rows = x.numel() // N
                if self.transform == "fft":
                    # radix-2: N*log2(N) butterflies, each 1 complex mul (4 mul
                    # + 2 add) plus 2 complex adds (4 add)
                    self.c["arith"] += int(rows * N * math.log2(N) * 10)
                else:
                    # board: direct DFT, N/2+1 output bins x N input samples,
                    # each an accumulate of x[t]*twiddle[f,t] = 2 mul + 2 add
                    bins = N // 2 + 1
                    self.c["arith"] += int(rows * bins * N * 4)
            return

        # ---- matmul / conv: mul+add per MAC, x4/x2 if complex
        if any(op == name or op in full.split(".")[-2:] for op in _MATMUL):
            n_out = self._numel(out)
            if n_out == 0:
                return
            k = self._reduction_len(name, args, out)
            macs = n_out * k
            self.c["arith"] += int(macs * (6 if self._is_complex(*args) else 2))
            return

        n = self._numel(out)
        if n == 0:
            return
        cplx = 2 if (isinstance(out, torch.Tensor) and out.is_complex()) else 1

        if any(op in name for op in _TRANS):
            self.c["trans"] += n
            if "softmax" in name:                 # exp, then a normalising div
                self.c["div"] += n
        elif any(op in name for op in _SQRT):
            self.c["sqrt"] += n
        elif any(op in name for op in _DIV):
            self.c["div"] += n * cplx
        elif any(name == op or name.startswith(op) for op in _ARITH):
            # a complex multiply is 4 real mul + 2 real add
            self.c["arith"] += n * (6 if self._is_complex(*args) and "mul" in name else cplx)
        elif any(op in name for op in _REDUCE):
            self.c["arith"] += self._numel(first) * cplx

    @staticmethod
    def _reduction_len(name, args, out):
        """Inner-product length of a matmul/conv, i.e. MACs per output element."""
        try:
            if "convolution" in name or "conv" in name:
                w = args[1]
                return int(w.numel() // w.shape[0])          # per output channel
            a = args[-2] if "addmm" in name or "baddbmm" in name else args[0]
            b = args[-1]
            if isinstance(b, torch.Tensor) and b.dim() >= 2:
                return int(b.shape[-2])
            if isinstance(a, torch.Tensor) and a.dim() >= 1:
                return int(a.shape[-1])
        except Exception:
            pass
        return 1


def base_args():
    return SimpleNamespace(
        root_path="data/", features="M", seq_len=96, label_len=96,
        d_state=8, kernel_size=7, period=12, num_bands=3,
        use_topk_gate=False, topk=2, use_residual_mlp=False, mlp_rank=16,
        use_norm=True, use_gate=True, use_robust_norm=False,
        revin_mode=None, winsor_bound=None, center_trim=None, channel_mode=None,
        gate_mode=None, dtk_act=None, head_mode=None, dtk_mode=None,
        lr=0.001, loss="mse", lambda_spec=0.01, optimizer="adam", weight_decay=0.0,
        batch_size=32, num_epochs=50, patience=10)


def build(model_name, data_name, pred_len, extra=None, D=None):
    if D is None:
        df, _, _ = data_factory.data_select(data_name, "data/")
        D = df.shape[1] - 1
    args = base_args()
    args.model = model_name
    for k, v in (get_method_defaults(model_name) or {}).items():
        setattr(args, k, v)
    if data_name:
        for k, v in (get_dataset_overrides(model_name, data_name) or {}).items():
            setattr(args, k, v)
    for k, v in (extra or {}).items():
        setattr(args, k, v)
    model = get_model(model_name, args, n_features=D, seq_len=args.seq_len,
                      pred_len=pred_len).float().eval()
    return model, D, args


def measure(model, D, seq_len, pred_len, transform="dft"):
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, seq_len, D)
    oc = OpCounter(transform=transform)
    with oc, torch.no_grad():
        model(x)
    weighted = sum(WEIGHTS[k] * v for k, v in oc.c.items())
    return params, oc.c, int(weighted)


def _fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
    slope = sxy / sxx if sxx else float("nan")
    return r, slope, my - slope * mx


def _matmul_only_macs(model, D, seq_len):
    """What a conventional counter would report, for contrast."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, seq_len, D)
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() // 2


def validate():
    """Regress each accounting against the four board latencies."""
    print("Board configs: D=7, L=96, H=96 (deployment/board/export_*.py)\n")
    rows = []
    for m, ms in BOARD_MS.items():
        model, _, _ = build(m, None, 96, extra=BOARD_CFG[m], D=7)
        params, c, w = measure(model, 7, 96, 96, transform="dft")
        _, _, w_fft = measure(model, 7, 96, 96, transform="fft")
        mac = _matmul_only_macs(model, 7, 96)
        rows.append((m, params, c, w, w_fft, mac, ms))

    print(f"{'model':13s} {'params':>7s} {'arith':>13s} {'div':>7s} {'sqrt':>6s} "
          f"{'trans':>7s} {'weighted':>13s} {'board ms':>9s} {'ns/op':>7s}")
    for m, p, c, w, wf, mac, ms in sorted(rows, key=lambda r: r[6]):
        print(f"{m:13s} {p:>7,} {c['arith']:>13,} {c['div']:>7,} {c['sqrt']:>6,} "
              f"{c['trans']:>7,} {w:>13,} {ms:>9.0f} {ms*1e6/w:>7.1f}")

    ys = [r[6] for r in rows]
    print()
    for label, xs in [
        ("matmul/conv MACs only (conventional)", [r[5] for r in rows]),
        ("all ops, transforms as FFT  N log2 N", [r[4] for r in rows]),
        ("all ops, transforms as DFT  N^2 (board)", [r[3] for r in rows]),
    ]:
        r, slope, icpt = _fit(xs, ys)
        print(f"  {label:42s} r={r:+.4f}  r^2={r*r:.4f}  "
              f"fit {slope*1e6:8.2f} ms/Mop  intercept {icpt:+7.1f} ms")

    print(f"\n  weights: {WEIGHTS}")
    print("  The board computes a direct O(N^2) DFT (fea_kernels.c:fea_rfft96),")
    print("  so charging N log2 N understates every frequency-domain model.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--pred_len", default="96,192,336,720")
    p.add_argument("--out", default="results/complexity.csv")
    p.add_argument("--validate", action="store_true",
                   help="regress the weighted count against board latency")
    args = p.parse_args()

    if args.validate:
        return validate()

    from baselines import list_models
    models = ([s.strip() for s in args.models.split(",")] if args.models
              else list_models())
    datasets = ([s.strip() for s in args.data.split(",")] if args.data
                else MFG + LTSF)
    preds = [int(s) for s in args.pred_len.split(",")]

    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "data", "pred_len", "D", "params",
                    "ops_arith", "ops_div", "ops_sqrt", "ops_trans",
                    "ops_weighted"])
        for m in models:
            for d in datasets:
                for h in preds:
                    try:
                        model, D, _ = build(m, d, h)
                        params, c, weighted = measure(model, D, 96, h)
                    except Exception as e:
                        print(f"  skip {m}/{d}/H{h}: {type(e).__name__}: {e}")
                        continue
                    w.writerow([m, d, h, D, params, c["arith"], c["div"],
                                c["sqrt"], c["trans"], weighted])
                    print(f"{m:14s}{d:13s} H{h:<4d} D={D:<4d} params={params:<8,} "
                          f"weighted={weighted:,}", flush=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
