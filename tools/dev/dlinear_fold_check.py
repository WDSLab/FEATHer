"""Check that DLinear (individual=False) is exactly ONE linear map.

DLinear = moving-average decomposition (linear: clamped centred mean) followed
by two Linear(L, H) heads that are summed.  So

    y = Ws (x - A x) + bs + Wt (A x) + bt = (Ws (I - A) + Wt A) x + (bs + bt)

with A the L x L moving-average matrix.  If that holds, a deployed DLinear
needs one L x H matrix, not two - half the flash and half the MACs, with the
same function.  This script builds the upstream model through our registry,
folds it, and compares outputs.  No training, no data.

    python tools/dev/dlinear_fold_check.py
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from baselines import get_model  # noqa: E402

L, D = 96, 7
FIRMWARE_KB = 10.4  # measured persistence build, results/board_footprint.csv


def moving_avg_matrix(model, L):
    """A[:, j] = moving average of the j-th unit vector, taken from the
    upstream module itself (so padding rules cannot drift)."""
    eye = torch.eye(L, dtype=torch.float64).T.unsqueeze(-1)  # (L batch, L, 1)
    trend = model.decompsition.moving_avg(eye)                # (L, L, 1)
    return trend.squeeze(-1).T                                # (L out, L in)


def fold(model, L):
    A = moving_avg_matrix(model, L)
    Ws = model.Linear_Seasonal.weight.double()
    Wt = model.Linear_Trend.weight.double()
    W = Ws @ (torch.eye(L, dtype=torch.float64) - A) + Wt @ A   # (H, L)
    b = (model.Linear_Seasonal.bias + model.Linear_Trend.bias).double()
    return W, b


def main():
    torch.manual_seed(0)
    print(f"L={L} D={D}")
    print(f"{'H':>4} {'max|diff| fp64':>15} {'max|diff| fp32':>15} "
          f"{'2 mats fp32':>12} {'1 mat fp32':>11} {'1 mat fp16':>11} {'1 mat int8':>11}")
    for H in (96, 192, 336, 720):
        args = argparse.Namespace(dlinear_individual=False)
        m = get_model("DLinear", args, D, L, H).double().eval()
        # non-trivial weights (default init is small and nearly symmetric)
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.1)
        W, b = fold(m, L)

        x = torch.randn(64, L, D, dtype=torch.float64).cumsum(1)  # random walks
        with torch.no_grad():
            y_ref = m(x)                                           # (B, H, D)
            y_fold = torch.einsum("hl,bld->bhd", W, x) + b[None, :, None]
        d64 = (y_ref - y_fold).abs().max().item()

        m32 = m.float()
        with torch.no_grad():
            y32 = m32(x.float())
            yf32 = torch.einsum("hl,bld->bhd", W.float(), x.float()) + b.float()[None, :, None]
        d32 = (y32 - yf32).abs().max().item()

        two = 2 * (L * H + H) * 4 / 1024
        one32 = (L * H + H) * 4 / 1024
        one16 = (L * H + H) * 2 / 1024
        one8 = (L * H + H * 4 + H * 4) / 1024   # int8 W + fp32 per-row scale + fp32 bias
        print(f"{H:>4} {d64:>15.2e} {d32:>15.2e} {two:>10.1f}KB {one32:>9.1f}KB "
              f"{one16:>9.1f}KB {one8:>9.1f}KB")
    print(f"(weights only; firmware base ~{FIRMWARE_KB} KB on top, flash = 128 KB)")


if __name__ == "__main__":
    main()
