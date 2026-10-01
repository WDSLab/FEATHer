# -*- coding: utf-8 -*-
"""Export float32 weights + golden vectors for the DLinear C port.

DLinear (Zeng et al., AAAI 2023) with individual=False — shared linear heads
across channels, which is the configuration our sweeps run
(baselines/DLinear/wrapper.py).

Why it is ported at all: at H=96 DLinear fits the F103RB (18,624 params =
74.5 KB of flash) AND beats FEATHer by 3.2%, so the deployability table has to
carry a measured row for it.  At H>=192 its weights alone exceed 128 KB, which
excludes it by arithmetic — no port needed there, and this script will refuse
to pretend otherwise.

Writes:
    golden/dlinear_weights.h   float32 weight arrays (flash constants)
    golden/dlinear_vectors.h   input + trend/seasonal + output

Run:
    python deployment/board/export_golden_dlinear.py
    python deployment/board/export_golden_dlinear.py --ckpt path/to.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines._import_helper import isolated_import  # noqa: E402

SEQ_LEN = 96
D_MODEL = 7                 # MetroPT3 deployment cell, same as the other ports
FLASH_BYTES = 128 * 1024
FIRMWARE_FLOOR = int(10.4 * 1024)      # measured, BOARD_RESULTS_20260824.md

WEIGHT_SEED = 2025
INPUT_SEED = 7
OUT_DIR = Path(__file__).resolve().parent / "golden"
_REPO = _ROOT / "baselines" / "DLinear" / "DLinear-main"


def emit_array(f, name: str, t: torch.Tensor) -> None:
    flat = t.detach().to(torch.float32).flatten().tolist()
    f.write(f"/* shape={list(t.shape)} */\n")
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}f" for v in flat[i:i + 6])
        f.write(f"  {chunk},\n")
    f.write("};\n\n")


def main() -> None:
    global D_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--d_model", type=int, default=D_MODEL,
                    help="channel count; the C header takes -D<PFX>_D to match")
    ap.add_argument("--force", action="store_true",
                    help="export even when the weights cannot fit flash")
    a = ap.parse_args()
    D_MODEL = a.d_model
    H = a.pred_len

    import argparse as _ap
    mod = isolated_import(str(_REPO), "models.DLinear")
    torch.manual_seed(WEIGHT_SEED)
    model = mod.Model(_ap.Namespace(seq_len=SEQ_LEN, pred_len=H,
                                    enc_in=D_MODEL, individual=False)).eval()

    n_params = sum(p.numel() for p in model.parameters())
    w_bytes = n_params * 4
    total = w_bytes + FIRMWARE_FLOOR
    print(f"params = {n_params}  weights = {w_bytes / 1024:.1f} KB  "
          f"+ firmware {FIRMWARE_FLOOR / 1024:.1f} KB = {total / 1024:.1f} KB "
          f"/ {FLASH_BYTES // 1024} KB flash")
    if total > FLASH_BYTES:
        msg = (f"DLinear at H={H} does not fit flash "
               f"({total / 1024:.1f} KB > {FLASH_BYTES // 1024} KB). "
               f"That exclusion is arithmetic and needs no port; the board "
               f"row for this horizon is 'not deployable', not a measurement.")
        if not a.force:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING (--force): " + msg)

    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"loaded checkpoint {a.ckpt}")

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)

    with torch.no_grad():
        seasonal, trend = model.decompsition(x)          # (1, L, D) each
        y = model(x)                                     # (1, H, D)

        # guard: the C port assumes replicate padding; confirm the two halves
        # really sum back to the input (they do by construction, but a wrong
        # padding mode here would silently change both parts)
        assert torch.allclose(seasonal + trend, x, atol=1e-6), \
            "decomposition does not reconstruct the input"

        # guard: the streaming claim — output step t must depend only on row t
        Ws, bs = model.Linear_Seasonal.weight, model.Linear_Seasonal.bias
        Wt, bt = model.Linear_Trend.weight, model.Linear_Trend.bias
        s_cm, t_cm = seasonal[0].T, trend[0].T           # (D, L)
        out = torch.empty(H, D_MODEL)
        for t in range(H):
            out[t] = s_cm @ Ws[t] + t_cm @ Wt[t] + bs[t] + bt[t]
        dmax = (out - y[0]).abs().max().item()
        print(f"streaming equivalence: max|diff| = {dmax:.3e}")
        assert dmax < 1e-4, "step-wise emission does not reproduce the forecast"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "dlinear_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dlinear.py"
                " - do not edit. */\n")
        src = a.ckpt if a.ckpt else f"random seed={WEIGHT_SEED}"
        f.write(f"/* source={src}, params={n_params}, L={SEQ_LEN}, H={H},"
                f" D={D_MODEL}, individual=False */\n")
        f.write("#ifndef DLINEAR_WEIGHTS_H\n#define DLINEAR_WEIGHTS_H\n\n")
        # Stamp the horizon the weights were generated at.  The port
        # static-asserts against it, so a stale golden header (weights
        # from one H, code built for another) fails to COMPILE instead of
        # silently producing a plausible wrong forecast.
        f.write(f"#define DL_H_GEN {H}\n\n")
        emit_array(f, "W_SEASONAL", Ws)      # (H, L) out,in
        emit_array(f, "B_SEASONAL", bs)      # (H,)
        emit_array(f, "W_TREND", Wt)         # (H, L)
        emit_array(f, "B_TREND", bt)         # (H,)
        f.write("#endif /* DLINEAR_WEIGHTS_H */\n")

    with (OUT_DIR / "dlinear_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dlinear.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED}, H={H},"
                f" streaming max|diff|={dmax:.3e} */\n")
        f.write("#ifndef DLINEAR_VECTORS_H\n#define DLINEAR_VECTORS_H\n\n")
        # Width the golden vectors were captured at; the test
        # _Static_asserts against it (a port built at another width
        # compiles fine and fails numerically, which reads as a code bug).
        f.write(f"#define DL_D_GEN {D_MODEL}\n\n")
        emit_array(f, "GD_X", x[0])              # (L,D) time-major
        emit_array(f, "GD_TREND", trend[0])      # (L,D) replicate-pad trap
        emit_array(f, "GD_SEAS", seasonal[0])    # (L,D)
        emit_array(f, "GD_Y", y[0])              # (H,D)
        f.write("#endif /* DLINEAR_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'dlinear_weights.h'}")
    print(f"wrote {OUT_DIR / 'dlinear_vectors.h'}")


if __name__ == "__main__":
    main()
