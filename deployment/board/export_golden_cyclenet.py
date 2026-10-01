# -*- coding: utf-8 -*-
"""Export float32 weights + golden vectors for the CycleNet C port.

CycleNet (Lin et al., NeurIPS 2024) with the LINEAR backbone and RevIN - the
configuration our sweeps run (baselines/CycleNet/wrapper.py).

Why it is ported (2026-09-30): its weights fit the F103RB at H = 96 and 192
(MetroPT3 cell: 44.5 / 80.9 KB), so without a port it sat in the tables as
"weights fit, SRAM unverified" and was left out of the deployable ranking for
a reason that was ours.  Same fairness argument as DLinear.  From H = 336 the
weights alone exceed 128 KB; this script refuses those horizons.

Weights are RANDOM, including the cycle table: upstream initialises the cycle
to zeros, and a zero cycle would let a port that ignores it (or indexes it
wrongly) pass the golden - the "blind golden" trap of 2026-09-14.

Writes:
    golden/cyclenet_weights.h   float32 weight arrays (flash constants)
    golden/cyclenet_vectors.h   input + de-cycled input + output

Run:
    python deployment/board/export_golden_cyclenet.py
    python deployment/board/export_golden_cyclenet.py --pred_len 192 --d_model 7
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
CYCLE = 298                 # CycleNet cycle for MetroPT3 (_DATASET_OVERRIDES)
FLASH_BYTES = 128 * 1024
FIRMWARE_FLOOR = int(10.4 * 1024)      # measured, BOARD_RESULTS_20260824.md

WEIGHT_SEED = 2025
INPUT_SEED = 7
OUT_DIR = Path(__file__).resolve().parent / "golden"
_REPO = _ROOT / "baselines" / "CycleNet" / "CycleNet-main"


def emit_array(f, name: str, t: torch.Tensor) -> None:
    flat = t.detach().to(torch.float32).flatten().tolist()
    f.write(f"/* shape={list(t.shape)} */\n")
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}f" for v in flat[i:i + 6])
        f.write(f"  {chunk},\n")
    f.write("};\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=D_MODEL,
                    help="channel count; the C header takes -DCN_D to match")
    ap.add_argument("--cycle", type=int, default=CYCLE)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--force", action="store_true",
                    help="export even when the weights cannot fit flash")
    a = ap.parse_args()
    D, H, C = a.d_model, a.pred_len, a.cycle

    import argparse as _ap
    mod = isolated_import(str(_REPO), "models.CycleNet")
    torch.manual_seed(WEIGHT_SEED)
    model = mod.Model(_ap.Namespace(seq_len=SEQ_LEN, pred_len=H, enc_in=D, cycle=C,
                                    model_type="linear", d_model=512,
                                    use_revin=True)).eval()

    n_params = sum(p.numel() for p in model.parameters())
    w_bytes = n_params * 4
    total = w_bytes + FIRMWARE_FLOOR
    print(f"params = {n_params}  weights = {w_bytes / 1024:.1f} KB  "
          f"+ firmware {FIRMWARE_FLOOR / 1024:.1f} KB = {total / 1024:.1f} KB "
          f"/ {FLASH_BYTES // 1024} KB flash")
    if total > FLASH_BYTES:
        msg = (f"CycleNet at H={H} does not fit flash "
               f"({total / 1024:.1f} KB > {FLASH_BYTES // 1024} KB). "
               f"That exclusion is arithmetic and needs no port.")
        if not a.force:
            raise SystemExit("REFUSING: " + msg)
        print("WARNING (--force): " + msg)

    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        sd = {k.replace("module.", "").replace("model.model.", "model."): v
              for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        print(f"loaded checkpoint {a.ckpt}")
    else:
        # the cycle starts at zeros upstream - fill it so the golden exercises it
        with torch.no_grad():
            model.cycleQueue.data.normal_(0.0, 0.5)

    g = torch.Generator().manual_seed(INPUT_SEED)
    # a non-zero level and scale per channel, so the RevIN path is exercised too
    x = (torch.randn(1, SEQ_LEN, D, generator=g) * 3.0
         + torch.linspace(-5.0, 5.0, D).view(1, 1, D))
    idx = torch.zeros(1, dtype=torch.long)          # phase 0, as in the sweeps

    with torch.no_grad():
        y = model(x, idx)                            # (1, H, D)

        # reference for the de-cycled input stage, and the streaming claim
        mean = x.mean(dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True) + 1e-5
        xs = (x - mean) / torch.sqrt(var) - model.cycleQueue(idx, SEQ_LEN)   # (1, L, D)
        W, b = model.model.weight, model.model.bias                           # (H, L), (H,)
        qout = model.cycleQueue((idx + SEQ_LEN) % C, H)[0]                     # (H, D)
        out = torch.empty(H, D)
        for t in range(H):
            out[t] = (xs[0].T @ W[t] + b[t] + qout[t]) * torch.sqrt(var[0, 0]) + mean[0, 0]
        dmax = (out - y[0]).abs().max().item()
        print(f"streaming equivalence: max|diff| = {dmax:.3e}")
        assert dmax < 1e-4, "step-wise emission does not reproduce the forecast"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "cyclenet_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_cyclenet.py"
                " - do not edit. */\n")
        src = a.ckpt if a.ckpt else f"random seed={WEIGHT_SEED}"
        f.write(f"/* source={src}, params={n_params}, L={SEQ_LEN}, H={H}, D={D},"
                f" cycle={C}, linear backbone, RevIN */\n")
        f.write("#ifndef CYCLENET_WEIGHTS_H\n#define CYCLENET_WEIGHTS_H\n\n")
        # stamps the port static-asserts against (stale header = compile error)
        f.write(f"#define CN_H_GEN {H}\n#define CN_D_W_GEN {D}\n#define CN_L_GEN {SEQ_LEN}\n")
        f.write(f"#define CN_CYCLE {C}\n\n")
        emit_array(f, "CN_W", W)                          # (H, L) out,in
        emit_array(f, "CN_B", b)                          # (H,)
        emit_array(f, "CN_Q", model.cycleQueue.data)      # (C, D)
        f.write("#endif /* CYCLENET_WEIGHTS_H */\n")

    with (OUT_DIR / "cyclenet_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_cyclenet.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED}, H={H},"
                f" streaming max|diff|={dmax:.3e} */\n")
        f.write("#ifndef CYCLENET_VECTORS_H\n#define CYCLENET_VECTORS_H\n\n")
        f.write(f"#define CN_D_GEN {D}\n\n")
        emit_array(f, "GC_X", x[0])            # (L, D) time-major
        emit_array(f, "GC_XS", xs[0])          # (L, D) normalised, de-cycled
        emit_array(f, "GC_Y", y[0])            # (H, D)
        f.write("#endif /* CYCLENET_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'cyclenet_weights.h'}")
    print(f"wrote {OUT_DIR / 'cyclenet_vectors.h'}")


if __name__ == "__main__":
    main()
