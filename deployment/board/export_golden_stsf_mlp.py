# -*- coding: utf-8 -*-
"""Export golden vectors + weights for the SparseTSF-MLP port.

Companion to export_golden_stsf.py, which covers the 41-parameter linear head.
This is the `mlp` variant: the phase-wise Linear(n -> m) becomes
Linear(n -> 128) -> ReLU -> Linear(128 -> m).

Sizes at the MetroPT3 deployment cell (D=7, period 24), against FEATHer:

    horizon      SparseTSF-MLP     FEATHer
    H=96               1,181          938     <- what the board runs
    H=720              4,535        1,042

The point is the same at either end: every other baseline on the board sits
one to two orders of magnitude below FEATHer (41, 49, 339 parameters), so a
latency win there could always be read as a size win. This is the only pairing
where both models carry a comparable budget, which is what lets the board
number speak about the architecture rather than the parameter count.

Uses the real upstream model through the benchmark wrapper (no
re-implementation drift); the staged replication is asserted equal to model(x)
before anything is written.

Run:  python deployment/board/export_golden_stsf_mlp.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.SparseTSF.wrapper import build  # noqa: E402
from export_golden import emit_array           # noqa: E402  (same dir)

SEQ_LEN = 96
PRED_LEN = 96
D_MODEL = 7
PERIOD = 24            # _DATASET_OVERRIDES (SparseTSF, MetroPT3)
D_HIDDEN = 128         # upstream default d_model for the mlp head

WEIGHT_SEED = 2025
INPUT_SEED = 7

OUT_DIR = Path(__file__).resolve().parent / "golden"


def main() -> None:
    global PRED_LEN
    global D_MODEL
    global PERIOD
    _cli = argparse.ArgumentParser()
    _cli.add_argument("--pred_len", type=int, default=PRED_LEN,
                      help="board horizon; the C header takes -D<PFX>_H to match")
    _cli.add_argument("--d_model", type=int, default=D_MODEL,
                      help="channel count; the C header takes -D<PFX>_D to match")
    # 2026-10-07: an industrial candidate setting (resolved as the experiments do);
    # only the hidden width changes the structure (Traffic: 512).  It goes to
    # golden/stsf_mlp_geom.h (SMLP_DM).  Omitted = the old default (128).
    _cli.add_argument("--data", default="MetroPT3")
    _cli.add_argument("--config", default=None,
                      help="industrial candidate (ETTh1 / ETTh2 / Weather / Electricity / Traffic)")
    _a = _cli.parse_args()
    PRED_LEN, D_MODEL = _a.pred_len, _a.d_model
    d_hidden = D_HIDDEN
    if _a.config:
        sys.path.insert(0, str(_ROOT))
        from tools.results.protocol import resolve
        _, _, ov = resolve("SparseTSF_MLP", "mfg", _a.data, PRED_LEN, "cand:" + _a.config)
        # 10-12: the period rule gives 48 on WindSCADA / BlastFurnace -> SMLP_P in the geom header
        PERIOD = int(ov.get("sparsetsf_period_len", PERIOD))
        d_hidden = int(ov.get("sparsetsf_d_model", D_HIDDEN))
    print(f"config={_a.config or 'default (old)'} d_hidden={d_hidden}")
    torch.manual_seed(WEIGHT_SEED)
    args = argparse.Namespace(
        sparsetsf_period_len=PERIOD,
        sparsetsf_model_type="mlp",
        sparsetsf_d_model=d_hidden,
    )
    model = build(args, D_MODEL, SEQ_LEN, PRED_LEN).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")
    # conv 25 + Linear(4->128) 640 + Linear(128->4) 516 = 1,181 at H=96.
    # geometry guard: these constants describe the FROZEN H=96 cell
    # (the reference the 08-24 board numbers were taken at).  At other
    # horizons the geometry legitimately changes - print it instead, and
    # the C header must be given matching -D overrides.
    if PRED_LEN == 96 and d_hidden == D_HIDDEN and PERIOD == 24:
        assert n_params == 1181, f"expected the 1,181-param mlp variant, got {n_params}"

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)

    n = SEQ_LEN // PERIOD
    with torch.no_grad():
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)          # (1,1,D)
        xc = (x - seq_mean).permute(0, 2, 1)                  # (1,D,L)
        xa = model.conv1d(xc.reshape(-1, 1, SEQ_LEN)).reshape(
            -1, D_MODEL, SEQ_LEN) + xc                        # (1,D,L)
        xb = xa.reshape(-1, n, PERIOD).permute(0, 2, 1)       # (D,w,n)
        hid = torch.relu(model.mlp[0](xb))                    # (D,w,128)
        yb = model.mlp[2](hid)                                # (D,w,m)
        y = yb.permute(0, 2, 1).reshape(1, D_MODEL, PRED_LEN)
        y = y.permute(0, 2, 1) + seq_mean                     # (1,H,D)

        y_ref = model(x)
        assert torch.equal(y, y_ref), "staged forward drifted from model.forward"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "stsf_mlp_geom.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_stsf_mlp.py - do not edit. */\n")
        f.write("#ifndef STSF_MLP_GEOM_H\n#define STSF_MLP_GEOM_H\n")
        f.write(f"#define SMLP_DM {d_hidden}\n#define SMLP_P {PERIOD}\n#endif /* STSF_MLP_GEOM_H */\n")

    with (OUT_DIR / "stsf_mlp_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_stsf_mlp.py"
                " - do not edit. */\n")
        f.write(f"/* seed={WEIGHT_SEED}, params={n_params} */\n")
        f.write("#ifndef STSF_MLP_WEIGHTS_H\n#define STSF_MLP_WEIGHTS_H\n\n")
        # Stamp the horizon the weights were generated at.  The port
        # static-asserts against it, so a stale golden header (weights
        # from one H, code built for another) fails to COMPILE instead of
        # silently producing a plausible wrong forecast.
        f.write(f"#define SMLP_H_GEN {PRED_LEN}\n#define SMLP_DM_GEN {d_hidden}\n\n")
        emit_array(f, "SMLP_W_CONV", model.conv1d.weight)     # (1,1,25)
        emit_array(f, "SMLP_W_FC1", model.mlp[0].weight)      # (128,4) out,in
        emit_array(f, "SMLP_B_FC1", model.mlp[0].bias)        # (128,)
        emit_array(f, "SMLP_W_FC2", model.mlp[2].weight)      # (4,128) out,in
        emit_array(f, "SMLP_B_FC2", model.mlp[2].bias)        # (4,)
        f.write("#endif /* STSF_MLP_WEIGHTS_H */\n")

    with (OUT_DIR / "stsf_mlp_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_stsf_mlp.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED} */\n")
        # SM_ prefix, not MG_: mixlinear_vectors.h already claims MG_X /
        # MG_MEAN / MG_Y, and main.c includes every model's vectors at once so
        # one build can switch between them. Colliding names are a redefinition
        # error at best and the wrong model's input at worst.
        f.write("#ifndef STSF_MLP_VECTORS_H\n#define STSF_MLP_VECTORS_H\n\n")
        # Width the golden vectors were captured at; the test
        # _Static_asserts against it (a port built at another width
        # compiles fine and fails numerically, which reads as a code bug).
        f.write(f"#define SMLP_D_GEN {D_MODEL}\n\n")
        emit_array(f, "SM_X", x[0])                # (96,7) time-major
        emit_array(f, "SM_MEAN", seq_mean[0, 0])   # (7,)
        emit_array(f, "SM_XAGG", xa[0])            # (7,96) channel-major
        # hid is (D, period, 128) -> flatten in the C loop's order
        emit_array(f, "SM_HID", hid.reshape(-1))   # (7*24*128,)
        emit_array(f, "SM_Y", y[0])                # (96,7)
        f.write("#endif /* STSF_MLP_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'stsf_mlp_weights.h'}")
    print(f"wrote {OUT_DIR / 'stsf_mlp_vectors.h'}")


if __name__ == "__main__":
    main()
