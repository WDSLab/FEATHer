# -*- coding: utf-8 -*-
"""Export weights + golden vectors for FEATHer-CI (channel_mode="ci_full").

Channels are folded into the batch, so every layer is built at WIDTH 1 and the weights are shared across channels:
golden/feather_weights.h holds the width-1 network (with its *_GEN stamps), the C port is feather_core.c compiled
with -DFEA2_D=1, driven per channel by src/feather_infer.c.  There is no second forward implementation.

Run:
    python deployment/board/export/export_golden_feather.py --pred_len 720 --d_model 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.FEATHer.FEATHer import FEATHer  # noqa: E402
from baselines.FEATHer.ablation import FEATHerNoGate  # noqa: E402
from golden_io import OUT_DIR, emit_array  # noqa: E402

# MetroPT3 deployment cell (D=7, FFT-detected period 24) - the same cell as the
# other ports, so latency is comparable across ports.
SEQ_LEN = 96
PERIOD = 24

# frozen architecture (configs/FEATHer.yaml), channel-independent
ARCH = dict(
    num_bands=3,
    point_kernel=3,
    use_residual_mlp=True,
    mlp_rank=16,
    channel_mode="ci_full",
)

WEIGHT_SEED = 2025
INPUT_SEED = 7


def add_arch_args(ap: argparse.ArgumentParser) -> None:
    """size / structure switches; the defaults are the frozen model and leave the headers byte-identical
    (nothing extra is stamped)."""
    # SPK segment length (0 = the period) and the shared aggregation kernel; the C header takes
    # -DFEA2_SEG / -DFEA2_AGG_SHARED, stamped as FEA2_SEG_GEN / FEA2_AGGS_GEN
    ap.add_argument("--head_seg", type=int, default=0)
    ap.add_argument("--agg_shared", type=int, default=0)
    # structural switches -> -DFEA2_KP / FEA2_NB / FEA2_USE_NORM / FEA2_GATE_NONE,
    # stamped as FEA2_KP_GEN / FEA2_NB_GEN / FEA2_NORM_GEN / FEA2_GNONE_GEN
    ap.add_argument("--point_kernel", type=int, default=3)
    ap.add_argument("--num_bands", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--use_norm", type=int, default=1)
    ap.add_argument("--gate_mode", default="banded", choices=("banded", "none"))
    # residual MLP off -> -DFEA2_USE_MLP=0, stamped as FEA2_MLP_GEN 0
    ap.add_argument("--use_residual_mlp", type=int, default=1)


def apply_arch_args(a, arch: dict, period: int) -> list:
    """fold the switches of add_arch_args into `arch`; returns the extra #define lines the weights header must
    carry"""
    assert a.point_kernel % 2 == 1, "even point_kernel widths cannot run (conv output length changes)"
    stamps = []
    if a.head_seg or a.agg_shared:
        arch["head_seg"], arch["agg_shared"] = a.head_seg, bool(a.agg_shared)
        stamps += [f"#define FEA2_SEG_GEN {a.head_seg or period}",
                   f"#define FEA2_AGGS_GEN {int(bool(a.agg_shared))}"]
    for key, val, dflt, macro in (("point_kernel", a.point_kernel, 3, "FEA2_KP_GEN"),
                                  ("num_bands", a.num_bands, 3, "FEA2_NB_GEN"),
                                  ("use_norm", bool(a.use_norm), True, "FEA2_NORM_GEN")):
        if val != dflt:
            arch[key] = val
            stamps.append(f"#define {macro} {int(val)}")
    if a.gate_mode == "none":
        stamps.append("#define FEA2_GNONE_GEN 1")
    if not a.use_residual_mlp:
        arch["use_residual_mlp"] = False
        stamps.append("#define FEA2_MLP_GEN 0")
    return stamps


def make_model(H: int, D: int, P: int, arch: dict, gate_none: bool) -> FEATHer:
    torch.manual_seed(WEIGHT_SEED)
    cls = FEATHerNoGate if gate_none else FEATHer
    return cls(seq_len=SEQ_LEN, pred_len=H, d_model=D, period=P, **{"use_norm": True, **arch}).eval()


def randomise_gate(m: FEATHer) -> None:
    if m.freq_gate is None:
        return
    torch.manual_seed(WEIGHT_SEED + 1)
    with torch.no_grad():
        m.freq_gate.proj.weight.normal_(0.0, 0.5)
        m.freq_gate.proj.bias.normal_(0.0, 0.5)


def write_weights(f, m: FEATHer, H: int, P: int, stamps: list) -> None:
    """stamps + weight arrays; the caller writes the guard"""
    # Stamp the horizon the weights were generated at.  The port
    # static-asserts against it, so a stale golden header (weights
    # from one H, code built for another) fails to COMPILE instead of
    # silently producing a plausible wrong forecast.
    f.write(f"#define FEA2_H_GEN {H}\n")
    # same guard for the residual-MLP rank: a rank-16 header built at
    # -DFEA2_MLP_R=4 would otherwise compile and give a wrong forecast
    if m.use_residual_mlp:
        f.write(f"#define FEA2_R_GEN {m.nl_down.out_features}\n")
    f.write(f"#define FEA2_P_GEN {P}\n")
    for s in stamps:
        f.write(s + "\n")
    f.write("\n")
    emit_array(f, "W_CONVP", m.conv_point.weight)          # (D,1,KP)
    emit_array(f, "B_CONVP", m.conv_point.bias)            # (D,)
    if m.conv_mid is not None:
        emit_array(f, "W_CONVM", m.conv_mid.weight)        # (D,1,5)
        emit_array(f, "B_CONVM", m.conv_mid.bias)          # (D,)
    if m.freq_gate is not None:                            # the C port runs the gate whenever the model has one
        emit_array(f, "W_GATE2", m.freq_gate.proj.weight)  # (S,S) out,in
        emit_array(f, "B_GATE2", m.freq_gate.proj.bias)    # (S,)
    if m.use_residual_mlp:
        emit_array(f, "W_NLDOWN2", m.nl_down.weight)       # (R,D) out,in
        emit_array(f, "B_NLDOWN2", m.nl_down.bias)         # (R,)
        emit_array(f, "W_NLUP2", m.nl_up.weight)           # (D,R) out,in
        emit_array(f, "B_NLUP2", m.nl_up.bias)             # (D,)
    emit_array(f, "W_AGG2", m.sparse_head.agg_conv.weight)  # (D,1,25)
    emit_array(f, "W_BB2", m.sparse_head.backbone.weight)   # (m,n) out,in
    f.write("#endif /* FEATHER_WEIGHTS_H */\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=7)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mlp_rank", type=int, default=ARCH["mlp_rank"],
                    help="residual MLP rank; the C header takes -DFEA2_MLP_R to match")
    ap.add_argument("--period", type=int, default=PERIOD,
                    help="SPK period; the C header takes -DFEA2_PERIOD to match")
    add_arch_args(ap)          # head_seg / agg_shared / point_kernel / num_bands / use_norm / gate_mode
    a = ap.parse_args()
    H, D, P = a.pred_len, a.d_model, a.period
    assert H % (a.head_seg or P) == 0, "pred_len must be divisible by the segment length"

    arch = dict(ARCH)
    arch["mlp_rank"] = a.mlp_rank
    stamps = apply_arch_args(a, arch, P)
    m = make_model(H, D, P, arch, a.gate_mode == "none")
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                          strict=True)
        print(f"loaded checkpoint {a.ckpt}")
    else:
        randomise_gate(m)

    n_params = sum(p.numel() for p in m.parameters())
    # ci_full builds every layer at width 1, so the parameter count must NOT
    # depend on D.  If it does, the variant was not actually folded.
    assert m.conv_point.weight.shape[0] == 1, "ci_full did not build at width 1"
    print(f"params = {n_params}  (L={SEQ_LEN} H={H} D={D} P={P}, width-1)")

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D, generator=g)
    with torch.no_grad():
        y = m(x)

        # guard: the C port runs the width-1 network once per channel and
        # concatenates.  Verify that is exactly what ci_full computes.
        cols = []
        for c in range(D):
            cols.append(m(x[:, :, c:c + 1]))
        y_loop = torch.cat(cols, dim=2)
        dmax = (y_loop - y).abs().max().item()
        print(f"per-channel loop equivalence: max|diff| = {dmax:.3e}")
        assert dmax < 1e-5, "ci_full is not a per-channel application"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "feather_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by export_golden_feather.py"
                " - do not edit.  WIDTH-1 (ci_full) weights. */\n")
        src = a.ckpt if a.ckpt else f"random seed={WEIGHT_SEED}"
        f.write(f"/* source={src}, params={n_params}, H={H}, ci_full */\n")
        f.write("#ifndef FEATHER_WEIGHTS_H\n#define FEATHER_WEIGHTS_H\n\n")
        write_weights(f, m, H, P, stamps)

    with (OUT_DIR / "feather_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by export_golden_feather.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED}, "
                f"H={H}, D={D}, loop max|diff|={dmax:.3e} */\n")
        f.write("#ifndef FEATHER_VECTORS_H\n#define FEATHER_VECTORS_H\n\n")
        f.write(f"#define GCI_D {D}\n#define GCI_H {H}\n\n")
        emit_array(f, "GCI_X", x[0])       # (L, D) time-major
        emit_array(f, "GCI_Y", y[0])       # (H, D) time-major
        f.write("#endif /* FEATHER_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'feather_weights.h'} (width-1)")
    print(f"wrote {OUT_DIR / 'feather_vectors.h'}")


if __name__ == "__main__":
    main()
