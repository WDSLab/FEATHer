# -*- coding: utf-8 -*-
"""Export float32 weights + golden vectors for the FEATHer in1 C port.

in1 = the shipping mix model with ONE change: the residual mixer reads the
normalized INPUT instead of the gated bands (dev arm mx_in1_r4, 2026-09-14):

    --no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none
    --channel_mode mix --mix_mode input --mix_pool 1 --mlp_rank R   (R = 4 by default)
    --gate_mode banded_sensor   (each sensor gates on its own spectrum)

    hidden = sum_s g_s * band_s(x_norm)                 per sensor
    hidden = hidden + W_up GELU(W_down x_norm[t] + b)   cross-sensor, per step
    y      = SPK(hidden) -> RevIN^-1

Because the correction depends on x_norm and not on the bands, the port can
keep the R-dim summary u[t] = GELU(W_down x_norm[t] + b) (L*R floats) and run
every other stage one sensor at a time, emitting that sensor's forecast before
moving on.  This script checks that claim in PyTorch (sensor-by-sensor
emission reproduces the forecast) and writes

    golden/feather2in_weights.h   float32 weight arrays
    golden/feather2in_vectors.h   input + intermediate stages + output

Run:
    python deployment/board/export_golden_feather2in.py
    python deployment/board/export_golden_feather2in.py --pred_len 720 --d_model 28
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.FEATHer.FEATHer import FEATHer  # noqa: E402

SEQ_LEN = 96
D_MODEL = 7
PERIOD = 24
MLP_RANK = 4

ARCH = dict(
    num_bands=3,
    point_kernel=3,
    dtk_mode="none",
    gate_mode="banded_sensor",
    use_in_proj=False,
    use_residual_mlp=True,
    channel_mode="mix",
    mix_mode="input",
    mix_pool=1,
    d_state=16,          # dead once dtk_mode="none"; constructor still takes it
    kernel_size=7,
)

WEIGHT_SEED = 2025
INPUT_SEED = 7
OUT_DIR = Path(__file__).resolve().parent / "golden"


def emit_array(f, name: str, t: torch.Tensor) -> None:
    flat = t.detach().to(torch.float32).flatten().tolist()
    f.write(f"/* shape={list(t.shape)} */\n")
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}f" for v in flat[i:i + 6])
        f.write(f"  {chunk},\n")
    f.write("};\n\n")


def build(pred_len: int, rank: int, ckpt: str | None) -> FEATHer:
    torch.manual_seed(WEIGHT_SEED)
    m = FEATHer(seq_len=SEQ_LEN, pred_len=pred_len, d_model=D_MODEL,
                use_norm=True, period=PERIOD, mlp_rank=rank, **ARCH).eval()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"checkpoint does not match the in1 architecture\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}")
        print(f"loaded checkpoint {ckpt}")
    else:
        # The banded gate is zero-initialised (exactly uniform), which would
        # let a wrong C gate pass.  Perturb it so the gate path is exercised.
        torch.manual_seed(WEIGHT_SEED + 1)
        with torch.no_grad():
            m.freq_gate.proj.weight.normal_(0.0, 0.5)
            m.freq_gate.proj.bias.normal_(0.0, 0.5)
        print("untrained: gate proj randomised so the gate path is covered")
    return m


def staged_forward(m: FEATHer, x: torch.Tensor, pred_len: int):
    """Mirror FEATHer.forward for the in1 config, returning every stage."""
    with torch.no_grad():
        means = x.mean(1, keepdim=True)
        stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - means) / stdev

        h_t = x_norm.permute(0, 2, 1)
        bP = m.conv_point(h_t).permute(0, 2, 1)
        bM = m.conv_mid(h_t).permute(0, 2, 1)
        bL = F.interpolate(m.pool_low(h_t), size=SEQ_LEN, mode="linear",
                           align_corners=False).permute(0, 2, 1)

        Freq = torch.fft.rfft(x_norm, dim=1)
        mag = torch.abs(Freq)
        P = mag ** 2                                        # (1, NF, D) per sensor
        total = P.sum(dim=1, keepdim=True) + 1e-6
        ratios = torch.stack([P[:, lo:hi].sum(dim=1)
                              for lo, hi in m.freq_gate._bounds], dim=1) / total
        feats = torch.log(ratios.flip(dims=[1]) + 1e-6)     # (1, S, D)
        feats = feats - feats.mean(dim=1, keepdim=True)
        gate = torch.softmax(m.freq_gate.proj(feats.transpose(1, 2)), dim=-1)  # (1, D, S)

        bands = torch.stack([bP, bM, bL], dim=1)
        hidden = (gate.permute(0, 2, 1).unsqueeze(2) * bands).sum(dim=1)

        u = F.gelu(m.nl_down(x_norm))                      # (1, L, R) summary
        mlp = hidden + m.nl_up(u)

        hh_t = mlp.permute(0, 2, 1)
        h_agg = hh_t + m.sparse_head.agg_conv(hh_t)        # (1, D, L)
        y = m.sparse_head(mlp)
        y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
        y = y + means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)

        assert torch.equal(y, m(x)), "staged forward drifted from model.forward"
    return dict(means=means, stdev=stdev, x_norm=x_norm, power=P, feats=feats,
                gate=gate, u=u, hidden=hidden, mlp=mlp, h_agg=h_agg, y=y)


def check_sensor_streaming(st, pred_len: int, m: FEATHer) -> float:
    """The claim the port rests on: once u is known, sensor d's whole forecast
    depends only on sensor d's own x_norm column and u, so sensors can be
    finished one at a time.  Rebuild each sensor from those two inputs alone
    and compare to the joint forecast."""
    with torch.no_grad():
        W = m.sparse_head.backbone.weight
        g = st["gate"][0]
        u = st["u"][0]
        n, mm = SEQ_LEN // PERIOD, pred_len // PERIOD
        worst = 0.0
        for d in range(D_MODEL):
            col = st["x_norm"][0, :, d].view(1, 1, SEQ_LEN)
            bP = F.conv1d(col, m.conv_point.weight[d:d + 1], m.conv_point.bias[d:d + 1],
                          padding=m.conv_point.padding)
            bM = F.conv1d(col, m.conv_mid.weight[d:d + 1], m.conv_mid.bias[d:d + 1],
                          padding=m.conv_mid.padding)
            bL = F.interpolate(F.avg_pool1d(col, 4, 4), size=SEQ_LEN, mode="linear",
                               align_corners=False)
            h = (g[d, 0] * bP + g[d, 1] * bM + g[d, 2] * bL)[0, 0]
            h = h + (u @ m.nl_up.weight[d] + m.nl_up.bias[d])
            agg = h + F.conv1d(h.view(1, 1, -1), m.sparse_head.agg_conv.weight[d:d + 1],
                               padding=m.sparse_head.agg_conv.padding)[0, 0]
            out = torch.empty(pred_len)
            for om in range(mm):
                for ph in range(PERIOD):
                    acc = sum(W[om, i] * agg[i * PERIOD + ph] for i in range(n))
                    out[om * PERIOD + ph] = acc * st["stdev"][0, 0, d] + st["means"][0, 0, d]
            worst = max(worst, (out - st["y"][0, :, d]).abs().max().item())
    return worst


def main() -> None:
    global D_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--d_model", type=int, default=D_MODEL)
    ap.add_argument("--mlp_rank", type=int, default=MLP_RANK)
    ap.add_argument("--ckpt", type=str, default=None)
    a = ap.parse_args()
    D_MODEL = a.d_model
    H, R = a.pred_len, a.mlp_rank
    assert H % PERIOD == 0, "pred_len must be divisible by period"

    m = build(H, R, a.ckpt)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params = {n_params}   (L={SEQ_LEN} H={H} D={D_MODEL} P={PERIOD} R={R})")

    # Per-sensor offsets and scales, so the normalization path is not an
    # identity on already-standardised noise.
    g = torch.Generator().manual_seed(INPUT_SEED)
    scale = 0.5 + 2.0 * torch.rand(D_MODEL, generator=g)
    offset = 5.0 * torch.randn(D_MODEL, generator=g)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g) * scale + offset
    st = staged_forward(m, x, H)

    dmax = check_sensor_streaming(st, H, m)
    print(f"sensor-by-sensor emission: max|diff| = {dmax:.3e}")
    assert dmax < 1e-4, "sensor-wise emission does not reproduce the forecast"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "feather2in_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_feather2in.py"
                " - do not edit. */\n")
        src = a.ckpt if a.ckpt else f"random seed={WEIGHT_SEED}"
        f.write(f"/* source={src}, params={n_params}, H={H}, R={R} */\n")
        f.write("#ifndef FEATHER2IN_WEIGHTS_H\n#define FEATHER2IN_WEIGHTS_H\n\n")
        f.write(f"#define FEA2I_H_GEN {H}\n#define FEA2I_R_GEN {R}\n\n")
        emit_array(f, "W_CONVP", m.conv_point.weight)           # (D,1,3)
        emit_array(f, "B_CONVP", m.conv_point.bias)
        emit_array(f, "W_CONVM", m.conv_mid.weight)             # (D,1,5)
        emit_array(f, "B_CONVM", m.conv_mid.bias)
        emit_array(f, "W_GATE", m.freq_gate.proj.weight)        # (S,S)
        emit_array(f, "B_GATE", m.freq_gate.proj.bias)
        emit_array(f, "W_NLDOWN", m.nl_down.weight)             # (R,D)
        emit_array(f, "B_NLDOWN", m.nl_down.bias)
        emit_array(f, "W_NLUP", m.nl_up.weight)                 # (D,R)
        emit_array(f, "B_NLUP", m.nl_up.bias)
        emit_array(f, "W_AGG", m.sparse_head.agg_conv.weight)   # (D,1,K)
        emit_array(f, "W_BB", m.sparse_head.backbone.weight)    # (m,n)
        f.write("#endif /* FEATHER2IN_WEIGHTS_H */\n")

    with (OUT_DIR / "feather2in_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_feather2in.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED}, H={H}, R={R},"
                f" sensor streaming max|diff|={dmax:.3e} */\n")
        f.write("#ifndef FEATHER2IN_VECTORS_H\n#define FEATHER2IN_VECTORS_H\n\n")
        f.write(f"#define FEA2I_D_GEN {D_MODEL}\n\n")
        emit_array(f, "GI_X", x[0])                            # (L,D) time-major
        emit_array(f, "GI_MEANS", st["means"][0, 0])
        emit_array(f, "GI_STDEV", st["stdev"][0, 0])
        emit_array(f, "GI_POWER", st["power"][0].T)            # (D,NF) per sensor
        emit_array(f, "GI_FEATS", st["feats"][0].T)            # (D,S)
        emit_array(f, "GI_GATE", st["gate"][0])                # (D,S)
        emit_array(f, "GI_U", st["u"][0])                      # (L,R) time-major
        # the port finishes one sensor at a time, so these are channel-major
        emit_array(f, "GI_HIDDEN", st["hidden"][0].T)          # (D,L)
        emit_array(f, "GI_MLP", st["mlp"][0].T)                # (D,L)
        emit_array(f, "GI_HAGG", st["h_agg"][0])               # (D,L)
        emit_array(f, "GI_Y", st["y"][0])                      # (H,D) time-major
        f.write("#endif /* FEATHER2IN_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'feather2in_weights.h'}")
    print(f"wrote {OUT_DIR / 'feather2in_vectors.h'}")
    print("gate (sensor 0) =", [round(v, 6) for v in st["gate"][0, 0].tolist()])


if __name__ == "__main__":
    main()
