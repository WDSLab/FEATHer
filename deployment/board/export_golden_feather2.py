# -*- coding: utf-8 -*-
"""Export float32 weights + golden vectors for the FEATHer v2 C port.

Builds the SHIPPING model frozen in docs/MODEL_FINAL.md:

    --num_bands 3 --point_kernel 3
    --gate_mode banded          (mfg: channel_mode=mix)

runs one forward pass stage by stage (mirroring FEATHer.forward), asserts the
staged replication equals model(x) bit-for-bit, ALSO asserts that emitting the
forecast one output period at a time reproduces the full forecast (this is the
streaming claim the C port rests on), and writes

    golden/feather2_weights.h   float32 weight arrays
    golden/feather2_vectors.h   input + every intermediate stage + output

Weights come from a real checkpoint when --ckpt is given, otherwise from a
fixed seed: latency / SRAM / flash and "same function as PyTorch" do not
depend on trained values, but the paper's board numbers should come from the
shipping checkpoint, so pass one for the final run.

Run:
    python deployment/board/export_golden_feather2.py
    python deployment/board/export_golden_feather2.py --pred_len 720
    python deployment/board/export_golden_feather2.py --ckpt path/to.pth
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
from baselines.FEATHer.ablation import FEATHerNoGate  # noqa: E402

# MetroPT3 deployment cell (D=7, FFT-detected period 24) - the same cell as the
# other ports, so latency is comparable across ports.
SEQ_LEN = 96
D_MODEL = 7
PERIOD = 24

# frozen architecture (docs/MODEL_FINAL.md)
ARCH = dict(
    num_bands=3,
    point_kernel=3,
    use_residual_mlp=True,
    mlp_rank=16,
    channel_mode="mix",
)

WEIGHT_SEED = 2025
INPUT_SEED = 7
OUT_DIR = Path(__file__).resolve().parent / "golden"
GATE_NONE = False       # gate_mode none (ablation.FEATHerNoGate); set by --gate_mode


def add_arch_args(ap: argparse.ArgumentParser) -> None:
    """size / structure switches shared with export_golden_feather2ci.py; the defaults are the
    frozen model and leave the headers byte-identical (nothing extra is stamped)."""
    # SPK segment length (0 = the period) and the shared aggregation kernel; the C header takes
    # -DFEA2_SEG / -DFEA2_AGG_SHARED, stamped as FEA2_SEG_GEN / FEA2_AGGS_GEN
    ap.add_argument("--head_seg", type=int, default=0)
    ap.add_argument("--agg_shared", type=int, default=0)
    # 10-14 structural switches -> -DFEA2_KP / FEA2_NB / FEA2_USE_NORM / FEA2_GATE_NONE,
    # stamped as FEA2_KP_GEN / FEA2_NB_GEN / FEA2_NORM_GEN / FEA2_GNONE_GEN
    ap.add_argument("--point_kernel", type=int, default=3)
    ap.add_argument("--num_bands", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--use_norm", type=int, default=1)
    ap.add_argument("--gate_mode", default="banded", choices=("banded", "none"))


def apply_arch_args(a, arch: dict, period: int) -> list:
    """fold the switches of add_arch_args into `arch` (and GATE_NONE); returns the extra
    #define lines the weights header must carry"""
    global GATE_NONE
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
    GATE_NONE = a.gate_mode == "none"
    if GATE_NONE:
        stamps.append("#define FEA2_GNONE_GEN 1")
    return stamps


def gate_on(m: FEATHer) -> bool:
    """the C port runs the spectrum gate only when it can differ from 1 (FEA2_GATE_ON)"""
    return m.freq_gate is not None and m.num_bands > 1


def make_model(H: int, D: int, P: int, arch: dict) -> FEATHer:
    torch.manual_seed(WEIGHT_SEED)
    cls = FEATHerNoGate if GATE_NONE else FEATHer
    return cls(seq_len=SEQ_LEN, pred_len=H, d_model=D, period=P, **{"use_norm": True, **arch}).eval()


def write_weights(f, m: FEATHer, H: int, P: int, stamps: list) -> None:
    """stamps + weight arrays (shared by the mix and CI exports); the caller writes the guard"""
    # Stamp the horizon the weights were generated at.  The port
    # static-asserts against it, so a stale golden header (weights
    # from one H, code built for another) fails to COMPILE instead of
    # silently producing a plausible wrong forecast.
    f.write(f"#define FEA2_H_GEN {H}\n")
    # same guard for the residual-MLP rank: a rank-16 header built at
    # -DFEA2_MLP_R=4 would otherwise compile and give a wrong forecast
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
    if gate_on(m):
        emit_array(f, "W_GATE2", m.freq_gate.proj.weight)  # (S,S) out,in
        emit_array(f, "B_GATE2", m.freq_gate.proj.bias)    # (S,)
    emit_array(f, "W_NLDOWN2", m.nl_down.weight)           # (R,D) out,in
    emit_array(f, "B_NLDOWN2", m.nl_down.bias)             # (R,)
    emit_array(f, "W_NLUP2", m.nl_up.weight)               # (D,R) out,in
    emit_array(f, "B_NLUP2", m.nl_up.bias)                 # (D,)
    emit_array(f, "W_AGG2", m.sparse_head.agg_conv.weight)  # (D,1,25)
    emit_array(f, "W_BB2", m.sparse_head.backbone.weight)   # (m,n) out,in
    f.write("#endif /* FEATHER2_WEIGHTS_H */\n")


def emit_array(f, name: str, t: torch.Tensor) -> None:
    flat = t.detach().to(torch.float32).flatten().tolist()
    f.write(f"/* shape={list(t.shape)} */\n")
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}f" for v in flat[i:i + 6])
        f.write(f"  {chunk},\n")
    f.write("};\n\n")


def build(pred_len: int, ckpt: str | None) -> FEATHer:
    m = make_model(pred_len, D_MODEL, PERIOD, ARCH)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"checkpoint does not match the frozen architecture\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}")
        print(f"loaded checkpoint {ckpt}")
    else:
        # The banded gate is zero-initialised BY DESIGN (it starts exactly
        # uniform, see FFTFrequencyGate.__init__), so an untrained model
        # returns [1/S, ..., 1/S] and the golden vectors would pass even if
        # the C gate maths were wrong.  Perturb proj so the test actually
        # exercises the log-energy-ratio path.  Trained checkpoints skip this.
        randomise_gate(m)
        print("untrained: gate proj randomised so the gate path is covered")
    return m


def randomise_gate(m: FEATHer) -> None:
    if m.freq_gate is None:
        return
    torch.manual_seed(WEIGHT_SEED + 1)
    with torch.no_grad():
        m.freq_gate.proj.weight.normal_(0.0, 0.5)
        m.freq_gate.proj.bias.normal_(0.0, 0.5)


def staged_forward(m: FEATHer, x: torch.Tensor, pred_len: int):
    """Mirror FEATHer.forward for the frozen config, returning every stage."""
    with torch.no_grad():
        NB, D = m.num_bands, x.size(2)
        if m.use_norm:
            means = x.mean(1, keepdim=True)
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = (x - means) / stdev
        else:                       # no RevIN: the C port leaves its captures at 0
            means = stdev = torch.zeros(1, 1, D)
            x_norm = x

        # no linear channel mixer before the bands
        h_t = x_norm.permute(0, 2, 1)

        # no temporal kernel after the band convs: band == conv out
        # (an absent band is exported as zeros: the C capture is never written)
        bP = m.conv_point(h_t).permute(0, 2, 1)
        bM = m.conv_mid(h_t).permute(0, 2, 1) if NB == 3 else torch.zeros_like(bP)
        if NB >= 2:
            bL_ds = m.pool_low(h_t).permute(0, 2, 1)
            bL = F.interpolate(bL_ds.permute(0, 2, 1), size=SEQ_LEN,
                               mode="linear", align_corners=False).permute(0, 2, 1)
        else:
            bL_ds, bL = torch.zeros(1, SEQ_LEN // 4, D), torch.zeros_like(bP)
        band_list = [bP] + ([bM] if NB == 3 else []) + ([bL] if NB >= 2 else [])

        if not gate_on(m):          # gate none / one band: weight 1, no spectrum in the port
            P = torch.zeros(1, SEQ_LEN // 2 + 1)
            feats = z = torch.zeros(1, NB)
            gate = torch.ones(1, NB)
            bands = torch.stack(band_list, dim=1)
            hidden = bands.sum(dim=1) if m.freq_gate is None else \
                (m.freq_gate(x_norm).view(1, NB, 1, 1) * bands).sum(dim=1)
            return finish(m, x, x_norm, pred_len, means, stdev, bP, bM, bL_ds, bL,
                          P, feats, z, gate, hidden)

        # banded gate internals
        Freq = torch.fft.rfft(x_norm, dim=1)
        mag = torch.abs(Freq)
        P = (mag ** 2).mean(dim=2)                          # (1, NF)
        total = P.sum(dim=1, keepdim=True) + 1e-6
        bounds = m.freq_gate._bounds                        # low -> high
        ratios = torch.stack([P[:, lo:hi].sum(dim=1) for lo, hi in bounds],
                             dim=1) / total
        feats = torch.log(ratios.flip(dims=[1]) + 1e-6)
        feats = feats - feats.mean(dim=1, keepdim=True)
        z = m.freq_gate.proj(feats)
        gate = torch.softmax(z, dim=-1)

        bands = torch.stack(band_list, dim=1)               # order matters
        hidden = (gate.view(1, NB, 1, 1) * bands).sum(dim=1)
        return finish(m, x, x_norm, pred_len, means, stdev, bP, bM, bL_ds, bL,
                      P, feats, z, gate, hidden)


def finish(m, x, x_norm, pred_len, means, stdev, bP, bM, bL_ds, bL, P, feats, z, gate, hidden):
    """residual MLP -> SPK head -> denorm, asserted equal to model(x)"""
    D_MODEL = x.size(2)
    with torch.no_grad():
        mlp = hidden + m.nl_up(F.gelu(m.nl_down(hidden)))

        hh_t = mlp.permute(0, 2, 1)
        if m.sparse_head.agg_shared:                       # one kernel for all channels
            h_agg = hh_t + m.sparse_head.agg_conv(hh_t.reshape(D_MODEL, 1, SEQ_LEN)).reshape(1, D_MODEL, SEQ_LEN)
        else:
            h_agg = hh_t + m.sparse_head.agg_conv(hh_t)     # (1, D, L)
        y_spk = m.sparse_head(mlp)

        if m.use_norm:
            y = y_spk * stdev[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, pred_len, 1)
        else:
            y = y_spk

        assert torch.equal(y, m(x)), "staged forward drifted from model.forward"
    return dict(means=means, stdev=stdev, x_norm=x_norm, bP=bP, bM=bM,
                bL_ds=bL_ds, bL=bL, power=P, feats=feats, z=z, gate=gate,
                hidden=hidden, mlp=mlp, h_agg=h_agg, y_spk=y_spk, y=y)


def check_streaming(st, pred_len: int, m: FEATHer) -> float:
    """The claim the whole port rests on.

    y[om*P + ph, d] = sum_in W_BB[om, in] * h_agg[d, in*P + ph]

    so an output step depends on exactly ONE backbone row; emitting P steps at
    a time is a loop reordering, not an approximation.  Verify by rebuilding
    the forecast block by block and comparing to the full-length one.
    """
    W = m.sparse_head.backbone.weight            # (m, n)
    h_agg, means, stdev = st["h_agg"][0], st["means"][0, 0], st["stdev"][0, 0]
    S = m.sparse_head.period                     # segment length (= PERIOD unless --head_seg)
    n, mm = SEQ_LEN // S, pred_len // S
    out = torch.empty(pred_len, D_MODEL)
    for om in range(mm):
        blk = torch.empty(S, D_MODEL)
        for ph in range(S):
            for d in range(D_MODEL):
                acc = sum(W[om, i] * h_agg[d, i * S + ph] for i in range(n))
                blk[ph, d] = acc * stdev[d] + means[d] if m.use_norm else acc
        out[om * S:(om + 1) * S] = blk
    return (out - st["y"][0]).abs().max().item()


def main() -> None:
    global D_MODEL, PERIOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mlp_rank", type=int, default=16,
                    help="residual MLP rank; the C header takes -DFEA2_MLP_R to match")
    ap.add_argument("--d_model", type=int, default=D_MODEL,
                    help="channel count; the C header takes -D<PFX>_D to match")
    # SPK period (FFT rule <= 48: 48 on WindSCADA / BlastFurnace); the C header
    # takes -DFEA2_PERIOD to match, stamped as FEA2_P_GEN
    ap.add_argument("--period", type=int, default=PERIOD)
    add_arch_args(ap)
    a = ap.parse_args()
    D_MODEL, PERIOD = a.d_model, a.period
    ARCH["mlp_rank"] = a.mlp_rank
    stamps = apply_arch_args(a, ARCH, PERIOD)       # none by default: the frozen golden stays byte-identical
    H = a.pred_len
    assert H % (a.head_seg or PERIOD) == 0, "pred_len must be divisible by the segment length"

    m = build(H, a.ckpt)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params = {n_params}   (L={SEQ_LEN} H={H} D={D_MODEL} P={PERIOD})")

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)
    st = staged_forward(m, x, H)

    dmax = check_streaming(st, H, m)
    print(f"streaming equivalence: max|diff| = {dmax:.3e}")
    assert dmax < 1e-5, "block-wise emission does not reproduce the forecast"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "feather2_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_feather2.py"
                " - do not edit. */\n")
        src = a.ckpt if a.ckpt else f"random seed={WEIGHT_SEED}"
        f.write(f"/* source={src}, params={n_params}, H={H} */\n")
        f.write("#ifndef FEATHER2_WEIGHTS_H\n#define FEATHER2_WEIGHTS_H\n\n")
        write_weights(f, m, H, PERIOD, stamps)

    with (OUT_DIR / "feather2_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_feather2.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED}, H={H},"
                f" streaming max|diff|={dmax:.3e} */\n")
        f.write("#ifndef FEATHER2_VECTORS_H\n#define FEATHER2_VECTORS_H\n\n")
        # Width the golden vectors were captured at; the test
        # _Static_asserts against it (a port built at another width
        # compiles fine and fails numerically, which reads as a code bug).
        f.write(f"#define FEA2_D_GEN {D_MODEL}\n\n")
        emit_array(f, "G2_X", x[0])                    # (L,D) time-major
        emit_array(f, "G2_MEANS", st["means"][0, 0])
        emit_array(f, "G2_STDEV", st["stdev"][0, 0])
        emit_array(f, "G2_XNORM", st["x_norm"][0])     # eps-inside-sqrt trap
        emit_array(f, "G2_POWER", st["power"][0])      # (NF,) channel-mean |X|^2
        emit_array(f, "G2_FEATS", st["feats"][0])      # (S,) centered log-ratio
        emit_array(f, "G2_GATE", st["gate"][0])        # (S,)
        emit_array(f, "G2_BP", st["bP"][0])            # (L,D)
        emit_array(f, "G2_BM", st["bM"][0])
        emit_array(f, "G2_BLDS", st["bL_ds"][0])       # (L/4,D)
        emit_array(f, "G2_BL", st["bL"][0])            # interp trap
        emit_array(f, "G2_HIDDEN", st["hidden"][0])    # gated sum, pre-MLP
        emit_array(f, "G2_MLP", st["mlp"][0])          # after residual MLP
        emit_array(f, "G2_HAGG", st["h_agg"][0])       # (D,L) channel-major
        emit_array(f, "G2_YSPK", st["y_spk"][0])       # (H,D)
        emit_array(f, "G2_Y", st["y"][0])              # (H,D) final
        f.write("#endif /* FEATHER2_VECTORS_H */\n")

    print(f"wrote {OUT_DIR / 'feather2_weights.h'}")
    print(f"wrote {OUT_DIR / 'feather2_vectors.h'}")
    print("gate =", [round(v, 6) for v in st["gate"][0].tolist()])


if __name__ == "__main__":
    main()
