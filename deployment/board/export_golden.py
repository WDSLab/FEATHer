# -*- coding: utf-8 -*-
"""Export float32 golden vectors + weights for the on-board C port.

Builds the SHIPPING FEATHer-N ??the model the accuracy tables report ??in its
MetroPT3 deployment cell (mix + residual MLP r16, d_state 16, kernel 7,
num_bands 4, period 24, D=7, L=H=96 -> 938 params), runs one forward pass
step-by-step (mirroring FEATHer.forward exactly), and dumps

    golden/feather_weights.h   float32 weight arrays (flash constants later)
    golden/feather_vectors.h   input + every intermediate stage + final output

The staged replication is asserted equal to model(x) before anything is
written, so the vectors cannot drift from the real model.

Weights are random-init under a fixed seed: latency/SRAM/flash and
"same function as PyTorch" do not depend on trained values
(EDGE_EXPERIMENT.md - smoke ckpt suffices; real ckpt swap is a re-run).

Run:  python deployment/board/export_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.FEATHer.FEATHer import FEATHer  # noqa: E402

# Shipping mfg config = baselines/__init__.py:_FEATHER_MFG_N, on the MetroPT3
# deployment cell (D=7, FFT-detected period 24). The 07-27 export used the
# pre-08 canonical (d8/p12/B3, no residual MLP, 453 params) and so measured a
# model the paper no longer reports.
SEQ_LEN = 96
PRED_LEN = 96
D_MODEL = 7
D_STATE = 16
KERNEL_SIZE = 7
PERIOD = 24
NUM_BANDS = 4
MLP_RANK = 16
USE_RESIDUAL_MLP = True

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


def main() -> None:
    torch.manual_seed(WEIGHT_SEED)
    model = FEATHer(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=D_MODEL,
        d_state=D_STATE, kernel_size=KERNEL_SIZE, use_norm=True,
        period=PERIOD, num_bands=NUM_BANDS,
        use_residual_mlp=USE_RESIDUAL_MLP, mlp_rank=MLP_RANK,
        channel_mode="mix",
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)

    # ---- staged forward, mirroring FEATHer.forward line by line ----
    with torch.no_grad():
        means = x.mean(1, keepdim=True)
        stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - means) / stdev

        x_proj = model.in_proj(x_norm)
        h_t = x_proj.permute(0, 2, 1)

        fP = model.conv_point(h_t).permute(0, 2, 1)
        oP = model.dense_kernel(fP)

        if NUM_BANDS == 4:
            fH = model.conv_high(h_t).permute(0, 2, 1)
            oH = model.dense_kernel(fH)

        fM = model.conv_mid(h_t).permute(0, 2, 1)
        oM = model.dense_kernel(fM)

        fL_ds = model.pool_low(h_t).permute(0, 2, 1)
        fL = F.interpolate(
            fL_ds.permute(0, 2, 1), size=SEQ_LEN,
            mode="linear", align_corners=False,
        ).permute(0, 2, 1)
        oL = model.dense_kernel(fL)

        # gate internals (mirrors FFTFrequencyGate.forward)
        Freq = torch.fft.rfft(x_norm, dim=1)          # (1, 49, D) complex
        mag = torch.abs(Freq)
        mag_mean = mag.mean(dim=2, keepdim=True).permute(0, 2, 1)  # (1,1,49)
        z = model.freq_gate.conv(mag_mean)
        z = model.freq_gate.pool(z).squeeze(-1)       # (1, 3)
        gate = torch.softmax(z, dim=-1)

        # Band order must match FEATHer.forward's bands_list exactly, since the
        # gate weights are indexed positionally.
        bands = torch.stack([oP, oH, oM, oL] if NUM_BANDS == 4 else [oP, oM, oL],
                            dim=1)
        bands_sum = (gate.view(1, NUM_BANDS, 1, 1) * bands).sum(dim=1)

        # residual MLP (FEATHer's only nonlinearity)
        mlp_lat = F.gelu(model.nl_down(bands_sum))
        hidden = bands_sum + model.nl_up(mlp_lat)

        # SPK internals
        hh_t = hidden.permute(0, 2, 1)
        h_agg = hh_t + model.sparse_head.agg_conv(hh_t)   # (1, D, L)
        y_spk = model.sparse_head(hidden)                 # (1, H, D)

        y = y_spk * stdev[:, 0, :].unsqueeze(1).repeat(1, PRED_LEN, 1)
        y = y + means[:, 0, :].unsqueeze(1).repeat(1, PRED_LEN, 1)

        # guard: staged replication must equal the real forward bit-for-bit
        y_ref = model(x)
        assert torch.equal(y, y_ref), "staged forward drifted from model.forward"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- weights header ----
    with (OUT_DIR / "feather_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden.py"
                " - do not edit. */\n")
        f.write(f"/* seed={WEIGHT_SEED}, params={n_params} */\n")
        f.write("#ifndef FEATHER_WEIGHTS_H\n#define FEATHER_WEIGHTS_H\n\n")
        emit_array(f, "W_INPROJ", model.in_proj.weight)        # (7,7) out,in
        emit_array(f, "B_INPROJ", model.in_proj.bias)          # (7,)
        emit_array(f, "W_CONVP", model.conv_point.weight)      # (7,1,1)
        emit_array(f, "B_CONVP", model.conv_point.bias)        # (7,)
        if NUM_BANDS == 4:
            emit_array(f, "W_CONVH", model.conv_high.weight)   # (7,1,3)
            emit_array(f, "B_CONVH", model.conv_high.bias)     # (7,)
        emit_array(f, "W_CONVM", model.conv_mid.weight)        # (7,1,5)
        emit_array(f, "B_CONVM", model.conv_mid.bias)          # (7,)
        emit_array(f, "W_DTK_IN", model.dense_kernel.in_proj.weight)   # (8,7)
        emit_array(f, "W_DTK_K", model.dense_kernel.kernel)            # (8,7)
        emit_array(f, "W_DTK_OUT", model.dense_kernel.out_proj.weight) # (7,8)
        emit_array(f, "W_GATE", model.freq_gate.conv.weight)   # (3,1,5)
        emit_array(f, "B_GATE", model.freq_gate.conv.bias)     # (3,)
        emit_array(f, "W_NLDOWN", model.nl_down.weight)        # (16,7) out,in
        emit_array(f, "B_NLDOWN", model.nl_down.bias)          # (16,)
        emit_array(f, "W_NLUP", model.nl_up.weight)            # (7,16) out,in
        emit_array(f, "B_NLUP", model.nl_up.bias)              # (7,)
        emit_array(f, "W_AGG", model.sparse_head.agg_conv.weight)      # (7,1,25)
        emit_array(f, "W_BB", model.sparse_head.backbone.weight)       # (4,4) out,in
        f.write("#endif /* FEATHER_WEIGHTS_H */\n")

    # ---- golden vectors header (input + every stage) ----
    with (OUT_DIR / "feather_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED} */\n")
        f.write("#ifndef FEATHER_VECTORS_H\n#define FEATHER_VECTORS_H\n\n")
        emit_array(f, "G_X", x[0])                 # (96,7) row-major (t,d)
        emit_array(f, "G_MEANS", means[0, 0])      # (7,)
        emit_array(f, "G_STDEV", stdev[0, 0])      # (7,)
        emit_array(f, "G_XNORM", x_norm[0])        # (96,7)  trap#3 eps-in-var
        emit_array(f, "G_XPROJ", x_proj[0])        # (96,7)
        emit_array(f, "G_FP", fP[0])               # (96,7)
        if NUM_BANDS == 4:
            emit_array(f, "G_FH", fH[0])           # (96,7)
        emit_array(f, "G_FM", fM[0])               # (96,7)
        emit_array(f, "G_FLDS", fL_ds[0])          # (24,7)
        emit_array(f, "G_FL", fL[0])               # (96,7)  trap#1 interp
        emit_array(f, "G_OP", oP[0])               # (96,7)  trap#2 causal DTK
        if NUM_BANDS == 4:
            emit_array(f, "G_OH", oH[0])
        emit_array(f, "G_OM", oM[0])
        emit_array(f, "G_OL", oL[0])
        emit_array(f, "G_RFFT_RE", Freq[0, :, 0].real)  # (49,) trap#4 rfft ch0
        emit_array(f, "G_RFFT_IM", Freq[0, :, 0].imag)  # (49,)
        emit_array(f, "G_MAGMEAN", mag_mean[0, 0])      # (49,)
        emit_array(f, "G_GATE_Z", z[0])            # (3,)
        emit_array(f, "G_GATE", gate[0])           # (3,)
        emit_array(f, "G_HIDDEN", bands_sum[0])    # (96,7)  gated, pre-MLP
        emit_array(f, "G_MLP", hidden[0])          # (96,7)  after residual MLP
        emit_array(f, "G_HAGG", h_agg[0])          # (7,96) channel-major
        emit_array(f, "G_YSPK", y_spk[0])          # (96,7)
        emit_array(f, "G_Y", y[0])                 # (96,7)
        f.write("#endif /* FEATHER_VECTORS_H */\n")

    # ---- 96-point DFT twiddle table (double-computed, float32-rounded:
    # identical constants on x86 and ARM, ~768 B of flash) ----
    import math
    with (OUT_DIR / "fea_twiddle.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden.py"
                " - do not edit. */\n")
        f.write("#ifndef FEA_TWIDDLE_H\n#define FEA_TWIDDLE_H\n\n")
        for name, fn in (("FEA_COS96", math.cos), ("FEA_SIN96", math.sin)):
            f.write(f"static const float {name}[96] = {{\n")
            for i in range(0, 96, 6):
                chunk = ", ".join(
                    f"{fn(2.0 * math.pi * j / 96.0):.9e}f"
                    for j in range(i, i + 6)
                )
                f.write(f"  {chunk},\n")
            f.write("};\n\n")
        f.write("#endif /* FEA_TWIDDLE_H */\n")

    print(f"wrote {OUT_DIR / 'feather_weights.h'}")
    print(f"wrote {OUT_DIR / 'feather_vectors.h'}")
    print(f"wrote {OUT_DIR / 'fea_twiddle.h'}")
    print("gate =", gate[0].tolist())


if __name__ == "__main__":
    main()
