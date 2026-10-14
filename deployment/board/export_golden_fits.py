# -*- coding: utf-8 -*-
"""Export golden vectors + weights for the FITS companion port.

Default config = benchmark defaults (shared upsampler, cut_freq = seq_len//4
= 24): complex Linear(24 -> 48) => 1,200 complex params (the registry count).

Uses the real upstream model via the benchmark wrapper; staged replication
is asserted equal to model(x) before writing. Also emits the 192-point DFT
twiddle tables (irfft length = seq_len + pred_len).

--config / --data export one of the industrial candidate settings
(resolved by tools/results/protocol.resolve, as the experiments do): cut_freq from
the official formula and the shared / per-channel upsampler.  The structural values
go to golden/fits_geom.h (FITS_CUT, FITS_IND), which the port includes.  Without
--config the export is the benchmark default above.

Run:  python deployment/board/export_golden_fits.py [--config Weather] [--data MetroPT3]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.FITS.wrapper import build  # noqa: E402
from tools.results.protocol import resolve  # noqa: E402
from golden_io import emit_array      # noqa: E402

SEQ_LEN = 96
PRED_LEN = 96
D_MODEL = 7

WEIGHT_SEED = 2025
INPUT_SEED = 7

OUT_DIR = Path(__file__).resolve().parent / "golden"


def randomize_golden_weights(model) -> None:
    """Make the golden weights exercise every arithmetic path of the port.

    The upstream init leaves the IMAGINARY part of the complex upsampler at 0,
    so every "- wi * im" / "+ wi * re" term would multiply zero and a sign or
    index error there would still pass.  Fill it with the same spread as the
    real part.  Uses its own generator, so the result does not depend on the
    global RNG state.
    """
    inner = model.inner
    gw = torch.Generator().manual_seed(WEIGHT_SEED + 1)
    ups = inner.freq_upsampler if inner.individual else [inner.freq_upsampler]
    with torch.no_grad():
        for up in ups:
            for p in (up.weight, up.bias):
                im = torch.randn(p.shape, generator=gw) * p.real.std()
                p.copy_(torch.complex(p.real.clone(), im))


def main() -> None:
    global PRED_LEN
    global D_MODEL
    _cli = argparse.ArgumentParser()
    _cli.add_argument("--pred_len", type=int, default=PRED_LEN,
                      help="board horizon; the C header takes -D<PFX>_H to match")
    _cli.add_argument("--d_model", type=int, default=D_MODEL,
                      help="channel count; the C header takes -D<PFX>_D to match")
    _cli.add_argument("--data", default="MetroPT3", help="industrial dataset (daily-cycle base_T)")
    _cli.add_argument("--config", default=None,
                      help="industrial candidate (ETTh1 / ETTm1 / Weather); omitted = old benchmark default")
    _a = _cli.parse_args()
    PRED_LEN, D_MODEL = _a.pred_len, _a.d_model
    ns = argparse.Namespace()
    if _a.config:
        _, _, ov = resolve("FITS", "mfg", _a.data, PRED_LEN, "cand:" + _a.config)
        ns = argparse.Namespace(**ov)
    torch.manual_seed(WEIGHT_SEED)
    model = build(ns, D_MODEL, SEQ_LEN, PRED_LEN).eval()
    randomize_golden_weights(model)
    inner = model.inner

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params} (complex -> {2 * n_params} floats)")

    cut = inner.dominance_freq
    ind = bool(inner.individual)
    up = int(cut * inner.length_ratio)
    print(f"config={_a.config or 'benchmark default'} cut={cut} up={up} individual={ind}")
    tn = SEQ_LEN + PRED_LEN
    # geometry guard for the reference H=96 default cell; elsewhere the
    # geometry legitimately changes (the C header takes matching -D overrides).
    if PRED_LEN == 96 and not _a.config:
        assert (cut, up) == (24, 48), f"unexpected FITS geometry {cut}/{up}"

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)

    with torch.no_grad():
        x_mean = torch.mean(x, dim=1, keepdim=True)
        xc = x - x_mean
        x_var = torch.var(xc, dim=1, keepdim=True) + 1e-5   # UNBIASED + eps
        sd = torch.sqrt(x_var)
        xn = xc / sd

        spec = torch.fft.rfft(xn, dim=1)                 # (1,49,D)
        low = spec[:, 0:cut, :]                          # (1,24,D)
        if ind:
            upspec = torch.stack([inner.freq_upsampler[i](low[:, :, i])
                                  for i in range(D_MODEL)], dim=-1)   # (1,UP,D)
        else:
            upspec = inner.freq_upsampler(
                low.permute(0, 2, 1)).permute(0, 2, 1)   # (1,48,D)
        padded = torch.zeros(
            [1, tn // 2 + 1, D_MODEL], dtype=upspec.dtype)
        padded[:, 0:up, :] = upspec
        low_xy = torch.fft.irfft(padded, dim=1)          # (1,192,D)
        low_xy = low_xy * inner.length_ratio             # x2.0
        xy = low_xy * sd + x_mean
        y = xy[:, -PRED_LEN:, :]

        y_ref = model(x)
        assert torch.equal(y, y_ref), "staged forward drifted from model.forward"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if ind:                                              # per channel: (D,UP,CUT) / (D,UP)
        W = torch.stack([u.weight for u in inner.freq_upsampler])
        B = torch.stack([u.bias for u in inner.freq_upsampler])
    else:
        W = inner.freq_upsampler.weight                  # (48,24) cfloat
        B = inner.freq_upsampler.bias                    # (48,)   cfloat

    with (OUT_DIR / "fits_geom.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_fits.py - do not edit. */\n")
        f.write(f"/* config={_a.config or 'benchmark default'} */\n")
        f.write("#ifndef FITS_GEOM_H\n#define FITS_GEOM_H\n")
        f.write(f"#define FITS_CUT {cut}\n#define FITS_IND {int(ind)}\n")
        f.write("#endif /* FITS_GEOM_H */\n")

    with (OUT_DIR / "fits_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_fits.py"
                " - do not edit. */\n")
        f.write(f"/* seed={WEIGHT_SEED}, params={n_params} complex "
                f"({2 * n_params} floats), imag randomized seed="
                f"{WEIGHT_SEED + 1} */\n")
        f.write("#ifndef FITS_WEIGHTS_H\n#define FITS_WEIGHTS_H\n\n")
        # Stamp the horizon the weights were generated at.  The port
        # static-asserts against it, so a stale golden header (weights
        # from one H, code built for another) fails to COMPILE instead of
        # silently producing a plausible wrong forecast.
        f.write(f"#define FITS_H_GEN {PRED_LEN}\n#define FITS_CUT_GEN {cut}\n"
                f"#define FITS_IND_GEN {int(ind)}\n#define FITS_WD_GEN {D_MODEL if ind else 1}\n\n")
        emit_array(f, "FITS_W_RE", W.real)               # (48,24)
        emit_array(f, "FITS_W_IM", W.imag)
        emit_array(f, "FITS_B_RE", B.real)               # (48,)
        emit_array(f, "FITS_B_IM", B.imag)
        f.write("#endif /* FITS_WEIGHTS_H */\n")

    with (OUT_DIR / "fits_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_fits.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED} */\n")
        f.write("#ifndef FITS_VECTORS_H\n#define FITS_VECTORS_H\n\n")
        # Width the golden vectors were captured at; the test
        # _Static_asserts against it (a port built at another width
        # compiles fine and fails numerically, which reads as a code bug).
        f.write(f"#define FITS_D_GEN {D_MODEL}\n\n")
        emit_array(f, "FG_X", x[0])                      # (96,7) time-major
        emit_array(f, "FG_MEAN", x_mean[0, 0])           # (7,)
        emit_array(f, "FG_SD", sd[0, 0])                 # (7,)
        emit_array(f, "FG_XN", xn[0])                    # (96,7)
        emit_array(f, "FG_UP_RE", upspec[0].real.permute(1, 0))  # (7,48)
        emit_array(f, "FG_UP_IM", upspec[0].imag.permute(1, 0))
        emit_array(f, "FG_Y", y[0])                      # (96,7)
        f.write("#endif /* FITS_VECTORS_H */\n")

    with (OUT_DIR / "fits_twiddle.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_fits.py"
                " - do not edit. */\n")
        f.write("#ifndef FITS_TWIDDLE_H\n#define FITS_TWIDDLE_H\n\n")
        for name, fn in (("FITS_COS192", math.cos), ("FITS_SIN192", math.sin)):
            f.write(f"static const float {name}[{tn}] = {{\n")
            for i in range(0, tn, 6):
                chunk = ", ".join(
                    f"{fn(2.0 * math.pi * j / tn):.9e}f"
                    for j in range(i, min(i + 6, tn))
                )
                f.write(f"  {chunk},\n")
            f.write("};\n\n")
        f.write("#endif /* FITS_TWIDDLE_H */\n")

    print(f"wrote {OUT_DIR / 'fits_weights.h'}")
    print(f"wrote {OUT_DIR / 'fits_vectors.h'}")
    print(f"wrote {OUT_DIR / 'fits_twiddle.h'}")


if __name__ == "__main__":
    main()
