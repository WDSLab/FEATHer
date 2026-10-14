# -*- coding: utf-8 -*-
"""Export golden vectors + weights for the DiPE-Linear companion port.

Default config = benchmark defaults: use_revin + static freq/time weights +
single-expert FFT big-conv, L = H = 96, D = 7.

Uses the real upstream model via the benchmark wrapper; staged replication
is asserted equal to model(x) (eval mode - dropout inactive) before writing.

Also emits the 193-point DFT twiddle tables the expert conv needs
(time_len = L + H - 1 + 2*pad = 193, pad = 1).

--config / --data export an industrial candidate setting (resolved by
tools/results/protocol.resolve): num_experts (static router, temperature 1 at test as
upstream's TemperatureScaling callback sets it) and use_revin.  They go to
golden/dipe_geom.h (DIPE_E, DIPE_REVIN).  Without --config: the default above.

Run:  python deployment/board/export_golden_dipe.py [--config ETTh2] [--data MetroPT3]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baselines.DiPE_Linear.wrapper import build  # noqa: E402
from tools.results.protocol import resolve       # noqa: E402
from golden_io import emit_array             # noqa: E402

SEQ_LEN = 96
PRED_LEN = 96
D_MODEL = 7

WEIGHT_SEED = 2025
INPUT_SEED = 7

OUT_DIR = Path(__file__).resolve().parent / "golden"


def randomize_golden_weights(model) -> None:
    """Make the golden weights exercise every arithmetic path of the port.

    DiPE's init is an identity: freq and time weights all 1, the expert passes
    only bin 0 (weight [1, 0, 0, ...], bias 0).  Exported as-is the forecast is
    ~the window mean, so the whole expert path - padding, the long DFT, the
    complex weights, the slice - would go untested (a hardcoded pad of 1,
    wrong for every H >= 192, would pass).  Randomize all four.  Uses its own
    generator, so the result does not depend on the global RNG state.
    """
    gw = torch.Generator().manual_seed(WEIGHT_SEED + 1)
    with torch.no_grad():
        for p in (getattr(model.freq_w, "weight", None), getattr(model.time_w, "weight", None)):
            if p is not None:              # use_*_w = False: upstream Identity, no weight
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=gw))
        for p, s in ((model.expert.weight, 0.3), (model.expert.bias, 0.05)):
            p.copy_(torch.complex(s * torch.randn(p.shape, generator=gw),
                                  s * torch.randn(p.shape, generator=gw)))


def main() -> None:
    global PRED_LEN
    global D_MODEL
    _cli = argparse.ArgumentParser()
    _cli.add_argument("--pred_len", type=int, default=PRED_LEN,
                      help="board horizon; the C header takes -D<PFX>_H to match")
    _cli.add_argument("--d_model", type=int, default=D_MODEL,
                      help="channel count; the C header takes -D<PFX>_D to match")
    _cli.add_argument("--data", default="MetroPT3")
    _cli.add_argument("--config", default=None,
                      help="industrial candidate (ETTh1 / ETTh2 / ETTm2 / Weather / Electricity / default)")
    _a = _cli.parse_args()
    PRED_LEN, D_MODEL = _a.pred_len, _a.d_model
    ns = argparse.Namespace()
    if _a.config:
        _, _, ov = resolve("DiPE_Linear", "mfg", _a.data, PRED_LEN, "cand:" + _a.config)
        ns = argparse.Namespace(**ov)
    torch.manual_seed(WEIGHT_SEED)
    model = build(ns, D_MODEL, SEQ_LEN, PRED_LEN).eval()
    randomize_golden_weights(model)
    E, revin = model.num_experts, bool(model.use_revin)
    ind_f, ind_t, ind_c = bool(model.individual_f), bool(model.individual_t), bool(model.individual_c)
    use_f, use_t = bool(model.use_freq_w), bool(model.use_time_w)
    # 10-14: one code for the individual_* / weight switches (dipe_infer.h DIPE_VAR)
    var = ind_f | ind_t << 1 | ind_c << 2 | (not use_f) << 3 | (not use_t) << 4
    if E > 1:
        model.temperature = 1.0          # test-time value (upstream TemperatureScaling)
    print(f"config={_a.config or 'default (old)'} experts={E} revin={revin} individual f/t/c="
          f"{int(ind_f)}{int(ind_t)}{int(ind_c)} freq_w={int(use_f)} time_w={int(use_t)}")

    n_params = sum(p.numel() for p in model.parameters())
    cplx = sum(p.numel() for p in model.parameters() if p.is_complex())
    print(f"params = {n_params} (complex entries: {cplx} -> "
          f"{n_params + cplx} floats)")

    ex = model.expert
    pad = ex.pad_len
    tlen = ex.time_len
    # geometry guard for the reference H=96 cell; at other horizons the
    # geometry legitimately changes (the C header takes matching -D overrides).
    if PRED_LEN == 96:
        assert (pad, tlen) == (1, 193), f"unexpected expert geometry {pad}/{tlen}"

    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(1, SEQ_LEN, D_MODEL, generator=g)

    with torch.no_grad():
        xr = x.permute(0, 2, 1).unsqueeze(1)          # n 1 c l
        if revin:
            x_mean = xr.mean(dim=-1, keepdim=True)
            x_std = xr.std(dim=-1, keepdim=True).clamp(min=1e-7)  # UNBIASED
            xn = (xr - x_mean) / x_std
        else:
            x_mean = torch.zeros(1, 1, D_MODEL, 1)
            x_std = torch.ones(1, 1, D_MODEL, 1)
            xn = xr

        # static router: per-channel softmax over experts mixes every weight
        if E > 1:
            rank = torch.softmax(model.route / model.temperature, dim=1)   # 1,E,C,1
            mix = lambda w: (w * rank).sum(dim=1, keepdim=True)
        else:
            mix = lambda w: w
        # individual_*: upstream mixes with static_route = eye(D) -> each channel its own row
        own = lambda w: (w * model.static_route).sum(dim=1, keepdim=True)
        # StaticFreqWeight: rfft -> real weight -> irfft (Identity when off)
        xf = torch.fft.irfft(torch.fft.rfft(xn) * (own if ind_f else mix)(model.freq_w.weight),
                             n=SEQ_LEN) if use_f else xn
        # StaticTimeWeight
        xt = xf * (own if ind_t else mix)(model.time_w.weight) if use_t else xf
        # FFTExpandBigConv1d
        xp = F.pad(xt, [pad, PRED_LEN - 1 + pad])
        Xf = torch.fft.rfft(xp)
        wsel = own if ind_c else mix
        Yf = Xf * wsel(ex.weight) + wsel(ex.bias)
        yt = torch.fft.irfft(Yf, n=tlen)
        yc = yt[..., -PRED_LEN - pad:-pad]

        y = yc * x_std + x_mean if revin else yc
        y = y.squeeze(1).permute(0, 2, 1)             # n l c

        y_ref = model(x)
        if E > 1:     # weights are mixed per channel: a different op order, so not bit-equal
            assert torch.allclose(y, y_ref, atol=1e-5), "staged forward drifted from model.forward"
        else:
            assert torch.equal(y, y_ref), "staged forward drifted from model.forward"

    with (OUT_DIR / "dipe_geom.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dipe.py - do not edit. */\n")
        f.write(f"/* config={_a.config or 'default (old)'} */\n")
        f.write("#ifndef DIPE_GEOM_H\n#define DIPE_GEOM_H\n")
        f.write(f"#define DIPE_E {E}\n#define DIPE_REVIN {int(revin)}\n")
        if var:                          # written only off the default: old headers stay identical
            f.write(f"#define DIPE_IND_F {int(ind_f)}\n#define DIPE_IND_T {int(ind_t)}\n"
                    f"#define DIPE_IND_C {int(ind_c)}\n#define DIPE_FREQ_W {int(use_f)}\n"
                    f"#define DIPE_TIME_W {int(use_t)}\n")
        f.write("#endif /* DIPE_GEOM_H */\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "dipe_weights.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dipe.py"
                " - do not edit. */\n")
        f.write(f"/* seed={WEIGHT_SEED}, params={n_params} "
                f"({n_params + cplx} floats), all weights randomized seed="
                f"{WEIGHT_SEED + 1} */\n")
        f.write("#ifndef DIPE_WEIGHTS_H\n#define DIPE_WEIGHTS_H\n\n")
        # Stamp the horizon the weights were generated at.  The port
        # static-asserts against it, so a stale golden header (weights
        # from one H, code built for another) fails to COMPILE instead of
        # silently producing a plausible wrong forecast.
        f.write(f"#define DIPE_H_GEN {PRED_LEN}\n")
        # ...and the expert padding, which changes with H (1/2/4/8 at
        # H=96/192/336/720); a hardcoded pad would build and be wrong.
        f.write(f"#define DIPE_PAD_GEN {pad}\n")
        f.write(f"#define DIPE_E_GEN {E}\n#define DIPE_REVIN_GEN {int(revin)}\n"
                f"#define DIPE_RD_GEN {D_MODEL if E > 1 or ind_f or ind_t or ind_c else 1}\n")
        if var:
            f.write(f"#define DIPE_VAR_GEN {var}\n")
        f.write("\n")
        if use_f:
            emit_array(f, "DIPE_W_FREQ", model.freq_w.weight)  # ([E|D,] 49) real
        if use_t:
            emit_array(f, "DIPE_W_TIME", model.time_w.weight)  # ([E|D,] 96) real
        emit_array(f, "DIPE_W_EXP_RE", ex.weight.real)         # ([E|D,] 97)
        emit_array(f, "DIPE_W_EXP_IM", ex.weight.imag)
        emit_array(f, "DIPE_B_EXP_RE", ex.bias.real)
        emit_array(f, "DIPE_B_EXP_IM", ex.bias.imag)
        if E > 1:
            emit_array(f, "DIPE_ROUTE", model.route[0, :, :, 0])   # (E, D) router logits
        f.write("#endif /* DIPE_WEIGHTS_H */\n")

    with (OUT_DIR / "dipe_vectors.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dipe.py"
                " - do not edit. */\n")
        f.write(f"/* weight_seed={WEIGHT_SEED}, input_seed={INPUT_SEED} */\n")
        f.write("#ifndef DIPE_VECTORS_H\n#define DIPE_VECTORS_H\n\n")
        # Width the golden vectors were captured at; the test
        # _Static_asserts against it (a port built at another width
        # compiles fine and fails numerically, which reads as a code bug).
        f.write(f"#define DIPE_D_GEN {D_MODEL}\n\n")
        emit_array(f, "DG_X", x[0])                # (96,7) time-major
        emit_array(f, "DG_MEAN", x_mean[0, 0, :, 0])   # (7,)
        emit_array(f, "DG_STD", x_std[0, 0, :, 0])     # (7,)
        emit_array(f, "DG_XN", xn[0, 0])           # (7,96) channel-major
        emit_array(f, "DG_XF", xf[0, 0])           # (7,96)
        emit_array(f, "DG_XT", xt[0, 0])           # (7,96)
        emit_array(f, "DG_Y", y[0])                # (96,7) time-major
        f.write("#endif /* DIPE_VECTORS_H */\n")

    with (OUT_DIR / "dipe_twiddle.h").open("w", newline="\n") as f:
        f.write("/* Auto-generated by deployment/board/export_golden_dipe.py"
                " - do not edit. */\n")
        f.write("#ifndef DIPE_TWIDDLE_H\n#define DIPE_TWIDDLE_H\n\n")
        n = tlen
        for name, fn in (("DIPE_COS193", math.cos), ("DIPE_SIN193", math.sin)):
            f.write(f"static const float {name}[{n}] = {{\n")
            for i in range(0, n, 6):
                chunk = ", ".join(
                    f"{fn(2.0 * math.pi * j / n):.9e}f"
                    for j in range(i, min(i + 6, n))
                )
                f.write(f"  {chunk},\n")
            f.write("};\n\n")
        f.write("#endif /* DIPE_TWIDDLE_H */\n")

    print(f"wrote {OUT_DIR / 'dipe_weights.h'}")
    print(f"wrote {OUT_DIR / 'dipe_vectors.h'}")
    print(f"wrote {OUT_DIR / 'dipe_twiddle.h'}")


if __name__ == "__main__":
    main()
