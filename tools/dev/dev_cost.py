"""Params + Cortex-M3 peak SRAM for a FEATHer config, without touching a board.

The SRAM model is read off the shipping C port (deployment/board/src/
feather_infer.{c,h}), not guessed:

    arena  = 4 * (3*L*D  +  max(L*d_state, L*mlp_rank)  +  2*D  +  2*num_bands)
             |     |          |                             |       |
             |     s_B/s_C/s_D working buffers              |       gate vectors
             |                s_lat (DTK latent; the MLP    means/stdev
             |                rank shares the same slot)
    peak   = arena + 4*L*D            (caller's input window)
                   + 4*max(0, H*D - L*D)   (output only overflows s_D once H > L)
                   + RUNTIME_B        (stack/runtime, fitted once)

Calibration: the board measured 19,072 B for L=96 H=96 D=7 d_state=16
mlp_rank=16 num_bands=4 (BOARD_RESULTS_20260824.md, model error 0.2%), which
pins RUNTIME_B. Run with --validate to re-check that anchor.

Why this matters for the redesign: every term above is proportional to L or H.
That is the whole reason FEATHer is RAM-last of the seven measured models and
cannot reach H>=192 on a 20 KB part, and it is a property of the batch-window
dataflow rather than of any component. Use --explain to see the split.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RUNTIME_B = 2088          # fitted to the 19,072 B board measurement
SRAM_BUDGET = 20 * 1024
FLASH_BUDGET = 128 * 1024
FIRMWARE_FLASH_B = 10.4 * 1024   # measured floor (persistence build)


def sram_breakdown(L, H, D, d_state, mlp_rank, num_bands, use_residual_mlp=True,
                   dtk=True, stream_out=False, period=24):
    # s_lat holds the DTK latent (L*d_state) and is reused for the residual
    # MLP's L*mlp_rank, hence the max. With dtk_mode="none" there is no latent
    # sequence at all; the residual MLP is pointwise, so a compact port needs
    # only a rank-sized scratch, not an L-long one. Model that as the MLP term
    # surviving only when the DTK is present.
    if dtk:
        lat = L * (max(d_state, mlp_rank) if use_residual_mlp else d_state)
    else:
        lat = mlp_rank if use_residual_mlp else 0
    # One scratch buffer per live band plus the accumulator; a single band
    # (num_bands=1) needs one fewer.
    work = (3 if num_bands > 1 else 2) * L * D
    misc = 2 * D + 2 * num_bands
    arena = 4 * (work + lat + misc)
    inp = 4 * L * D
    if stream_out:
        # SPK is phase-independent: y[j*w+p] = <W[j], x_p>, so inverting the
        # loop nest to run over OUTPUT periods emits the forecast in chunks of
        # `period` steps instead of materialising all H. Verified numerically
        # on 2026-08-25 (pure loop reordering, max|diff| 2.68e-07), and it needs
        # no change to the PyTorch model — only the C port. The output slot then
        # costs one chunk and stops depending on H.
        out = max(0, 4 * period * D - 4 * L * D)
    else:
        out = 4 * max(0, H * D - L * D)
    return {
        "work_buffers": 4 * work,
        "latent": 4 * lat,
        "misc": 4 * misc,
        "arena": arena,
        "input": inp,
        "output_overflow": out,
        "runtime": RUNTIME_B,
        "peak": arena + inp + out + RUNTIME_B,
    }


def count_params(**kw):
    import torch  # noqa: F401  (imported lazily so --explain works without torch)
    from baselines.FEATHer.FEATHer import FEATHer
    m = FEATHer(**kw)
    return sum(p.numel() for p in m.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--d_model", type=int, default=7, help="channels D")
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--mlp_rank", type=int, default=16)
    p.add_argument("--num_bands", type=int, default=4)
    p.add_argument("--period", type=int, default=24)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_dtk", action="store_true",
                   help="dtk_mode=none (the redA configuration): removes the "
                        "L*d_state latent sequence entirely.")
    p.add_argument("--point_kernel", type=int, default=1)
    p.add_argument("--no_in_proj", action="store_true")
    p.add_argument("--spk_hidden", type=int, default=0)
    p.add_argument("--stream_out", action="store_true",
                   help="Assume the chunked SPK output loop (verified "
                        "2026-08-25): the output slot costs one period, not H.")
    p.add_argument("--horizons", default="96,192,336,720")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="Re-check the model against the board measurement.")
    a = p.parse_args()

    if a.validate:
        b = sram_breakdown(96, 96, 7, 16, 16, 4)
        meas = 19072
        err = 100 * (b["peak"] - meas) / meas
        print("board anchor  L96 H96 D7 s16 r16 nb4")
        print("  predicted %6d B   measured %6d B   error %+.2f%%" % (b["peak"], meas, err))
        print("  %s" % ("PASS" if abs(err) < 1.0 else "FAIL"))
        return

    try:
        params = count_params(
            seq_len=a.seq_len, pred_len=a.pred_len, d_model=a.d_model,
            d_state=a.d_state, kernel_size=a.kernel_size, period=a.period,
            num_bands=a.num_bands, use_residual_mlp=not a.no_residual,
            mlp_rank=a.mlp_rank, channel_mode="mix",
            dtk_mode="none" if a.no_dtk else "full",
            point_kernel=a.point_kernel, use_in_proj=not a.no_in_proj,
            spk_hidden=a.spk_hidden)
    except Exception as exc:                       # torch missing / bad config
        params = None
        print("[params unavailable: %s]" % exc)

    print("L=%d D=%d d_state=%d mlp_rank=%d bands=%d period=%d"
          % (a.seq_len, a.d_model, a.d_state, a.mlp_rank, a.num_bands, a.period))
    if params is not None:
        flash = FIRMWARE_FLASH_B + 4 * params
        print("params=%d   est. flash=%.1f KB / 128 KB" % (params, flash / 1024))
    print()
    print("%6s %10s %10s %8s" % ("H", "peak SRAM", "of 20 KB", "fits?"))
    for H in [int(x) for x in a.horizons.split(",")]:
        b = sram_breakdown(a.seq_len, H, a.d_model, a.d_state, a.mlp_rank,
                           a.num_bands, not a.no_residual, dtk=not a.no_dtk,
                           stream_out=a.stream_out, period=a.period)
        print("%6d %8.1f KB %9.0f%% %8s"
              % (H, b["peak"] / 1024, 100 * b["peak"] / SRAM_BUDGET,
                 "yes" if b["peak"] <= SRAM_BUDGET else "NO"))
        if a.explain:
            for k in ("work_buffers", "latent", "misc", "input",
                      "output_overflow", "runtime"):
                if b[k]:
                    print("        %-16s %7.2f KB  (%.0f%%)"
                          % (k, b[k] / 1024, 100 * b[k] / b["peak"]))


if __name__ == "__main__":
    main()

