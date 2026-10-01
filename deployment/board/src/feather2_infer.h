/*
 * FEATHer v2 (2026-09-04 shipping model) float32 inference for Cortex-M3.
 *
 * Model, as frozen in MODEL_FINAL.md:
 *     --no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none
 *     --gate_mode banded          (mfg: channel_mode=mix)
 *
 *   RevIN -> [no in_proj] -> 3 bands (conv k=3 | conv k=5 | AvgPool4+interp)
 *         -> [no DTK] -> banded FFT gate -> weighted sum
 *         -> residual MLP (D->R->GELU->D) -> SPK head -> RevIN^-1
 *
 * Deliberate departures from src/feather_infer.c (the 938-parameter 08-24
 * port, kept for provenance):
 *
 *   (1) OUTPUT STREAMING.  The SPK backbone maps input periods -> output
 *       periods per PHASE, so output step (om*P + ph) of channel d depends
 *       only on backbone row `om` and on channel d.  Emitting one period of
 *       one channel at a time is a pure loop reordering (verified in PyTorch,
 *       max|diff| 2.7e-07), and it removes H from the arena entirely: the old
 *       port parked H*D floats (19.7 KB at H=720) that this one never
 *       allocates.
 *
 *   (2) BAND ACCUMULATE-IN-PLACE.  Each band is scaled by its gate weight
 *       and added to the hidden accumulator as it is produced, so the
 *       L*D band scratch buffer disappears.  Legal because the gate is
 *       computed from x_norm (not from the bands) and each band conv reads
 *       only x_norm.
 *
 *   (3) the residual MLP bottleneck is per-timestep (R floats) instead of
 *       L*R (1,536 floats = 6 KB in the old port) — the MLP is applied
 *       independently at each step, so nothing needs to be materialised.
 *
 *   (4) ONE (L, D) BUFFER (2026-09-14, equal-effort rule, see
 *       deployment/board/README.md).  The residual MLP needs every channel of
 *       h[t] at once, so the hidden state is held for all channels - that is
 *       the structural cost of mixing.  Nothing else is: x_norm is parked in
 *       the hidden accumulator until the gate is known, the bands and the SPK
 *       aggregation run one channel at a time, and the output block is one
 *       channel wide and lives in the scratch.  The previous build also kept
 *       a separate (L, D) x_norm buffer, a (LP, D) pooled band and a (P, D)
 *       output block, i.e. 1,352 B per channel instead of 776 B (input window
 *       included).  No stage is recomputed; the forecast is bit-identical.
 *
 * Arena is HORIZON-INDEPENDENT.  See FEA2_ARENA_BYTES.
 *
 * Layout: time-major buf[t*D + d] throughout.
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef FEATHER2_INFER_H
#define FEATHER2_INFER_H

/* Board builds: deployment/board/board_prep.py writes board_cfg.h into the
 * CubeIDE project (never into this repo), carrying the geometry of the model
 * under test.  x86 tests and measure_footprint.py do not see it. */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#include <stddef.h>

#include "fea_kernels.h"

/* ---- geometry (overridable at build time, e.g. -DFEA2_H=720) ---- */
#ifndef FEA2_L
#define FEA2_L 96                       /* seq_len                        */
#endif
#ifndef FEA2_H
#define FEA2_H 96                       /* pred_len                       */
#endif
#ifndef FEA2_D
#define FEA2_D 7                        /* channels (MetroPT3 cell)       */
#endif
#ifndef FEA2_PERIOD
#define FEA2_PERIOD 24                  /* SPK period (FFT-detected)      */
#endif

#define FEA2_NB 3                       /* bands: POINT / MID / LOW       */
#define FEA2_KP 3                       /* POINT conv kernel              */
#define FEA2_KM 5                       /* MID   conv kernel              */
#ifndef FEA2_MLP_R                      /* overridable (2026-09-22 R=4 check) */
#define FEA2_MLP_R 16                   /* residual MLP bottleneck rank   */
#endif

#define FEA2_NF (FEA2_L / 2 + 1)                 /* rfft bins  = 49       */
#define FEA2_LP (FEA2_L / 4)                     /* pooled LOW = 24       */
#define FEA2_N  (FEA2_L / FEA2_PERIOD)           /* input periods         */
#define FEA2_M  (FEA2_H / FEA2_PERIOD)           /* output periods        */
#define FEA2_AGG_K (2 * (FEA2_PERIOD / 2) + 1)   /* SPK agg conv kernel   */

/* ---- static arena, in floats ----
 *   s_C       L*D   x_norm -> hidden accumulator -> h_agg  (the ONLY D*L)
 *   s_scratch       ONE buffer shared by five SEQUENTIAL stages (below)
 *   + mean/stdev (2D), gate (NB)
 * No term contains H; the only term that scales with D beyond 2D is s_C.
 *
 * The scratch stages never coexist.  The forward pass touches them strictly
 * in this order and never revisits an earlier one:
 *   1. s_spec 3*NF+64  gate rfft re/im/power + fast-FFT scratch (banded_gate)
 *   2. s_row  L     one channel's x_norm   } per channel, both live at once,
 *      s_pool LP    that channel pooled    } so they sit side by side
 *   3. s_mlp  R     residual MLP bottleneck (per timestep)
 *   4. s_row  L     SPK aggregation: one channel's pre-conv column
 *   5. s_blk  P     one output period of one channel (fea2_block, after
 *                   fea2_begin has returned - the streaming window)
 * so one buffer of max(3*NF+64, L+LP, R, P) floats serves all five.  The same
 * overlay is applied to the MixLinear / FITS / DiPE ports: it is an
 * implementation tidy-up, and giving it only to our own model would make the
 * footprint table measure our effort rather than the designs.
 */
#define FEA2_MAX2(a, b) ((a) > (b) ? (a) : (b))
/* 3*NF + 64: the gate's re/im/power plus the fast FFT's 2 x 32 scratch */
#define FEA2_SCRATCH_N FEA2_MAX2(FEA2_MAX2(3 * FEA2_NF + 64, FEA2_L + FEA2_LP), \
                                 FEA2_MAX2(FEA2_MLP_R, FEA2_PERIOD))
#define FEA2_ARENA_N (FEA2_L * FEA2_D + FEA2_SCRATCH_N \
                      + 2 * FEA2_D + FEA2_NB)
#define FEA2_ARENA_BYTES (FEA2_ARENA_N * (int)sizeof(float))

/* ---- streaming API ----
 *
 * Typical board use — never materialises the forecast:
 *
 *     fea2_begin(x);                                  // x is read only
 *     for (int om = 0; om < FEA2_M; ++om)
 *         for (int d = 0; d < FEA2_D; ++d)
 *             sink_write(om, d, fea2_block(om, d));   // FEA2_PERIOD floats
 *
 * fea2_begin() runs everything up to and including the SPK aggregation and
 * leaves h_agg in the arena; fea2_block() then costs one backbone row for one
 * channel.  The returned pointer is valid until the next fea2_block() or
 * fea2_begin() call.
 *
 * fea2_input() returns the arena's (L, D) buffer.  A caller that must stage
 * the window itself (e.g. gather one channel out of a wider buffer) may write
 * it there and call fea2_begin(fea2_input()); it is normalized in place.
 */
void fea2_begin(const float *x);
const float *fea2_block(int om, int d);
float *fea2_input(void);

/* convenience wrapper for the golden test / batch eval: streams every block
 * into a caller-owned (FEA2_H * FEA2_D) buffer.  The full-length buffer is
 * the CALLER's — the model never allocates one. */
void fea2_forward(const float *x, float *y);

/* stage capture for golden comparison; only with -DFEA2_CAPTURE_STAGES */
#ifdef FEA2_CAPTURE_STAGES
extern float fea2_st_xnorm[FEA2_L * FEA2_D];
extern float fea2_st_mean[FEA2_D];
extern float fea2_st_stdev[FEA2_D];
extern float fea2_st_power[FEA2_NF];      /* channel-mean power spectrum */
extern float fea2_st_feats[FEA2_NB];      /* centered log energy ratios  */
extern float fea2_st_gate[FEA2_NB];
extern float fea2_st_bP[FEA2_L * FEA2_D];
extern float fea2_st_bM[FEA2_L * FEA2_D];
extern float fea2_st_bL[FEA2_L * FEA2_D];
extern float fea2_st_hidden[FEA2_L * FEA2_D];   /* after gated sum      */
extern float fea2_st_mlp[FEA2_L * FEA2_D];      /* after residual MLP   */
extern float fea2_st_hagg[FEA2_D * FEA2_L];     /* channel-major        */
#endif

#endif /* FEATHER2_INFER_H */
