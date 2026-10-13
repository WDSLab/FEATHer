/*
 * DiPE-Linear float32 inference - companion model (optional tier).
 * Single-expert paper-default config: RevIN (unbiased std, clamp 1e-7,
 * no eps) -> static freq weight (rfft96 . w . irfft96) -> static time
 * weight -> FFT big-conv (193-point DFT, complex weight + bias) -> denorm.
 */
#ifndef DIPE_INFER_H
#define DIPE_INFER_H

#include "fea_emit.h"

/* Board builds: deployment/board/board_prep.py writes board_cfg.h into the
 * CubeIDE project (never into this repo), carrying the geometry of the model
 * under test.  x86 tests and measure_footprint.py do not see it. */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#include "fea_kernels.h"

/* experts (static router) and RevIN on/off: written by the export into
 * golden/dipe_geom.h (2026-10-07, candidate settings); the fallback is the
 * single-expert default this port was written at */
#if defined(__has_include)
#if __has_include("dipe_geom.h")
#include "dipe_geom.h"
#endif
#endif
#ifndef DIPE_E
#define DIPE_E 1
#endif
#ifndef DIPE_REVIN
#define DIPE_REVIN 1
#endif

#ifndef DIPE_L
#define DIPE_L 96
#endif
#ifndef DIPE_H
#define DIPE_H 96
#endif
#ifndef DIPE_D
#define DIPE_D 7
#endif
/* Expert padding, as upstream FFTExpandBigConv1d computes it:
 *     pad = max(floor((L + H - 1) / 100), 1)
 * = 1 / 2 / 4 / 8 at H = 96 / 192 / 336 / 720.  This used to be a literal 1,
 * which is only right at H=96: every longer horizon ran a shorter DFT than
 * the model and produced a wrong forecast - and still passed its golden test,
 * because the exported weights were DiPE's identity init (fixed 2026-09-14;
 * the export now stamps DIPE_PAD_GEN and the port asserts against it). */
#define DIPE_PAD_RAW ((DIPE_L + DIPE_H - 1) / 100)
#define DIPE_PAD (DIPE_PAD_RAW > 1 ? DIPE_PAD_RAW : 1)
#define DIPE_TN (DIPE_L + DIPE_H - 1 + 2 * DIPE_PAD)   /* = 193 at H=96 */
#define DIPE_TF (DIPE_TN / 2 + 1)                       /* = 97  at H=96 */

/* x: (DIPE_L, DIPE_D) time-major -> y: (DIPE_H, DIPE_D) time-major */
void dipe_forward(const float *x, float *y);
/* one channel d -> yc[0..H-1] (stream per channel) */
void dipe_forward_ch(const float *x, int d, float *yc);
/* one channel d, each value to emit(d, t, v) as it is final (10-13, no H buffer) */
void dipe_forward_emit(const float *x, int d, fea_emit_fn emit);

/* stage buffers for golden comparison (channel-major (D, L)) - test build
 * only (the board build works one channel at a time and never holds these) */
#ifdef DIPE_CAPTURE_STAGES
extern float dipe_st_mean[DIPE_D];
extern float dipe_st_std[DIPE_D];
extern float dipe_st_xn[DIPE_D * DIPE_L];
extern float dipe_st_xf[DIPE_D * DIPE_L];
extern float dipe_st_xt[DIPE_D * DIPE_L];
#endif

#endif /* DIPE_INFER_H */
