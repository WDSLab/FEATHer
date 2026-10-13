/*
 * FITS float32 inference - companion model (estimate-footnote upgraded to
 * an optional measured row). Shared-upsampler benchmark default:
 * RevIN (unbiased var + 1e-5) -> rfft96 -> keep 24 low bins ->
 * complex Linear(24 -> UP) -> zero-pad to TF bins -> irfft(TN) ->
 * x(TN/L) energy compensation -> last H -> denorm.
 * (UP = 48, TF = 97, TN = 192 at H=96.)
 */
#ifndef FITS_INFER_H
#define FITS_INFER_H

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

#ifndef FITS_L
#define FITS_L 96
#endif
#ifndef FITS_H
#define FITS_H 96
#endif
#ifndef FITS_D
#define FITS_D 7
#endif
/* kept low-freq bins and shared / per-channel upsampler: written by the
 * export into golden/fits_geom.h (2026-10-07, candidate settings); the
 * fallback is the benchmark default this port was written at */
#if defined(__has_include)
#if __has_include("fits_geom.h")
#include "fits_geom.h"
#endif
#endif
#ifndef FITS_CUT
#define FITS_CUT 24
#endif
#ifndef FITS_IND
#define FITS_IND 0
#endif
/* Upsampled bin count.  FITS scales the kept low-frequency block by the
 * same ratio the sequence grows: up = cut * (L+H) / L.  At H=96 that is
 * 24*192/96 = 48 (the frozen value this port was written at); at H=720 it
 * is 24*816/96 = 204.  Hardcoding 48 silently produced a wrong forecast
 * at every other horizon (caught 2026-09-04 by the H=720 golden run). */
#define FITS_UP ((FITS_CUT * FITS_TN) / FITS_L)
#define FITS_TN (FITS_L + FITS_H)          /* irfft length = 192 at H=96 */
#define FITS_TF (FITS_TN / 2 + 1)          /* padded spectrum bins (the
                                            * padding is implicit, never
                                            * allocated) */

/* x: (FITS_L, FITS_D) time-major -> y: (FITS_H, FITS_D) time-major */
void fits_forward(const float *x, float *y);
/* one channel d -> yc[0..H-1] (stream per channel) */
void fits_forward_ch(const float *x, int d, float *yc);
/* one channel d, each value to emit(d, t, v) as it is final (10-13, no H buffer) */
void fits_forward_emit(const float *x, int d, fea_emit_fn emit);

/* stage buffers for golden comparison - test build only (the board build
 * works one channel at a time and never holds these) */
#ifdef FITS_CAPTURE_STAGES
extern float fits_st_mean[FITS_D];
extern float fits_st_sd[FITS_D];
extern float fits_st_xn[FITS_L * FITS_D];          /* time-major   */
extern float fits_st_up_re[FITS_D * FITS_UP];      /* channel-major */
extern float fits_st_up_im[FITS_D * FITS_UP];
#endif

#endif /* FITS_INFER_H */
