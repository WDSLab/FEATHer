/*
 * FEATHer in1 float32 inference for Cortex-M3 (2026-09-16 candidate, not the
 * frozen model).
 *
 *   RevIN -> 3 bands (conv k=3 | conv k=5 | AvgPool4+interp)
 *         -> PER-SENSOR banded gate (each sensor's own spectrum)
 *         -> hidden += W_up GELU(W_down x_norm[t] + b)   (D -> R -> D)
 *         -> SPK head -> RevIN^-1
 *
 * = baselines/FEATHer/FEATHer.py with mix_mode="input", mix_pool=1,
 *   mlp_rank=R, gate_mode="banded_sensor" (export_golden_feather2in.py).
 *
 * Sensor coupling.  The gate is per sensor, so sensor d's gate is computed
 * while sensor d is processed.  The ONLY stage that reads every sensor is the
 * R-dim summary u[t] = GELU(W_down x_norm[t] + b) (L*R floats); after it,
 * each sensor is finished and emitted before the next starts.  Because u needs
 * x_norm of all sensors and each sensor's bands need its own x_norm again,
 * x_norm is used in two passes, scheduled by FEA2I_RECOMPUTE_NORM:
 *
 *   0  STRICT  x_norm computed once and kept as an (L, D) buffer (obeys rule 4
 *              of deployment/board/README.md: never recompute to save memory).
 *   1  RENORM  x_norm recomputed per sensor in the second pass from the
 *              caller's raw window (2*L*D extra sub/div per inference); no
 *              (L, D) buffer.  A declared memory<->compute trade.
 *
 * Both give the same forecast.  The caller's time-major window x[t*D + d]
 * must stay valid from fea2i_begin() until the last sensor is finished.
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef FEATHER2IN_INFER_H
#define FEATHER2IN_INFER_H

#include <stddef.h>

#include "fea_kernels.h"

#ifndef FEA2I_L
#define FEA2I_L 96
#endif
#ifndef FEA2I_H
#define FEA2I_H 96
#endif
#ifndef FEA2I_D
#define FEA2I_D 7
#endif
#ifndef FEA2I_PERIOD
#define FEA2I_PERIOD 24
#endif
#ifndef FEA2I_R
#define FEA2I_R 4
#endif
#ifndef FEA2I_RECOMPUTE_NORM
#define FEA2I_RECOMPUTE_NORM 1
#endif

#define FEA2I_NB 3
#define FEA2I_KP 3
#define FEA2I_KM 5

#define FEA2I_NF (FEA2I_L / 2 + 1)
#define FEA2I_LP (FEA2I_L / 4)
#define FEA2I_N  (FEA2I_L / FEA2I_PERIOD)
#define FEA2I_M  (FEA2I_H / FEA2I_PERIOD)
#define FEA2I_AGG_K (2 * (FEA2I_PERIOD / 2) + 1)

/* ---- static arena, in floats ----
 *   s_U   L*R  cross-sensor summary u[t]                (both schedules)
 *   s_X   L*D  x_norm -> hidden -> h_agg                (STRICT only)
 *   s_h   L    current sensor's hidden -> h_agg         (RENORM only)
 *   s_scratch, stages that never coexist:
 *     RENORM  pass 1: row L.  pass 2: row L + spectrum 3*NF (gate), then
 *             row L + pooled LOW LP (LP sits on the dead spectrum), then the
 *             agg copy L, then the output block P on the row.
 *     STRICT  pass 2: spectrum 3*NF (reads s_X directly), then row L + LP,
 *             then the agg copy L, then the block P.
 *   + mean/stdev (2D), current sensor's gate (NB)
 * No term contains H; in RENORM only 2D grows with the sensor count. */
#define FEA2I_MAX2(a, b) ((a) > (b) ? (a) : (b))
#if FEA2I_RECOMPUTE_NORM
#define FEA2I_SCRATCH_N (FEA2I_L + FEA2I_MAX2(3 * FEA2I_NF, FEA2I_LP))
#define FEA2I_ARENA_N (FEA2I_L * FEA2I_R + FEA2I_L + FEA2I_SCRATCH_N \
                       + 2 * FEA2I_D + FEA2I_NB)
#else
#define FEA2I_SCRATCH_N FEA2I_MAX2(3 * FEA2I_NF, FEA2I_L + FEA2I_LP)
#define FEA2I_ARENA_N (FEA2I_L * FEA2I_D + FEA2I_L * FEA2I_R + FEA2I_SCRATCH_N \
                       + 2 * FEA2I_D + FEA2I_NB)
#endif
#define FEA2I_ARENA_BYTES (FEA2I_ARENA_N * (int)sizeof(float))

/* ---- streaming API ----
 *     fea2i_begin(x);                          // stats + summary u
 *     for (int d = 0; d < FEA2I_D; ++d) {
 *         fea2i_channel(d);                    // gate, bands, correction, agg
 *         for (int om = 0; om < FEA2I_M; ++om)
 *             sink_write(om, d, fea2i_block(om));
 *     }
 */
void fea2i_begin(const float *x);
void fea2i_channel(int d);
const float *fea2i_block(int om);
void fea2i_forward(const float *x, float *y);

#ifdef FEA2I_CAPTURE_STAGES
extern float fea2i_st_mean[FEA2I_D];
extern float fea2i_st_stdev[FEA2I_D];
extern float fea2i_st_power[FEA2I_D * FEA2I_NF];  /* per sensor, channel-major */
extern float fea2i_st_feats[FEA2I_D * FEA2I_NB];
extern float fea2i_st_gate[FEA2I_D * FEA2I_NB];
extern float fea2i_st_u[FEA2I_L * FEA2I_R];       /* time-major */
extern float fea2i_st_hidden[FEA2I_D * FEA2I_L];  /* channel-major */
extern float fea2i_st_mlp[FEA2I_D * FEA2I_L];
extern float fea2i_st_hagg[FEA2I_D * FEA2I_L];
#endif

#endif /* FEATHER2IN_INFER_H */
