/*
 * FEATHer float32 inference for Cortex-M3 (soft-float) - week-1 x86 build.
 *
 * Data layout convention (matches PyTorch (L, D) row-major, batch = 1):
 *   time-major buffers  buf[t * FEA_D + d]
 *   channel-major (SPK) buf[d * FEA_L + t]
 *
 * All math is plain float32; no libm beyond expf/sqrtf. Compile both x86
 * and ARM with -O2 -ffp-contract=off -fno-fast-math (EDGE_EXPERIMENT.md §7).
 */
#ifndef FEATHER_INFER_H
#define FEATHER_INFER_H

#include "fea_kernels.h"

#define FEA_L 96            /* seq_len                     */
#define FEA_H 96            /* pred_len                    */
#define FEA_D 7             /* input channels (MetroPT3)   */
#define FEA_S 8             /* DTK latent d_state          */
#define FEA_K 7             /* DTK kernel_size             */
#define FEA_PERIOD 12       /* SPK period                  */
#define FEA_NB 3            /* num_bands (POINT/MID/LOW)   */
#define FEA_NF (FEA_L / 2 + 1)      /* rfft bins = 49      */
#define FEA_LP (FEA_L / 4)          /* pooled LOW len = 24 */
#define FEA_N (FEA_L / FEA_PERIOD)  /* input periods = 8   */
#define FEA_M (FEA_H / FEA_PERIOD)  /* output periods = 8  */
#define FEA_AGG_K (2 * (FEA_PERIOD / 2) + 1)  /* = 13      */

/* static arena footprint (B + C + D + LAT + stats + gate), bytes */
#define FEA_ARENA_BYTES \
    ((3 * FEA_L * FEA_D + FEA_L * FEA_S + 2 * FEA_D + 2 * FEA_NB) \
     * (int)sizeof(float))

/* primitives live in fea_kernels.h (shared with SparseTSF & co.) */

/* ---- full forward ----
 * Board entry point: x (FEA_L * FEA_D time-major) lives in the caller's
 * arena slot, is normalized IN PLACE and consumed; returns a pointer to
 * the forecast (FEA_H * FEA_D) in an internal buffer, valid until the
 * next call. Static arena = 2,804 floats (10.96 KB); +input = 13.6 KB.
 */
float *feather_forward_ip(float *x);

/* const-input convenience wrapper (adds one 672-float copy buffer) */
void feather_forward(const float *x, float *y);

/* stage buffers for golden-vector comparison - ONLY defined when the
 * translation unit is built with -DFEA_CAPTURE_STAGES (test builds);
 * the board build compiles the exact same code path without copies.  */
extern float fea_st_xnorm[FEA_L * FEA_D];
extern float fea_st_mean[FEA_D];
extern float fea_st_stdev[FEA_D];
extern float fea_st_xproj[FEA_L * FEA_D];
extern float fea_st_fP[FEA_L * FEA_D];
extern float fea_st_fM[FEA_L * FEA_D];
extern float fea_st_fLds[FEA_LP * FEA_D];
extern float fea_st_fL[FEA_L * FEA_D];
extern float fea_st_oP[FEA_L * FEA_D];
extern float fea_st_oM[FEA_L * FEA_D];
extern float fea_st_oL[FEA_L * FEA_D];
extern float fea_st_rfft_re[FEA_NF];   /* channel 0, for trap#4 test */
extern float fea_st_rfft_im[FEA_NF];
extern float fea_st_magmean[FEA_NF];
extern float fea_st_gate_z[FEA_NB];
extern float fea_st_gate[FEA_NB];
extern float fea_st_hidden[FEA_L * FEA_D];
extern float fea_st_hagg[FEA_D * FEA_L];   /* channel-major */
extern float fea_st_yspk[FEA_H * FEA_D];

#endif /* FEATHER_INFER_H */
