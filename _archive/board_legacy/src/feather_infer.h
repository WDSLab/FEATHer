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

/* Shipping FEATHer-N as frozen in baselines/__init__.py:_FEATHER_MFG_N —
 * d_state 16, kernel 7, num_bands 4, residual MLP rank 16, channel_mode mix —
 * on the MetroPT3 deployment cell (D=7, period 24, H=96) = 938 parameters.
 *
 * The 07-27 port predates two model changes and measured a different network:
 * the residual MLP landed 07-31 and num_bands=4 / d_state=16 came out of the
 * 08-11 grid search, so the board was running a 453-parameter 3-band model
 * with no nonlinearity while the paper reported this one.
 */
#define FEA_L 96            /* seq_len                          */
#define FEA_H 96            /* pred_len                         */
#define FEA_D 7             /* input channels (MetroPT3)        */
#define FEA_S 16            /* DTK latent d_state               */
#define FEA_K 7             /* DTK kernel_size                  */
#define FEA_PERIOD 24       /* SPK period (MetroPT3 FFT-detected) */
#define FEA_NB 4            /* num_bands: 4 = POINT/HIGH/MID/LOW, 3 drops HIGH */
#define FEA_MLP_R 16        /* residual MLP bottleneck rank     */
#define FEA_NF (FEA_L / 2 + 1)      /* rfft bins = 49      */
#define FEA_LP (FEA_L / 4)          /* pooled LOW len = 24 */
#define FEA_N (FEA_L / FEA_PERIOD)  /* input periods = 4   */
#define FEA_M (FEA_H / FEA_PERIOD)  /* output periods = 4  */
#define FEA_AGG_K (2 * (FEA_PERIOD / 2) + 1)  /* = 25      */

/* Static arena (B + C + D + LAT + stats + gate), bytes. LAT is sized by the
 * largest transient that parks there: the DTK latent (L*S), the residual-MLP
 * bottleneck (L*R), the SPK aggregation output (D*L) and the gate spectra
 * (3*NF) never overlap, so one buffer at the max serves all four. */
#define FEA_LAT_N ((FEA_L * FEA_S) > (FEA_L * FEA_MLP_R) \
                   ? (FEA_L * FEA_S) : (FEA_L * FEA_MLP_R))
#define FEA_ARENA_BYTES \
    ((3 * FEA_L * FEA_D + FEA_LAT_N + 2 * FEA_D + 2 * FEA_NB) \
     * (int)sizeof(float))

/* primitives live in fea_kernels.h (shared with SparseTSF & co.) */

/* ---- full forward ----
 * Board entry point: x (FEA_L * FEA_D time-major) lives in the caller's
 * arena slot, is normalized IN PLACE and consumed; returns a pointer to
 * the forecast (FEA_H * FEA_D) in an internal buffer, valid until the
 * next call.
 *
 * Static arena at d_state 16 / num_bands 4 = 3,574 floats (14.0 KB), of
 * which s_lat is 6 KB. Against the F103RB's 20 KB that leaves room for
 * main.c's output buffer and the stack, but not for a separate input copy —
 * see feather_forward.
 */
float *feather_forward_ip(float *x);

/* const-input convenience wrapper. Stages through the arena rather than a
 * private buffer, which is what keeps the board build inside 20 KB. */
void feather_forward(const float *x, float *y);

/* stage buffers for golden-vector comparison - ONLY defined when the
 * translation unit is built with -DFEA_CAPTURE_STAGES (test builds);
 * the board build compiles the exact same code path without copies.  */
extern float fea_st_xnorm[FEA_L * FEA_D];
extern float fea_st_mean[FEA_D];
extern float fea_st_stdev[FEA_D];
extern float fea_st_xproj[FEA_L * FEA_D];
extern float fea_st_fP[FEA_L * FEA_D];
#if FEA_NB == 4
extern float fea_st_fH[FEA_L * FEA_D];   /* HIGH band exists only at 4 bands */
extern float fea_st_oH[FEA_L * FEA_D];
#endif
extern float fea_st_fM[FEA_L * FEA_D];
extern float fea_st_fLds[FEA_LP * FEA_D];
extern float fea_st_fL[FEA_L * FEA_D];
extern float fea_st_oP[FEA_L * FEA_D];
extern float fea_st_oM[FEA_L * FEA_D];
extern float fea_st_oL[FEA_L * FEA_D];
extern float fea_st_mlp[FEA_L * FEA_D];   /* hidden after residual MLP */
extern float fea_st_rfft_re[FEA_NF];   /* channel 0, for trap#4 test */
extern float fea_st_rfft_im[FEA_NF];
extern float fea_st_magmean[FEA_NF];
extern float fea_st_gate_z[FEA_NB];
extern float fea_st_gate[FEA_NB];
extern float fea_st_hidden[FEA_L * FEA_D];
extern float fea_st_hagg[FEA_D * FEA_L];   /* channel-major */
extern float fea_st_yspk[FEA_H * FEA_D];

#endif /* FEATHER_INFER_H */
