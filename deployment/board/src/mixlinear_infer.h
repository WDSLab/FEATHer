/*
 * MixLinear (ICLR 2026, 49 params, D-independent) float32 inference -
 * FEATHer's closest sub-1K rival for the deployment table.
 * Config = SparseTSF-symmetric period_len=24 (seq=96, pred=96):
 *   seg_x = seg_y = 4, sqrt = 2, lpf = 4, conv k = 25, alpha = 0.5.
 *
 * Forward = mean-sub -> shared conv1d(1->1) residual -> segment (P x seg_x)
 *   -> TIME branch : factored Linear(2,2) x2 over the sqrt-split seg axis
 *   -> FREQ branch : len-4 FFT -> complex Linear(4->2->4) -> len-4 iFFT(real)
 *   -> mix (alpha) + seq_mean.
 * All matmuls tiny; verified bit-exact against the staged torch reference
 * (deployment/board/_mixlinear_scalar.py, |diff| ~ 1e-7).
 */
#ifndef MIXLINEAR_INFER_H
#define MIXLINEAR_INFER_H

#include "fea_kernels.h"

#define ML_L 96
#define ML_H 96
#define ML_D 7
#define ML_P 24                       /* period_len              */
#define ML_SEGX (ML_L / ML_P)         /* input segments  = 4     */
#define ML_SEGY (ML_H / ML_P)         /* output segments = 4     */
#define ML_SQ 2                       /* sqrt(seg) split         */
#define ML_LPF 4                      /* kept FFT bins (= seg_x) */
#define ML_K (ML_P + 1)               /* conv kernel = 25        */

/* x: (ML_L, ML_D) time-major -> y: (ML_H, ML_D) time-major */
void mixlinear_forward(const float *x, float *y);

/* stage buffers for golden comparison */
extern float mixlinear_st_mean[ML_D];
extern float mixlinear_st_xt[ML_H * ML_D];   /* time branch, time-major */
extern float mixlinear_st_xf[ML_H * ML_D];   /* freq branch, time-major */

#endif /* MIXLINEAR_INFER_H */
