/*
 * DiPE-Linear float32 inference - companion model (optional tier).
 * Single-expert paper-default config: RevIN (unbiased std, clamp 1e-7,
 * no eps) -> static freq weight (rfft96 . w . irfft96) -> static time
 * weight -> FFT big-conv (193-point DFT, complex weight + bias) -> denorm.
 */
#ifndef DIPE_INFER_H
#define DIPE_INFER_H

#include "fea_kernels.h"

#define DIPE_L 96
#define DIPE_H 96
#define DIPE_D 7
#define DIPE_PAD 1                              /* floor((L+H-1)/100)  */
#define DIPE_TN (DIPE_L + DIPE_H - 1 + 2 * DIPE_PAD)   /* = 193       */
#define DIPE_TF (DIPE_TN / 2 + 1)                       /* = 97        */

/* x: (DIPE_L, DIPE_D) time-major -> y: (DIPE_H, DIPE_D) time-major */
void dipe_forward(const float *x, float *y);

/* stage buffers for golden comparison (channel-major (D, L)) */
extern float dipe_st_mean[DIPE_D];
extern float dipe_st_std[DIPE_D];
extern float dipe_st_xn[DIPE_D * DIPE_L];
extern float dipe_st_xf[DIPE_D * DIPE_L];
extern float dipe_st_xt[DIPE_D * DIPE_L];

#endif /* DIPE_INFER_H */
