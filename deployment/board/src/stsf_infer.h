/*
 * SparseTSF (linear, 41 params) float32 inference - companion model for
 * the deployment table (EDGE_EXPERIMENT.md decision #3).
 * Config = shipping MetroPT3 cell: period_len=24 -> n=m=4, conv k=25.
 */
#ifndef STSF_INFER_H
#define STSF_INFER_H

#include "fea_kernels.h"

#define STSF_L 96
#define STSF_H 96
#define STSF_D 7
#define STSF_P 24                       /* period_len                */
#define STSF_N (STSF_L / STSF_P)        /* input segments  = 4       */
#define STSF_M (STSF_H / STSF_P)        /* output segments = 4       */
#define STSF_K (1 + 2 * (STSF_P / 2))   /* agg conv kernel = 25      */

/* x: (STSF_L, STSF_D) time-major -> y: (STSF_H, STSF_D) time-major */
void stsf_forward(const float *x, float *y);

/* stage buffers for golden comparison */
extern float stsf_st_mean[STSF_D];
extern float stsf_st_xagg[STSF_D * STSF_L];   /* channel-major, conv+residual */

#endif /* STSF_INFER_H */
