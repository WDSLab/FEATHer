/*
 * SparseTSF-MLP float32 inference — the same-size rival.
 *
 * SparseTSF ships two heads. The linear one (stsf_infer.c, 41 params) is the
 * paper's headline model; this is the `mlp` variant, which replaces the
 * phase-wise Linear(n -> m) with Linear(n -> 128) -> ReLU -> Linear(128 -> m).
 * At the MetroPT3 deployment cell that is 1,181 parameters against FEATHer's
 * 938 (and 4,535 vs 1,042 at H=720), so it is the only baseline in the same
 * budget as FEATHer rather than one to two orders below it — which makes it
 * the one board comparison that answers "is FEATHer faster because it is
 * better, or only because it is smaller?".
 *
 * Everything upstream of the head is identical to the linear variant
 * (mean-subtract, shared k=25 'same' conv, residual, phase folding), so this
 * file reuses the same weights layout and only diverges at the projection.
 */
#ifndef STSF_MLP_INFER_H
#define STSF_MLP_INFER_H

#include "fea_kernels.h"

#define SMLP_L 96
#define SMLP_H 96
#define SMLP_D 7
#define SMLP_P 24                       /* period_len                 */
#define SMLP_N (SMLP_L / SMLP_P)        /* input segments  = 4        */
#define SMLP_M (SMLP_H / SMLP_P)        /* output segments = 4        */
#define SMLP_K (1 + 2 * (SMLP_P / 2))   /* agg conv kernel = 25       */
#define SMLP_DM 128                     /* d_model, upstream default  */

/* x: (SMLP_L, SMLP_D) time-major -> y: (SMLP_H, SMLP_D) time-major */
void stsf_mlp_forward(const float *x, float *y);

/* stage buffers for golden comparison */
extern float smlp_st_mean[SMLP_D];
extern float smlp_st_xagg[SMLP_D * SMLP_L];   /* channel-major, conv+residual */
extern float smlp_st_hid[SMLP_D * SMLP_P * SMLP_DM];  /* post-ReLU hidden */

#endif /* STSF_MLP_INFER_H */
