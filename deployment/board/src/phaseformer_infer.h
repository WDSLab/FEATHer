/*
 * PhaseFormer (ICLR 2026) float32 inference - chip-pool baseline added 2026-10-06/07.
 * Upstream: baselines/PhaseFormer/PhaseFormer-main/models/PhaseFormer.py
 * (commit ed1db61) + layers/SelfAttention_Family.py (AttentionLayer, FullAttention).
 *
 * The widths differ between the authors' settings, so the export script
 * writes them into golden/phaseformer_weights.h: PF_P (period), PF_DL (latent),
 * PF_NL (routing layers), PF_R (routers), PF_NH (heads), PF_PE (phase
 * positional embedding on/off).  Only horizon, input length and width are
 * build-time geometry.
 */
#ifndef PHASEFORMER_INFER_H
#define PHASEFORMER_INFER_H

#include "fea_emit.h"

#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#ifndef PF_L
#define PF_L 96
#endif
#ifndef PF_H
#define PF_H 96
#endif
#ifndef PF_D
#define PF_D 7
#endif

/* x: (PF_L, PF_D) time-major -> y: (PF_H, PF_D) time-major */
void phaseformer_forward(const float *x, float *y);
/* one channel d -> yc[0..H-1] (stream per channel) */
void phaseformer_forward_ch(const float *x, int d, float *yc);
/* one channel d, each value to emit(d, t, v) as it is final (10-13, no H buffer) */
void phaseformer_forward_emit(const float *x, int d, fea_emit_fn emit);

#ifdef PF_CAPTURE_STAGES
#include "phaseformer_weights.h"
extern float pf_st_mu[PF_D];
extern float pf_st_sigma[PF_D];
extern float pf_st_z0[PF_D * PF_P * PF_DL];    /* after the embedding      */
extern float pf_st_zl[PF_D * PF_P * PF_DL];    /* after the last routing   */
#endif

#endif /* PHASEFORMER_INFER_H */
