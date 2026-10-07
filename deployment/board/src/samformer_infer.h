/*
 * SAMformer (ICML 2024) float32 inference - reference model added 2026-10-06/07.
 * Upstream: baselines/SAMformer/SAMformer-main/samformer_pytorch/samformer/samformer.py
 * (SAMFormerArchitecture, commit 71f10ea) + utils/revin.py.
 *
 * The one lightweight baseline that MIXES channels: every channel is an
 * attention token (features = its L normalised samples).  Every author setting
 * we run has the same structure (hid_dim 16; the candidates differ only in
 * lr / SAM rho), written by the export as SAM_HID.
 */
#ifndef SAMFORMER_INFER_H
#define SAMFORMER_INFER_H

#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#ifndef SAM_L
#define SAM_L 96
#endif
#ifndef SAM_H
#define SAM_H 96
#endif
#ifndef SAM_D
#define SAM_D 7
#endif

/* x: (SAM_L, SAM_D) time-major -> y: (SAM_H, SAM_D) time-major */
void samformer_forward(const float *x, float *y);

#ifdef SAM_CAPTURE_STAGES
#include "samformer_weights.h"
extern float sam_st_mean[SAM_D];
extern float sam_st_std[SAM_D];
extern float sam_st_attw[SAM_D * SAM_D];    /* softmax rows, query-major */
extern float sam_st_out[SAM_D * SAM_L];     /* x_norm + attention, channel-major */
#endif

#endif /* SAMFORMER_INFER_H */
