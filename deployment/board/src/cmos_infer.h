/*
 * CMoS (ICML 2025) float32 inference - chip-pool baseline added 2026-10-06/07.
 * Upstream: baselines/CMoS/CMoS-main/model/CMoS/Model.py (commit b696a0c).
 *
 * The structural values differ between the authors' per-dataset settings
 * (seg_size 2..24, num_map 2..8, kernel 4..24, stride 2..4), so they are NOT
 * defined here: the export script writes them into golden/cmos_weights.h
 * (CMOS_SEG / CMOS_NMAP / CMOS_K / CMOS_STRIDE) together with the weights.
 * Only the horizon and the width are build-time geometry, as for every port.
 */
#ifndef CMOS_INFER_H
#define CMOS_INFER_H

#include "fea_emit.h"

/* Board builds: board_prep.py writes board_cfg.h into the CubeIDE project
 * (never into this repo).  x86 tests and measure_footprint.py do not see it. */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#ifndef CMOS_L
#define CMOS_L 96
#endif
#ifndef CMOS_H
#define CMOS_H 96
#endif
#ifndef CMOS_D
#define CMOS_D 7
#endif

/* x: (CMOS_L, CMOS_D) time-major -> y: (CMOS_H, CMOS_D) time-major */
void cmos_forward(const float *x, float *y);
/* one channel d -> yc[0..H-1] (stream per channel) */
void cmos_forward_ch(const float *x, int d, float *yc);
/* one channel d, each value to emit(d, t, v) as it is final (10-13, no H buffer) */
void cmos_forward_emit(const float *x, int d, fea_emit_fn emit);

/* stage buffers for golden comparison - test build only (the board build
 * works one channel at a time and never holds these) */
#ifdef CMOS_CAPTURE_STAGES
#include "cmos_weights.h"
extern float cmos_st_mean[CMOS_D];
extern float cmos_st_std[CMOS_D];
extern float cmos_st_conv[CMOS_D * CMOS_CONV_DIM];   /* channel-major */
extern float cmos_st_gate[CMOS_D * CMOS_NMAP];       /* softmax output */
#endif

#endif /* CMOS_INFER_H */
