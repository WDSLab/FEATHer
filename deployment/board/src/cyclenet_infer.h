/*
 * CycleNet (Lin et al., NeurIPS 2024) float32 inference for Cortex-M3,
 * linear backbone - the variant our sweeps run (baselines/CycleNet/wrapper.py).
 *
 * Ported for FAIRNESS (2026-09-30), for the same reason DLinear was: its
 * weights fit the F103RB at H = 96 and 192, so leaving it without a port kept
 * it out of the deployable comparison for a reason that was ours, not the
 * model's.  From H = 336 its weights alone exceed the flash and no build is
 * possible; export_golden_cyclenet.py refuses those horizons.
 *
 * Structure (upstream models/CycleNet.py, model_type = linear, use_revin = 1):
 *
 *   mean, var = over the L inputs of one channel   (var unbiased, + 1e-5)
 *   xs[i]     = (x[i] - mean) / sqrt(var) - Q[(phase + i) % C][c]
 *   y[t]      = b[t] + sum_i W[t][i] * xs[i]  +  Q[(phase + L + t) % C][c]
 *   out[t]    = y[t] * sqrt(var) + mean
 *
 * Q is the learned recurrent cycle, one column per channel (C x D).  `phase`
 * is the position of the window inside the cycle.  Our training and test
 * loaders do not carry it, so the sweeps ran with phase = 0 for every window
 * (a documented limitation, same treatment as TQNet); pass 0 to reproduce the
 * reported accuracy.
 *
 * Same rules as every other port (deployment/board/README.md):
 *   CHANNEL AT A TIME - nothing couples two channels except that each has its
 *   own column of Q, so one channel is normalised, de-cycled and streamed
 *   before the next is touched.
 *   OUTPUT STREAMING - step t reads row t of W and one entry of Q.
 *   No model-specific rewrite (the H x L matrix stays as trained, fp32).
 *
 * Layout: time-major buf[t*D + d].
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef CYCLENET_INFER_H
#define CYCLENET_INFER_H

/* board builds: geometry from board_cfg.h (written by board_prep.py) */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#include <stddef.h>

#ifndef CN_L
#define CN_L 96                         /* seq_len                        */
#endif
#ifndef CN_H
#define CN_H 96                         /* pred_len                       */
#endif
#ifndef CN_D
#define CN_D 7                          /* channels (MetroPT3 cell)       */
#endif

/* ---- static arena, in floats ----
 *   s_xs   L   normalised, de-cycled input of ONE channel
 * plus two scalars (mean, std).  Nothing contains H or D; the H x L matrix and
 * the C x D cycle live in FLASH. */
#define CN_ARENA_N (CN_L + 2)
#define CN_ARENA_BYTES (CN_ARENA_N * (int)sizeof(float))

/* ---- streaming API (mirrors dl_begin / dl_step) ----
 *
 *     for (int c = 0; c < CN_D; ++c) {
 *         cn_begin(x, c, 0);                 // one channel, phase 0
 *         for (int t = 0; t < CN_H; ++t)
 *             sink_write(t, c, cn_step(t));
 *     }
 */
void cn_begin(const float *x, int c, int phase);
float cn_step(int t);

/* convenience wrapper for the golden test: streams into a caller-owned
 * (CN_H * CN_D) buffer, phase 0.  That buffer belongs to the CALLER. */
void cn_forward(const float *x, float *y);

/* stage capture for golden comparison; only with -DCN_CAPTURE_STAGES */
#ifdef CN_CAPTURE_STAGES
extern float cn_st_xs[CN_L * CN_D];
#endif

#endif /* CYCLENET_INFER_H */
