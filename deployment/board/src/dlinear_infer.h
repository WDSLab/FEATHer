/*
 * DLinear (Zeng et al., AAAI 2023) float32 inference for Cortex-M3.
 *
 * Ported for FAIRNESS, not because it fits everywhere: at H=96 DLinear IS
 * deployable (18,624 params = 74.5 KB of flash, inside the F103RB's 128 KB)
 * and it is MORE accurate than FEATHer there, so leaving it off the board
 * would look like dropping the inconvenient competitor.  From H=192 its
 * weights alone are 149 KB and no build is possible — that exclusion follows
 * from the parameter count and needs no port.
 *
 * Structure (individual=False, the config our sweeps use — shared linear
 * heads across channels; see baselines/DLinear/wrapper.py):
 *
 *   trend    = AvgPool1d(k=25, stride=1) over replicate-padded x
 *   seasonal = x - trend
 *   y[t]     = Linear_Seasonal(seasonal)[t] + Linear_Trend(trend)[t]
 *
 * There is NO instance normalization — DLinear works on raw values.
 *
 * OUTPUT STREAMING, same as feather2_infer.c.  Output step t reads row t of
 * both weight matrices and nothing else, so the horizon loop can be turned
 * inside out here exactly as it can for FEATHer's SPK head.  This is
 * deliberate: the streaming rewrite is a property of per-output-step linear
 * heads (DLinear, SparseTSF, FEATHer all have one), NOT a FEATHer advantage,
 * and the deployability comparison is only honest if every model gets it.
 * What differs between them is how the WEIGHTS grow with the horizon —
 * DLinear's are O(L*H), FEATHer's are O(1) in H.
 *
 * CHANNEL AT A TIME (2026-09-14).  No term couples two channels, so the port
 * decomposes one channel and streams its H values before touching the next —
 * the same order as feather2ci_infer.c.  Values are bit-identical to the
 * earlier time-major build; only the emission order changed.
 *
 * Layout: time-major buf[t*D + d].
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef DLINEAR_INFER_H
#define DLINEAR_INFER_H

/* Board builds: deployment/board/board_prep.py writes board_cfg.h into the
 * CubeIDE project (never into this repo), carrying the geometry of the model
 * under test.  x86 tests and measure_footprint.py do not see it. */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#include <stddef.h>

#ifndef DL_L
#define DL_L 96                         /* seq_len                        */
#endif
#ifndef DL_H
#define DL_H 96                         /* pred_len                       */
#endif
#ifndef DL_D
#define DL_D 7                          /* channels (MetroPT3 cell)       */
#endif

#define DL_K 25                         /* moving-average kernel (fixed)  */
#define DL_PAD ((DL_K - 1) / 2)         /* = 12, replicate on both ends   */

/* ---- static arena, in floats ----
 *   s_trend  L   moving average, one channel
 *   s_seas   L   x - trend, one channel
 * No term contains H (the H*L weight matrices live in FLASH, not SRAM) and
 * none contains D. */
#define DL_ARENA_N (2 * DL_L)
#define DL_ARENA_BYTES (DL_ARENA_N * (int)sizeof(float))

/* Flash cost of the two heads, for the deployability table. */
#define DL_WEIGHT_BYTES ((int)(2 * (DL_H * DL_L + DL_H) * sizeof(float)))

/* ---- streaming API (mirrors fea2ci_begin / fea2ci_block) ----
 *
 *     for (int c = 0; c < DL_D; ++c) {
 *         dl_begin(x, c);                    // decompose one channel
 *         for (int t = 0; t < DL_H; ++t)
 *             sink_write(t, c, dl_step(t));  // one forecast value at a time
 *     }
 */
void dl_begin(const float *x, int c);
float dl_step(int t);

/* convenience wrapper for the golden test: streams into a caller-owned
 * (DL_H * DL_D) buffer.  That buffer belongs to the CALLER. */
void dl_forward(const float *x, float *y);

/* stage capture for golden comparison; only with -DDL_CAPTURE_STAGES */
#ifdef DL_CAPTURE_STAGES
extern float dl_st_trend[DL_L * DL_D];
extern float dl_st_seas[DL_L * DL_D];
#endif

#endif /* DLINEAR_INFER_H */
