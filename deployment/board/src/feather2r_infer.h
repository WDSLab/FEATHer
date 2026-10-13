/*
 * FEATHer-mix, RECOMPUTE schedule - the port the frozen model ships with
 * (2026-09-23 re-freeze: mlp_rank 4, build with -DFEA2_MLP_R=4).
 *
 * Geometry, weights and constants are shared with feather2_infer.h; only the
 * schedule differs (see the header comment of feather2r_infer.c).  No (L, D)
 * buffer is held besides the caller's input window, so SRAM grows by one
 * input row per sensor - 43 sensors fit a 20 KB part at R = 4.
 *
 * Typical board use - never materialises the forecast:
 *
 *     fea2r_begin(x);                          // x = caller's (L, D) window
 *     for (int d = 0; d < FEA2_D; ++d) {
 *         fea2r_channel(d);                    // one sensor, recomputed
 *         for (int om = 0; om < FEA2_M; ++om)
 *             sink_write(om, d, fea2r_block(om));   // FEA2_SEG floats
 *     }
 *
 * x must stay valid until the last fea2r_block() call.  The returned block
 * pointer is valid until the next fea2r_block() / fea2r_channel() call.
 */
#ifndef FEATHER2R_INFER_H
#define FEATHER2R_INFER_H

#include "feather2_infer.h"

void fea2r_begin(const float *x);
void fea2r_channel(int d);
const float *fea2r_block(int om);

/* convenience wrapper for tests: streams every block into a caller-owned
 * (FEA2_H * FEA2_D) buffer */
void fea2r_forward(const float *x, float *y);

#endif /* FEATHER2R_INFER_H */
