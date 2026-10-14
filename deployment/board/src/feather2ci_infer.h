/*
 * FEATHer-CI: the channel-INDEPENDENT variant (channel_mode="ci_full").
 *
 * In PyTorch this is not a different network - FEATHer.forward folds the
 * channel axis into the batch (x -> (B*D, L, 1)) so every layer is built at
 * width 1 and sees one univariate series.  The C port therefore reuses
 * feather2_infer.c compiled at FEA2_D=1 and calls it once per channel; there
 * is no second implementation to keep in sync.
 *
 * Why it exists: the mixing variant holds an (L x D) hidden state because the
 * residual MLP mixes channels at every timestep, so its SRAM grows with D and
 * it stops fitting a 20 KB part beyond some channel count (measured in
 * results/board_footprint.csv).  This variant's working set is D-independent
 * apart from the caller's input buffer, so it covers the high-channel cells
 * the mixing variant cannot reach - at a measured accuracy cost (Steel +16%,
 * WindSCADA +11% on mfg-6).
 *
 * The weights are SHARED across channels (one width-1 network), exactly as in
 * training - do not export per-channel copies.
 *
 * Build: compile feather2_infer.c with -DFEA2_D=1 alongside this file, and
 * give FEA2CI_D the real channel count.
 */
#ifndef FEATHER2CI_INFER_H
#define FEATHER2CI_INFER_H

#include "feather2_infer.h"

/* BOARD_SKIP_CI: set by board_prep.py when another model is under test - the
 * CubeIDE project compiles every port, but this one only exists at width 1. */
#if FEA2_D != 1 && !defined(BOARD_SKIP_CI)
#error "FEATHer-CI requires the width-1 network: build feather2_infer.c with -DFEA2_D=1"
#endif

#ifndef FEA2CI_D
#define FEA2CI_D 7                      /* real channel count */
#endif

/* Working set = the width-1 arena and nothing else: each channel is gathered
 * into that arena's own input buffer and blocks are returned without a copy.
 * The only D-dependent term is the caller's input buffer. */
#define FEA2CI_ARENA_N FEA2_ARENA_N
#define FEA2CI_ARENA_BYTES (FEA2CI_ARENA_N * (int)sizeof(float))

/* Streaming, per channel:
 *
 *     for (int c = 0; c < FEA2CI_D; ++c) {
 *         fea2ci_begin(x, c);                       // x is (L, FEA2CI_D)
 *         for (int om = 0; om < FEA2_M; ++om)
 *             sink_write(fea2ci_block(om), FEA2_SEG);
 *     }
 *
 * Channels are independent, so they may also be processed as they arrive.
 */
void fea2ci_begin(const float *x, int c);
const float *fea2ci_block(int om);

/* convenience: full (FEA2_H * FEA2CI_D) forecast into a caller-owned buffer */
void fea2ci_forward(const float *x, float *y);

#endif /* FEATHER2CI_INFER_H */
