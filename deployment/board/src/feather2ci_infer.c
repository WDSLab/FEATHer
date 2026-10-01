/*
 * FEATHer-CI - channel loop around the width-1 network.  See the header.
 *
 * This file contains no model maths: it gathers one channel into the width-1
 * network's own input buffer, hands it to the forward in feather2_infer.c,
 * and scatters the emitted blocks back into the interleaved output.  Keeping
 * the arithmetic in one place is deliberate - the two variants differ only in
 * whether channels are mixed, so a second copy of the forward would be a
 * standing risk of the two drifting apart.
 */
#include "feather2ci_infer.h"

#ifndef BOARD_SKIP_CI   /* see feather2ci_infer.h */

void fea2ci_begin(const float *x, int c)
{
    /* Stage the channel straight into the width-1 arena.  fea2_begin() takes
     * its statistics before it writes and then normalizes in place, so no
     * gathered column of our own is needed (2026-09-14: that copy, and a copy
     * of every emitted block, were the last buffers this wrapper held). */
    float *col = fea2_input();
    for (int t = 0; t < FEA2_L; ++t)
        col[t] = x[t * FEA2CI_D + c];
    fea2_begin(col);
}

const float *fea2ci_block(int om)
{
    return fea2_block(om, 0);              /* FEA2_PERIOD * 1, no copy */
}

void fea2ci_forward(const float *x, float *y)
{
    for (int c = 0; c < FEA2CI_D; ++c) {
        fea2ci_begin(x, c);
        for (int om = 0; om < FEA2_M; ++om) {
            const float *blk = fea2ci_block(om);
            for (int ph = 0; ph < FEA2_PERIOD; ++ph)
                y[(om * FEA2_PERIOD + ph) * FEA2CI_D + c] = blk[ph];
        }
    }
}

#endif /* BOARD_SKIP_CI */
