/*
 * CycleNet inference - streaming, one channel at a time.  See cyclenet_infer.h.
 *
 * Mirrors baselines/CycleNet/CycleNet-main/models/CycleNet.py with
 * model_type = linear and use_revin = True, which is what our sweeps run.
 */
#include <math.h>

#include "cyclenet_infer.h"
#include "cyclenet_weights.h"   /* generated: golden/cyclenet_weights.h */

/* The golden headers are regenerated per horizon and width.  A port built
 * against weights exported for another H or D compiles fine and returns a
 * plausible, wrong forecast, so refuse it at compile time.  (The cycle is a
 * C x D table, so unlike DLinear the WEIGHTS depend on D as well.) */
_Static_assert(CN_H == CN_H_GEN,
               "CN_H does not match the horizon the weights were exported at - re-run export_golden_cyclenet.py --pred_len");
_Static_assert(CN_D == CN_D_W_GEN,
               "CN_D does not match the width the weights were exported at - re-run export_golden_cyclenet.py --d_model");
_Static_assert(CN_L == CN_L_GEN,
               "CN_L does not match the input length the weights were exported at");

#ifdef CN_CAPTURE_STAGES
float cn_st_xs[CN_L * CN_D];
#endif

/* ---- the arena (horizon- AND width-independent) ---- */
static float s_xs[CN_L];
static float s_mean, s_std;
static int s_c, s_phase;

void cn_begin(const float *x, int c, int phase)
{
    /* instance norm over the window of channel c.  torch.var default is the
     * UNBIASED estimate (divide by L - 1); upstream then adds 1e-5 before the
     * square root. */
    float sum = 0.0f;
    for (int i = 0; i < CN_L; ++i)
        sum += x[i * CN_D + c];
    const float mean = sum / (float)CN_L;

    float ss = 0.0f;
    for (int i = 0; i < CN_L; ++i) {
        const float d = x[i * CN_D + c] - mean;
        ss += d * d;
    }
    const float std = sqrtf(ss / (float)(CN_L - 1) + 1e-5f);

    s_mean = mean;
    s_std = std;
    s_c = c;
    s_phase = phase;

    for (int i = 0; i < CN_L; ++i) {
        const int q = (phase + i) % CN_CYCLE;
        const float v = (x[i * CN_D + c] - mean) / std - CN_Q[q * CN_D + c];
        s_xs[i] = v;
#ifdef CN_CAPTURE_STAGES
        cn_st_xs[i * CN_D + c] = v;
#endif
    }
}

/*
 * One output value: step t of the channel last passed to cn_begin().
 * Reads row t of W, one bias and one cycle entry - nothing else.
 */
float cn_step(int t)
{
    const float *w = CN_W + (size_t)t * CN_L;
    float acc = CN_B[t];
    for (int i = 0; i < CN_L; ++i)
        acc += w[i] * s_xs[i];
    acc += CN_Q[((s_phase + CN_L + t) % CN_CYCLE) * CN_D + s_c];
    return acc * s_std + s_mean;
}

/* convenience: stream every value into a caller-owned full-length buffer */
void cn_forward(const float *x, float *y)
{
    for (int c = 0; c < CN_D; ++c) {
        cn_begin(x, c, 0);
        for (int t = 0; t < CN_H; ++t)
            y[(size_t)t * CN_D + c] = cn_step(t);
    }
}
