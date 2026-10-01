/*
 * DLinear inference - streaming arena build.  See dlinear_infer.h.
 *
 * Mirrors baselines/DLinear/DLinear-main/models/DLinear.py with
 * individual=False (shared heads), which is what our sweeps run.
 */
#include "dlinear_infer.h"
#include "dlinear_weights.h"   /* generated: golden/dlinear_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(DL_H == DL_H_GEN,
               "DL_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");

#ifdef DL_CAPTURE_STAGES
float dl_st_trend[DL_L * DL_D];
float dl_st_seas[DL_L * DL_D];
#endif

/* ---- the arena (horizon- AND width-independent) ----
 *
 * PER-CHANNEL WORKING SET (2026-09-14) - equal-effort rule, see
 * deployment/board/README.md.  individual=False means every channel goes
 * through the same two heads with no cross-channel term, so the decomposition
 * only ever needs one channel.  The previous build decomposed all D channels
 * up front (2*D*L floats, 21.5 KB at D=28) to stream time-major steps; that
 * put the port over the 20 KB SRAM budget on BlastFurnace - an artefact of the
 * port, not of the model.  Same per-channel treatment as the FITS / DiPE /
 * MixLinear / SparseTSF ports; full-width copies are test-build only. */
static float s_trend[DL_L];
static float s_seas[DL_L];

/*
 * Series decomposition of ONE channel.
 *
 * Upstream pads by REPLICATION — front = x[0] repeated (k-1)/2 times, end =
 * x[L-1] repeated (k-1)/2 times — then runs AvgPool1d(k, stride=1) over the
 * padded sequence.  That is exactly a centered moving average with clamped
 * indices, which is how it is written here (no padded copy needed):
 *
 *     trend[t] = (1/k) * sum_{j=0}^{k-1} x[clamp(t - pad + j, 0, L-1)]
 *
 * NOT zero padding — using zeros here would pull the trend toward 0 at both
 * ends and silently change the forecast.
 */
void dl_begin(const float *x, int c)
{
    const float inv_k = 1.0f / (float)DL_K;
    for (int t = 0; t < DL_L; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < DL_K; ++j) {
            int s = t - DL_PAD + j;
            if (s < 0) s = 0;
            if (s > DL_L - 1) s = DL_L - 1;
            acc += x[s * DL_D + c];
        }
        float tr = acc * inv_k;
        s_trend[t] = tr;
        s_seas[t] = x[t * DL_D + c] - tr;
#ifdef DL_CAPTURE_STAGES
        dl_st_trend[t * DL_D + c] = tr;
        dl_st_seas[t * DL_D + c] = s_seas[t];
#endif
    }
}

/*
 * One output value: step t of the channel last passed to dl_begin().
 *
 *   y[t] = sum_i Ws[t][i] * seasonal[i] + bs[t]
 *        + sum_i Wt[t][i] * trend[i]    + bt[t]
 *
 * Row t of each weight matrix is touched and nothing else, so the horizon
 * loop is free to run outside and nothing but the returned scalar is live.
 */
float dl_step(int t)
{
    const float *ws = W_SEASONAL + (size_t)t * DL_L;
    const float *wt = W_TREND + (size_t)t * DL_L;
    float acc = B_SEASONAL[t] + B_TREND[t];

    for (int i = 0; i < DL_L; ++i)
        acc += ws[i] * s_seas[i] + wt[i] * s_trend[i];
    return acc;
}

/* convenience: stream every value into a caller-owned full-length buffer */
void dl_forward(const float *x, float *y)
{
    for (int c = 0; c < DL_D; ++c) {
        dl_begin(x, c);
        for (int t = 0; t < DL_H; ++t)
            y[(size_t)t * DL_D + c] = dl_step(t);
    }
}
