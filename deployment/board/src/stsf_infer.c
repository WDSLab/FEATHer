/*
 * SparseTSF (linear) forward, mirroring the upstream
 * baselines/SparseTSF/SparseTSF-main/models/SparseTSF.py exactly:
 *   mean-subtract -> shared 1x25 'same' conv + residual ->
 *   phase-wise shared linear n->m -> add mean back.
 */
#include "stsf_infer.h"
#include "stsf_weights.h"      /* generated: golden/stsf_weights.h */

float stsf_st_mean[STSF_D];
float stsf_st_xagg[STSF_D * STSF_L];

static float s_xc[STSF_D * STSF_L];    /* centered, channel-major */
static float s_cv[STSF_L];             /* conv scratch, one channel */

void stsf_forward(const float *x, float *y)
{
    /* per-channel mean only (SparseTSF has no variance norm) */
    fea_instnorm_stats(x, STSF_L, STSF_D, stsf_st_mean, 0);

    for (int d = 0; d < STSF_D; ++d)
        for (int t = 0; t < STSF_L; ++t)
            s_xc[d * STSF_L + t] = x[t * STSF_D + d] - stsf_st_mean[d];

    /* one shared conv kernel applied to every channel + residual */
    for (int d = 0; d < STSF_D; ++d) {
        fea_dwconv_same(s_xc + d * STSF_L, s_cv, STSF_W_CONV, 0,
                        STSF_L, 1, STSF_K);
        for (int t = 0; t < STSF_L; ++t)
            stsf_st_xagg[d * STSF_L + t] = s_xc[d * STSF_L + t] + s_cv[t];
    }

    /* phase-wise shared linear: v[n] -> out[m] per (channel, phase) */
    for (int d = 0; d < STSF_D; ++d) {
        for (int ph = 0; ph < STSF_P; ++ph) {
            float v[STSF_N];
            for (int in = 0; in < STSF_N; ++in)
                v[in] = stsf_st_xagg[d * STSF_L + in * STSF_P + ph];
            for (int om = 0; om < STSF_M; ++om) {
                float acc = 0.0f;
                for (int in = 0; in < STSF_N; ++in)
                    acc += STSF_W_LIN[om * STSF_N + in] * v[in];
                y[(om * STSF_P + ph) * STSF_D + d] = acc + stsf_st_mean[d];
            }
        }
    }
}
