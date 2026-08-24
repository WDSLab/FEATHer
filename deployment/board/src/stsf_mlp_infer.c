/*
 * SparseTSF-MLP forward, mirroring
 * baselines/SparseTSF/SparseTSF-main/models/SparseTSF.py with
 * model_type='mlp':
 *   mean-subtract -> shared 1x25 'same' conv + residual ->
 *   phase-wise  Linear(n -> d_model) -> ReLU -> Linear(d_model -> m)
 *   -> add mean back.
 *
 * Only the head differs from stsf_infer.c; the front end is byte-identical,
 * which is what makes the two a controlled pair on the board.
 */
#include "stsf_mlp_infer.h"
#include "stsf_mlp_weights.h"   /* generated: golden/stsf_mlp_weights.h */

float smlp_st_mean[SMLP_D];
float smlp_st_xagg[SMLP_D * SMLP_L];
float smlp_st_hid[SMLP_D * SMLP_P * SMLP_DM];

static float s_xc[SMLP_D * SMLP_L];    /* centered, channel-major */
static float s_cv[SMLP_L];             /* conv scratch, one channel */
static float s_h[SMLP_DM];             /* hidden for one (channel, phase) */

void stsf_mlp_forward(const float *x, float *y)
{
    /* per-channel mean only (SparseTSF has no variance norm) */
    fea_instnorm_stats(x, SMLP_L, SMLP_D, smlp_st_mean, 0);

    for (int d = 0; d < SMLP_D; ++d)
        for (int t = 0; t < SMLP_L; ++t)
            s_xc[d * SMLP_L + t] = x[t * SMLP_D + d] - smlp_st_mean[d];

    /* one shared conv kernel applied to every channel + residual */
    for (int d = 0; d < SMLP_D; ++d) {
        fea_dwconv_same(s_xc + d * SMLP_L, s_cv, SMLP_W_CONV, 0,
                        SMLP_L, 1, SMLP_K);
        for (int t = 0; t < SMLP_L; ++t)
            smlp_st_xagg[d * SMLP_L + t] = s_xc[d * SMLP_L + t] + s_cv[t];
    }

    /* phase-wise MLP: v[n] -> hidden[128] -> out[m], per (channel, phase) */
    for (int d = 0; d < SMLP_D; ++d) {
        for (int ph = 0; ph < SMLP_P; ++ph) {
            float v[SMLP_N];
            for (int in = 0; in < SMLP_N; ++in)
                v[in] = smlp_st_xagg[d * SMLP_L + in * SMLP_P + ph];

            for (int h = 0; h < SMLP_DM; ++h) {
                float acc = SMLP_B_FC1[h];
                for (int in = 0; in < SMLP_N; ++in)
                    acc += SMLP_W_FC1[h * SMLP_N + in] * v[in];
                s_h[h] = acc > 0.0f ? acc : 0.0f;         /* ReLU */
            }
            for (int hh = 0; hh < SMLP_DM; ++hh)
                smlp_st_hid[(d * SMLP_P + ph) * SMLP_DM + hh] = s_h[hh];

            for (int om = 0; om < SMLP_M; ++om) {
                float acc = SMLP_B_FC2[om];
                for (int h = 0; h < SMLP_DM; ++h)
                    acc += SMLP_W_FC2[om * SMLP_DM + h] * s_h[h];
                y[(om * SMLP_P + ph) * SMLP_D + d] = acc + smlp_st_mean[d];
            }
        }
    }
}
