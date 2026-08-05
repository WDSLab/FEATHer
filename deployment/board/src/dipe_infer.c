/*
 * DiPE-Linear forward, mirroring the upstream
 * baselines/DiPE_Linear/DiPE_Linear-main/timeprophet/models/DiPE.py
 * (DiPE.forward, num_experts=1, individual_* all False, eval mode -
 * dropout inactive). Every channel is independent.
 */
#include <math.h>

#include "dipe_infer.h"
#include "dipe_weights.h"      /* generated: golden/dipe_weights.h */
#include "fea_twiddle.h"       /* 96-point tables (freq_w path)    */
#include "dipe_twiddle.h"      /* 193-point tables (expert path)   */

float dipe_st_mean[DIPE_D];
float dipe_st_std[DIPE_D];
float dipe_st_xn[DIPE_D * DIPE_L];
float dipe_st_xf[DIPE_D * DIPE_L];
float dipe_st_xt[DIPE_D * DIPE_L];

static float s_re[DIPE_TF], s_im[DIPE_TF];    /* max(49, 97) bins  */
static float s_pad[DIPE_TN];                  /* padded expert in  */
static float s_out[DIPE_TN];                  /* expert irfft out  */

void dipe_forward(const float *x, float *y)
{
    for (int d = 0; d < DIPE_D; ++d) {
        /* RevIN stats: UNBIASED std (n-1), no eps, clamp min 1e-7 */
        float m = 0.0f;
        for (int t = 0; t < DIPE_L; ++t)
            m += x[t * DIPE_D + d];
        m /= (float)DIPE_L;
        float v = 0.0f;
        for (int t = 0; t < DIPE_L; ++t) {
            float c = x[t * DIPE_D + d] - m;
            v += c * c;
        }
        float sd = sqrtf(v / (float)(DIPE_L - 1));
        if (sd < 1e-7f) sd = 1e-7f;
        dipe_st_mean[d] = m;
        dipe_st_std[d] = sd;

        float *xn = dipe_st_xn + d * DIPE_L;
        for (int t = 0; t < DIPE_L; ++t)
            xn[t] = (x[t * DIPE_D + d] - m) / sd;

        /* static freq weight: rfft96 -> real per-bin weight -> irfft96 */
        fea_rfft96(xn, 1, s_re, s_im);
        for (int f = 0; f < FEA_DFT_BINS; ++f) {
            s_re[f] *= DIPE_W_FREQ[f];
            s_im[f] *= DIPE_W_FREQ[f];
        }
        float *xf = dipe_st_xf + d * DIPE_L;
        fea_idft_half(s_re, s_im, DIPE_L, FEA_COS96, FEA_SIN96, xf);

        /* static time weight */
        float *xt = dipe_st_xt + d * DIPE_L;
        for (int t = 0; t < DIPE_L; ++t)
            xt[t] = xf[t] * DIPE_W_TIME[t];

        /* FFT big-conv: pad [PAD | x | H-1+PAD] -> 193-pt DFT ->
         * complex weight + bias -> inverse -> slice [-H-PAD : -PAD] */
        for (int t = 0; t < DIPE_TN; ++t)
            s_pad[t] = 0.0f;
        for (int t = 0; t < DIPE_L; ++t)
            s_pad[DIPE_PAD + t] = xt[t];
        fea_dft_half(s_pad, 1, DIPE_TN, DIPE_COS193, DIPE_SIN193,
                     s_re, s_im);
        for (int f = 0; f < DIPE_TF; ++f) {
            float ar = s_re[f], ai = s_im[f];
            s_re[f] = ar * DIPE_W_EXP_RE[f] - ai * DIPE_W_EXP_IM[f]
                    + DIPE_B_EXP_RE[f];
            s_im[f] = ar * DIPE_W_EXP_IM[f] + ai * DIPE_W_EXP_RE[f]
                    + DIPE_B_EXP_IM[f];
        }
        fea_idft_half(s_re, s_im, DIPE_TN, DIPE_COS193, DIPE_SIN193,
                      s_out);

        /* slice + denormalize */
        for (int j = 0; j < DIPE_H; ++j)
            y[j * DIPE_D + d] =
                s_out[DIPE_TN - DIPE_H - DIPE_PAD + j] * sd + m;
    }
}
