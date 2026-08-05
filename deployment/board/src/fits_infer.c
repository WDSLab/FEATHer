/*
 * FITS forward, mirroring the upstream
 * baselines/FITS/FITS-main/models/FITS.py (shared upsampler) + the
 * benchmark wrapper's last-pred_len slice. Every channel is independent.
 */
#include <math.h>

#include "fits_infer.h"
#include "fits_weights.h"      /* generated: golden/fits_weights.h */
#include "fea_twiddle.h"       /* 96-point tables (analysis rfft)  */
#include "fits_twiddle.h"      /* 192-point tables (synthesis)     */

float fits_st_mean[FITS_D];
float fits_st_sd[FITS_D];
float fits_st_xn[FITS_L * FITS_D];
float fits_st_up_re[FITS_D * FITS_UP];
float fits_st_up_im[FITS_D * FITS_UP];

static float s_xn[FITS_L];                 /* one channel, contiguous */
static float s_re[FITS_TF], s_im[FITS_TF]; /* spectra scratch         */
static float s_xy[FITS_TN];                /* irfft(192) output       */

void fits_forward(const float *x, float *y)
{
    for (int d = 0; d < FITS_D; ++d) {
        /* RevIN: mean, UNBIASED var + 1e-5 inside sqrt */
        float m = 0.0f;
        for (int t = 0; t < FITS_L; ++t)
            m += x[t * FITS_D + d];
        m /= (float)FITS_L;
        float v = 0.0f;
        for (int t = 0; t < FITS_L; ++t) {
            float c = x[t * FITS_D + d] - m;
            v += c * c;
        }
        float sd = sqrtf(v / (float)(FITS_L - 1) + 1e-5f);
        fits_st_mean[d] = m;
        fits_st_sd[d] = sd;

        for (int t = 0; t < FITS_L; ++t) {
            s_xn[t] = (x[t * FITS_D + d] - m) / sd;
            fits_st_xn[t * FITS_D + d] = s_xn[t];
        }

        /* analysis rfft96; only the first FITS_CUT bins are used (LPF) */
        fea_rfft96(s_xn, 1, s_re, s_im);

        /* complex Linear(cut -> up): out = W in + b */
        float *ur = fits_st_up_re + d * FITS_UP;
        float *ui = fits_st_up_im + d * FITS_UP;
        for (int o = 0; o < FITS_UP; ++o) {
            float ar = FITS_B_RE[o], ai = FITS_B_IM[o];
            for (int i = 0; i < FITS_CUT; ++i) {
                float wr = FITS_W_RE[o * FITS_CUT + i];
                float wi = FITS_W_IM[o * FITS_CUT + i];
                ar += wr * s_re[i] - wi * s_im[i];
                ai += wr * s_im[i] + wi * s_re[i];
            }
            ur[o] = ar;
            ui[o] = ai;
        }

        /* zero-pad to FITS_TF bins, irfft(192), x2 energy compensation */
        for (int f = 0; f < FITS_TF; ++f) {
            s_re[f] = (f < FITS_UP) ? ur[f] : 0.0f;
            s_im[f] = (f < FITS_UP) ? ui[f] : 0.0f;
        }
        fea_idft_half(s_re, s_im, FITS_TN, FITS_COS192, FITS_SIN192, s_xy);

        /* last pred_len samples + denorm (length_ratio = 2.0) */
        for (int j = 0; j < FITS_H; ++j)
            y[j * FITS_D + d] =
                s_xy[FITS_TN - FITS_H + j] * 2.0f * sd + m;
    }
}
