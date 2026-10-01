/*
 * DiPE-Linear forward, mirroring the upstream
 * baselines/DiPE_Linear/DiPE_Linear-main/timeprophet/models/DiPE.py
 * (DiPE.forward, num_experts=1, individual_* all False, eval mode -
 * dropout inactive). Every channel is independent.
 */
#include <math.h>

#include "dipe_infer.h"
#include "dipe_weights.h"      /* generated: golden/dipe_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(DIPE_H == DIPE_H_GEN,
               "DIPE_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
/* The 96-point freq_w tables stay inside fea_kernels.o (fea_irfft96);
 * including fea_twiddle.h here put a second 768 B copy into this object. */
#include "dipe_twiddle.h"      /* TN-point tables (expert path)    */

/* The expert geometry depends on H through the padding.  Refuse to build
 * against weights / twiddles generated for a different padding or length
 * instead of silently running the wrong DFT. */
_Static_assert(DIPE_PAD == DIPE_PAD_GEN,
               "DIPE_PAD does not match the padding the weights were exported at - re-run export_golden_dipe.py --pred_len");
_Static_assert(sizeof(DIPE_COS193) / sizeof(DIPE_COS193[0]) == DIPE_TN,
               "dipe_twiddle.h was generated for a different expert length (DIPE_TN)");
_Static_assert(sizeof(DIPE_W_EXP_RE) / sizeof(DIPE_W_EXP_RE[0]) == DIPE_TF,
               "dipe_weights.h expert size does not match DIPE_TF");

/* PER-CHANNEL WORKING SET (2026-09-04) - equal-effort rule, see
 * deployment/board/README.md.  DiPE is channel-independent, so nothing is
 * held for all D at once; full-width copies are test-build only.
 *
 * OUTPUT SIDE (2026-09-14).  Only what the forecast uses is computed or held:
 *   - xn, xf and xt share one row: the rfft consumes xn before the inverse
 *     writes xf, and the time weight is applied in place;
 *   - the expert's padded input [PAD zeros | xt | H-1+PAD zeros] is never
 *     built: the long DFT visits only the L real samples (the zeros added
 *     nothing to any bin);
 *   - the long inverse evaluates only the H samples the slice keeps,
 *     straight into y (the other L-1+2*PAD were computed and discarded).
 * Every value that IS computed keeps its terms and their order, so the
 * forecast is bit-identical.  The arena falls from 3*L + 2*TF + TN floats
 * (7.8 KB at H=720) to L + 2*TF (3.7 KB). */
static float s_x[DIPE_L];                     /* xn -> xf -> xt, in place */
static float s_re[DIPE_TF], s_im[DIPE_TF];    /* spectra; TF >= 49 bins   */
/* the 96-point transforms borrow bins [49, 81) of these as FFT scratch */
_Static_assert(DIPE_TF >= FEA_DFT_BINS + 32, "DiPE spectrum too short for FFT scratch");

#ifdef DIPE_CAPTURE_STAGES
float dipe_st_mean[DIPE_D];
float dipe_st_std[DIPE_D];
float dipe_st_xn[DIPE_D * DIPE_L];
float dipe_st_xf[DIPE_D * DIPE_L];
float dipe_st_xt[DIPE_D * DIPE_L];
#endif

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
#ifdef DIPE_CAPTURE_STAGES
        dipe_st_mean[d] = m;
        dipe_st_std[d] = sd;
#endif

        for (int t = 0; t < DIPE_L; ++t)
            s_x[t] = (x[t * DIPE_D + d] - m) / sd;
#ifdef DIPE_CAPTURE_STAGES
        for (int t = 0; t < DIPE_L; ++t)
            dipe_st_xn[d * DIPE_L + t] = s_x[t];
#endif

        /* static freq weight: rfft96 -> real per-bin weight -> irfft96 */
        fea_rfft96_fast(s_x, 1, FEA_DFT_BINS, s_re, s_im,
                        s_re + FEA_DFT_BINS, s_im + FEA_DFT_BINS);   /* free tail */
        for (int f = 0; f < FEA_DFT_BINS; ++f) {
            s_re[f] *= DIPE_W_FREQ[f];
            s_im[f] *= DIPE_W_FREQ[f];
        }
        fea_irfft96_fast(s_re, s_im, s_x, s_re + FEA_DFT_BINS, s_im + FEA_DFT_BINS);
#ifdef DIPE_CAPTURE_STAGES
        for (int t = 0; t < DIPE_L; ++t)
            dipe_st_xf[d * DIPE_L + t] = s_x[t];
#endif

        /* static time weight */
        for (int t = 0; t < DIPE_L; ++t)
            s_x[t] = s_x[t] * DIPE_W_TIME[t];
#ifdef DIPE_CAPTURE_STAGES
        for (int t = 0; t < DIPE_L; ++t)
            dipe_st_xt[d * DIPE_L + t] = s_x[t];
#endif

        /* FFT big-conv: pad [PAD | xt | H-1+PAD] -> TN-pt DFT ->
         * complex weight + bias -> inverse -> slice [-H-PAD : -PAD].
         * The padding is implicit: only the L real samples are visited. */
        fea_dft_bins(s_x, DIPE_L, DIPE_PAD, DIPE_TN, DIPE_TF,
                     DIPE_COS193, DIPE_SIN193, s_re, s_im);
        for (int f = 0; f < DIPE_TF; ++f) {
            float ar = s_re[f], ai = s_im[f];
            s_re[f] = ar * DIPE_W_EXP_RE[f] - ai * DIPE_W_EXP_IM[f]
                    + DIPE_B_EXP_RE[f];
            s_im[f] = ar * DIPE_W_EXP_IM[f] + ai * DIPE_W_EXP_RE[f]
                    + DIPE_B_EXP_IM[f];
        }

        /* the kept slice only, denormalized straight into y */
        for (int j = 0; j < DIPE_H; ++j) {
            float s = fea_idft_half_at(s_re, s_im, DIPE_TN, DIPE_TF,
                                       DIPE_COS193, DIPE_SIN193,
                                       DIPE_TN - DIPE_H - DIPE_PAD + j);
            y[j * DIPE_D + d] = s * sd + m;
        }
    }
}
