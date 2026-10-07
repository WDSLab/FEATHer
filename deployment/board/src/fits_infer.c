/*
 * FITS forward, mirroring the upstream
 * baselines/FITS/FITS-main/models/FITS.py (shared upsampler) + the
 * benchmark wrapper's last-pred_len slice. Every channel is independent.
 */
#include <math.h>

#include "fits_infer.h"
#include "fits_weights.h"      /* generated: golden/fits_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(FITS_H == FITS_H_GEN,
               "FITS_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
/* The 96-point analysis tables stay inside fea_kernels.o (fea_rfft96_bins);
 * including fea_twiddle.h here put a second 768 B copy into this object. */
#include "fits_twiddle.h"      /* TN-point tables (synthesis)      */

_Static_assert(FITS_CUT == FITS_CUT_GEN && FITS_IND == FITS_IND_GEN,
               "fits_geom.h does not match the exported weights - re-run export_golden_fits.py");
_Static_assert(!FITS_IND || FITS_D == FITS_WD_GEN,
               "FITS_D does not match the width the per-channel weights were exported at");

/* PER-CHANNEL WORKING SET (2026-09-04).  FITS computes each channel
 * independently, so nothing needs to be held for all D at once.  The arrays
 * below are one channel wide; the full-width copies exist only in the test
 * build, for golden comparison.  Holding them for every channel cost
 * D*UP*2 floats (11.4 KB at H=720) and pushed FITS over the 20 KB SRAM
 * budget - an artefact of the port, not of the model.
 *
 * OUTPUT SIDE (2026-09-14).  Only what the forecast uses is computed or held:
 *   - the analysis DFT evaluates the CUT kept bins, not all 49 (the low-pass
 *     threw the other 25 away);
 *   - the upsampled spectrum is never copied into a zero-padded TF-bin
 *     buffer, and synthesis does not sum the padding bins (they are zero);
 *   - synthesis evaluates only the H output samples, straight into y (the
 *     first L of the TN samples were computed and sliced off).
 * Every value that IS computed keeps its terms and their order, so the
 * forecast is bit-identical.  The arena falls from 2*TF + max(L, 2*UP, TN)
 * floats (6.5 KB at H=720) to max(L, 2*UP) + 2*CUT (1.8 KB).
 *
 * The same transformation is applied to every port in this directory; see
 * deployment/board/README.md "equal-effort rule". */

/* Within one channel s_xn is consumed by the analysis DFT before the
 * upsampled spectrum is written, so the two never coexist and share one
 * buffer of max(L, 2*UP) floats.  s_lo spans both stages and stays apart. */
#define FITS_MAX2(a, b) ((a) > (b) ? (a) : (b))
#define FITS_SCRATCH_N FITS_MAX2(FITS_L, 2 * FITS_UP)
static float s_scratch[FITS_SCRATCH_N];
#define s_xn    s_scratch                          /* 1. normalized channel */
#define s_up_re s_scratch                          /* 2. upsampled spectrum */
#define s_up_im (s_scratch + FITS_UP)
static float s_lo_re[FITS_CUT], s_lo_im[FITS_CUT]; /* kept analysis bins    */
/* fast rfft96 scratch (2 x 32).  During the analysis transform s_scratch holds
 * only s_xn [0, L); its tail [L, 2*UP) is free and is borrowed when it is long
 * enough (H >= 224), as DiPE and FEATHer do - otherwise FITS gets (and is
 * charged for) its own 64 floats. */
#if FITS_SCRATCH_N >= FITS_L + 64
#define s_fftw (s_scratch + FITS_L)
#else
static float s_fftw[64];
#endif

#ifdef FITS_CAPTURE_STAGES
float fits_st_mean[FITS_D];
float fits_st_sd[FITS_D];
float fits_st_xn[FITS_L * FITS_D];
float fits_st_up_re[FITS_D * FITS_UP];
float fits_st_up_im[FITS_D * FITS_UP];
#endif

/* Energy compensation.  The factor is the length ratio (L+H)/L, not a
 * constant 2 — that is only its value at H=96.  Hardcoding 2 scaled the whole
 * forecast by 2L/(L+H) at every other horizon (H=192 came out 1.5x too small;
 * caught 2026-09-04 by the H=192 golden run). */
#define FITS_ECOMP ((float)FITS_TN / (float)FITS_L)

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
#ifdef FITS_CAPTURE_STAGES
        fits_st_mean[d] = m;
        fits_st_sd[d] = sd;
#endif

        for (int t = 0; t < FITS_L; ++t) {
            s_xn[t] = (x[t * FITS_D + d] - m) / sd;
#ifdef FITS_CAPTURE_STAGES
            fits_st_xn[t * FITS_D + d] = s_xn[t];
#endif
        }

        /* analysis rfft96, only the FITS_CUT bins the low-pass keeps */
        fea_rfft96_fast(s_xn, 1, FITS_CUT, s_lo_re, s_lo_im, s_fftw, s_fftw + 32);

        /* complex Linear(cut -> up): out = W in + b (one per channel when
         * individual, as the official Weather script) */
        const int wo = FITS_IND ? d * FITS_UP * FITS_CUT : 0;
        const int bo = FITS_IND ? d * FITS_UP : 0;
        for (int o = 0; o < FITS_UP; ++o) {
            float ar = FITS_B_RE[bo + o], ai = FITS_B_IM[bo + o];
            for (int i = 0; i < FITS_CUT; ++i) {
                float wr = FITS_W_RE[wo + o * FITS_CUT + i];
                float wi = FITS_W_IM[wo + o * FITS_CUT + i];
                ar += wr * s_lo_re[i] - wi * s_lo_im[i];
                ai += wr * s_lo_im[i] + wi * s_lo_re[i];
            }
            s_up_re[o] = ar;
            s_up_im[o] = ai;
        }
#ifdef FITS_CAPTURE_STAGES
        for (int o = 0; o < FITS_UP; ++o) {
            fits_st_up_re[d * FITS_UP + o] = s_up_re[o];
            fits_st_up_im[d * FITS_UP + o] = s_up_im[o];
        }
#endif

        /* irfft(FITS_TN) of the spectrum zero-padded to FITS_TF bins, last
         * pred_len samples only, energy compensation, denorm */
        for (int j = 0; j < FITS_H; ++j) {
            float s = fea_idft_half_at(s_up_re, s_up_im, FITS_TN, FITS_UP,
                                       FITS_COS192, FITS_SIN192,
                                       FITS_TN - FITS_H + j);
            y[j * FITS_D + d] = s * FITS_ECOMP * sd + m;
        }
    }
}
