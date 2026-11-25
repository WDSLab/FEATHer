/*
 * FITS forward, mirroring the upstream
 * baselines/FITS/FITS-main/models/FITS.py (shared upsampler) + the
 * benchmark wrapper's last-pred_len slice. Every channel is independent.
 */
#include <math.h>

#include "fits_infer.h"
#include "mcu_pt.h"
#include "fits_weights.h"      /* generated: golden/fits_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(FITS_H == FITS_H_GEN,
               "FITS_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
#include "mcu_rfft96_plan.h"   /* generated: analysis rfft(96) plan  */
#include "fits_fft.h"          /* generated: synthesis irfft(TN) plan */

_Static_assert(FITS_CUT == FITS_CUT_GEN && FITS_IND == FITS_IND_GEN,
               "fits_geom.h does not match the exported weights - re-run export_golden_fits.py");
_Static_assert(!FITS_IND || FITS_D == FITS_WD_GEN,
               "FITS_D does not match the width the per-channel weights were exported at");

/* PER-CHANNEL WORKING SET.  FITS computes each channel
 * independently, so nothing needs to be held for all D at once.  The arrays
 * below are one channel wide; the full-width copies exist only in the test
 * build, for golden comparison.  Holding them for every channel cost
 * D*UP*2 floats (11.4 KB at H=720) and pushed FITS over the 20 KB SRAM
 * budget - an artefact of the port, not of the model.
 *
 * OUTPUT SIDE.  The analysis rfft96 returns only the CUT kept
 * bins (the low-pass threw the others away), and the upsampled spectrum is
 * never copied into a zero-padded TF-bin buffer.
 *
 * TRANSFORMS: rfft(96) and irfft(TN) are pocketfft's real
 * transforms (src/mcu_pocket.c), the FFT PyTorch runs; the 1 / TN scale is the transform's.
 * All TN samples are computed, the last H are the forecast.
 *
 * The same rules hold for every port in this directory (equal-effort rule:
 * every port gets the same memory tidy-up). */

/* Two buffers serve one channel.  s_a: the normalised channel, transformed in place into its
 * spectrum (pocketfft's r0 r1 i1 ... sits at s_a + 1, so bin k is the pair s_a[2k], s_a[2k+1]),
 * next to that transform's work buffer; dead once the upsampler has read it, it is then the work
 * buffer of the synthesis.  s_p: the upsampled spectrum as (re, im) pairs, zero above UP bins,
 * transformed in place (from s_p + 1) into the TN samples. */
#define FITS_MAX2(a, b) ((a) > (b) ? (a) : (b))
_Static_assert(FITS_L == 96 && FITS_FFT_TN == FITS_TN && FITS_CUT <= FITS_L / 2 + 1 && 2 * FITS_UP <= FITS_TN,
               "fits_fft.h does not match FITS_TN - re-run export_golden_fits.py");
static float s_a[FITS_MAX2(2 * (FITS_L / 2 + 1) + FITS_L, FITS_TN)];
static float s_p[FITS_TN + 1];
#define s_xn (s_a + 1)
static float s_mean[FITS_D];

#ifdef FITS_CAPTURE_STAGES
float fits_st_mean[FITS_D];
float fits_st_sd[FITS_D];
float fits_st_xn[FITS_L * FITS_D];
float fits_st_up_re[FITS_D * FITS_UP];
float fits_st_up_im[FITS_D * FITS_UP];
#endif

/* Energy compensation.  The factor is the length ratio (L+H)/L, not a
 * constant 2 — that is only its value at H=96. */
#define FITS_ECOMP ((float)FITS_TN / (float)FITS_L)

static void fits_run(const float *x, float *y, int d0, int d1, int ys, mcu_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
        /* RevIN as upstream: x - mean, torch.var of the centred series (UNBIASED) + 1e-5, sqrt taken at the
         * normalisation and again at the output.  mean and var = PyTorch's kernels (mcu_pt.c): the means of
         * all channels in one call, at the first channel, as torch.mean computes them */
        if (d == 0)
            pt_mean_t(x, s_mean, FITS_L, FITS_D);
        const float m = s_mean[d];
        for (int t = 0; t < FITS_L; ++t)
            s_xn[t] = x[t * FITS_D + d] - m;
        float v;
        pt_var(s_xn, &v, 1, 0, FITS_L, 1, 1);
        const float var = v + 1e-5f;
        float sd = sqrtf(var);
#ifdef FITS_CAPTURE_STAGES
        fits_st_mean[d] = m;
        fits_st_sd[d] = sd;
#endif

        for (int t = 0; t < FITS_L; ++t) {
            s_xn[t] = s_xn[t] / sd;
#ifdef FITS_CAPTURE_STAGES
            fits_st_xn[t * FITS_D + d] = s_xn[t];
#endif
        }

        /* analysis rfft96; the low-pass keeps the first FITS_CUT bins */
        pk_rfft_forward(&MCU_RFFT96_F, s_xn, s_a + 2 * (FITS_L / 2 + 1), 1.0f);
        s_a[0] = s_a[1];
        s_a[1] = 0.0f;
        s_a[FITS_L + 1] = 0.0f;

        /* complex Linear(cut -> up): out = W in + b (one per channel when
         * individual, as the official Weather script) */
        const int wo = FITS_IND ? d * FITS_UP * FITS_CUT : 0;
        const int bo = FITS_IND ? d * FITS_UP : 0;
        for (int o = 0; o < FITS_UP; ++o) {
            float ar = FITS_B_RE[bo + o], ai = FITS_B_IM[bo + o];
            for (int i = 0; i < FITS_CUT; ++i) {
                float wr = FITS_W_RE[wo + o * FITS_CUT + i];
                float wi = FITS_W_IM[wo + o * FITS_CUT + i];
                ar += wr * s_a[2 * i] - wi * s_a[2 * i + 1];
                ai += wr * s_a[2 * i + 1] + wi * s_a[2 * i];
            }
            s_p[2 * o] = ar;
            s_p[2 * o + 1] = ai;
        }
#ifdef FITS_CAPTURE_STAGES
        for (int o = 0; o < FITS_UP; ++o) {
            fits_st_up_re[d * FITS_UP + o] = s_p[2 * o];
            fits_st_up_im[d * FITS_UP + o] = s_p[2 * o + 1];
        }
#endif

        /* irfft(FITS_TN) of the spectrum zero-padded to FITS_TF bins, last
         * pred_len samples only, energy compensation, denorm */
        for (int k = 2 * FITS_UP; k < FITS_TN + 1; ++k)
            s_p[k] = 0.0f;
        s_p[1] = s_p[0];                       /* r0 (the imaginary part of bin 0 is not used) */
        pk_rfft_backward(&FITS_IRFFT_B, s_p + 1, s_a, 1.0f / (float)FITS_TN);
        /* upstream scales and de-normalises all TN samples (the wrapper keeps the last H) and returns
         * low_xy * sqrt(var) as a second output: all of it is computed, as PyTorch does */
        sd = sqrtf(var);
        const float sd2 = sqrtf(var);
        const float ecomp = mcu_rt(FITS_ECOMP);
        for (int n = 0; n < FITS_TN; ++n) {
            const float lo = s_p[1 + n] * ecomp;
            const float v = lo * sd + m;
            mcu_drop(lo * sd2);
            if (n >= FITS_TN - FITS_H)
                MCU_OUT(yd, ys, emit, d, n - (FITS_TN - FITS_H), v);
            else
                mcu_drop(v);
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void fits_forward(const float *x, float *y) { fits_run(x, y, 0, FITS_D, FITS_D, 0); }

/* One channel d: its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer instead of holding H x D. */
void fits_forward_ch(const float *x, int d, float *yc) { fits_run(x, yc, d, d + 1, 1, 0); }

/* One channel d: each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (mcu_emit.h).  Same values as fits_forward_ch, bit for bit. */
void fits_forward_emit(const float *x, int d, mcu_emit_fn emit) { fits_run(x, 0, d, d + 1, 1, emit); }
