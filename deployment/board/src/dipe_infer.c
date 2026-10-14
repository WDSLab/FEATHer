/*
 * DiPE-Linear forward, mirroring the upstream
 * baselines/DiPE_Linear/DiPE_Linear-main/timeprophet/models/DiPE.py
 * (DiPE.forward, eval mode - dropout inactive). Every channel is independent.
 * num_experts / RevIN / individual_* / use_freq_w / use_time_w: dipe_infer.h.
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
_Static_assert(sizeof(DIPE_W_EXP_RE) / sizeof(DIPE_W_EXP_RE[0]) == (DIPE_IND_C ? DIPE_D : DIPE_E) * DIPE_TF,
               "dipe_weights.h expert size does not match DIPE_TF");
_Static_assert(DIPE_E == DIPE_E_GEN && DIPE_REVIN == DIPE_REVIN_GEN,
               "dipe_geom.h does not match the exported weights - re-run export_golden_dipe.py");
_Static_assert((DIPE_E == 1 && !(DIPE_IND_F || DIPE_IND_T || DIPE_IND_C)) || DIPE_D == DIPE_RD_GEN,
               "DIPE_D does not match the width the router / per-channel weights were exported at");
#ifndef DIPE_VAR_GEN                 /* individual_* / weight switches, stamped only off the default */
#define DIPE_VAR_GEN 0
#endif
_Static_assert(DIPE_VAR == DIPE_VAR_GEN, "dipe_geom.h does not match the exported weights (individual_* / use_*_w)");

/* Static router (num_experts > 1).  Upstream mixes every weight per channel on
 * every forward: w_c = sum_e softmax_e(route[:, c] / T) * w_e, T = 1 at test.
 * The mix is evaluated where each weight is read (each is read once per
 * channel), so it needs no buffer and nothing is computed twice. */
#if DIPE_E > 1
static float s_r[DIPE_E];
#define DIPE_MIX(arr, n, i) dipe_mix((arr), (n), (i))
static inline float dipe_mix(const float *w, int n, int i)
{
    float acc = 0.0f;
    for (int e = 0; e < DIPE_E; ++e)
        acc += w[e * n + i] * s_r[e];
    return acc;
}
#else
#define DIPE_MIX(arr, n, i) ((arr)[i])
#endif
/* weight i of channel d: its own row when individual (upstream mixes with eye(D), i.e. picks
 * row d exactly), else the router mix / the shared row */
#define DIPE_W(ind, arr, n, d, i) ((ind) ? (arr)[(d) * (n) + (i)] : DIPE_MIX(arr, n, i))

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

static void dipe_run(const float *x, float *y, int d0, int d1, int ys, fea_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
#if DIPE_E > 1
        {   /* this channel's expert weights: softmax over the router logits */
            float mx = -INFINITY, se = 0.0f;
            for (int e = 0; e < DIPE_E; ++e)
                if (DIPE_ROUTE[e * DIPE_D + d] > mx) mx = DIPE_ROUTE[e * DIPE_D + d];
            for (int e = 0; e < DIPE_E; ++e) {
                s_r[e] = expf(DIPE_ROUTE[e * DIPE_D + d] - mx);
                se += s_r[e];
            }
            for (int e = 0; e < DIPE_E; ++e)
                s_r[e] /= se;
        }
#endif
#if DIPE_REVIN
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
#else   /* no RevIN (official Weather / Electricity scripts) */
#ifdef DIPE_CAPTURE_STAGES
        dipe_st_mean[d] = 0.0f;
        dipe_st_std[d] = 1.0f;
#endif
        for (int t = 0; t < DIPE_L; ++t)
            s_x[t] = x[t * DIPE_D + d];
#endif
#ifdef DIPE_CAPTURE_STAGES
        for (int t = 0; t < DIPE_L; ++t)
            dipe_st_xn[d * DIPE_L + t] = s_x[t];
#endif

        /* static freq weight: rfft96 -> real per-bin weight -> irfft96 */
#if DIPE_FREQ_W
        fea_rfft96_fast(s_x, 1, FEA_DFT_BINS, s_re, s_im,
                        s_re + FEA_DFT_BINS, s_im + FEA_DFT_BINS);   /* free tail */
        for (int f = 0; f < FEA_DFT_BINS; ++f) {
            const float w = DIPE_W(DIPE_IND_F, DIPE_W_FREQ, FEA_DFT_BINS, d, f);
            s_re[f] *= w;
            s_im[f] *= w;
        }
        fea_irfft96_fast(s_re, s_im, s_x, s_re + FEA_DFT_BINS, s_im + FEA_DFT_BINS);
#endif
#ifdef DIPE_CAPTURE_STAGES
        for (int t = 0; t < DIPE_L; ++t)
            dipe_st_xf[d * DIPE_L + t] = s_x[t];
#endif

        /* static time weight */
#if DIPE_TIME_W
        for (int t = 0; t < DIPE_L; ++t)
            s_x[t] = s_x[t] * DIPE_W(DIPE_IND_T, DIPE_W_TIME, DIPE_L, d, t);
#endif
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
            const float wr = DIPE_W(DIPE_IND_C, DIPE_W_EXP_RE, DIPE_TF, d, f);
            const float wi = DIPE_W(DIPE_IND_C, DIPE_W_EXP_IM, DIPE_TF, d, f);
            s_re[f] = ar * wr - ai * wi + DIPE_W(DIPE_IND_C, DIPE_B_EXP_RE, DIPE_TF, d, f);
            s_im[f] = ar * wi + ai * wr + DIPE_W(DIPE_IND_C, DIPE_B_EXP_IM, DIPE_TF, d, f);
        }

        /* the kept slice only, denormalized straight into y */
        for (int j = 0; j < DIPE_H; ++j) {
            float s = fea_idft_half_at(s_re, s_im, DIPE_TN, DIPE_TF,
                                       DIPE_COS193, DIPE_SIN193,
                                       DIPE_TN - DIPE_H - DIPE_PAD + j);
#if DIPE_REVIN
            FEA_OUT(yd, ys, emit, d, j, s * sd + m);
#else
            FEA_OUT(yd, ys, emit, d, j, s);
#endif
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void dipe_forward(const float *x, float *y) { dipe_run(x, y, 0, DIPE_D, DIPE_D, 0); }

/* One channel d (10-12): its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer (porting rule 1) instead of holding H x D. */
void dipe_forward_ch(const float *x, int d, float *yc) { dipe_run(x, yc, d, d + 1, 1, 0); }

/* One channel d (10-13): each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (fea_emit.h).  Same values as dipe_forward_ch, bit for bit. */
void dipe_forward_emit(const float *x, int d, fea_emit_fn emit) { dipe_run(x, 0, d, d + 1, 1, emit); }
