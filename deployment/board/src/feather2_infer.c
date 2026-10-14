/*
 * FEATHer v2 inference - streaming arena build.  See feather2_infer.h for
 * the model definition and the departures from the 08-24 port.
 *
 * Mirrors baselines/FEATHer/FEATHer.py forward() for
 *   use_in_proj=False, num_bands=3, point_kernel=3, dtk_mode="none",
 *   gate_mode="banded", use_residual_mlp=True, channel_mode="mix".
 * 10-14: num_bands / point_kernel / use_norm / gate_mode none are build switches
 * (FEA2_NB / FEA2_KP / FEA2_USE_NORM / FEA2_GATE_NONE, feather2_infer.h).
 */
#include <math.h>
#include <string.h>

#include "feather2_infer.h"
#include "feather2_weights.h"   /* generated: golden/feather2_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
#ifdef FEA2_P_GEN                       /* stamped by the export since 10-12 */
_Static_assert(FEA2_PERIOD == FEA2_P_GEN, "feather2_weights.h was exported at another period");
#endif
#ifdef FEA2_SEG_GEN                     /* 10-13 size dial, stamped by the export */
_Static_assert(FEA2_SEG == FEA2_SEG_GEN, "feather2_weights.h was exported at another SPK segment length");
_Static_assert(FEA2_AGG_SHARED == FEA2_AGGS_GEN, "feather2_weights.h was exported with another agg-kernel sharing");
#else                                   /* a canonical golden: the dial must be at its default */
_Static_assert(FEA2_SEG == FEA2_PERIOD && FEA2_AGG_SHARED == 0,
               "FEA2_SEG / FEA2_AGG_SHARED set but the weights were exported without --head_seg / --agg_shared");
#endif
_Static_assert(FEA2_H == FEA2_H_GEN,
               "FEA2_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
_Static_assert(FEA2_MLP_R == FEA2_R_GEN,
               "FEA2_MLP_R does not match the rank the weights were exported at - re-run export_golden_*.py --mlp_rank");
/* 10-14 structural switches: stamped only when not the frozen value */
#ifndef FEA2_NB_GEN
#define FEA2_NB_GEN 3
#endif
#ifndef FEA2_KP_GEN
#define FEA2_KP_GEN 3
#endif
#ifndef FEA2_NORM_GEN
#define FEA2_NORM_GEN 1
#endif
#ifndef FEA2_GNONE_GEN
#define FEA2_GNONE_GEN 0
#endif
_Static_assert(FEA2_NB == FEA2_NB_GEN && FEA2_KP == FEA2_KP_GEN && FEA2_USE_NORM == FEA2_NORM_GEN
               && FEA2_GATE_NONE == FEA2_GNONE_GEN,
               "FEA2_NB / FEA2_KP / FEA2_USE_NORM / FEA2_GATE_NONE do not match the exported weights");
_Static_assert(FEA2_KP % 2 == 1 && FEA2_NB >= 1 && FEA2_NB <= 3, "point_kernel must be odd, num_bands 1..3");

#ifdef FEA2_CAPTURE_STAGES
float fea2_st_xnorm[FEA2_L * FEA2_D];
float fea2_st_mean[FEA2_D];
float fea2_st_stdev[FEA2_D];
float fea2_st_power[FEA2_NF];
float fea2_st_feats[FEA2_NB];
float fea2_st_gate[FEA2_NB];
float fea2_st_bP[FEA2_L * FEA2_D];
float fea2_st_bM[FEA2_L * FEA2_D];
float fea2_st_bL[FEA2_L * FEA2_D];
float fea2_st_hidden[FEA2_L * FEA2_D];
float fea2_st_mlp[FEA2_L * FEA2_D];
float fea2_st_hagg[FEA2_D * FEA2_L];
#define CAP(dst, src, n) memcpy((dst), (src), (n) * sizeof(float))
#else
#define CAP(dst, src, n) ((void)0)
#endif

/* ---- the arena (horizon-independent) ----
 * s_C is the ONLY (L, D) buffer.  The residual MLP mixes channels at every
 * timestep, so the hidden state has to be held for all channels at once -
 * that is the model's structural cost.  Everything else is one channel wide:
 * s_C first holds x_norm (free until the bands are gated), then the hidden
 * state, then h_agg. */
static float s_C[FEA2_L * FEA2_D];        /* x_norm -> hidden -> h_agg       */

/* Five stages that never coexist share one buffer - see the lifetime
 * argument in feather2_infer.h.  Aliases keep the call sites readable and
 * make the sharing explicit at every use; do NOT introduce a use that
 * straddles two stages without re-checking that comment. */
static float s_scratch[FEA2_SCRATCH_N];
#define s_spec s_scratch                  /* 1. gate re / im / power       */
#define s_row  s_scratch                  /* 2,4. one channel's row (L)    */
#define s_pool (s_scratch + FEA2_L)       /* 2. LOW pooled, beside s_row   */
#define s_mlp  s_scratch                  /* 3. MLP bottleneck             */
#define s_blk  s_scratch                  /* 5. one output period, one ch. */
#if FEA2_USE_NORM
static float s_mean[FEA2_D], s_stdev[FEA2_D];
#endif
#if FEA2_GATE_ON
static float s_g[FEA2_NB];
#define FEA2_G(i) s_g[i]
#else
#define FEA2_G(i) 1.0f                    /* gate none / one band: weight 1 */
#endif

/* ------------------------------------------------------------------ *
 *  band helpers - one channel in, accumulated into its column of the  *
 *  time-major (L, D) buffer; a band is never materialised             *
 * ------------------------------------------------------------------ */

/* col[t][d] += scale * ( sum_j w[d][j] * row[t+j-pad] + b[d] )
 * 'same' zero padding, identical to fea_dwconv_same. */
static void dwconv_row_acc(const float *row, float *col, int d,
                           const float *w, const float *b, int k, float scale)
{
    const int pad = k / 2;
    for (int t = 0; t < FEA2_L; ++t) {
        float acc = b ? b[d] : 0.0f;
        for (int j = 0; j < k; ++j) {
            int s = t + j - pad;
            if (s >= 0 && s < FEA2_L)
                acc += w[d * k + j] * row[s];
        }
        col[t * FEA2_D + d] += scale * acc;
    }
}

/* col[t][d] += scale * interp_linear(src)[t]
 * align_corners=False, matches ATen upsample_linear1d (and
 * fea_interp_linear, which this deliberately duplicates to fuse the add). */
static void interp_row_acc(const float *src, float *col, int d,
                           int len_in, int len_out, float scale)
{
    const float sc = (float)len_in / (float)len_out;
    for (int t = 0; t < len_out; ++t) {
        float p = ((float)t + 0.5f) * sc - 0.5f;
        if (p < 0.0f) p = 0.0f;
        int i0 = (int)p;
        int i1 = (i0 + 1 < len_in) ? i0 + 1 : len_in - 1;
        float f = p - (float)i0;
        col[t * FEA2_D + d] += scale * (src[i0] * (1.0f - f) + src[i1] * f);
    }
}

/* ------------------------------------------------------------------ *
 *  banded FFT gate                                                    *
 *                                                                     *
 *  P     = mean_d |rfft(x_norm[:,d])|^2                (NF bins)      *
 *  ratio = slice energy / total,  slices low->high, equal width       *
 *  band s owns slice (NB-1-s)  =>  band 0 (POINT) is the HIGHEST      *
 *  feats = log(ratio + 1e-6), then mean-centered                      *
 *  g     = softmax(W feats + b)                                       *
 * ------------------------------------------------------------------ */
#if FEA2_GATE_ON
static void banded_gate(const float *xn)
{
    float *re = s_spec;
    float *im = s_spec + FEA2_NF;
    float *P  = s_spec + 2 * FEA2_NF;

    for (int f = 0; f < FEA2_NF; ++f)
        P[f] = 0.0f;
    for (int d = 0; d < FEA2_D; ++d) {
        fea_rfft96_fast(xn + d, FEA2_D, FEA2_NF, re, im,
                        s_spec + 3 * FEA2_NF, s_spec + 3 * FEA2_NF + 32);
        for (int f = 0; f < FEA2_NF; ++f)
            P[f] += re[f] * re[f] + im[f] * im[f];
    }
    for (int f = 0; f < FEA2_NF; ++f)
        P[f] /= (float)FEA2_D;
    CAP(fea2_st_power, P, FEA2_NF);

    float total = 1e-6f;
    for (int f = 0; f < FEA2_NF; ++f)
        total += P[f];

    /* equal-width slices, low -> high (mirrors _fft_bounds "linear" and
     * utils/losses.spectral_separation_loss_scales) */
    const int step = FEA2_NF / FEA2_NB;
    float feats[FEA2_NB];
    for (int s = 0; s < FEA2_NB; ++s) {
        int lo = s * step;
        int hi = (s == FEA2_NB - 1) ? FEA2_NF : (s + 1) * step;
        float e = 0.0f;
        for (int f = lo; f < hi; ++f)
            e += P[f];
        /* flip: band (NB-1-s) owns slice s */
        feats[FEA2_NB - 1 - s] = logf(e / total + 1e-6f);
    }
    float m = 0.0f;
    for (int s = 0; s < FEA2_NB; ++s)
        m += feats[s];
    m /= (float)FEA2_NB;
    for (int s = 0; s < FEA2_NB; ++s)
        feats[s] -= m;
    CAP(fea2_st_feats, feats, FEA2_NB);

    float z[FEA2_NB];
    for (int i = 0; i < FEA2_NB; ++i) {
        float acc = B_GATE2[i];
        for (int j = 0; j < FEA2_NB; ++j)
            acc += W_GATE2[i * FEA2_NB + j] * feats[j];
        z[i] = acc;
    }
    fea_softmax(z, s_g, FEA2_NB);
    CAP(fea2_st_gate, s_g, FEA2_NB);
}
#endif /* FEA2_GATE_ON */

/* The (L, D) arena buffer, handed out so a caller that has to stage the input
 * window anyway (FEATHer-CI gathering one channel) writes it here instead of
 * into a buffer of its own.  fea2_begin(fea2_input()) is valid: the
 * statistics are taken before anything is written, and each sample is read
 * before it is overwritten in place. */
float *fea2_input(void)
{
    return s_C;
}

/* ------------------------------------------------------------------ *
 *  fea2_begin - everything up to and including the SPK aggregation.   *
 *  Leaves h_agg (time-major, L*D) in s_C, ready for fea2_block().     *
 *  x is only read; it may be fea2_input() (normalized in place).      *
 * ------------------------------------------------------------------ */
void fea2_begin(const float *x)
{
    /* half-RevIN normalize.  The mean-detach / stdev-attached asymmetry in
     * FEATHer.py is a TRAINING-time choice (which statistic carries
     * gradient) and has no forward effect, so inference is the plain
     * instance norm below.  x_norm is parked in s_C: the hidden accumulator
     * is not needed until the bands are gated, so it holds x_norm instead of
     * a second (L, D) buffer. */
#if FEA2_USE_NORM
    fea_instnorm_stats(x, FEA2_L, FEA2_D, s_mean, s_stdev);
    for (int t = 0; t < FEA2_L; ++t)
        for (int d = 0; d < FEA2_D; ++d)
            s_C[t * FEA2_D + d] = (x[t * FEA2_D + d] - s_mean[d]) / s_stdev[d];
    CAP(fea2_st_mean, s_mean, FEA2_D);
    CAP(fea2_st_stdev, s_stdev, FEA2_D);
#else                                     /* no RevIN: the bands read x itself */
    for (int i = 0; i < FEA2_L * FEA2_D; ++i)
        s_C[i] = x[i];
#endif
    CAP(fea2_st_xnorm, s_C, FEA2_L * FEA2_D);

    /* the gate reads x_norm, NOT the bands - this is what makes
     * accumulate-in-place legal */
#if FEA2_GATE_ON
    banded_gate(s_C);
#elif defined(FEA2_CAPTURE_STAGES)
    for (int s = 0; s < FEA2_NB; ++s)
        fea2_st_gate[s] = 1.0f;
#endif

#ifdef FEA2_CAPTURE_STAGES
    /* unscaled bands, for golden comparison only - the board build does not
     * compile this and never materialises a band */
    for (int i = 0; i < FEA2_L * FEA2_D; ++i) {
        fea2_st_bP[i] = 0.0f;
        fea2_st_bM[i] = 0.0f;
        fea2_st_bL[i] = 0.0f;
    }
#endif

    /* Bands, one channel at a time: lift the channel's x_norm out into s_row,
     * clear its column, and fold the gated bands back in.  Each cell receives
     * its additions in PyTorch's stacking order [oP, oM, oL], exactly as the
     * all-channel build did, so the hidden state is bit-identical.  With
     * dtk_mode="none" dense_kernel is Identity, so band == conv out. */
    for (int d = 0; d < FEA2_D; ++d) {
        for (int t = 0; t < FEA2_L; ++t) {
            s_row[t] = s_C[t * FEA2_D + d];
            s_C[t * FEA2_D + d] = 0.0f;
        }
        dwconv_row_acc(s_row, s_C, d, W_CONVP, B_CONVP, FEA2_KP, FEA2_G(0)); /* POINT */
#if FEA2_NB == 3
        dwconv_row_acc(s_row, s_C, d, W_CONVM, B_CONVM, FEA2_KM, FEA2_G(1)); /* MID   */
#endif
#if FEA2_NB >= 2
        fea_avgpool4(s_row, s_pool, FEA2_L, 1);                              /* LOW   */
        interp_row_acc(s_pool, s_C, d, FEA2_LP, FEA2_L, FEA2_G(FEA2_NB - 1));
#endif
#ifdef FEA2_CAPTURE_STAGES
        dwconv_row_acc(s_row, fea2_st_bP, d, W_CONVP, B_CONVP, FEA2_KP, 1.0f);
#if FEA2_NB == 3
        dwconv_row_acc(s_row, fea2_st_bM, d, W_CONVM, B_CONVM, FEA2_KM, 1.0f);
#endif
#if FEA2_NB >= 2
        interp_row_acc(s_pool, fea2_st_bL, d, FEA2_LP, FEA2_L, 1.0f);
#endif
#endif
    }
    CAP(fea2_st_hidden, s_C, FEA2_L * FEA2_D);

    /* residual MLP over the channel axis, applied independently per step:
     *   h[t] += W_up( GELU( W_down h[t] + b_down ) ) + b_up
     * exact-erf GELU (torch default approximate='none').  The bottleneck is
     * R floats, not L*R - nothing needs to be materialised across time.
     * This is the one stage that needs every channel of h[t] at once. */
    for (int t = 0; t < FEA2_L; ++t) {
        const float *h = s_C + t * FEA2_D;
        for (int r = 0; r < FEA2_MLP_R; ++r) {
            float acc = B_NLDOWN2[r];
            for (int d = 0; d < FEA2_D; ++d)
                acc += W_NLDOWN2[r * FEA2_D + d] * h[d];
            s_mlp[r] = fea_gelu(acc);
        }
        for (int d = 0; d < FEA2_D; ++d) {
            float acc = B_NLUP2[d];
            for (int r = 0; r < FEA2_MLP_R; ++r)
                acc += W_NLUP2[d * FEA2_MLP_R + r] * s_mlp[r];
            s_C[t * FEA2_D + d] += acc;
        }
    }
    CAP(fea2_st_mlp, s_C, FEA2_L * FEA2_D);

    /* SPK aggregation: h_agg = h + agg_conv(h), depthwise over time, done in
     * place one column at a time.  s_row keeps the pre-conv column; the sum
     * acc + h is the same IEEE addition as fea_dwconv_same followed by +=. */
    const int agg_pad = FEA2_AGG_K / 2;
    for (int d = 0; d < FEA2_D; ++d) {
        for (int t = 0; t < FEA2_L; ++t)
            s_row[t] = s_C[t * FEA2_D + d];
        for (int t = 0; t < FEA2_L; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < FEA2_AGG_K; ++j) {
                int s = t + j - agg_pad;
                if (s >= 0 && s < FEA2_L)
                    acc += W_AGG2[FEA2_AGG_ROW(d) * FEA2_AGG_K + j] * s_row[s];
            }
            s_C[t * FEA2_D + d] = acc + s_row[t];
        }
    }
#ifdef FEA2_CAPTURE_STAGES
    for (int d = 0; d < FEA2_D; ++d)
        for (int t = 0; t < FEA2_L; ++t)
            fea2_st_hagg[d * FEA2_L + t] = s_C[t * FEA2_D + d];
#endif
}

/* ------------------------------------------------------------------ *
 *  fea2_block - one output period of one channel.                     *
 *                                                                     *
 *  y[om*P + ph][d] = sum_in W_BB[om][in] * h_agg[in*P + ph][d]        *
 *                                                                     *
 *  Output step (om*P+ph) touches ONLY backbone row om and channel d,   *
 *  so the forecast can be emitted P steps at a time per channel.       *
 *  Denormalization is per-channel and step-independent, so it applies  *
 *  to the block unchanged.                                            *
 * ------------------------------------------------------------------ */
const float *fea2_block(int om, int d)
{
    for (int ph = 0; ph < FEA2_SEG; ++ph) {
        float acc = 0.0f;
        for (int in = 0; in < FEA2_N; ++in)
            acc += W_BB2[om * FEA2_N + in]
                 * s_C[(in * FEA2_SEG + ph) * FEA2_D + d];
#if FEA2_USE_NORM
        s_blk[ph] = acc * s_stdev[d] + s_mean[d];
#else
        s_blk[ph] = acc;
#endif
    }
    return s_blk;
}

/* convenience: stream every block into a caller-owned full-length buffer.
 * The full-length buffer belongs to the CALLER (test harness / batch eval);
 * the model itself never allocates one. */
void fea2_forward(const float *x, float *y)
{
    fea2_begin(x);
    for (int om = 0; om < FEA2_M; ++om)
        for (int d = 0; d < FEA2_D; ++d) {
            const float *blk = fea2_block(om, d);
            for (int ph = 0; ph < FEA2_SEG; ++ph)
                y[((size_t)om * FEA2_SEG + ph) * FEA2_D + d] = blk[ph];
        }
}
