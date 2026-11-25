/*
 * FEATHer inference - streaming arena build.  See feather_core.h for
 * the model definition and the memory layout.
 *
 * Mirrors baselines/FEATHer/FEATHer.py forward() for the frozen defaults
 *   num_bands=3, point_kernel=3, gate_mode="banded", use_residual_mlp=True.
 * num_bands / point_kernel / use_norm / gate_mode none are build switches
 * (FEA2_NB / FEA2_KP / FEA2_USE_NORM / FEA2_GATE_NONE, feather_core.h).
 */
#include <math.h>
#include <string.h>

#include "feather_core.h"
#include "mcu_emit.h"            /* mcu_rt */
#include "mcu_pt.h"              /* PyTorch's reduction / softmax kernels */
#include "feather_weights.h"   /* generated: golden/feather_weights.h */
/* ci_full folds the channels into the batch: the gate's mean over the (size-1) channel axis runs over the whole
 * (channels, bins, 1) tensor, so channel c is a slice of that loop */
#if FEA2_D == 1 && defined(FEA2CI_D)
#define FEA2_NBATCH FEA2CI_D
#else
#define FEA2_NBATCH 1
#endif
#if FEA2_GATE_ON
_Static_assert(FEA2_D == 1, "the gate is written for ci_full (width 1)");
#include "mcu_rfft96_plan.h"    /* generated: golden/mcu_rfft96_plan.h (export/pocket_plan.py) */
_Static_assert(FEA2_L == 96, "the gate's FFT plan is the 96-point one");
#endif

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
#ifdef FEA2_P_GEN                       /* stamped by the export */
_Static_assert(FEA2_PERIOD == FEA2_P_GEN, "feather_weights.h was exported at another period");
#endif
#ifdef FEA2_SEG_GEN                     /* size keys, stamped by the export */
_Static_assert(FEA2_SEG == FEA2_SEG_GEN, "feather_weights.h was exported at another SPK segment length");
_Static_assert(FEA2_AGG_SHARED == FEA2_AGGS_GEN, "feather_weights.h was exported with another agg-kernel sharing");
#else                                   /* a canonical golden: the dial must be at its default */
_Static_assert(FEA2_SEG == FEA2_PERIOD && FEA2_AGG_SHARED == 0,
               "FEA2_SEG / FEA2_AGG_SHARED set but the weights were exported without --head_seg / --agg_shared");
#endif
_Static_assert(FEA2_H == FEA2_H_GEN,
               "FEA2_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
#ifndef FEA2_MLP_GEN                    /* stamped by the export only without the residual MLP */
#define FEA2_MLP_GEN 1
#endif
_Static_assert(FEA2_USE_MLP == FEA2_MLP_GEN, "FEA2_USE_MLP does not match the weights (export --use_residual_mlp)");
#if FEA2_USE_MLP
_Static_assert(FEA2_MLP_R == FEA2_R_GEN,
               "FEA2_MLP_R does not match the rank the weights were exported at - re-run export_golden_*.py --mlp_rank");
#endif
/* structural switches: stamped only when not the frozen value */
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
 * argument in feather_core.h.  Aliases keep the call sites readable and
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
static float s_mean_b[FEA2_NBATCH];       /* the means of all channels (torch.mean over the folded batch) */
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
 * 'same' zero padding, identical to mcu_dwconv_same. */
static void dwconv_row_acc(const float *row, float *col, int d,
                           const float *w, const float *b, int k, float scale)
{
    const int pad = k / 2;
    const float zero = mcu_rt(0.0f);            /* the padding: multiplied like any sample, as PyTorch does */
    for (int t = 0; t < FEA2_L; ++t) {
        float acc = b ? b[d] : 0.0f;
        for (int j = 0; j < k; ++j) {
            int s = t + j - pad;
            acc += w[d * k + j] * ((s >= 0 && s < FEA2_L) ? row[s] : zero);
        }
        col[t * FEA2_D + d] += scale * acc;
    }
}

/* col[t][d] += scale * interp_linear(src)[t]
 * align_corners=False, matches ATen upsample_linear1d. */
#if FEA2_NB >= 2
/* Source index and the two weights of every output step: ATen computes them once per forward
 * for all channels, so they are set once per forecast (fea2_interp_setup) and each
 * channel reads them - before, every channel computed them again. */
static unsigned char s_i0[FEA2_L];
static float s_w0[FEA2_L], s_w1[FEA2_L];
#endif
void fea2_interp_setup(void)
{
#if FEA2_NB >= 2
    const float sc = mcu_rt((float)FEA2_LP) / mcu_rt((float)FEA2_L);
    for (int t = 0; t < FEA2_L; ++t) {           /* area_pixel_compute_source_index + guard_index_and_lambda */
        float p = sc * ((float)t + 0.5f) - 0.5f;
        if (p < 0.0f) p = 0.0f;
        int i0 = (int)floorf(p);
        if (i0 > FEA2_LP - 1) i0 = FEA2_LP - 1;
        float l1 = p - (float)i0;
        l1 = l1 < 0.0f ? 0.0f : l1;              /* std::max(., 0) */
        l1 = 1.0f < l1 ? 1.0f : l1;              /* std::min(., 1) */
        s_i0[t] = (unsigned char)i0;
        s_w1[t] = l1;
        s_w0[t] = 1.0f - l1;
    }
#endif
}
#if FEA2_NB >= 2
static void interp_row_acc(const float *src, float *col, int d, float scale)
{
    for (int t = 0; t < FEA2_L; ++t) {
        int i0 = s_i0[t];
        int i1 = (i0 + 1 < FEA2_LP) ? i0 + 1 : FEA2_LP - 1;
        col[t * FEA2_D + d] += scale * (src[i0] * s_w0[t] + src[i1] * s_w1[t]);
    }
}
#endif

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
static void banded_gate(const float *xn, int c)
{
    float *sp = s_spec;                       /* the spectrum as NF (re, im) pairs */
    float *pw = s_spec + 2 * FEA2_NF;         /* re^2 + im^2 */
    float *ch = s_spec + 3 * FEA2_NF;         /* the FFT's work buffer, L */

    /* pocketfft returns r0 r1 i1 r2 i2 ... r48: transformed at sp + 1, the pairs then line up from sp[2] on */
    for (int t = 0; t < FEA2_L; ++t)
        sp[1 + t] = xn[t];
    pk_rfft_forward(&MCU_RFFT96_F, sp + 1, ch, 1.0f);
    sp[0] = sp[1];
    sp[1] = 0.0f;
    sp[2 * FEA2_NF - 1] = 0.0f;
    for (int f = 0; f < FEA2_NF; ++f)
        pw[f] = sp[2 * f] * sp[2 * f] + sp[2 * f + 1] * sp[2 * f + 1];
    /* P = (...).mean(dim=2): this channel's rows of the (batch, NF, 1) reduction (mcu_pt.c) */
    {
        const int shape[3] = {FEA2_NBATCH, FEA2_NF, 1}, stride[3] = {FEA2_NF, 1, 1};
        pt_mean_part(pw, c * FEA2_NF, ch, 3, shape, stride, 1u << 2, c * FEA2_NF, (c + 1) * FEA2_NF);
    }
    const float *P = ch;                      /* the FFT's work buffer is free again */
    CAP(fea2_st_power, P, FEA2_NF);

    float total;                              /* P.sum(dim=1) + 1e-6 */
    {
        const int shape[2] = {1, FEA2_NF}, stride[2] = {FEA2_NF, 1};
        pt_sum(P, &total, 2, shape, stride, 1u << 1);
    }
    total += 1e-6f;

    /* equal-width slices, low -> high (mirrors _fft_bounds "linear" and
     * utils/losses.spectral_separation_loss_scales); each P[:, lo:hi].sum(dim=1) */
    const int step = FEA2_NF / FEA2_NB;
    float feats[FEA2_NB];
    for (int s = 0; s < FEA2_NB; ++s) {
        int lo = s * step;
        int hi = (s == FEA2_NB - 1) ? FEA2_NF : (s + 1) * step;
        const int shape[2] = {1, hi - lo}, stride[2] = {FEA2_NF, 1};
        float e;
        pt_sum(P + lo, &e, 2, shape, stride, 1u << 1);
        /* flip: band (NB-1-s) owns slice s */
        feats[FEA2_NB - 1 - s] = logf(e / total + 1e-6f);
    }
    float m;
    {
        const int shape[2] = {1, FEA2_NB}, stride[2] = {FEA2_NB, 1};
        pt_mean(feats, &m, 2, shape, stride, 1u << 1);
    }
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
    pt_softmax(z, s_g, 1, FEA2_NB);
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

/* RevIN means: x.mean(1) of the (channels, L, 1) view of the window runs over all channels in one call
 * (torch.mean's kernel, the same loop as the (1, L, channels) layout of the other ports), so it is taken
 * once per forecast from the whole window x (L, channels). */
void fea2_norm_setup(const float *x)
{
#if FEA2_USE_NORM
    pt_mean_t(x, s_mean_b, FEA2_L, FEA2_NBATCH);
#else
    (void)x;
#endif
}

/* ------------------------------------------------------------------ *
 *  fea2_begin - everything up to and including the SPK aggregation.   *
 *  Leaves h_agg (time-major, L*D) in s_C, ready for fea2_block().     *
 *  x is only read; it may be fea2_input() (normalized in place).      *
 * ------------------------------------------------------------------ */
void fea2_begin(const float *x, int c)
{
    /* half-RevIN normalize.  The mean-detach / stdev-attached asymmetry in
     * FEATHer.py is a TRAINING-time choice (which statistic carries
     * gradient) and has no forward effect, so inference is the plain
     * instance norm below.  x_norm is parked in s_C: the hidden accumulator
     * is not needed until the bands are gated, so it holds x_norm instead of
     * a second (L, D) buffer. */
#if FEA2_USE_NORM
    {                                     /* x.mean(1) (fea2_norm_setup) and torch.var(x, 1, unbiased=False): mcu_pt.c */
        _Static_assert(FEA2_D == 1, "RevIN here is written for width 1");
        s_mean[0] = s_mean_b[c];
        float v;
        pt_var(x, &v, 1, 0, FEA2_L, 1, 0);
        s_stdev[0] = sqrtf(v + 1e-5f);
    }
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
    banded_gate(s_C, c);
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
     * its additions in PyTorch's stacking order [oP, oM, oL], so the hidden
     * state is the same as PyTorch's sum. */
    const float zero = mcu_rt(0.0f);
    for (int d = 0; d < FEA2_D; ++d) {
        for (int t = 0; t < FEA2_L; ++t) {
            s_row[t] = s_C[t * FEA2_D + d];
            s_C[t * FEA2_D + d] = zero;
        }
        dwconv_row_acc(s_row, s_C, d, W_CONVP, B_CONVP, FEA2_KP, FEA2_G(0)); /* POINT */
#if FEA2_NB == 3
        dwconv_row_acc(s_row, s_C, d, W_CONVM, B_CONVM, FEA2_KM, FEA2_G(1)); /* MID   */
#endif
#if FEA2_NB >= 2
        mcu_avgpool4(s_row, s_pool, FEA2_L, 1);                              /* LOW   */
        interp_row_acc(s_pool, s_C, d, FEA2_G(FEA2_NB - 1));
#endif
        /* (g * bands).sum(dim=1) over the stacked (batch, S, L, 1) tensor = SumKernel's outer sum, 32 time steps a
         * group: per step 0 + p_0 + .. + p_S-1 (above), the three empty cascade levels, then the store into the
         * zeroed output (mcu_pt.c; checked by test/test_pt_default.py) */
        _Static_assert(FEA2_L % 32 == 0 && FEA2_NB < 16, "the band sum's cascade shape");
        for (int t = 0; t < FEA2_L; ++t)
            s_C[t * FEA2_D + d] = zero + (((s_C[t * FEA2_D + d] + zero) + zero) + zero);
#ifdef FEA2_CAPTURE_STAGES
        dwconv_row_acc(s_row, fea2_st_bP, d, W_CONVP, B_CONVP, FEA2_KP, 1.0f);
#if FEA2_NB == 3
        dwconv_row_acc(s_row, fea2_st_bM, d, W_CONVM, B_CONVM, FEA2_KM, 1.0f);
#endif
#if FEA2_NB >= 2
        interp_row_acc(s_pool, fea2_st_bL, d, 1.0f);
#endif
#endif
    }
    CAP(fea2_st_hidden, s_C, FEA2_L * FEA2_D);

    /* residual MLP over the channel axis, applied independently per step:
     *   h[t] += W_up( GELU( W_down h[t] + b_down ) ) + b_up
     * exact-erf GELU (torch default approximate='none').  The bottleneck is
     * R floats, not L*R - nothing needs to be materialised across time.
     * This is the one stage that needs every channel of h[t] at once. */
#if FEA2_USE_MLP
    for (int t = 0; t < FEA2_L; ++t) {
        const float *h = s_C + t * FEA2_D;
        for (int r = 0; r < FEA2_MLP_R; ++r) {
            float acc = B_NLDOWN2[r];
            for (int d = 0; d < FEA2_D; ++d)
                acc += W_NLDOWN2[r * FEA2_D + d] * h[d];
            s_mlp[r] = mcu_gelu(acc);
        }
        for (int d = 0; d < FEA2_D; ++d) {
            float acc = B_NLUP2[d];
            for (int r = 0; r < FEA2_MLP_R; ++r)
                acc += W_NLUP2[d * FEA2_MLP_R + r] * s_mlp[r];
            s_C[t * FEA2_D + d] += acc;
        }
    }
#endif
    CAP(fea2_st_mlp, s_C, FEA2_L * FEA2_D);

    /* SPK aggregation: h_agg = h + agg_conv(h), depthwise over time, done in
     * place one column at a time.  s_row keeps the pre-conv column; the sum
     * acc + h is the same IEEE addition as mcu_dwconv_same followed by +=. */
    const int agg_pad = FEA2_AGG_K / 2;
    for (int d = 0; d < FEA2_D; ++d) {
        for (int t = 0; t < FEA2_L; ++t)
            s_row[t] = s_C[t * FEA2_D + d];
        for (int t = 0; t < FEA2_L; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < FEA2_AGG_K; ++j) {
                int s = t + j - agg_pad;
                acc += W_AGG2[FEA2_AGG_ROW(d) * FEA2_AGG_K + j] * ((s >= 0 && s < FEA2_L) ? s_row[s] : zero);
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
