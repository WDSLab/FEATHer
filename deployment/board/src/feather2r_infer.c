/*
 * FEATHer v2 (the 09-04 frozen mix model, 452p) - RECOMPUTE schedule.
 * 2026-09-22.  Same function as feather2_infer.c, built only to answer a
 * fairness question: the candidate (feather2in_infer.c) reaches 43 sensors per
 * chip by recomputing x_norm per sensor instead of holding it (an exception to
 * port rule 4).  What does the SAME concession buy the old model?
 *
 * feather2_infer.c holds one (L, D) buffer (x_norm -> hidden -> h_agg) because
 * the residual MLP mixes the gated band output h[t, :] of every channel at each
 * step.  Here nothing (L, D) is kept besides the caller's input window:
 *
 *   begin   pass 0  per sensor: x_norm row -> rfft -> accumulate mean power
 *                   -> banded gate g (needs every sensor, so it comes first)
 *           pass 1  per sensor: x_norm row -> gated bands h_d (L) ->
 *                   U[t][r] += W_down[r][d] h_d[t]      (U = L x R; R = 16 for the 09-04
 *                   model, R = 4 for the 2026-09-23 re-freeze via -DFEA2_MLP_R=4)
 *                   then U = GELU(U + b_down)  (bias added first, see below)
 *   channel(d)      recompute x_norm row and h_d, add W_up U + b_up,
 *                   SPK aggregation -> one column h_agg_d (L)
 *   block(om)       one output period of the current sensor
 *
 * i.e. the band convolutions run TWICE per sensor and x_norm three times.
 * Additions happen in the same order as feather2_infer.c (U starts at b_down
 * and adds sensors in ascending d), so the forecast is bit-identical; the test
 * test/test_feather2r.c checks that against the original port and the golden.
 */
#include <math.h>
#include <string.h>

#include "feather2_infer.h"
#include "feather2r_infer.h"
/* FEA2_FRONT_NONE (2026-10-07): the control arm without the band bank and the
 * FFT gate (FEATHer.py front="none": hidden = x_norm), to measure what bands +
 * gate cost on the chip.  Everything else - residual mixer, SPK head, this
 * recompute schedule - is the frozen model's.  Off = byte-identical port. */
#ifdef FEA2_FRONT_NONE
#include "feather2fn_weights.h"
_Static_assert(FEA2_FN_GEN == 1, "feather2fn_weights.h was not exported with front=none");
#else
#include "feather2_weights.h"
#endif

_Static_assert(FEA2_H == FEA2_H_GEN,
               "FEA2_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
_Static_assert(FEA2_MLP_R == FEA2_R_GEN,
               "FEA2_MLP_R does not match the rank the weights were exported at - re-run export_golden_*.py --mlp_rank");

static const float *s_x;                   /* caller's (L, D) window, read only */
static float s_mean[FEA2_D], s_stdev[FEA2_D];
#ifndef FEA2_FRONT_NONE
static float s_g[FEA2_NB];
#endif
static float s_U[FEA2_L * FEA2_MLP_R];     /* the only buffer that scales with R  */
static float s_row[FEA2_L];                /* one sensor's x_norm                  */
static float s_col[FEA2_L];                /* one sensor's hidden -> h_agg         */
/* pass 0 needs re/im/power beside s_row; pass 1/2 need the pooled LOW branch and
 * the agg-conv copy.  They never coexist, so they share one buffer. */
#define FEA2R_SCR_N FEA2_MAX2(3 * FEA2_NF, FEA2_MAX2(FEA2_LP, FEA2_MAX2(FEA2_L, FEA2_PERIOD)))
static float s_scr[FEA2R_SCR_N];
static int s_cur = -1;

static void norm_row(int d)
{
    for (int t = 0; t < FEA2_L; ++t)
        s_row[t] = (s_x[t * FEA2_D + d] - s_mean[d]) / s_stdev[d];
}

#ifdef FEA2_FRONT_NONE
/* no bands, no gate: the hidden row of sensor d is its x_norm row */
static void gated_bands(int d)
{
    (void)d;
    for (int t = 0; t < FEA2_L; ++t)
        s_col[t] = s_row[t];
}
#else

/* col[t] += scale * (sum_j w[d][j] row[t+j-pad] + b[d]), 'same' zero padding */
static void dwconv_acc(const float *row, float *col, int d,
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
        col[t] += scale * acc;
    }
}

static void interp_acc(const float *src, float *col, int len_in, int len_out, float scale)
{
    const float sc = (float)len_in / (float)len_out;
    for (int t = 0; t < len_out; ++t) {
        float p = ((float)t + 0.5f) * sc - 0.5f;
        if (p < 0.0f) p = 0.0f;
        int i0 = (int)p;
        int i1 = (i0 + 1 < len_in) ? i0 + 1 : len_in - 1;
        float f = p - (float)i0;
        col[t] += scale * (src[i0] * (1.0f - f) + src[i1] * f);
    }
}

/* h_d = g0*POINT + g1*MID + g2*LOW of the x_norm row in s_row, into s_col */
static void gated_bands(int d)
{
    for (int t = 0; t < FEA2_L; ++t)
        s_col[t] = 0.0f;
    dwconv_acc(s_row, s_col, d, W_CONVP, B_CONVP, FEA2_KP, s_g[0]);
    dwconv_acc(s_row, s_col, d, W_CONVM, B_CONVM, FEA2_KM, s_g[1]);
    fea_avgpool4(s_row, s_scr, FEA2_L, 1);
    interp_acc(s_scr, s_col, FEA2_LP, FEA2_L, s_g[2]);
}

static void banded_gate(void)
{
    float *re = s_scr, *im = s_scr + FEA2_NF, *P = s_scr + 2 * FEA2_NF;
    for (int f = 0; f < FEA2_NF; ++f)
        P[f] = 0.0f;
    for (int d = 0; d < FEA2_D; ++d) {
        norm_row(d);
        fea_rfft96_fast(s_row, 1, FEA2_NF, re, im,   /* s_col is free in pass 0 */
                        s_col, s_col + 32);
        for (int f = 0; f < FEA2_NF; ++f)
            P[f] += re[f] * re[f] + im[f] * im[f];
    }
    for (int f = 0; f < FEA2_NF; ++f)
        P[f] /= (float)FEA2_D;
    float total = 1e-6f;
    for (int f = 0; f < FEA2_NF; ++f)
        total += P[f];
    const int step = FEA2_NF / FEA2_NB;
    float feats[FEA2_NB];
    for (int s = 0; s < FEA2_NB; ++s) {
        int lo = s * step;
        int hi = (s == FEA2_NB - 1) ? FEA2_NF : (s + 1) * step;
        float e = 0.0f;
        for (int f = lo; f < hi; ++f)
            e += P[f];
        feats[FEA2_NB - 1 - s] = logf(e / total + 1e-6f);
    }
    float m = 0.0f;
    for (int s = 0; s < FEA2_NB; ++s)
        m += feats[s];
    m /= (float)FEA2_NB;
    for (int s = 0; s < FEA2_NB; ++s)
        feats[s] -= m;
    float z[FEA2_NB];
    for (int i = 0; i < FEA2_NB; ++i) {
        float acc = B_GATE2[i];
        for (int j = 0; j < FEA2_NB; ++j)
            acc += W_GATE2[i * FEA2_NB + j] * feats[j];
        z[i] = acc;
    }
    fea_softmax(z, s_g, FEA2_NB);
}
#endif /* FEA2_FRONT_NONE */

void fea2r_begin(const float *x)
{
    s_x = x;
    s_cur = -1;
    fea_instnorm_stats(x, FEA2_L, FEA2_D, s_mean, s_stdev);
#ifndef FEA2_FRONT_NONE
    banded_gate();                                   /* pass 0 */
#endif

    for (int t = 0; t < FEA2_L; ++t)                 /* pass 1 */
        for (int r = 0; r < FEA2_MLP_R; ++r)
            s_U[t * FEA2_MLP_R + r] = B_NLDOWN2[r];
    for (int d = 0; d < FEA2_D; ++d) {
        norm_row(d);
        gated_bands(d);
        for (int t = 0; t < FEA2_L; ++t)
            for (int r = 0; r < FEA2_MLP_R; ++r)
                s_U[t * FEA2_MLP_R + r] += W_NLDOWN2[r * FEA2_D + d] * s_col[t];
    }
    for (int i = 0; i < FEA2_L * FEA2_MLP_R; ++i)
        s_U[i] = fea_gelu(s_U[i]);
}

void fea2r_channel(int d)
{
    norm_row(d);
    gated_bands(d);                                  /* second time for this sensor */
    for (int t = 0; t < FEA2_L; ++t) {
        float acc = B_NLUP2[d];
        for (int r = 0; r < FEA2_MLP_R; ++r)
            acc += W_NLUP2[d * FEA2_MLP_R + r] * s_U[t * FEA2_MLP_R + r];
        s_col[t] += acc;
    }
    const int agg_pad = FEA2_AGG_K / 2;
    float *keep = s_scr;
    for (int t = 0; t < FEA2_L; ++t)
        keep[t] = s_col[t];
    for (int t = 0; t < FEA2_L; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < FEA2_AGG_K; ++j) {
            int s = t + j - agg_pad;
            if (s >= 0 && s < FEA2_L)
                acc += W_AGG2[d * FEA2_AGG_K + j] * keep[s];
        }
        s_col[t] = acc + keep[t];
    }
    s_cur = d;
}

/* one output period of the current sensor; valid until the next call */
const float *fea2r_block(int om)
{
    float *blk = s_scr;
    for (int ph = 0; ph < FEA2_PERIOD; ++ph) {
        float acc = 0.0f;
        for (int in = 0; in < FEA2_N; ++in)
            acc += W_BB2[om * FEA2_N + in] * s_col[in * FEA2_PERIOD + ph];
        blk[ph] = acc * s_stdev[s_cur] + s_mean[s_cur];
    }
    return blk;
}

void fea2r_forward(const float *x, float *y)
{
    fea2r_begin(x);
    for (int d = 0; d < FEA2_D; ++d) {
        fea2r_channel(d);
        for (int om = 0; om < FEA2_M; ++om) {
            const float *blk = fea2r_block(om);
            for (int ph = 0; ph < FEA2_PERIOD; ++ph)
                y[((size_t)om * FEA2_PERIOD + ph) * FEA2_D + d] = blk[ph];
        }
    }
}
