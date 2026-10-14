/*
 * FEATHer in1 inference - sensor-by-sensor streaming build with a per-sensor
 * banded gate.  See feather2in_infer.h for the model and the two schedules.
 *
 * Mirrors baselines/FEATHer/FEATHer.py forward() for
 *   use_in_proj=False, num_bands=3, point_kernel=3, dtk_mode="none",
 *   gate_mode="banded_sensor", use_residual_mlp=True, channel_mode="mix",
 *   mix_mode="input", mix_pool=1.
 */
#include <math.h>
#include <string.h>

#include "feather2in_infer.h"
#include "feather2in_weights.h"

_Static_assert(FEA2I_H == FEA2I_H_GEN,
               "FEA2I_H does not match the horizon the weights were exported at - re-run export_golden_feather2in.py --pred_len");
_Static_assert(FEA2I_R == FEA2I_R_GEN,
               "FEA2I_R does not match the summary size the weights were exported at - re-run export_golden_feather2in.py --mlp_rank");

#ifdef FEA2I_CAPTURE_STAGES
float fea2i_st_mean[FEA2I_D];
float fea2i_st_stdev[FEA2I_D];
float fea2i_st_power[FEA2I_D * FEA2I_NF];
float fea2i_st_feats[FEA2I_D * FEA2I_NB];
float fea2i_st_gate[FEA2I_D * FEA2I_NB];
float fea2i_st_u[FEA2I_L * FEA2I_R];
float fea2i_st_hidden[FEA2I_D * FEA2I_L];
float fea2i_st_mlp[FEA2I_D * FEA2I_L];
float fea2i_st_hagg[FEA2I_D * FEA2I_L];
#define CAP(dst, src, n) memcpy((dst), (src), (n) * sizeof(float))
#else
#define CAP(dst, src, n) ((void)0)
#endif

static float s_U[FEA2I_L * FEA2I_R];
#if FEA2I_RECOMPUTE_NORM
static float s_h[FEA2I_L];
#else
static float s_X[FEA2I_L * FEA2I_D];
#endif
static float s_scratch[FEA2I_SCRATCH_N];
static float s_mean[FEA2I_D], s_stdev[FEA2I_D];
static float s_g[FEA2I_NB];
static const float *s_x;
static int s_cur;

#if FEA2I_RECOMPUTE_NORM
#define COL(d)    (s_h)
#define STRIDE    1
#else
#define COL(d)    (s_X + (d))
#define STRIDE    FEA2I_D
#endif

static void dwconv_row_acc(const float *row, float *col, int stride, int d,
                           const float *w, const float *b, int k, float scale)
{
    const int pad = k / 2;
    for (int t = 0; t < FEA2I_L; ++t) {
        float acc = b ? b[d] : 0.0f;
        for (int j = 0; j < k; ++j) {
            int s = t + j - pad;
            if (s >= 0 && s < FEA2I_L)
                acc += w[d * k + j] * row[s];
        }
        col[t * stride] += scale * acc;
    }
}

static void interp_row_acc(const float *src, float *col, int stride,
                           int len_in, int len_out, float scale)
{
    const float sc = (float)len_in / (float)len_out;
    for (int t = 0; t < len_out; ++t) {
        float p = ((float)t + 0.5f) * sc - 0.5f;
        if (p < 0.0f) p = 0.0f;
        int i0 = (int)p;
        int i1 = (i0 + 1 < len_in) ? i0 + 1 : len_in - 1;
        float f = p - (float)i0;
        col[t * stride] += scale * (src[i0] * (1.0f - f) + src[i1] * f);
    }
}

/* Sensor d's banded gate from its own (strided) x_norm, into s_g:
 * power -> equal-width slice energy ratios -> centered log -> proj -> softmax.
 * `spec` is 3*NF floats of scratch that nothing else uses at this point. */
static void sensor_gate(const float *xn, int stride, int d, float *spec)
{
    float *re = spec, *im = spec + FEA2I_NF, *P = spec + 2 * FEA2I_NF;
    fea_rfft96(xn, stride, re, im);
    for (int f = 0; f < FEA2I_NF; ++f)
        P[f] = re[f] * re[f] + im[f] * im[f];
#ifdef FEA2I_CAPTURE_STAGES
    memcpy(fea2i_st_power + d * FEA2I_NF, P, FEA2I_NF * sizeof(float));
#endif
    float total = 1e-6f;
    for (int f = 0; f < FEA2I_NF; ++f)
        total += P[f];

    const int step = FEA2I_NF / FEA2I_NB;
    float feats[FEA2I_NB];
    for (int s = 0; s < FEA2I_NB; ++s) {
        int lo = s * step;
        int hi = (s == FEA2I_NB - 1) ? FEA2I_NF : (s + 1) * step;
        float e = 0.0f;
        for (int f = lo; f < hi; ++f)
            e += P[f];
        feats[FEA2I_NB - 1 - s] = logf(e / total + 1e-6f);
    }
    float m = 0.0f;
    for (int s = 0; s < FEA2I_NB; ++s)
        m += feats[s];
    m /= (float)FEA2I_NB;
    for (int s = 0; s < FEA2I_NB; ++s)
        feats[s] -= m;

    float z[FEA2I_NB];
    for (int i = 0; i < FEA2I_NB; ++i) {
        float acc = B_GATE[i];
        for (int j = 0; j < FEA2I_NB; ++j)
            acc += W_GATE[i * FEA2I_NB + j] * feats[j];
        z[i] = acc;
    }
    fea_softmax(z, s_g, FEA2I_NB);
#ifdef FEA2I_CAPTURE_STAGES
    memcpy(fea2i_st_feats + d * FEA2I_NB, feats, FEA2I_NB * sizeof(float));
    memcpy(fea2i_st_gate + d * FEA2I_NB, s_g, FEA2I_NB * sizeof(float));
#endif
}

#if FEA2I_RECOMPUTE_NORM
static void norm_col(int d, float *row)
{
    for (int t = 0; t < FEA2I_L; ++t)
        row[t] = (s_x[t * FEA2I_D + d] - s_mean[d]) / s_stdev[d];
}
#endif

/* fea2i_begin - statistics and the cross-sensor summary
 * u[t][r] = GELU(b[r] + sum_d W_down[r][d] x_norm[t][d]), additions in
 * ascending d in both schedules (bit-identical u).  No FFT here any more:
 * the gate is per sensor and is computed in fea2i_channel. */
void fea2i_begin(const float *x)
{
    s_x = x;
    s_cur = -1;
    fea_instnorm_stats(x, FEA2I_L, FEA2I_D, s_mean, s_stdev);
    CAP(fea2i_st_mean, s_mean, FEA2I_D);
    CAP(fea2i_st_stdev, s_stdev, FEA2I_D);

    for (int t = 0; t < FEA2I_L; ++t)
        for (int r = 0; r < FEA2I_R; ++r)
            s_U[t * FEA2I_R + r] = B_NLDOWN[r];

#if FEA2I_RECOMPUTE_NORM
    float *row = s_scratch;
    for (int d = 0; d < FEA2I_D; ++d) {
        norm_col(d, row);                         /* first of two passes */
        for (int t = 0; t < FEA2I_L; ++t)
            for (int r = 0; r < FEA2I_R; ++r)
                s_U[t * FEA2I_R + r] += W_NLDOWN[r * FEA2I_D + d] * row[t];
    }
#else
    for (int t = 0; t < FEA2I_L; ++t)
        for (int d = 0; d < FEA2I_D; ++d)
            s_X[t * FEA2I_D + d] = (x[t * FEA2I_D + d] - s_mean[d]) / s_stdev[d];
    for (int d = 0; d < FEA2I_D; ++d)
        for (int t = 0; t < FEA2I_L; ++t)
            for (int r = 0; r < FEA2I_R; ++r)
                s_U[t * FEA2I_R + r] += W_NLDOWN[r * FEA2I_D + d] * s_X[t * FEA2I_D + d];
#endif

    for (int i = 0; i < FEA2I_L * FEA2I_R; ++i)
        s_U[i] = fea_gelu(s_U[i]);
    CAP(fea2i_st_u, s_U, FEA2I_L * FEA2I_R);
}

/* fea2i_channel - sensor d: own gate, bands, correction, SPK aggregation. */
void fea2i_channel(int d)
{
    float *row  = s_scratch;
    float *pool = s_scratch + FEA2I_L;
    float *col  = COL(d);

#if FEA2I_RECOMPUTE_NORM
    norm_col(d, row);                             /* second pass */
    sensor_gate(row, 1, d, s_scratch + FEA2I_L);  /* spectrum beside the row */
    for (int t = 0; t < FEA2I_L; ++t)
        col[t] = 0.0f;
#else
    sensor_gate(s_X + d, FEA2I_D, d, s_scratch);  /* reads the kept x_norm */
    for (int t = 0; t < FEA2I_L; ++t) {
        row[t] = col[t * STRIDE];
        col[t * STRIDE] = 0.0f;
    }
#endif

    dwconv_row_acc(row, col, STRIDE, d, W_CONVP, B_CONVP, FEA2I_KP, s_g[0]);
    dwconv_row_acc(row, col, STRIDE, d, W_CONVM, B_CONVM, FEA2I_KM, s_g[1]);
    fea_avgpool4(row, pool, FEA2I_L, 1);
    interp_row_acc(pool, col, STRIDE, FEA2I_LP, FEA2I_L, s_g[2]);
#ifdef FEA2I_CAPTURE_STAGES
    for (int t = 0; t < FEA2I_L; ++t)
        fea2i_st_hidden[d * FEA2I_L + t] = col[t * STRIDE];
#endif

    for (int t = 0; t < FEA2I_L; ++t) {
        float acc = B_NLUP[d];
        for (int r = 0; r < FEA2I_R; ++r)
            acc += W_NLUP[d * FEA2I_R + r] * s_U[t * FEA2I_R + r];
        col[t * STRIDE] += acc;
    }
#ifdef FEA2I_CAPTURE_STAGES
    for (int t = 0; t < FEA2I_L; ++t)
        fea2i_st_mlp[d * FEA2I_L + t] = col[t * STRIDE];
#endif

    const int agg_pad = FEA2I_AGG_K / 2;
    for (int t = 0; t < FEA2I_L; ++t)
        row[t] = col[t * STRIDE];
    for (int t = 0; t < FEA2I_L; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < FEA2I_AGG_K; ++j) {
            int s = t + j - agg_pad;
            if (s >= 0 && s < FEA2I_L)
                acc += W_AGG[d * FEA2I_AGG_K + j] * row[s];
        }
        col[t * STRIDE] = acc + row[t];
    }
#ifdef FEA2I_CAPTURE_STAGES
    for (int t = 0; t < FEA2I_L; ++t)
        fea2i_st_hagg[d * FEA2I_L + t] = col[t * STRIDE];
#endif
    s_cur = d;
}

const float *fea2i_block(int om)
{
    const float *col = COL(s_cur);
    float *blk = s_scratch;
    for (int ph = 0; ph < FEA2I_PERIOD; ++ph) {
        float acc = 0.0f;
        for (int in = 0; in < FEA2I_N; ++in)
            acc += W_BB[om * FEA2I_N + in]
                 * col[(in * FEA2I_PERIOD + ph) * STRIDE];
        blk[ph] = acc * s_stdev[s_cur] + s_mean[s_cur];
    }
    return blk;
}

void fea2i_forward(const float *x, float *y)
{
    fea2i_begin(x);
    for (int d = 0; d < FEA2I_D; ++d) {
        fea2i_channel(d);
        for (int om = 0; om < FEA2I_M; ++om) {
            const float *blk = fea2i_block(om);
            for (int ph = 0; ph < FEA2I_PERIOD; ++ph)
                y[((size_t)om * FEA2I_PERIOD + ph) * FEA2I_D + d] = blk[ph];
        }
    }
}
