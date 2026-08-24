/*
 * Shared float32 primitives - see fea_kernels.h. Numerical conventions
 * (padding, trims, eps placement, DFT twiddles) are pinned by the golden
 * tests in test/; do not "simplify" without re-running them.
 */
#include <math.h>

#include "fea_kernels.h"
#include "fea_twiddle.h"       /* generated: golden/fea_twiddle.h */

void fea_dwconv_same(const float *x, float *y, const float *w,
                     const float *b, int len, int nch, int k)
{
    int pad = k / 2;
    for (int d = 0; d < nch; ++d) {
        for (int t = 0; t < len; ++t) {
            float acc = b ? b[d] : 0.0f;
            for (int j = 0; j < k; ++j) {
                int src = t + j - pad;
                if (src >= 0 && src < len)
                    acc += w[d * k + j] * x[src * nch + d];
            }
            y[t * nch + d] = acc;
        }
    }
}

void fea_dwconv_causal(const float *x, float *y, const float *w,
                       int len, int nch, int k)
{
    /* PyTorch: conv1d(padding=k-1) then trim [:L]  =>  output t sees
     * inputs t-(k-1) .. t only. */
    for (int s = 0; s < nch; ++s) {
        for (int t = 0; t < len; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) {
                int src = t + j - (k - 1);
                if (src >= 0)
                    acc += w[s * k + j] * x[src * nch + s];
            }
            y[t * nch + s] = acc;
        }
    }
}

void fea_linear_t(const float *x, float *y, const float *w,
                  const float *b, int len, int nin, int nout)
{
    for (int t = 0; t < len; ++t) {
        for (int o = 0; o < nout; ++o) {
            float acc = b ? b[o] : 0.0f;
            for (int i = 0; i < nin; ++i)
                acc += w[o * nin + i] * x[t * nin + i];
            y[t * nout + o] = acc;
        }
    }
}

void fea_avgpool4(const float *x, float *y, int len, int nch)
{
    int lo = len / 4;
    for (int d = 0; d < nch; ++d)
        for (int p = 0; p < lo; ++p) {
            float acc = 0.0f;
            for (int j = 0; j < 4; ++j)
                acc += x[(4 * p + j) * nch + d];
            y[p * nch + d] = acc * 0.25f;
        }
}

void fea_interp_linear(const float *x, float *y,
                       int len_in, int len_out, int nch)
{
    /* src = (dst + 0.5) * scale - 0.5, clamped at 0
     * (align_corners=False, matches ATen upsample_linear1d). */
    float scale = (float)len_in / (float)len_out;
    for (int t = 0; t < len_out; ++t) {
        float src = ((float)t + 0.5f) * scale - 0.5f;
        if (src < 0.0f) src = 0.0f;
        int i0 = (int)src;
        int i1 = (i0 + 1 < len_in) ? i0 + 1 : len_in - 1;
        float f = src - (float)i0;
        for (int d = 0; d < nch; ++d)
            y[t * nch + d] = x[i0 * nch + d] * (1.0f - f)
                           + x[i1 * nch + d] * f;
    }
}

void fea_rfft96(const float *x, int stride, float *re, float *im)
{
    /* exact 96-point DFT, twiddles from the flash table
     * (NO 128 zero-padding - EDGE_EXPERIMENT.md decision #2). */
    for (int f = 0; f < FEA_DFT_BINS; ++f) {
        float ar = 0.0f, ai = 0.0f;
        for (int t = 0; t < FEA_DFT_N; ++t) {
            int idx = (f * t) % FEA_DFT_N;
            float v = x[t * stride];
            ar += v * FEA_COS96[idx];
            ai -= v * FEA_SIN96[idx];
        }
        re[f] = ar;
        im[f] = ai;
    }
}

void fea_dft_half(const float *x, int stride, int n,
                  const float *cos_tab, const float *sin_tab,
                  float *re, float *im)
{
    int bins = n / 2 + 1;
    for (int f = 0; f < bins; ++f) {
        float ar = 0.0f, ai = 0.0f;
        for (int t = 0; t < n; ++t) {
            int idx = (int)(((long)f * t) % n);
            float v = x[t * stride];
            ar += v * cos_tab[idx];
            ai -= v * sin_tab[idx];
        }
        re[f] = ar;
        im[f] = ai;
    }
}

void fea_idft_half(const float *re, const float *im, int n,
                   const float *cos_tab, const float *sin_tab,
                   float *out)
{
    /* x[t] = (1/n) * [ re[0]
     *                + 2*sum_{f=1}^{mid-1} (re[f] cos - im[f] sin)
     *                + (n even ? re[n/2] * (-1)^t : last f doubled) ] */
    int bins = n / 2 + 1;
    int even = (n % 2 == 0);
    float inv_n = 1.0f / (float)n;
    for (int t = 0; t < n; ++t) {
        float acc = re[0];
        int last = even ? bins - 1 : bins;   /* exclusive of Nyquist */
        for (int f = 1; f < last; ++f) {
            int idx = (int)(((long)f * t) % n);
            acc += 2.0f * (re[f] * cos_tab[idx] - im[f] * sin_tab[idx]);
        }
        if (even)
            acc += re[bins - 1] * ((t & 1) ? -1.0f : 1.0f);
        out[t] = acc * inv_n;
    }
}

void fea_softmax(const float *x, float *y, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        y[i] = expf(x[i] - mx);
        sum += y[i];
    }
    for (int i = 0; i < n; ++i)
        y[i] /= sum;
}

float fea_gelu(float x)
{
    /* 0.5*x*(1+erf(x/sqrt(2))) — torch's default (approximate='none').
     * 1/sqrt(2) written out so the constant folds identically on both
     * toolchains under -fno-fast-math. */
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

void fea_instnorm_stats(const float *x, int len, int nch,
                        float *mean, float *stdev)
{
    for (int d = 0; d < nch; ++d) {
        float m = 0.0f;
        for (int t = 0; t < len; ++t)
            m += x[t * nch + d];
        m /= (float)len;
        mean[d] = m;
        if (stdev) {
            float v = 0.0f;
            for (int t = 0; t < len; ++t) {
                float c = x[t * nch + d] - m;
                v += c * c;
            }
            v /= (float)len;                /* biased (unbiased=False) */
            stdev[d] = sqrtf(v + 1e-5f);    /* eps inside sqrt         */
        }
    }
}
