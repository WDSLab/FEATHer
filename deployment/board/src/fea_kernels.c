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

void fea_dft_bins(const float *x, int len, int offset, int n, int nbins,
                  const float *cos_tab, const float *sin_tab,
                  float *re, float *im)
{
    /* The first nbins bins of the n-point DFT of a signal that is zero
     * everywhere except x[0..len), which sits at sample `offset`.  Same
     * per-term arithmetic as fea_dft_half / fea_rfft96; the zero samples are
     * simply not visited, so no padded copy is built and nothing is computed
     * for them. */
    for (int f = 0; f < nbins; ++f) {
        float ar = 0.0f, ai = 0.0f;
        for (int i = 0; i < len; ++i) {
            int idx = (int)(((long)f * (offset + i)) % n);
            float v = x[i];
            ar += v * cos_tab[idx];
            ai -= v * sin_tab[idx];
        }
        re[f] = ar;
        im[f] = ai;
    }
}

float fea_idft_half_at(const float *re, const float *im, int n, int nbins,
                       const float *cos_tab, const float *sin_tab, int t)
{
    /* Sample t of fea_idft_half, for a spectrum whose bins >= nbins are all
     * zero (nbins = n/2+1 for a full spectrum).  Same terms in the same order
     * as fea_idft_half; the zero bins are skipped instead of summed, and only
     * the requested sample is evaluated. */
    int bins = n / 2 + 1;
    int even = (n % 2 == 0);
    int last = even ? bins - 1 : bins;       /* exclusive of Nyquist */
    int stop = nbins < last ? nbins : last;
    float inv_n = 1.0f / (float)n;
    float acc = re[0];
    for (int f = 1; f < stop; ++f) {
        int idx = (int)(((long)f * t) % n);
        acc += 2.0f * (re[f] * cos_tab[idx] - im[f] * sin_tab[idx]);
    }
    if (even && nbins >= bins)
        acc += re[bins - 1] * ((t & 1) ? -1.0f : 1.0f);
    return acc * inv_n;
}

void fea_rfft96_bins(const float *x, int nbins, float *re, float *im)
{
    /* fea_dft_bins over the exact 96-point grid, using this file's twiddle
     * table.  A model that referenced FEA_COS96 / FEA_SIN96 itself carried a
     * second 768 B copy of them in its own object (found 2026-09-14). */
    fea_dft_bins(x, FEA_DFT_N, 0, FEA_DFT_N, nbins,
                 FEA_COS96, FEA_SIN96, re, im);
}

void fea_irfft96(const float *re, const float *im, float *out)
{
    /* fea_idft_half at n = 96 with this file's twiddle table (see above) */
    fea_idft_half(re, im, FEA_DFT_N, FEA_COS96, FEA_SIN96, out);
}

/* ---- fast 96-point transforms: 96 = 3 x 32 (see fea_kernels.h) ---- */

/* in-place radix-2 32-point complex FFT; inverse = conjugate twiddles, no
 * 1/N.  Twiddles W32^j = W96^(3j) come from the same flash table. */
static void fea_fft32(float *zr, float *zi, int inverse)
{
    for (int i = 1, j = 0; i < 32; ++i) {          /* bit reversal */
        int bit = 16;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j) {
            float t = zr[i]; zr[i] = zr[j]; zr[j] = t;
            t = zi[i]; zi[i] = zi[j]; zi[j] = t;
        }
    }
    for (int len = 2; len <= 32; len <<= 1) {
        const int half = len >> 1;
        const int step = FEA_DFT_N / len;          /* W_len^j = W96^(j*96/len) */
        for (int i = 0; i < 32; i += len) {
            for (int j = 0; j < half; ++j) {
                const float c = FEA_COS96[j * step];
                const float s = inverse ? FEA_SIN96[j * step] : -FEA_SIN96[j * step];
                const int a = i + j, b = a + half;
                const float tr = zr[b] * c - zi[b] * s;
                const float ti = zr[b] * s + zi[b] * c;
                zr[b] = zr[a] - tr;
                zi[b] = zi[a] - ti;
                zr[a] += tr;
                zi[a] += ti;
            }
        }
    }
}

void fea_rfft96_fast(const float *x, int stride, int nbins,
                     float *re, float *im, float *wr, float *wi)
{
#ifdef FEA_DFT_DIRECT
    (void)wr; (void)wi;
    for (int f = 0; f < nbins; ++f) {               /* = fea_rfft96, nbins bins */
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
#else
    /* X[k] = sum_r W96^(r k) Z_r[k mod 32],  Z_r = FFT32(x[3m + r]) */
    for (int k = 0; k < nbins; ++k)
        re[k] = im[k] = 0.0f;
    for (int r = 0; r < 3; ++r) {
        for (int m = 0; m < 32; ++m) {
            wr[m] = x[(3 * m + r) * stride];
            wi[m] = 0.0f;
        }
        fea_fft32(wr, wi, 0);
        for (int k = 0; k < nbins; ++k) {
            const int q = k & 31;
            const int idx = (r * k) % FEA_DFT_N;
            const float c = FEA_COS96[idx], s = FEA_SIN96[idx];   /* W = c - i s */
            re[k] += wr[q] * c + wi[q] * s;
            im[k] += wi[q] * c - wr[q] * s;
        }
    }
#endif
}

void fea_irfft96_fast(const float *re, const float *im, float *out,
                      float *wr, float *wi)
{
#ifdef FEA_DFT_DIRECT
    (void)wr; (void)wi;
    fea_irfft96(re, im, out);
#else
    /* torch.fft.irfft semantics: Hermitian half-spectrum, imag of DC and of
     * the Nyquist bin ignored.  x[t] = (1/96) sum_r Re(e^(+i 2pi r t/96)
     * U_r[t mod 32]),  U_r = IFFT32(X[3m + r]) over the full spectrum. */
    for (int t = 0; t < FEA_DFT_N; ++t)
        out[t] = 0.0f;
    for (int r = 0; r < 3; ++r) {
        for (int m = 0; m < 32; ++m) {
            const int k = 3 * m + r;
            if (k == 0 || k == FEA_DFT_N / 2) {
                wr[m] = re[k];
                wi[m] = 0.0f;
            } else if (k < FEA_DFT_N / 2) {
                wr[m] = re[k];
                wi[m] = im[k];
            } else {
                wr[m] = re[FEA_DFT_N - k];
                wi[m] = -im[FEA_DFT_N - k];
            }
        }
        fea_fft32(wr, wi, 1);
        for (int t = 0; t < FEA_DFT_N; ++t) {
            const int q = t & 31;
            const int idx = (r * t) % FEA_DFT_N;
            out[t] += wr[q] * FEA_COS96[idx] - wi[q] * FEA_SIN96[idx];
        }
    }
    const float inv_n = 1.0f / (float)FEA_DFT_N;
    for (int t = 0; t < FEA_DFT_N; ++t)
        out[t] *= inv_n;
#endif
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
