/*
 * Shared float32 primitives for all on-board models (FEATHer, SparseTSF,
 * DiPE-Linear...). Weight-free: everything is passed by pointer, so each
 * model's firmware image links only its own flash constants.
 *
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef FEA_KERNELS_H
#define FEA_KERNELS_H

/* Board builds: board_prep.py may define FEA_DFT_DIRECT in board_cfg.h (A/B
 * timing of the 96-point transforms); absent in the repo, so x86 tests and
 * measure_footprint.py never see it. */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#define FEA_DFT_N 96                    /* DFT length (= seq_len)  */
#define FEA_DFT_BINS (FEA_DFT_N / 2 + 1)

/* depthwise conv over time, 'same' zero padding; x,y are (len, nch)
 * time-major interleaved; w is (nch, k); b may be NULL              */
void fea_dwconv_same(const float *x, float *y, const float *w,
                     const float *b, int len, int nch, int k);

/* causal depthwise conv: PyTorch conv1d(padding=k-1) then [:len]    */
void fea_dwconv_causal(const float *x, float *y, const float *w,
                       int len, int nch, int k);

/* per-timestep linear: y[t] = W x[t] + b; W is (nout, nin)          */
void fea_linear_t(const float *x, float *y, const float *w,
                  const float *b, int len, int nin, int nout);

/* AvgPool1d(kernel=4, stride=4), (len, nch) -> (len/4, nch)         */
void fea_avgpool4(const float *x, float *y, int len, int nch);

/* F.interpolate(mode=linear, align_corners=False)                   */
void fea_interp_linear(const float *x, float *y,
                       int len_in, int len_out, int nch);

/* exact 96-point real DFT of one strided channel (NO zero padding)  */
void fea_rfft96(const float *x, int stride, float *re, float *im);

/* generic real DFT: n-point, n/2+1 bins; twiddle tables are n-entry
 * cos/sin of 2*pi*i/n (flash constants, emitted by the export script) */
void fea_dft_half(const float *x, int stride, int n,
                  const float *cos_tab, const float *sin_tab,
                  float *re, float *im);

/* inverse of fea_dft_half (torch.fft.irfft semantics: Hermitian
 * half-spectrum assumed; imag of DC / even-n Nyquist ignored)       */
void fea_idft_half(const float *re, const float *im, int n,
                   const float *cos_tab, const float *sin_tab,
                   float *out);

/* first nbins bins of the n-point DFT of a signal that is zero except
 * x[0..len) placed at sample `offset` (implicit zero padding: the zeros are
 * never stored or visited).  Same per-term arithmetic as fea_dft_half.   */
void fea_dft_bins(const float *x, int len, int offset, int n, int nbins,
                  const float *cos_tab, const float *sin_tab,
                  float *re, float *im);

/* sample t of fea_idft_half for a spectrum whose bins >= nbins are zero
 * (nbins = n/2+1 for a full one) - evaluates one output sample and skips
 * the zero bins, so a caller that keeps a slice computes only the slice.  */
float fea_idft_half_at(const float *re, const float *im, int n, int nbins,
                       const float *cos_tab, const float *sin_tab, int t);

/* 96-point transforms that keep the twiddle tables inside fea_kernels.o.
 * Models call these instead of passing FEA_COS96 / FEA_SIN96 themselves, so
 * the 768 B tables are not duplicated into every object that needs them.  */
void fea_rfft96_bins(const float *x, int nbins, float *re, float *im);
void fea_irfft96(const float *re, const float *im, float *out);

/* ---- fast 96-point transforms (2026-09-29) ----
 * Same results as fea_rfft96 / fea_rfft96_bins / fea_irfft96 up to float
 * rounding, via 96 = 3 x 32: three radix-2 32-point FFTs of the decimated
 * sequences, combined with W96^(r k).  About 5x fewer multiplies than the
 * direct form.  Every port that takes a full 96-point transform calls these
 * (FEATHer's gate, DiPE's frequency weight, FITS's analysis), so the speed-up
 * is the same for every model that has such a transform - transforms whose
 * length grows with H stay in the direct, one-sample-at-a-time form for
 * everyone (a fast one would need an O(H) buffer).
 *
 * wr / wi: 32 floats each of CALLER-provided scratch.  Each port passes memory
 * that is free at that moment, or a buffer it declares (and is charged for).
 * re / im / out / x must not overlap wr / wi.
 *
 * -DFEA_DFT_DIRECT routes both to the direct form (board A/B timing).     */
void fea_rfft96_fast(const float *x, int stride, int nbins,
                     float *re, float *im, float *wr, float *wi);
void fea_irfft96_fast(const float *re, const float *im, float *out,
                      float *wr, float *wi);

/* softmax over n elements (max-subtracted, matches torch.softmax)   */
void fea_softmax(const float *x, float *y, int n);

/* GELU, exact erf form: 0.5*x*(1+erf(x/sqrt(2))). This is what
 * torch.nn.functional.gelu computes by default (approximate='none') — the
 * cheaper tanh approximation is off by ~1e-3 near |x|~1 and would blow the
 * golden-vector tolerance (atol 1e-5). erff() is a libm call and costs
 * accordingly on soft-float, which is why the op counter charges it 15x. */
float fea_gelu(float x);

/* per-channel mean + stdev over time; biased var, eps inside sqrt:
 * stdev = sqrt(var + 1e-5). stdev may be NULL (mean-only models)    */
void fea_instnorm_stats(const float *x, int len, int nch,
                        float *mean, float *stdev);

#endif /* FEA_KERNELS_H */
