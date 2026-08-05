/*
 * Shared float32 primitives for all on-board models (FEATHer, SparseTSF,
 * DiPE-Linear...). Weight-free: everything is passed by pointer, so each
 * model's firmware image links only its own flash constants.
 *
 * Compile with -O2 -ffp-contract=off -fno-fast-math on every target.
 */
#ifndef FEA_KERNELS_H
#define FEA_KERNELS_H

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

/* softmax over n elements (max-subtracted, matches torch.softmax)   */
void fea_softmax(const float *x, float *y, int n);

/* per-channel mean + stdev over time; biased var, eps inside sqrt:
 * stdev = sqrt(var + 1e-5). stdev may be NULL (mean-only models)    */
void fea_instnorm_stats(const float *x, int len, int nch,
                        float *mean, float *stdev);

#endif /* FEA_KERNELS_H */
