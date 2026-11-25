/* PyTorch 2.5.1 DEFAULT-build CPU kernels (no SIMD; Vectorized<float> = 8 lanes done one by one), ported step by step
 * so that the same arithmetic runs in the same order (checked bit-exact against torch: test/test_pt.c + test/test_pt_default.py).
 * Strides in elements.  Reductions: shape / stride of the PyTorch tensor, red = bit mask of the reduced dims,
 * out = the result, contiguous (keepdim layout). */
#ifndef MCU_PT_H
#define MCU_PT_H

#define PT_MAXD 4

void pt_sum(const float *x, float *out, int nd, const int *shape, const int *stride, unsigned red);
/* only the outputs [lo, hi) of the whole tensor's reduction (same arithmetic for each): x[g - xoff] = element g,
 * out[o - lo] = output o */
void pt_sum_part(const float *x, int xoff, float *out, int nd, const int *shape, const int *stride, unsigned red,
                 int lo, int hi);
void pt_mean_part(const float *x, int xoff, float *out, int nd, const int *shape, const int *stride, unsigned red,
                  int lo, int hi);
void pt_mean(const float *x, float *out, int nd, const int *shape, const int *stride, unsigned red);
void pt_mean_t(const float *x, float *out, int l, int d);              /* x (1, l, d) contiguous, mean over dim 1 */
/* var over one dim: row r starts at x + r * rstep, n values stride apart (WelfordOps in double) */
void pt_var(const float *x, float *out, int rows, int rstep, int n, int stride, int correction);
void pt_layer_norm(const float *x, float *y, int rows, int n, const float *g, const float *b, float eps);
void pt_softmax(const float *x, float *y, int rows, int k);            /* last dim, contiguous rows */
float pt_norm2(const float *x, int n);                                 /* ord 2 over one contiguous row */

#endif
