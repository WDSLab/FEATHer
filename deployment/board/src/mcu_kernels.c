/*
 * Shared float32 primitives - see mcu_kernels.h. Numerical conventions
 * (padding, trims, eps placement) are pinned by the golden
 * tests in test/; do not "simplify" without re-running them.
 */
#include <math.h>

#include "mcu_kernels.h"
#include "mcu_emit.h"              /* mcu_rt */

void mcu_dwconv_same(const float *x, float *y, const float *w,
                     const float *b, int len, int nch, int k)
{
    int pad = k / 2;
    const float zero = mcu_rt(0.0f);            /* the padding: multiplied like any sample, as PyTorch does */
    for (int d = 0; d < nch; ++d) {
        for (int t = 0; t < len; ++t) {
            float acc = b ? b[d] : 0.0f;
            for (int j = 0; j < k; ++j) {
                int src = t + j - pad;
                acc += w[d * k + j] * ((src >= 0 && src < len) ? x[src * nch + d] : zero);
            }
            y[t * nch + d] = acc;
        }
    }
}

void mcu_avgpool4(const float *x, float *y, int len, int nch)
{
    int lo = len / 4;
    const float n4 = mcu_rt(4.0f);                 /* avg_pool divides by the kernel size */
    const float zero = mcu_rt(0.0f);
    for (int d = 0; d < nch; ++d)
        for (int p = 0; p < lo; ++p) {
            float acc = 0.0f;
            for (int j = 0; j < 4; ++j)
                acc += x[(4 * p + j) * nch + d];
            y[p * nch + d] = zero + acc / n4;       /* ATen: output = 0, then output += sum / k */
        }
}

float mcu_gelu(float x)
{
    /* 0.5*x*(1+erf(x/sqrt(2))) — torch's default (approximate='none').
     * 1/sqrt(2) written out so the constant folds identically on both
     * toolchains under -fno-fast-math. */
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
