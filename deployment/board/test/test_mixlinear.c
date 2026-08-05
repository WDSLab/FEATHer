/*
 * x86 golden-vector test for the MixLinear companion port.
 * Mixed tolerance atol=1e-5, rtol=1e-4 (same as the other model tests).
 */
#include <math.h>
#include <stdio.h>

#include "mixlinear_infer.h"
#include "mixlinear_vectors.h"

#define ATOL 1e-5f
#define RTOL 1e-4f

static int g_fail = 0;

static void check(const char *name, const float *got,
                  const float *ref, int n)
{
    int bad = 0, bad_idx = -1;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ad = fabsf(got[i] - ref[i]);
        float rd = ad / (fabsf(ref[i]) + 1e-12f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (ad > ATOL + RTOL * fabsf(ref[i])) {
            if (!bad) bad_idx = i;
            ++bad;
        }
    }
    if (bad) {
        ++g_fail;
        printf("FAIL  %-5s n=%d bad=%d first@%d got=%.6e ref=%.6e\n",
               name, n, bad, bad_idx,
               (double)got[bad_idx], (double)ref[bad_idx]);
    } else {
        printf("PASS  %-5s n=%d max_abs=%.2e max_rel=%.2e\n",
               name, n, (double)max_abs, (double)max_rel);
    }
}

int main(void)
{
    static float y[ML_H * ML_D];
    mixlinear_forward(MG_X, y);

    check("mean", mixlinear_st_mean, MG_MEAN, ML_D);
    check("xt",   mixlinear_st_xt,   MG_XT,   ML_H * ML_D);
    check("xf",   mixlinear_st_xf,   MG_XF,   ML_H * ML_D);
    check("y",    y,                 MG_Y,    ML_H * ML_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nMixLinear ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
