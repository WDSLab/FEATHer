/*
 * x86 golden-vector test for the SparseTSF companion port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>

#include "stsf_infer.h"
#include "stsf_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(STSF_D == STSF_D_GEN,
               "STSF_D does not match the width the golden vectors were exported at - re-run export_golden_stsf.py --d_model");

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
        printf("FAIL  %-6s n=%d bad=%d first@%d got=%.6e ref=%.6e\n",
               name, n, bad, bad_idx,
               (double)got[bad_idx], (double)ref[bad_idx]);
    } else {
        printf("PASS  %-6s n=%d max_abs=%.2e max_rel=%.2e\n",
               name, n, (double)max_abs, (double)max_rel);
    }
}

int main(void)
{
    static float y[STSF_H * STSF_D];
    stsf_forward(SG_X, y);

    check("mean", stsf_st_mean, SG_MEAN, STSF_D);
    check("xagg", stsf_st_xagg, SG_XAGG, STSF_D * STSF_L);
    check("y",    y,            SG_Y,    STSF_H * STSF_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nSparseTSF ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
