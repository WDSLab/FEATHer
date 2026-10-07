/*
 * x86 golden-vector test for the PhaseFormer port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>

#include "phaseformer_infer.h"
#include "phaseformer_vectors.h"

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
        printf("FAIL  %-6s n=%d bad=%d first@%d got=%.6e ref=%.6e max_abs=%.2e\n",
               name, n, bad, bad_idx,
               (double)got[bad_idx], (double)ref[bad_idx], (double)max_abs);
    } else {
        printf("PASS  %-6s n=%d max_abs=%.2e max_rel=%.2e\n",
               name, n, (double)max_abs, (double)max_rel);
    }
}

/* the golden vectors are captured at one width; PhaseFormer's weights are
 * shared across channels, so only the test needs it */
_Static_assert(PF_D == PF_D_GEN,
               "PF_D does not match the width the golden vectors were exported at - re-run export_golden_phaseformer.py --d_model");

int main(void)
{
    static float y[PF_H * PF_D];
    phaseformer_forward(PG_X, y);

    printf("PhaseFormer L=%d H=%d D=%d period=%d layers=%d latent=%d routers=%d heads=%d pos=%d\n",
           PF_L, PF_H, PF_D, PF_P, PF_NL, PF_DL, PF_R, PF_NH, PF_PE);
    check("mu",    pf_st_mu,    PG_MU,    PF_D);
    check("sigma", pf_st_sigma, PG_SIGMA, PF_D);
    check("z0",    pf_st_z0,    PG_Z0,    PF_D * PF_P * PF_DL);
    check("zl",    pf_st_zl,    PG_ZL,    PF_D * PF_P * PF_DL);
    check("y",     y,           PG_Y,     PF_H * PF_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nPhaseFormer ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
