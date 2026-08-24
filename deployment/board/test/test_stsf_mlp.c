/*
 * x86 golden-vector test for the SparseTSF-MLP port: run the C forward once
 * and compare every stage against the PyTorch dump
 * (golden/stsf_mlp_vectors.h).
 *
 * Pass criterion (EDGE_EXPERIMENT.md decision #4): element-wise mixed
 * tolerance  |c - ref| <= atol + rtol * |ref|,  atol=1e-5, rtol=1e-4.
 *
 * Build & run:  make -C deployment/board build/test_stsf_mlp && \
 *               deployment/board/build/test_stsf_mlp
 */
#include <math.h>
#include <stdio.h>

#include "stsf_mlp_infer.h"
#include "stsf_mlp_vectors.h"

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
        printf("FAIL  %-8s n=%d bad=%d first@%d got=%.6e ref=%.6e "
               "(max_abs=%.2e max_rel=%.2e)\n",
               name, n, bad, bad_idx,
               (double)got[bad_idx], (double)ref[bad_idx],
               (double)max_abs, (double)max_rel);
    } else {
        printf("PASS  %-8s n=%d max_abs=%.2e max_rel=%.2e\n",
               name, n, (double)max_abs, (double)max_rel);
    }
}

int main(void)
{
    static float y[SMLP_H * SMLP_D];
    stsf_mlp_forward(SM_X, y);

    check("mean", smlp_st_mean, SM_MEAN, SMLP_D);
    check("xagg", smlp_st_xagg, SM_XAGG, SMLP_D * SMLP_L);
    check("hid",  smlp_st_hid,  SM_HID,  SMLP_D * SMLP_P * SMLP_DM);
    check("y",    y,            SM_Y,    SMLP_H * SMLP_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nSparseTSF-MLP ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
