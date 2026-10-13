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
#include <string.h>

#include "stsf_mlp_infer.h"
#include "stsf_mlp_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(SMLP_D == SMLP_D_GEN,
               "SMLP_D does not match the width the golden vectors were exported at - re-run export_golden_stsf_mlp.py --d_model");

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

/* 10-13: the per-value emit API must hand out exactly the buffered values (bit for bit),
 * each (d, t) once.  Rebuilt time-major from the sink and compared with memcmp. */
static float g_ye[SMLP_H * SMLP_D];
static int g_cnt[SMLP_H * SMLP_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * SMLP_D + d] = v;
    g_cnt[t * SMLP_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < SMLP_D; ++d)
        stsf_mlp_forward_emit(SM_X, d, emit_sink);
    for (int i = 0; i < SMLP_H * SMLP_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", SMLP_H * SMLP_D);
    return 0;
}

int main(void)
{
    static float y[SMLP_H * SMLP_D];
    stsf_mlp_forward(SM_X, y);
    g_fail += check_emit(y);

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
