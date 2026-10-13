/*
 * x86 golden-vector test for the DiPE-Linear companion port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dipe_infer.h"
#include "dipe_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(DIPE_D == DIPE_D_GEN,
               "DIPE_D does not match the width the golden vectors were exported at - re-run export_golden_dipe.py --d_model");

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

/* 10-13: the per-value emit API must hand out exactly the buffered values (bit for bit),
 * each (d, t) once.  Rebuilt time-major from the sink and compared with memcmp. */
static float g_ye[DIPE_H * DIPE_D];
static int g_cnt[DIPE_H * DIPE_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * DIPE_D + d] = v;
    g_cnt[t * DIPE_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < DIPE_D; ++d)
        dipe_forward_emit(DG_X, d, emit_sink);
    for (int i = 0; i < DIPE_H * DIPE_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", DIPE_H * DIPE_D);
    return 0;
}

int main(void)
{
    static float y[DIPE_H * DIPE_D];
    dipe_forward(DG_X, y);
    g_fail += check_emit(y);

    check("mean", dipe_st_mean, DG_MEAN, DIPE_D);
    check("std",  dipe_st_std,  DG_STD,  DIPE_D);
    check("xn",   dipe_st_xn,   DG_XN,   DIPE_D * DIPE_L);
    check("xf",   dipe_st_xf,   DG_XF,   DIPE_D * DIPE_L);  /* irfft96  */
    check("xt",   dipe_st_xt,   DG_XT,   DIPE_D * DIPE_L);
    check("y",    y,            DG_Y,    DIPE_H * DIPE_D);  /* 193-pt   */

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nDiPE-Linear ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
