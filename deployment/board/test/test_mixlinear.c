/*
 * x86 golden-vector test for the MixLinear companion port.
 * Mixed tolerance atol=1e-5, rtol=1e-4 (same as the other model tests).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "mixlinear_infer.h"
#include "mixlinear_vectors.h"

/* The golden vectors are regenerated per (horizon, width).  Comparing a
 * port built at one width against vectors captured at another compiles
 * fine and reports a numerical failure that looks like a code bug, so
 * refuse it at compile time - the guard the other ports carry on H. */
_Static_assert(ML_D == ML_D_GEN,
               "ML_D does not match the width the golden vectors were exported at - re-run export_golden_mixlinear.py --d_model");
_Static_assert(ML_H == ML_H_GEN,
               "ML_H does not match the horizon the golden vectors were exported at - re-run export_golden_mixlinear.py --pred_len");

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
static float g_ye[ML_H * ML_D];
static int g_cnt[ML_H * ML_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * ML_D + d] = v;
    g_cnt[t * ML_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < ML_D; ++d)
        mixlinear_forward_emit(MG_X, d, emit_sink);
    for (int i = 0; i < ML_H * ML_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", ML_H * ML_D);
    return 0;
}

int main(void)
{
    static float y[ML_H * ML_D];
    mixlinear_forward(MG_X, y);
    g_fail += check_emit(y);

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
