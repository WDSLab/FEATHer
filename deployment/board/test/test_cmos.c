/*
 * x86 golden-vector test for the CMoS port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cmos_infer.h"
#include "cmos_vectors.h"

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

/* 10-13: the per-value emit API must hand out exactly the buffered values (bit for bit),
 * each (d, t) once.  Rebuilt time-major from the sink and compared with memcmp. */
static float g_ye[CMOS_H * CMOS_D];
static int g_cnt[CMOS_H * CMOS_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * CMOS_D + d] = v;
    g_cnt[t * CMOS_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < CMOS_D; ++d)
        cmos_forward_emit(CG_X, d, emit_sink);
    for (int i = 0; i < CMOS_H * CMOS_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", CMOS_H * CMOS_D);
    return 0;
}

int main(void)
{
    static float y[CMOS_H * CMOS_D];
    cmos_forward(CG_X, y);
    g_fail += check_emit(y);

    printf("CMoS L=%d H=%d D=%d seg=%d num_map=%d kernel=%d stride=%d\n",
           CMOS_L, CMOS_H, CMOS_D, CMOS_SEG, CMOS_NMAP, CMOS_K, CMOS_STRIDE);
    check("mean", cmos_st_mean, CG_MEAN, CMOS_D);
    check("std",  cmos_st_std,  CG_STD,  CMOS_D);
    check("conv", cmos_st_conv, CG_CONV, CMOS_D * CMOS_CONV_DIM);
    check("gate", cmos_st_gate, CG_GATE, CMOS_D * CMOS_NMAP);
    check("y",    y,            CG_Y,    CMOS_H * CMOS_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nCMoS ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
