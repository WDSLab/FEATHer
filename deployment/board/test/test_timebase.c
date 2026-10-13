/*
 * x86 golden-vector test for the TimeBase port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "timebase_infer.h"
#include "timebase_vectors.h"

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
static float g_ye[TB_H * TB_D];
static int g_cnt[TB_H * TB_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * TB_D + d] = v;
    g_cnt[t * TB_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < TB_D; ++d)
        timebase_forward_emit(TG_X, d, emit_sink);
    for (int i = 0; i < TB_H * TB_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", TB_H * TB_D);
    return 0;
}

int main(void)
{
    static float y[TB_H * TB_D];
    timebase_forward(TG_X, y);
    g_fail += check_emit(y);

    printf("TimeBase L=%d H=%d D=%d period=%d basis=%d individual=%d\n",
           TB_L, TB_H, TB_D, TB_P, TB_B, TB_IND);
    check("pmean", tb_st_pmean, TG_PMEAN, TB_D * TB_P);
    check("basis", tb_st_basis, TG_BASIS, TB_D * TB_P * TB_B);
    check("y",     y,           TG_Y,     TB_H * TB_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nTimeBase ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
