/*
 * x86 golden-vector test for the FITS companion port.
 * Same mixed tolerance as test_golden.c (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "fits_infer.h"
#include "fits_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(FITS_D == FITS_D_GEN,
               "FITS_D does not match the width the golden vectors were exported at - re-run export_golden_fits.py --d_model");

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
static float g_ye[FITS_H * FITS_D];
static int g_cnt[FITS_H * FITS_D];
static void emit_sink(int d, int t, float v)
{
    g_ye[t * FITS_D + d] = v;
    g_cnt[t * FITS_D + d]++;
}

static int check_emit(const float *y)
{
    for (int d = 0; d < FITS_D; ++d)
        fits_forward_emit(FG_X, d, emit_sink);
    for (int i = 0; i < FITS_H * FITS_D; ++i)
        if (g_cnt[i] != 1) {
            printf("FAIL  emit   cell %d emitted %d times\n", i, g_cnt[i]);
            return 1;
        }
    if (memcmp(g_ye, y, sizeof g_ye) != 0) {
        printf("FAIL  emit   values differ from the buffered forward\n");
        return 1;
    }
    printf("PASS  emit   n=%d bit-identical to the buffered forward\n", FITS_H * FITS_D);
    return 0;
}

int main(void)
{
    static float y[FITS_H * FITS_D];
    fits_forward(FG_X, y);
    g_fail += check_emit(y);

    check("mean",  fits_st_mean,  FG_MEAN,  FITS_D);
    check("sd",    fits_st_sd,    FG_SD,    FITS_D);
    check("xn",    fits_st_xn,    FG_XN,    FITS_L * FITS_D);
    check("up_re", fits_st_up_re, FG_UP_RE, FITS_D * FITS_UP);
    check("up_im", fits_st_up_im, FG_UP_IM, FITS_D * FITS_UP);
    check("y",     y,             FG_Y,     FITS_H * FITS_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nFITS ALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
