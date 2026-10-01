/*
 * Golden test for the DLinear port.
 *
 * Checks the decomposition (replicate padding is the trap — zero padding
 * would drag the trend toward 0 at both ends), the forecast, and the
 * step-wise streaming path.
 *
 * Build (x86): see Makefile target `dlinear`
 */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "dlinear_infer.h"
#include "dlinear_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(DL_D == DL_D_GEN,
               "DL_D does not match the width the golden vectors were exported at - re-run export_golden_dlinear.py --d_model");

static int g_fail = 0;

static void check(const char *name, const float *got, const float *want,
                  int n, float atol)
{
    float worst = 0.0f;
    int at = -1;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > worst || isnan(d)) { worst = d; at = i; }
    }
    int ok = (worst <= atol) && !isnan(worst);
    if (!ok) g_fail = 1;
    printf("  %-10s %-4s max|diff| = %.3e  (n=%d, atol=%.0e)",
           name, ok ? "PASS" : "FAIL", worst, n, atol);
    if (!ok)
        printf("   worst at %d: got %.9g want %.9g", at, got[at], want[at]);
    printf("\n");
}

int main(void)
{
    const float atol = 1e-4f;   /* looser than FEATHer: 96-term dot products
                                 * on raw (un-normalized) values accumulate
                                 * more than the normalized pipeline does */
    printf("DLinear golden test  (L=%d H=%d D=%d, decomp k=%d)\n",
           DL_L, DL_H, DL_D, DL_K);
    printf("static arena = %d floats = %d bytes = %.2f KB   [H-independent]\n",
           DL_ARENA_N, DL_ARENA_BYTES, DL_ARENA_BYTES / 1024.0);
    printf("head weights = %d bytes = %.1f KB   [GROWS with H]\n\n",
           DL_WEIGHT_BYTES, DL_WEIGHT_BYTES / 1024.0);

    static float y[DL_H * DL_D];          /* CALLER's buffer */
    dl_forward(GD_X, y);

    printf("stages:\n");
    check("trend", dl_st_trend, GD_TREND, DL_L * DL_D, atol);
    check("seas",  dl_st_seas,  GD_SEAS,  DL_L * DL_D, atol);
    check("y",     y,           GD_Y,     DL_H * DL_D, atol);

    /* streaming: one value at a time, channel by channel, must land where
     * the dense forecast has it */
    printf("\nstreaming (%d channels x %d steps):\n", DL_D, DL_H);
    float worst = 0.0f;
    for (int c = 0; c < DL_D; ++c) {
        dl_begin(GD_X, c);
        for (int t = 0; t < DL_H; ++t) {
            float df = fabsf(dl_step(t) - GD_Y[t * DL_D + c]);
            if (df > worst || isnan(df)) worst = df;
        }
    }
    int ok = worst <= atol;
    if (!ok) g_fail = 1;
    printf("  %-10s %-4s max|diff| = %.3e   (live window = 1 float = %d B)\n",
           "steps", ok ? "PASS" : "FAIL", worst, (int)sizeof(float));
    printf("  %-10s      full forecast would be %d floats = %d B\n",
           "vs dense", DL_H * DL_D, (int)(DL_H * DL_D * sizeof(float)));

    printf("\n%s\n", g_fail ? "*** FAIL ***" : "ALL PASS");
    return g_fail;
}
