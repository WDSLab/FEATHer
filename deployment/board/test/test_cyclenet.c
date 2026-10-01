/*
 * Golden test for the CycleNet port.
 *
 * Checks the de-cycled input (RevIN with the UNBIASED variance, then the
 * cycle subtracted at phase 0), the forecast, and the step-wise streaming path.
 * The golden weights are random INCLUDING the cycle, so a port that ignored
 * the cycle or indexed it wrongly would fail here.
 *
 * Build (x86): see Makefile target `cyclenet`
 */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "cyclenet_infer.h"
#include "cyclenet_vectors.h"

_Static_assert(CN_D == CN_D_GEN,
               "CN_D does not match the width the golden vectors were exported at - re-run export_golden_cyclenet.py --d_model");

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
    const float atol = 1e-4f;
    printf("CycleNet golden test  (L=%d H=%d D=%d, linear backbone, RevIN)\n",
           CN_L, CN_H, CN_D);
    printf("static arena = %d floats = %d bytes = %.2f KB   [H- and D-independent]\n\n",
           CN_ARENA_N, CN_ARENA_BYTES, CN_ARENA_BYTES / 1024.0);

    static float y[CN_H * CN_D];          /* CALLER's buffer */
    cn_forward(GC_X, y);

    printf("stages:\n");
    check("xs", cn_st_xs, GC_XS, CN_L * CN_D, atol);
    check("y",  y,        GC_Y,  CN_H * CN_D, atol);

    printf("\nstreaming (%d channels x %d steps):\n", CN_D, CN_H);
    float worst = 0.0f;
    for (int c = 0; c < CN_D; ++c) {
        cn_begin(GC_X, c, 0);
        for (int t = 0; t < CN_H; ++t) {
            float df = fabsf(cn_step(t) - GC_Y[t * CN_D + c]);
            if (df > worst || isnan(df)) worst = df;
        }
    }
    int ok = worst <= atol;
    if (!ok) g_fail = 1;
    printf("  %-10s %-4s max|diff| = %.3e\n", "steps", ok ? "PASS" : "FAIL", worst);

    printf("\n%s\n", g_fail ? "*** FAIL ***" : "ALL PASS");
    return g_fail;
}
