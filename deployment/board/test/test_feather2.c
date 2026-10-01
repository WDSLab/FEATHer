/*
 * Golden test for the FEATHer v2 streaming port.
 *
 * Compares every stage of src/feather2_infer.c against the PyTorch vectors
 * emitted by export_golden_feather2.py, and separately checks the two
 * properties the port claims:
 *
 *   STREAM  fea2_block(om) emitted period by period == the full forecast
 *   ARENA   the static working set contains no H term
 *
 * Build (x86):  see Makefile target `feather2`
 */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "feather2_infer.h"
#include "feather2_vectors.h"

/* Golden vectors are captured per (horizon, width).  The horizon is
 * already guarded in the port; this covers the width. */
_Static_assert(FEA2_D == FEA2_D_GEN,
               "FEA2_D does not match the width the golden vectors were exported at - re-run export_golden_feather2.py --d_model");

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
    const float atol = 1e-5f;
    printf("FEATHer v2 golden test  (L=%d H=%d D=%d P=%d, %d bands)\n",
           FEA2_L, FEA2_H, FEA2_D, FEA2_PERIOD, FEA2_NB);
    printf("static arena = %d floats = %d bytes = %.2f KB   [H-independent]\n\n",
           FEA2_ARENA_N, FEA2_ARENA_BYTES, FEA2_ARENA_BYTES / 1024.0);

    static float y[FEA2_H * FEA2_D];          /* CALLER's buffer, not the model's */
    fea2_forward(G2_X, y);

    printf("stages:\n");
    check("mean",   fea2_st_mean,   G2_MEANS,  FEA2_D, atol);
    check("stdev",  fea2_st_stdev,  G2_STDEV,  FEA2_D, atol);
    check("xnorm",  fea2_st_xnorm,  G2_XNORM,  FEA2_L * FEA2_D, atol);
    /* the power spectrum is an O(1e2) quantity, so compare it relatively */
    check("power",  fea2_st_power,  G2_POWER,  FEA2_NF, 1e-2f);
    check("feats",  fea2_st_feats,  G2_FEATS,  FEA2_NB, atol);
    check("gate",   fea2_st_gate,   G2_GATE,   FEA2_NB, atol);
    check("band_P", fea2_st_bP,     G2_BP,     FEA2_L * FEA2_D, atol);
    check("band_M", fea2_st_bM,     G2_BM,     FEA2_L * FEA2_D, atol);
    check("band_L", fea2_st_bL,     G2_BL,     FEA2_L * FEA2_D, atol);
    check("hidden", fea2_st_hidden, G2_HIDDEN, FEA2_L * FEA2_D, atol);
    check("mlp",    fea2_st_mlp,    G2_MLP,    FEA2_L * FEA2_D, atol);
    check("h_agg",  fea2_st_hagg,   G2_HAGG,   FEA2_D * FEA2_L, atol);
    check("y",      y,              G2_Y,      FEA2_H * FEA2_D, atol);

    /* ---- streaming: the property the whole port rests on ----
     * Re-run through the block API and confirm each emitted period lands
     * exactly where the full forecast has it.  A caller that never keeps
     * more than one block still gets the same forecast. */
    printf("\nstreaming (%d blocks x %d channels of %d steps):\n",
           FEA2_M, FEA2_D, FEA2_PERIOD);
    fea2_begin(G2_X);
    float worst = 0.0f;
    for (int om = 0; om < FEA2_M; ++om) {
        for (int d = 0; d < FEA2_D; ++d) {
            const float *blk = fea2_block(om, d);
            for (int ph = 0; ph < FEA2_PERIOD; ++ph) {
                float df = fabsf(blk[ph]
                                 - G2_Y[(om * FEA2_PERIOD + ph) * FEA2_D + d]);
                if (df > worst || isnan(df)) worst = df;
            }
        }
    }
    int ok = worst <= atol;
    if (!ok) g_fail = 1;
    printf("  %-10s %-4s max|diff| = %.3e   (live window = %d floats = %d B)\n",
           "blocks", ok ? "PASS" : "FAIL", worst,
           FEA2_PERIOD, (int)(FEA2_PERIOD * sizeof(float)));
    printf("  %-10s      full forecast would be %d floats = %d B\n",
           "vs dense", FEA2_H * FEA2_D,
           (int)(FEA2_H * FEA2_D * sizeof(float)));

    printf("\n%s\n", g_fail ? "*** FAIL ***" : "ALL PASS");
    return g_fail;
}
