/*
 * Golden test for the FEATHer in1 port (feather2in_infer.c).
 *
 * Compares every stage against the PyTorch vectors from
 * export_golden_feather2in.py, then re-runs through the streaming API
 * (sensor by sensor, one output period at a time) and checks it lands on the
 * same forecast.  Build both schedules with `make feather2in`.
 */
#include <stdio.h>
#include <math.h>

#include "feather2in_infer.h"
#include "feather2in_vectors.h"

_Static_assert(FEA2I_D == FEA2I_D_GEN,
               "FEA2I_D does not match the width the golden vectors were exported at - re-run export_golden_feather2in.py --d_model");

static int g_fail = 0;

static void check(const char *name, const float *got, const float *want,
                  int n, float atol)
{
    float worst = 0.0f;
    int at = -1;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
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
    printf("FEATHer in1 golden test  (L=%d H=%d D=%d P=%d R=%d)  schedule: %s\n",
           FEA2I_L, FEA2I_H, FEA2I_D, FEA2I_PERIOD, FEA2I_R,
           FEA2I_RECOMPUTE_NORM ? "RENORM (x_norm recomputed per sensor)"
                                : "STRICT (x_norm kept, L x D)");
    printf("static arena = %d floats = %d bytes = %.2f KB   [H-independent]\n\n",
           FEA2I_ARENA_N, FEA2I_ARENA_BYTES, FEA2I_ARENA_BYTES / 1024.0);

    static float y[FEA2I_H * FEA2I_D];        /* CALLER's buffer */
    fea2i_forward(GI_X, y);

    printf("stages:\n");
    check("mean",   fea2i_st_mean,   GI_MEANS,  FEA2I_D, atol);
    check("stdev",  fea2i_st_stdev,  GI_STDEV,  FEA2I_D, atol);
    check("power",  fea2i_st_power,  GI_POWER,  FEA2I_D * FEA2I_NF, 1e-2f);
    check("feats",  fea2i_st_feats,  GI_FEATS,  FEA2I_D * FEA2I_NB, atol);
    check("gate",   fea2i_st_gate,   GI_GATE,   FEA2I_D * FEA2I_NB, atol);
    check("u",      fea2i_st_u,      GI_U,      FEA2I_L * FEA2I_R, atol);
    check("hidden", fea2i_st_hidden, GI_HIDDEN, FEA2I_D * FEA2I_L, atol);
    check("mlp",    fea2i_st_mlp,    GI_MLP,    FEA2I_D * FEA2I_L, atol);
    check("h_agg",  fea2i_st_hagg,   GI_HAGG,   FEA2I_D * FEA2I_L, atol);
    check("y",      y,               GI_Y,      FEA2I_H * FEA2I_D, atol);

    printf("\nstreaming (%d sensors x %d blocks of %d steps):\n",
           FEA2I_D, FEA2I_M, FEA2I_PERIOD);
    fea2i_begin(GI_X);
    float worst = 0.0f;
    for (int d = 0; d < FEA2I_D; ++d) {
        fea2i_channel(d);
        for (int om = 0; om < FEA2I_M; ++om) {
            const float *blk = fea2i_block(om);
            for (int ph = 0; ph < FEA2I_PERIOD; ++ph) {
                float df = fabsf(blk[ph]
                                 - GI_Y[(om * FEA2I_PERIOD + ph) * FEA2I_D + d]);
                if (df > worst) worst = df;
            }
        }
    }
    int ok = worst <= atol;
    if (!ok) g_fail = 1;
    printf("  %-10s %-4s max|diff| = %.3e\n", "blocks", ok ? "PASS" : "FAIL", worst);

    printf("\n%s\n", g_fail ? "*** FAIL ***" : "ALL PASS");
    return g_fail;
}
