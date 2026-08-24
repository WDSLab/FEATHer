/*
 * x86 golden-vector test: run the C forward once and compare EVERY stage
 * against PyTorch dumps (golden/feather_vectors.h).
 *
 * Pass criterion (EDGE_EXPERIMENT.md decision #4): element-wise mixed
 * tolerance  |c - ref| <= atol + rtol * |ref|,  atol=1e-5, rtol=1e-4.
 *
 * Build & run:  make -C deployment/board test   (WSL / any gcc)
 */
#include <math.h>
#include <stdio.h>

#include "feather_infer.h"
#include "feather_vectors.h"

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
        printf("FAIL  %-10s n=%d bad=%d first@%d got=%.6e ref=%.6e "
               "(max_abs=%.2e max_rel=%.2e)\n",
               name, n, bad, bad_idx,
               (double)got[bad_idx], (double)ref[bad_idx],
               (double)max_abs, (double)max_rel);
    } else {
        printf("PASS  %-10s n=%d max_abs=%.2e max_rel=%.2e\n",
               name, n, (double)max_abs, (double)max_rel);
    }
}

int main(void)
{
    static float y[FEA_H * FEA_D];
    feather_forward(G_X, y);

    /* order = forward order; a first FAIL localizes the broken layer */
    check("mean",    fea_st_mean,    G_MEANS,   FEA_D);
    check("stdev",   fea_st_stdev,   G_STDEV,   FEA_D);   /* trap#3 */
    check("xnorm",   fea_st_xnorm,   G_XNORM,   FEA_L * FEA_D);
    check("xproj",   fea_st_xproj,   G_XPROJ,   FEA_L * FEA_D);
    check("fP",      fea_st_fP,      G_FP,      FEA_L * FEA_D);
    check("fH",      fea_st_fH,      G_FH,      FEA_L * FEA_D);
    check("fM",      fea_st_fM,      G_FM,      FEA_L * FEA_D);
    check("fLds",    fea_st_fLds,    G_FLDS,    FEA_LP * FEA_D);
    check("fL",      fea_st_fL,      G_FL,      FEA_L * FEA_D);   /* trap#1 */
    check("oP",      fea_st_oP,      G_OP,      FEA_L * FEA_D);   /* trap#2 */
    check("oH",      fea_st_oH,      G_OH,      FEA_L * FEA_D);
    check("oM",      fea_st_oM,      G_OM,      FEA_L * FEA_D);
    check("oL",      fea_st_oL,      G_OL,      FEA_L * FEA_D);
    check("rfft_re", fea_st_rfft_re, G_RFFT_RE, FEA_NF);          /* trap#4 */
    check("rfft_im", fea_st_rfft_im, G_RFFT_IM, FEA_NF);
    check("magmean", fea_st_magmean, G_MAGMEAN, FEA_NF);
    check("gate_z",  fea_st_gate_z,  G_GATE_Z,  FEA_NB);
    check("gate",    fea_st_gate,    G_GATE,    FEA_NB);
    check("hidden",  fea_st_hidden,  G_HIDDEN,  FEA_L * FEA_D);
    check("mlp",     fea_st_mlp,     G_MLP,     FEA_L * FEA_D);
    check("hagg",    fea_st_hagg,    G_HAGG,    FEA_D * FEA_L);
    check("yspk",    fea_st_yspk,    G_YSPK,    FEA_H * FEA_D);
    check("y",       y,              G_Y,       FEA_H * FEA_D);

    if (g_fail) {
        printf("\n%d stage(s) FAILED\n", g_fail);
        return 1;
    }
    printf("\nALL STAGES PASS (atol=%.0e rtol=%.0e)\n",
           (double)ATOL, (double)RTOL);
    return 0;
}
