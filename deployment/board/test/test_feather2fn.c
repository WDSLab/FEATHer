/*
 * FEATHer-mix without bands + FFT gate (FEA2_FRONT_NONE build of the shipping
 * recompute port) against its PyTorch golden (front="none").
 * Same mixed tolerance as the other ports (atol=1e-5, rtol=1e-4).
 */
#include <math.h>
#include <stdio.h>

#include "feather2r_infer.h"
#include "feather2fn_vectors.h"

_Static_assert(FEA2_D == FEA2_D_GEN,
               "FEA2_D does not match the width the golden vectors were exported at - re-run export_golden_feather2fn.py --d_model");

int main(void)
{
    static float y[FEA2_H * FEA2_D];
    fea2r_forward(G2_X, y);
    int bad = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < FEA2_H * FEA2_D; ++i) {
        float ad = fabsf(y[i] - G2_Y[i]);
        if (ad > max_abs) max_abs = ad;
        if (ad > 1e-5f + 1e-4f * fabsf(G2_Y[i])) ++bad;
    }
    printf("FEATHer front=none (H=%d D=%d R=%d): bad=%d max|diff| %.3e  %s\n",
           FEA2_H, FEA2_D, FEA2_MLP_R, bad, (double)max_abs, bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
