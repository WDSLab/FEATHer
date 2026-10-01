/* The recompute schedule of the 09-04 mix model (src/feather2r_infer.c) must
 * give the same forecast as the original port (src/feather2_infer.c) and the
 * PyTorch golden.  Both ports are linked into this one test. */
#include <math.h>
#include <stdio.h>

#include "feather2_infer.h"
#include "feather2_vectors.h"
#include "feather2r_infer.h"

int main(void)
{
    static float y_orig[FEA2_H * FEA2_D], y_rec[FEA2_H * FEA2_D];
    fea2_forward(G2_X, y_orig);
    fea2r_forward(G2_X, y_rec);
    float d_orig = 0.0f, d_gold = 0.0f;
    int exact = 1;
    for (int i = 0; i < FEA2_H * FEA2_D; ++i) {
        float a = fabsf(y_rec[i] - y_orig[i]), b = fabsf(y_rec[i] - G2_Y[i]);
        if (a > d_orig) d_orig = a;
        if (b > d_gold) d_gold = b;
        if (y_rec[i] != y_orig[i]) exact = 0;
    }
    int ok = exact && d_gold <= 1e-5f;
    printf("feather2 recompute (H=%d D=%d): vs original max|diff| %.3e (%s), vs golden %.3e  %s\n",
           FEA2_H, FEA2_D, d_orig, exact ? "bit-identical" : "NOT identical", d_gold,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
