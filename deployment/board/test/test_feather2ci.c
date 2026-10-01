/*
 * Golden test for FEATHer-CI (the channel-independent variant).
 *
 * The point of this port is that its working set does NOT grow with the
 * channel count - only the caller's input buffer does - so it reaches the
 * high-D cells the mixing variant cannot.  The printout states both, so the
 * deployability table can be read straight off it.
 */
#include <stdio.h>
#include <math.h>

#include "feather2ci_infer.h"
#include "feather2ci_vectors.h"

int main(void)
{
    printf("FEATHer-CI golden test  (L=%d H=%d D=%d P=%d)\n",
           FEA2_L, FEA2_H, FEA2CI_D, FEA2_PERIOD);
    printf("working set = %d floats = %.2f KB   [independent of D and H]\n",
           FEA2CI_ARENA_N, FEA2CI_ARENA_BYTES / 1024.0);
    printf("input buffer (caller) = %d B; dense forecast would be %d B\n\n",
           FEA2_L * FEA2CI_D * 4, FEA2_H * FEA2CI_D * 4);

    if (FEA2CI_D != GCI_D || FEA2_H != GCI_H) {
        printf("*** geometry mismatch: built D=%d H=%d, vectors D=%d H=%d\n",
               FEA2CI_D, FEA2_H, GCI_D, GCI_H);
        return 1;
    }

    static float y[GCI_H * GCI_D];
    fea2ci_forward(GCI_X, y);

    float worst = 0.0f;
    int at = -1;
    for (int i = 0; i < GCI_H * GCI_D; ++i) {
        float d = fabsf(y[i] - GCI_Y[i]);
        if (d > worst || isnan(d)) { worst = d; at = i; }
    }
    int ok = (worst <= 1e-5f) && !isnan(worst);
    printf("  %-10s %-4s max|diff| = %.3e  (n=%d)",
           "y", ok ? "PASS" : "FAIL", worst, GCI_H * GCI_D);
    if (!ok)
        printf("   worst at %d: got %.9g want %.9g", at, y[at], GCI_Y[at]);
    printf("\n\n%s\n", ok ? "ALL PASS" : "*** FAIL ***");
    return ok ? 0 : 1;
}
