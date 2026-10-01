/*
 * Replay real test windows through the deployed port and compare against the
 * PyTorch predictions produced by the SAME trained checkpoint.
 *
 * The golden test proves the port computes the right function on one synthetic
 * window with random weights.  This proves it on real inputs with the shipping
 * weights - i.e. that the accuracy the paper reports is produced by the code
 * that runs on the part.
 *
 * Inputs come from verify_deployed_accuracy.py:
 *   golden/real_x.bin   float32 (N, L, D)  real test inputs
 *   golden/real_p.bin   float32 (N, H, D)  PyTorch predictions
 *   golden/real_ref.h   geometry + the PyTorch test MSE
 *
 * Tolerance is on the PREDICTIONS, not on a metric: masked cells and the
 * worker's aggregation stay on the Python side, so nothing about the metric
 * is reimplemented here where it could quietly diverge.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "feather2_infer.h"
#include "real_ref.h"

/* -DREALCHECK_RECOMPUTE replays the shipping recompute port (feather2r),
 * the default replays the original (L, D)-arena port (feather2). */
#ifdef REALCHECK_RECOMPUTE
#include "feather2r_infer.h"
#define RC_FORWARD fea2r_forward
#else
#define RC_FORWARD fea2_forward
#endif

static float *load(const char *path, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    float *buf = malloc(n * sizeof(float));
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        printf("%s: expected %zu floats, got %zu\n", path, n, got);
        exit(2);
    }
    return buf;
}

int main(void)
{
    if (REAL_D != FEA2_D || REAL_H != FEA2_H || REAL_L != FEA2_L) {
        printf("geometry mismatch: built L=%d H=%d D=%d, data L=%d H=%d D=%d\n",
               FEA2_L, FEA2_H, FEA2_D, REAL_L, REAL_H, REAL_D);
        return 1;
    }

    const size_t xn = (size_t)REAL_N * REAL_L * REAL_D;
    const size_t pn = (size_t)REAL_N * REAL_H * REAL_D;
    float *X = load("golden/real_x.bin", xn);
    float *P = load("golden/real_p.bin", pn);
    static float y[FEA2_H * FEA2_D];

    printf("replaying %d real test windows  (L=%d H=%d D=%d)\n",
           REAL_N, REAL_L, REAL_H, REAL_D);
    printf("PyTorch test MSE on the full test set = %.6f\n\n", REAL_PT_MSE);

    double worst = 0.0, sumsq = 0.0;
    int at_w = -1, at_i = -1;
    for (int w = 0; w < REAL_N; ++w) {
        RC_FORWARD(X + (size_t)w * REAL_L * REAL_D, y);
        for (int i = 0; i < REAL_H * REAL_D; ++i) {
            double ref = P[(size_t)w * REAL_H * REAL_D + i];
            double d = fabs((double)y[i] - ref);
            sumsq += d * d;
            if (d > worst || isnan(d)) { worst = d; at_w = w; at_i = i; }
        }
    }
    double rms = sqrt(sumsq / (double)pn);

    /* 1e-4 absolute on denormalized predictions: the port accumulates a 96-term
     * dot product per output step in float32 in a different order than PyTorch,
     * and the values are on the data's own scale, not normalized. */
    const double tol = 1e-4;
    int ok = (worst <= tol) && !isnan(worst);
    printf("  max|C - PyTorch| = %.3e   (worst: window %d, cell %d)\n",
           worst, at_w, at_i);
    printf("  rms|C - PyTorch| = %.3e   over %zu predictions\n", rms, pn);
    printf("  tolerance        = %.0e\n\n", tol);
    printf("%s\n", ok ? "REALCHECK PASS - the deployed port reproduces the "
                        "reported model on real data"
                      : "*** REALCHECK FAIL ***");
    free(X); free(P);
    return ok ? 0 : 1;
}
