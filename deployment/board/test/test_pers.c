/*
 * Persistence baseline sanity test - pure index logic, so the check is
 * exact equality against a directly-constructed expectation (no PyTorch
 * golden needed; definition confirmed against
 * tools/paper/persistence_baseline.py: "forecast = last observed input
 * step, repeated across the horizon").
 */
#include <stdio.h>

#include "pers_infer.h"

#define L 96
#define H 96
#define D 7

int main(void)
{
    static float x[L * D], y[H * D];
    for (int t = 0; t < L; ++t)
        for (int d = 0; d < D; ++d)
            x[t * D + d] = (float)(t * 100 + d);

    pers_forward(x, y, L, H, D);

    for (int t = 0; t < H; ++t)
        for (int d = 0; d < D; ++d)
            if (y[t * D + d] != x[(L - 1) * D + d]) {
                printf("FAIL  pers @(%d,%d)\n", t, d);
                return 1;
            }
    printf("PASS  pers n=%d (exact copy-last)\n", H * D);
    return 0;
}
