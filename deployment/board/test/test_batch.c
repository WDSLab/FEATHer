/*
 * Week-2 acceptance test: run 100 windows through every ported model and
 * compare the aggregate test MSE against PyTorch to 4 decimal places
 * (EDGE_EXPERIMENT.md decision #4, second half).
 *
 * Reads golden/batch_x.bin + batch_t.bin (float32 raw); run from
 * deployment/board (the Makefile does).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "feather_infer.h"
#include "stsf_infer.h"
#include "dipe_infer.h"
#include "fits_infer.h"
#include "pers_infer.h"
#include "batch_ref.h"

#define L 96
#define H 96
#define D 7
#define W_ELEMS (L * D)
#define TOL 0.5e-4          /* 4-decimal agreement */

static float *load(const char *path, long n_floats)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    float *buf = (float *)malloc((size_t)n_floats * sizeof(float));
    if (fread(buf, sizeof(float), (size_t)n_floats, fp)
            != (size_t)n_floats) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(fp);
    return buf;
}

typedef void (*fwd_fn)(const float *x, float *y);

static void pers_fwd(const float *x, float *y)
{
    pers_forward(x, y, L, H, D);
}

static int run(const char *name, fwd_fn fn, const float *X,
               const float *T, double ref)
{
    static float y[H * D];
    double sse = 0.0;
    for (int w = 0; w < BATCH_N; ++w) {
        fn(X + (long)w * W_ELEMS, y);
        const float *t = T + (long)w * W_ELEMS;
        for (int i = 0; i < H * D; ++i) {
            double dlt = (double)y[i] - (double)t[i];
            sse += dlt * dlt;
        }
    }
    double mse = sse / ((double)BATCH_N * H * D);
    double diff = fabs(mse - ref);
    int ok = diff < TOL;
    printf("%s  %-8s C=%.6f  torch=%.6f  |diff|=%.2e\n",
           ok ? "PASS" : "FAIL", name, mse, ref, diff);
    return ok ? 0 : 1;
}

int main(void)
{
    float *X = load("golden/batch_x.bin", (long)BATCH_N * W_ELEMS);
    float *T = load("golden/batch_t.bin", (long)BATCH_N * W_ELEMS);

    int fails = 0;
    fails += run("FEATHer", feather_forward, X, T, BATCH_REF_FEATHER);
    fails += run("SparseTSF", stsf_forward, X, T, BATCH_REF_STSF);
    fails += run("DiPE", dipe_forward, X, T, BATCH_REF_DIPE);
    fails += run("FITS", fits_forward, X, T, BATCH_REF_FITS);
    fails += run("persist", pers_fwd, X, T, BATCH_REF_PERS);

    printf("\nFEATHer static arena = %d bytes (+%d input) "
           "of 20480 SRAM\n", FEA_ARENA_BYTES,
           (int)(W_ELEMS * sizeof(float)));

    free(X);
    free(T);
    if (fails) {
        printf("%d model(s) FAILED\n", fails);
        return 1;
    }
    printf("BATCH ALL PASS (%d windows, 4-decimal MSE agreement)\n",
           BATCH_N);
    return 0;
}
