/* mcu_pt.c on the calls recorded by test/test_pt_default.py --c:  test_pt in.bin out.bin */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mcu_pt.h"

static int rd(FILE *f, void *p, size_t n)
{
    return fread(p, 1, n, f) == n;
}

int main(int argc, char **argv)
{
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    int ncase;
    if (!fi || !fo || argc < 3 || !rd(fi, &ncase, 4)) return 1;
    for (int c = 0; c < ncase; ++c) {
        int op, ni, nf, a[16];
        if (!rd(fi, &op, 4) || !rd(fi, &ni, 4) || ni > 16 || !rd(fi, a, 4 * (size_t)ni) || !rd(fi, &nf, 4)) return 2;
        float *x = malloc(4 * (size_t)nf + 64);
        if (!rd(fi, x, 4 * (size_t)nf)) return 3;
        int nout = 0;
        float *y = NULL;
        if (op <= 1) {                                       /* sum / mean: nd, shape[4], stride[4], red */
            const int nd = a[0];
            nout = 1;
            for (int d = 0; d < nd; ++d)
                if (!(a[9] >> d & 1)) nout *= a[1 + d];
            y = malloc(4 * (size_t)nout);
            (op ? pt_mean : pt_sum)(x, y, nd, a + 1, a + 5, (unsigned)a[9]);
        } else if (op == 2) {                                /* var: rows, rstep, n, stride, correction */
            nout = a[0];
            y = malloc(4 * (size_t)nout);
            pt_var(x, y, a[0], a[1], a[2], a[3], a[4]);
        } else if (op == 3) {                                /* layer norm: rows, n | x, g, b, eps */
            nout = a[0] * a[1];
            y = malloc(4 * (size_t)nout);
            pt_layer_norm(x, y, a[0], a[1], x + nout, x + nout + a[1], x[nout + 2 * a[1]]);
        } else if (op == 4) {                                /* softmax: rows, k */
            nout = a[0] * a[1];
            y = malloc(4 * (size_t)nout);
            pt_softmax(x, y, a[0], a[1]);
        } else if (op == 5) {                                /* vector norm: rows, n */
            nout = a[0];
            y = malloc(4 * (size_t)nout);
            for (int r = 0; r < a[0]; ++r) y[r] = pt_norm2(x + r * a[1], a[1]);
        }
        fwrite(&nout, 4, 1, fo);
        fwrite(y, 4, (size_t)nout, fo);
        free(x);
        free(y);
    }
    fclose(fo);
    return 0;
}
