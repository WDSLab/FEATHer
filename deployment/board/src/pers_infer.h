/*
 * Persistence (copy-last) baseline - the zero-parameter floor row of the
 * deployment table. Matches tools/paper/persistence_baseline.py: the
 * forecast repeats the last observed timestep for every horizon step.
 */
#ifndef PERS_INFER_H
#define PERS_INFER_H

static inline void pers_forward(const float *x, float *y,
                                int seq_len, int pred_len, int nch)
{
    const float *last = x + (seq_len - 1) * nch;
    for (int t = 0; t < pred_len; ++t)
        for (int d = 0; d < nch; ++d)
            y[t * nch + d] = last[d];
}

#endif /* PERS_INFER_H */
