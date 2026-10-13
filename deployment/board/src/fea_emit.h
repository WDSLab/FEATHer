/*
 * Per-value output sink for the full-buffer baseline ports (2026-10-13).
 *
 * Every forecast value of these ports is final the moment it is computed (no
 * later pass adds to it), so the caller does not need an H-float buffer to hold a
 * channel: <pfx>_forward_emit hands each value to emit(d, t, v) as it is produced,
 * with its time index t, and the caller writes it out (the receiver reassembles by
 * index, as replay.py already does for FEATHer).  Porting rule 1 applied to the
 * output side the same way FEATHer emits its blocks.  The arithmetic is unchanged:
 * the value passed is the one the buffered API stores, bit for bit.
 */
#ifndef FEA_EMIT_H
#define FEA_EMIT_H

typedef void (*fea_emit_fn)(int d, int t, float v);

/* store into the caller's buffer (buffered APIs) or hand to the sink (emit API) */
#define FEA_OUT(yd, ys, emit, d, t, v)                     \
    do {                                                   \
        const float fea_out_v_ = (v);                      \
        if (emit) (emit)((d), (t), fea_out_v_);            \
        else (yd)[(t) * (ys)] = fea_out_v_;                \
    } while (0)

#endif /* FEA_EMIT_H */
