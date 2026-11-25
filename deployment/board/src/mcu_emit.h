/*
 * Per-value output sink for the full-buffer baseline ports.
 *
 * Every forecast value of these ports is final the moment it is computed (no
 * later pass adds to it), so the caller does not need an H-float buffer to hold a
 * channel: <pfx>_forward_emit hands each value to emit(d, t, v) as it is produced,
 * with its time index t, and the caller writes it out (the receiver reassembles by
 * index, as replay.py does for FEATHer).  The porting rule (same computation,
 * memory only reordered) applied to the output side, as FEATHer emits its blocks.  The arithmetic is unchanged:
 * the value passed is the one the buffered API stores, bit for bit.
 */
#ifndef MCU_EMIT_H
#define MCU_EMIT_H

typedef void (*mcu_emit_fn)(int d, int t, float v);

/* store into the caller's buffer (buffered APIs) or hand to the sink (emit API) */
#define MCU_OUT(yd, ys, emit, d, t, v)                     \
    do {                                                   \
        const float mcu_out_v_ = (v);                      \
        if (emit) (emit)((d), (t), mcu_out_v_);            \
        else (yd)[(t) * (ys)] = mcu_out_v_;                \
    } while (0)

/* A count PyTorch reads from a tensor's shape at run time (the n of a mean, a pool size, a length ratio) is read
 * at run time here too: through a volatile, so the compiler cannot fold it (x / 4 -> x * 0.25, x / 1 -> x,
 * x * 2 -> x + x) and the port does the operation PyTorch does. */
static inline float mcu_rt(float v)
{
    volatile float r = v;
    return r;
}

/* A value PyTorch computes and then does not use (sliced away, or returned next to the forecast): the port
 * computes it too and drops it here - the volatile store keeps the compiler from removing the computation. */
static inline void mcu_drop(float v)
{
    volatile float r = v;
    (void)r;
}

#endif /* MCU_EMIT_H */
