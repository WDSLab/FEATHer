/*
 * TimeBase (ICML 2025) float32 inference.
 * Upstream: baselines/TimeBase/TimeBase-main/models/TimeBase.py (commit 369b330).
 *
 * period_len, basis_num and `individual` differ between settings, so the
 * export script writes them into golden/timebase_weights.h (TB_P / TB_B /
 * TB_IND) with the weights.  Only horizon, input length and width are
 * build-time geometry.
 */
#ifndef TIMEBASE_INFER_H
#define TIMEBASE_INFER_H

#include "mcu_emit.h"

#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

#ifndef TB_L
#define TB_L 96
#endif
#ifndef TB_H
#define TB_H 96
#endif
#ifndef TB_D
#define TB_D 7
#endif

/* x: (TB_L, TB_D) time-major -> y: (TB_H, TB_D) time-major */
void timebase_forward(const float *x, float *y);
/* one channel d -> yc[0..H-1] (stream per channel) */
void timebase_forward_ch(const float *x, int d, float *yc);
/* one channel d, each value to emit(d, t, v) as it is final (no H buffer) */
void timebase_forward_emit(const float *x, int d, mcu_emit_fn emit);

/* upstream's second output (use_orthogonal), valid after the last channel has been run */
extern float timebase_orth;

#ifdef TB_CAPTURE_STAGES
#include "timebase_weights.h"
extern float tb_st_pmean[TB_D * TB_P];             /* per-phase means, channel-major */
extern float tb_st_basis[TB_D * TB_P * TB_B];      /* basis coefficients (d, p, b)   */
#endif

#endif /* TIMEBASE_INFER_H */
