/*
 * TimeBase forward, mirroring baselines/TimeBase/TimeBase-main/models/TimeBase.py
 * with use_period_norm = 1 (every official script):
 *   x (one channel) -> (period_len p, seg_num n) view, x[n*P + p] ->
 *   per-phase mean over the n segments subtracted ->
 *   ts2basis Linear(n_x -> basis) -> basis2ts Linear(basis -> n_y) ->
 *   per-phase mean added back -> sample t = j*P + p.
 * individual = 1 (official Weather script) gives every channel its own pair of
 * linears; otherwise one pair is shared.  The orthogonality term is a
 * training loss only.
 */
#include "timebase_infer.h"
#include "timebase_weights.h"      /* generated: golden/timebase_weights.h */

_Static_assert(TB_H == TB_H_GEN,
               "TB_H does not match the horizon the weights were exported at - re-run export_golden_timebase.py --pred_len");
_Static_assert(TB_L == TB_L_GEN,
               "TB_L does not match the input length the weights were exported at - re-run export_golden_timebase.py --seq_len");
_Static_assert(!TB_IND || TB_D == TB_D_GEN,
               "TB_D does not match the width the per-channel weights were exported at - re-run export_golden_timebase.py --d_model");
/* upstream pads the input / trims the output when P does not divide L / H;
 * every setting we run has P | L and P | H, so that path is not ported */
_Static_assert(TB_L % TB_P == 0 && TB_H % TB_P == 0,
               "period_len must divide L and H (the padding path is not ported)");

#define TB_NX (TB_L / TB_P)     /* input segments  */
#define TB_NY (TB_H / TB_P)     /* output segments */

#if TB_IND
#define TB_WSTRIDE_X (TB_B * TB_NX)
#define TB_WSTRIDE_Y (TB_NY * TB_B)
#define TB_CH(d) (d)
#else
#define TB_CH(d) 0
#define TB_WSTRIDE_X 0
#define TB_WSTRIDE_Y 0
#endif

/* PER-CHANNEL, PER-PHASE WORKING SET - equal-effort rule (README).
 * TimeBase is channel-independent and every phase p is an independent
 * n_x -> n_y map, so only one phase's centred segment values (kept: read by
 * all basis rows, rule 4) and its basis coefficients are live at a time.
 * Outputs go straight to the caller's buffer (rule 2). */
#ifdef TB_CAPTURE_STAGES
float tb_st_pmean[TB_D * TB_P];
float tb_st_basis[TB_D * TB_P * TB_B];
#endif

static float s_v[TB_NX];      /* one phase, centred */
static float s_b[TB_B];       /* its basis coefficients */

static void timebase_run(const float *x, float *y, int d0, int d1, int ys, fea_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
        const float *w1 = &TB_W1[TB_CH(d) * TB_WSTRIDE_X];
        const float *b1 = &TB_B1[TB_CH(d) * TB_B];
        const float *w2 = &TB_W2[TB_CH(d) * TB_WSTRIDE_Y];
        const float *b2 = &TB_B2[TB_CH(d) * TB_NY];
        for (int p = 0; p < TB_P; ++p) {
            float m = 0.0f;
            for (int n = 0; n < TB_NX; ++n)
                m += x[(n * TB_P + p) * TB_D + d];
            m /= (float)TB_NX;
            for (int n = 0; n < TB_NX; ++n)
                s_v[n] = x[(n * TB_P + p) * TB_D + d] - m;

            for (int b = 0; b < TB_B; ++b) {
                float acc = b1[b];
                for (int n = 0; n < TB_NX; ++n)
                    acc += w1[b * TB_NX + n] * s_v[n];
                s_b[b] = acc;
            }
#ifdef TB_CAPTURE_STAGES
            tb_st_pmean[d * TB_P + p] = m;
            for (int b = 0; b < TB_B; ++b)
                tb_st_basis[(d * TB_P + p) * TB_B + b] = s_b[b];
#endif
            for (int j = 0; j < TB_NY; ++j) {
                float acc = b2[j];
                for (int b = 0; b < TB_B; ++b)
                    acc += w2[j * TB_B + b] * s_b[b];
                FEA_OUT(yd, ys, emit, d, j * TB_P + p, acc + m);
            }
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void timebase_forward(const float *x, float *y) { timebase_run(x, y, 0, TB_D, TB_D, 0); }

/* One channel d (10-12): its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer (porting rule 1) instead of holding H x D. */
void timebase_forward_ch(const float *x, int d, float *yc) { timebase_run(x, yc, d, d + 1, 1, 0); }

/* One channel d (10-13): each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (fea_emit.h).  Same values as timebase_forward_ch, bit for bit. */
void timebase_forward_emit(const float *x, int d, fea_emit_fn emit) { timebase_run(x, 0, d, d + 1, 1, emit); }
