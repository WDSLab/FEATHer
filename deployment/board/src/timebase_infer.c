/*
 * TimeBase forward, mirroring baselines/TimeBase/TimeBase-main/models/TimeBase.py
 * with use_period_norm = 1 (every official script; 0 = one mean per channel, TB_PNORM):
 *   x (one channel) -> (period_len p, seg_num n) view, x[n*P + p] ->
 *   per-phase mean over the n segments subtracted ->
 *   ts2basis Linear(n_x -> basis) -> basis2ts Linear(basis -> n_y) ->
 *   per-phase mean added back -> sample t = j*P + p.
 * individual = 1 (official Weather script) gives every channel its own pair of
 * linears; otherwise one pair is shared.  With use_orthogonal upstream's forward also
 * computes the orthogonality term of the basis and returns it next to the forecast: the
 * port computes it too (TB_ORTH: every operation of PyTorch's forward).
 */
#include <math.h>

#include "timebase_infer.h"
#include "mcu_pt.h"
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
/* use_period_norm = 0 (chip-tier search): upstream _normalize_input then subtracts ONE
 * mean per channel (over all L samples) instead of one per phase, and adds it back to every
 * output.  Stamped by the export only when 0. */
#ifndef TB_PNORM
#define TB_PNORM 1
#endif

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

/* PER-CHANNEL, PER-PHASE WORKING SET - equal-effort rule (same tidy-up for every port).
 * TimeBase is channel-independent and every phase p is an independent
 * n_x -> n_y map, so only one phase's centred segment values (kept: read by
 * all basis rows, not recomputed) and its basis coefficients are live at a time.
 * Outputs go straight to the caller's buffer. */
#ifdef TB_CAPTURE_STAGES
float tb_st_pmean[TB_D * TB_P];
float tb_st_basis[TB_D * TB_P * TB_B];
#endif

#ifndef TB_ORTH
#define TB_ORTH 0
#endif
static float s_v[TB_NX];      /* one phase, centred */
#if !TB_PNORM
static float s_ch[TB_L];      /* the channel as one contiguous row */
#endif
#if TB_ORTH
static float s_bas[TB_P * TB_B];   /* this channel's basis coefficients, every phase (the term reads them all) */
static float s_nrm[TB_D];          /* each channel's norm */
float timebase_orth;               /* upstream's second output, set after the last channel */
#else
static float s_b[TB_B];       /* its basis coefficients */
#endif

#if TB_ORTH
/* one element of G = B^T B.  Not inlined: G[i][j] and G[j][i] are equal, and PyTorch computes both. */
static float __attribute__((noinline)) gram(const float *ci, const float *cj)
{
    float g = 0.0f;
    for (int p = 0; p < TB_P; ++p)
        g += ci[p * TB_B] * cj[p * TB_B];
    return g;
}
#endif

static void timebase_run(const float *x, float *y, int d0, int d1, int ys, mcu_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
        const float *w1 = &TB_W1[TB_CH(d) * TB_WSTRIDE_X];
        const float *b1 = &TB_B1[TB_CH(d) * TB_B];
        const float *w2 = &TB_W2[TB_CH(d) * TB_WSTRIDE_Y];
        const float *b2 = &TB_B2[TB_CH(d) * TB_NY];
        /* means = torch.mean's kernel (mcu_pt.c).  Period norm: the (c, period, n) view of x, one call per
         * channel (PyTorch's loop over the channels runs the same per-channel loop).  Else: the channel mean over
         * a contiguous (b, c, L) tensor -> the channel copied to one contiguous row */
#if TB_PNORM
        float pm[TB_P];
        {
            const int shape[2] = {TB_P, TB_NX}, stride[2] = {TB_D, TB_P * TB_D};
            pt_mean(x + d, pm, 2, shape, stride, 1u << 1);
        }
#else
        float mc;
        {
            const int shape[2] = {1, TB_L}, stride[2] = {TB_L, 1};
            for (int p = 0; p < TB_P; ++p)             /* upstream's row is the (p, n) view flattened: phase-major */
                for (int n = 0; n < TB_NX; ++n)
                    s_ch[p * TB_NX + n] = x[(n * TB_P + p) * TB_D + d];
            pt_mean(s_ch, &mc, 2, shape, stride, 1u << 1);
        }
#endif
        for (int p = 0; p < TB_P; ++p) {
#if TB_PNORM
            const float m = pm[p];
#else
            const float m = mc;
#endif
            for (int n = 0; n < TB_NX; ++n)
                s_v[n] = x[(n * TB_P + p) * TB_D + d] - m;
#if TB_ORTH
            float *s_b = &s_bas[p * TB_B];
#endif

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
                MCU_OUT(yd, ys, emit, d, j * TB_P + p, acc + m);
            }
        }
#if TB_ORTH
        /* cal_orthogonal_loss: G = B^T B (B = this channel's P x basis coefficients), G minus its diagonal,
         * Frobenius norm; the mean over the channels after the last one */
        const float zero = mcu_rt(0.0f);
#if TB_B == 1                                       /* torch.norm over 1 x 1: the absolute value, nothing computed */
        const float g = gram(s_bas, s_bas);
        s_nrm[d] = fabsf(g - g);
#else                                               /* torch.norm over the last two dims: 8 lanes, then the tail */
        float acc[8];
        for (int l = 0; l < 8; ++l)
            acc[l] = zero;
        const int nn = TB_B * TB_B, nv = nn - nn % 8;
        for (int k = 0; k < nn; ++k) {
            const int i = k / TB_B, j = k % TB_B;
            const float g = gram(s_bas + i, s_bas + j);
            const float off = g - ((i == j) ? g : zero);
            if (k < nv) {
                acc[k % 8] += off * off;
            } else {
                if (k == nv)
                    for (int l = 1; l < 8; ++l)
                        acc[0] = acc[0] + acc[l];
                acc[0] += off * off;
            }
        }
        if (nv == nn)
            for (int l = 1; l < 8; ++l)
                acc[0] = acc[0] + acc[l];
        s_nrm[d] = sqrtf(acc[0]);
#endif
        if (d == TB_D - 1) {                        /* loss.mean() over the channels */
            const int shape[1] = {TB_D}, stride[1] = {1};
            pt_mean(s_nrm, &timebase_orth, 1, shape, stride, 1u);
        }
#endif
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void timebase_forward(const float *x, float *y) { timebase_run(x, y, 0, TB_D, TB_D, 0); }

/* One channel d: its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer instead of holding H x D. */
void timebase_forward_ch(const float *x, int d, float *yc) { timebase_run(x, yc, d, d + 1, 1, 0); }

/* One channel d: each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (mcu_emit.h).  Same values as timebase_forward_ch, bit for bit. */
void timebase_forward_emit(const float *x, int d, mcu_emit_fn emit) { timebase_run(x, 0, d, d + 1, 1, emit); }
