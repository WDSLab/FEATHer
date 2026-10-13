/*
 * CMoS forward, mirroring baselines/CMoS/CMoS-main/model/CMoS/Model.py:
 *   per-channel mean/std norm (biased var, eps 1e-10 inside sqrt) ->
 *   per-channel conv1d (1 -> 1, kernel K, stride S, no padding) ->
 *   shared gate Linear(conv_dim -> num_map) + softmax ->
 *   num_map shared chunk mappings Linear(L/seg -> H/seg), applied to every
 *   offset s inside a chunk -> gate-weighted sum -> de-norm.
 *
 * Upstream builds num_map + 1 mappings but forward() uses only the first
 * num_map; the spare one is not exported (it is never computed).
 */
#include "cmos_infer.h"
#include "cmos_weights.h"      /* generated: golden/cmos_weights.h */

#include <math.h>

_Static_assert(CMOS_H == CMOS_H_GEN,
               "CMOS_H does not match the horizon the weights were exported at - re-run export_golden_cmos.py --pred_len");
_Static_assert(CMOS_L == CMOS_L_GEN,
               "CMOS_L does not match the input length the weights were exported at - re-run export_golden_cmos.py --seq_len");
_Static_assert(CMOS_D == CMOS_D_GEN,
               "CMOS_D does not match the width the weights were exported at (CMoS has one conv per channel) - re-run export_golden_cmos.py --d_model");
_Static_assert(CMOS_L % CMOS_SEG == 0 && CMOS_H % CMOS_SEG == 0,
               "seg_size must divide L and H");

#define CMOS_NI (CMOS_L / CMOS_SEG)     /* input chunks  */
#define CMOS_NO (CMOS_H / CMOS_SEG)     /* output chunks */

/* PER-CHANNEL WORKING SET - equal-effort rule, deployment/board/README.md.
 * CMoS is channel-independent (each channel's conv, gate and mappings read
 * only that channel), so one channel is processed at a time.  The normalised
 * channel is kept (it is read by the conv and by every mapping: rule 4, no
 * recompute).  The forecast is written straight to the caller's buffer one
 * sample at a time (rule 2: no H-length temporary). */
#ifdef CMOS_CAPTURE_STAGES
float cmos_st_mean[CMOS_D];
float cmos_st_std[CMOS_D];
float cmos_st_conv[CMOS_D * CMOS_CONV_DIM];
float cmos_st_gate[CMOS_D * CMOS_NMAP];
#endif

static float s_xn[CMOS_L];            /* one channel, normalised           */
static float s_cv[CMOS_CONV_DIM];     /* conv output, one channel          */
static float s_g[CMOS_NMAP];          /* gate logits -> softmax weights    */

static void cmos_run(const float *x, float *y, int d0, int d1, int ys, fea_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
        /* mean, then biased variance of the centred channel */
        float m = 0.0f;
        for (int t = 0; t < CMOS_L; ++t)
            m += x[t * CMOS_D + d];
        m /= (float)CMOS_L;
        float v = 0.0f;
        for (int t = 0; t < CMOS_L; ++t) {
            float c = x[t * CMOS_D + d] - m;
            s_xn[t] = c;
            v += c * c;
        }
        v /= (float)CMOS_L;
        const float sd = sqrtf(v + 1e-10f);
        for (int t = 0; t < CMOS_L; ++t)
            s_xn[t] /= sd;

        /* this channel's own conv (valid, strided) */
        const float *wc = &CMOS_W_CONV[d * CMOS_K];
        for (int o = 0; o < CMOS_CONV_DIM; ++o) {
            float acc = CMOS_B_CONV[d];
            for (int k = 0; k < CMOS_K; ++k)
                acc += wc[k] * s_xn[o * CMOS_STRIDE + k];
            s_cv[o] = acc;
        }

        /* shared gate + softmax over the mappings */
        float mx = -INFINITY;
        for (int n = 0; n < CMOS_NMAP; ++n) {
            float acc = CMOS_B_GATE[n];
            for (int o = 0; o < CMOS_CONV_DIM; ++o)
                acc += CMOS_W_GATE[n * CMOS_CONV_DIM + o] * s_cv[o];
            s_g[n] = acc;
            if (acc > mx) mx = acc;
        }
        float se = 0.0f;
        for (int n = 0; n < CMOS_NMAP; ++n) {
            s_g[n] = expf(s_g[n] - mx);
            se += s_g[n];
        }
        for (int n = 0; n < CMOS_NMAP; ++n)
            s_g[n] /= se;

#ifdef CMOS_CAPTURE_STAGES
        cmos_st_mean[d] = m;
        cmos_st_std[d] = sd;
        for (int o = 0; o < CMOS_CONV_DIM; ++o)
            cmos_st_conv[d * CMOS_CONV_DIM + o] = s_cv[o];
        for (int n = 0; n < CMOS_NMAP; ++n)
            cmos_st_gate[d * CMOS_NMAP + n] = s_g[n];
#endif

        /* output sample t = j*seg + s: every mapping n maps the chunk
         * sequence at offset s (x[i*seg + s], i < L/seg) to chunk j; the
         * gate weights then mix the num_map results (the einsum). */
        for (int j = 0; j < CMOS_NO; ++j) {
            for (int s = 0; s < CMOS_SEG; ++s) {
                float out = 0.0f;
                for (int n = 0; n < CMOS_NMAP; ++n) {
                    const float *w = &CMOS_W_MAP[(n * CMOS_NO + j) * CMOS_NI];
                    float acc = CMOS_B_MAP[n * CMOS_NO + j];
                    for (int i = 0; i < CMOS_NI; ++i)
                        acc += w[i] * s_xn[i * CMOS_SEG + s];
                    out += acc * s_g[n];
                }
                FEA_OUT(yd, ys, emit, d, j * CMOS_SEG + s, out * sd + m);
            }
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void cmos_forward(const float *x, float *y) { cmos_run(x, y, 0, CMOS_D, CMOS_D, 0); }

/* One channel d (10-12): its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer (porting rule 1) instead of holding H x D. */
void cmos_forward_ch(const float *x, int d, float *yc) { cmos_run(x, yc, d, d + 1, 1, 0); }

/* One channel d (10-13): each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (fea_emit.h).  Same values as cmos_forward_ch, bit for bit. */
void cmos_forward_emit(const float *x, int d, fea_emit_fn emit) { cmos_run(x, 0, d, d + 1, 1, emit); }
