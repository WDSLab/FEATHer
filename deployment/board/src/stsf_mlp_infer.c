/*
 * SparseTSF-MLP forward, mirroring
 * baselines/SparseTSF/SparseTSF-main/models/SparseTSF.py with
 * model_type='mlp':
 *   mean-subtract -> shared 1x25 'same' conv + residual ->
 *   phase-wise  Linear(n -> d_model) -> ReLU -> Linear(d_model -> m)
 *   -> add mean back.
 *
 * Only the head differs from stsf_infer.c; the front end is byte-identical,
 * which is what makes the two a controlled pair on the board.
 */
#include <string.h>

#include "stsf_mlp_infer.h"
#include "stsf_mlp_weights.h"   /* generated: golden/stsf_mlp_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(SMLP_H == SMLP_H_GEN,
               "SMLP_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");
_Static_assert(SMLP_DM == SMLP_DM_GEN,
               "stsf_mlp_geom.h does not match the hidden width of the exported weights - re-run export_golden_stsf_mlp.py");

/* PER-CHANNEL WORKING SET (2026-09-14) - equal-effort rule, see
 * deployment/board/README.md.  Identical front end to stsf_infer.c and the
 * same fix: one channel wide instead of 2*D*L floats.  Kept in lock-step with
 * the linear variant so the pair stays a controlled comparison.
 *
 * The full hidden tensor is D*period*128 floats = 86 KB — four times the
 * board's entire SRAM. It exists only so the x86 golden test can check every
 * unit of the MLP, so it is compiled in exclusively for that build, the same
 * way the stage buffers above it are. The board build computes the
 * identical arithmetic and keeps just the 128-float working row. */
#ifdef SMLP_CAPTURE_STAGES
float smlp_st_mean[SMLP_D];
float smlp_st_xagg[SMLP_D * SMLP_L];
float smlp_st_hid[SMLP_D * SMLP_P * SMLP_DM];
#define SMLP_CAP(d, ph, src) \
    memcpy(smlp_st_hid + ((d) * SMLP_P + (ph)) * SMLP_DM, (src), \
           SMLP_DM * sizeof(float))
#else
#define SMLP_CAP(d, ph, src) ((void)0)
#endif

static float s_xc[SMLP_L];    /* one channel: centered, then conv+residual */
static float s_cv[SMLP_L];    /* conv output, one channel                  */
static float s_h[SMLP_DM];    /* hidden for one (channel, phase)           */

static void stsf_mlp_run(const float *x, float *y, int d0, int d1, int ys, fea_emit_fn emit)
{
    for (int d = d0; d < d1; ++d) {
        float *yd = y ? ((ys == 1) ? y : y + d) : 0;   /* this channel's outputs */
        /* per-channel mean only (SparseTSF has no variance norm).  Same
         * accumulation order as fea_instnorm_stats, so the value is
         * bit-identical to the all-channel build. */
        float m = 0.0f;
        for (int t = 0; t < SMLP_L; ++t)
            m += x[t * SMLP_D + d];
        m /= (float)SMLP_L;

        for (int t = 0; t < SMLP_L; ++t)
            s_xc[t] = x[t * SMLP_D + d] - m;

        /* one shared conv kernel + residual, folded back into s_xc */
        fea_dwconv_same(s_xc, s_cv, SMLP_W_CONV, 0, SMLP_L, 1, SMLP_K);
        for (int t = 0; t < SMLP_L; ++t)
            s_xc[t] += s_cv[t];

#ifdef SMLP_CAPTURE_STAGES
        smlp_st_mean[d] = m;
        for (int t = 0; t < SMLP_L; ++t)
            smlp_st_xagg[d * SMLP_L + t] = s_xc[t];
#endif

        /* phase-wise MLP: v[n] -> hidden[128] -> out[m], per phase */
        for (int ph = 0; ph < SMLP_P; ++ph) {
            float v[SMLP_N];
            for (int in = 0; in < SMLP_N; ++in)
                v[in] = s_xc[in * SMLP_P + ph];

            for (int h = 0; h < SMLP_DM; ++h) {
                float acc = SMLP_B_FC1[h];
                for (int in = 0; in < SMLP_N; ++in)
                    acc += SMLP_W_FC1[h * SMLP_N + in] * v[in];
                s_h[h] = acc > 0.0f ? acc : 0.0f;         /* ReLU */
            }
            SMLP_CAP(d, ph, s_h);

            for (int om = 0; om < SMLP_M; ++om) {
                float acc = SMLP_B_FC2[om];
                for (int h = 0; h < SMLP_DM; ++h)
                    acc += SMLP_W_FC2[om * SMLP_DM + h] * s_h[h];
                FEA_OUT(yd, ys, emit, d, om * SMLP_P + ph, acc + m);
            }
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void stsf_mlp_forward(const float *x, float *y) { stsf_mlp_run(x, y, 0, SMLP_D, SMLP_D, 0); }

/* One channel d (10-12): its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer (porting rule 1) instead of holding H x D. */
void stsf_mlp_forward_ch(const float *x, int d, float *yc) { stsf_mlp_run(x, yc, d, d + 1, 1, 0); }

/* One channel d (10-13): each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (fea_emit.h).  Same values as stsf_mlp_forward_ch, bit for bit. */
void stsf_mlp_forward_emit(const float *x, int d, fea_emit_fn emit) { stsf_mlp_run(x, 0, d, d + 1, 1, emit); }
