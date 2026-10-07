/*
 * SAMformer forward, mirroring SAMFormerArchitecture.forward (use_revin=True):
 *   RevIN norm per channel (biased var, eps 1e-5 inside sqrt, affine w/b) ->
 *   channel tokens: q, k = Linear(L -> hid), v = Linear(L -> L) ->
 *   softmax(q k^T / sqrt(hid)) over the D channels, attention = A v ->
 *   out = x_norm + attention -> Linear(L -> H) ->
 *   RevIN denorm ((y - b) / (w + eps^2) * std + mean).
 */
#include "samformer_infer.h"
#include "samformer_weights.h"      /* generated: golden/samformer_weights.h */

#include <math.h>

_Static_assert(SAM_H == SAM_H_GEN,
               "SAM_H does not match the horizon the weights were exported at - re-run export_golden_samformer.py --pred_len");
_Static_assert(SAM_L == SAM_L_GEN,
               "SAM_L does not match the input length the weights were exported at - re-run export_golden_samformer.py --seq_len");
_Static_assert(SAM_D == SAM_D_GEN,
               "SAM_D does not match the width the weights were exported at (RevIN affine is per channel) - re-run export_golden_samformer.py --d_model");

#define SAM_EPS 1e-5f

/* WORKING SET - equal-effort rule (README).  SAMformer mixes channels, so it
 * holds what the mixing needs for all D channels and nothing more: the
 * normalised input (read by k, v and the residual), the keys and the values.
 * A query is used once, so it is made per channel when its row is computed.
 * The residual sum overwrites that channel's normalised row (no longer read:
 * all keys and values are made by then), and the forecast goes straight to
 * the caller's buffer.  No recomputation. */
#ifdef SAM_CAPTURE_STAGES
float sam_st_mean[SAM_D];
float sam_st_std[SAM_D];
float sam_st_attw[SAM_D * SAM_D];
float sam_st_out[SAM_D * SAM_L];
#endif

static float s_xn[SAM_D * SAM_L];     /* normalised input, channel-major; row i
                                         becomes x_norm_i + attention_i       */
static float s_v[SAM_D * SAM_L];      /* values                               */
static float s_k[SAM_D * SAM_HID];    /* keys                                 */
static float s_mean[SAM_D], s_std[SAM_D];
static float s_q[SAM_HID];
static float s_a[SAM_D];              /* one softmax row                      */

void samformer_forward(const float *x, float *y)
{
    /* RevIN statistics + normalisation */
    for (int d = 0; d < SAM_D; ++d) {
        float m = 0.0f;
        for (int t = 0; t < SAM_L; ++t)
            m += x[t * SAM_D + d];
        m /= (float)SAM_L;
        float v = 0.0f;
        for (int t = 0; t < SAM_L; ++t) {
            float c = x[t * SAM_D + d] - m;
            v += c * c;
        }
        v /= (float)SAM_L;
        const float sd = sqrtf(v + SAM_EPS);
        s_mean[d] = m;
        s_std[d] = sd;
        for (int t = 0; t < SAM_L; ++t)
            s_xn[d * SAM_L + t] = (x[t * SAM_D + d] - m) / sd * SAM_RV_W[d] + SAM_RV_B[d];
    }

    /* keys and values of every channel token */
    for (int d = 0; d < SAM_D; ++d) {
        const float *xd = &s_xn[d * SAM_L];
        for (int h = 0; h < SAM_HID; ++h) {
            float acc = SAM_BK[h];
            for (int t = 0; t < SAM_L; ++t)
                acc += SAM_WK[h * SAM_L + t] * xd[t];
            s_k[d * SAM_HID + h] = acc;
        }
        for (int o = 0; o < SAM_L; ++o) {
            float acc = SAM_BV[o];
            for (int t = 0; t < SAM_L; ++t)
                acc += SAM_WV[o * SAM_L + t] * xd[t];
            s_v[d * SAM_L + o] = acc;
        }
    }

    const float scale = 1.0f / sqrtf((float)SAM_HID);
    for (int i = 0; i < SAM_D; ++i) {
        float *xi = &s_xn[i * SAM_L];
        for (int h = 0; h < SAM_HID; ++h) {
            float acc = SAM_BQ[h];
            for (int t = 0; t < SAM_L; ++t)
                acc += SAM_WQ[h * SAM_L + t] * xi[t];
            s_q[h] = acc;
        }
        float mx = -INFINITY;
        for (int j = 0; j < SAM_D; ++j) {
            float acc = 0.0f;
            for (int h = 0; h < SAM_HID; ++h)
                acc += s_q[h] * s_k[j * SAM_HID + h];
            s_a[j] = acc * scale;
            if (s_a[j] > mx) mx = s_a[j];
        }
        float se = 0.0f;
        for (int j = 0; j < SAM_D; ++j) {
            s_a[j] = expf(s_a[j] - mx);
            se += s_a[j];
        }
        for (int j = 0; j < SAM_D; ++j)
            s_a[j] /= se;

        /* residual: x_norm_i + sum_j A_ij v_j, in place */
        for (int t = 0; t < SAM_L; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < SAM_D; ++j)
                acc += s_a[j] * s_v[j * SAM_L + t];
            xi[t] += acc;
        }
#ifdef SAM_CAPTURE_STAGES
        sam_st_mean[i] = s_mean[i];
        sam_st_std[i] = s_std[i];
        for (int j = 0; j < SAM_D; ++j)
            sam_st_attw[i * SAM_D + j] = s_a[j];
        for (int t = 0; t < SAM_L; ++t)
            sam_st_out[i * SAM_L + t] = xi[t];
#endif

        /* linear forecaster + RevIN denorm */
        const float aw = SAM_RV_W[i] + SAM_EPS * SAM_EPS;
        for (int o = 0; o < SAM_H; ++o) {
            float acc = SAM_BF[o];
            for (int t = 0; t < SAM_L; ++t)
                acc += SAM_WF[o * SAM_L + t] * xi[t];
            y[o * SAM_D + i] = (acc - SAM_RV_B[i]) / aw * s_std[i] + s_mean[i];
        }
    }
}
