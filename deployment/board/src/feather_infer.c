/*
 * FEATHer float32 inference - ARENA build (week-2 buffer schedule).
 *
 * Mirrors baselines/FEATHer/FEATHer.py forward() exactly. One code path
 * serves both builds:
 *   - board build: compact static arena, no stage copies
 *   - test build (-DFEA_CAPTURE_STAGES): same computation, plus memcpy
 *     of every intermediate into fea_st_* for golden comparison
 *
 * Buffer schedule (floats):
 *   x (caller, 672)  normalized IN PLACE, free after gate + in_proj
 *   B (672)  xproj   - band source; free after LOW pool -> reused as
 *                      SPK channel-major h / h_agg
 *   C (672)  hidden  - gated band accumulator
 *   D (672)  band scratch (fP/fM/fL -> DTK out) -> reused as yspk / y
 *   LAT (768) DTK latent (causal conv runs IN PLACE via ring buffer);
 *             free outside DTK -> reused for gate spectra (49*3), LOW
 *             pooled seq (168), SPK agg-conv output (672)
 *   + mean/stdev (14), gate z/g (6)
 * Static total = 2,804 floats = 10.96 KB; + caller input 672 floats
 * = 13.6 KB end-to-end (target <=16 KB, SRAM 20 KB). Verified by
 * FEA_ARENA_BYTES in the header and the batch test printout.
 */
#include <math.h>
#include <string.h>

#include "feather_infer.h"
#include "feather_weights.h"   /* generated: golden/feather_weights.h */

#ifdef FEA_CAPTURE_STAGES
float fea_st_xnorm[FEA_L * FEA_D];
float fea_st_mean[FEA_D];
float fea_st_stdev[FEA_D];
float fea_st_xproj[FEA_L * FEA_D];
float fea_st_fP[FEA_L * FEA_D];
#if FEA_NB == 4
float fea_st_fH[FEA_L * FEA_D];
#endif
float fea_st_fM[FEA_L * FEA_D];
float fea_st_fLds[FEA_LP * FEA_D];
float fea_st_fL[FEA_L * FEA_D];
float fea_st_oP[FEA_L * FEA_D];
#if FEA_NB == 4
float fea_st_oH[FEA_L * FEA_D];
#endif
float fea_st_oM[FEA_L * FEA_D];
float fea_st_oL[FEA_L * FEA_D];
float fea_st_mlp[FEA_L * FEA_D];
float fea_st_rfft_re[FEA_NF];
float fea_st_rfft_im[FEA_NF];
float fea_st_magmean[FEA_NF];
float fea_st_gate_z[FEA_NB];
float fea_st_gate[FEA_NB];
float fea_st_hidden[FEA_L * FEA_D];
float fea_st_hagg[FEA_D * FEA_L];
float fea_st_yspk[FEA_H * FEA_D];
#define CAP(dst, src, n) memcpy((dst), (src), (n) * sizeof(float))
#else
#define CAP(dst, src, n) ((void)0)
#endif

/* ---- the arena ---- */
static float s_B[FEA_L * FEA_D];      /* xproj -> SPK h/h_agg      */
static float s_C[FEA_L * FEA_D];      /* hidden accumulator        */
static float s_D[FEA_L * FEA_D];      /* band scratch -> yspk -> y */
static float s_lat[FEA_LAT_N];        /* DTK latent / gate / pool /
                                         SPK agg out / MLP bottleneck */
static float s_mean[FEA_D], s_stdev[FEA_D];
static float s_gz[FEA_NB], s_g[FEA_NB];

/* causal depthwise conv IN PLACE (ring buffer of the last k-1 inputs);
 * saves a second 768-float latent buffer */
static void dwconv_causal_ip(float *x, const float *w, int len,
                             int nch, int k)
{
    for (int s = 0; s < nch; ++s) {
        float hist[FEA_K];               /* k <= FEA_K by construction */
        for (int j = 0; j < k; ++j)
            hist[j] = 0.0f;
        for (int t = 0; t < len; ++t) {
            float cur = x[t * nch + s];
            hist[t % k] = cur;
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) {
                int src = t - (k - 1) + j;   /* input index for tap j */
                if (src >= 0)
                    acc += w[s * k + j] * hist[src % k];
            }
            x[t * nch + s] = acc;
        }
    }
}

/* DTK: (L,D) src -> (L,D) dst (src == dst allowed) */
static void dtk(const float *src, float *dst)
{
    fea_linear_t(src, s_lat, W_DTK_IN, 0, FEA_L, FEA_D, FEA_S);
    dwconv_causal_ip(s_lat, W_DTK_K, FEA_L, FEA_S, FEA_K);
    fea_linear_t(s_lat, dst, W_DTK_OUT, 0, FEA_L, FEA_S, FEA_D);
}

/* FFT gate on x_norm; spectra scratch lives in s_lat (free here) */
static void freq_gate(const float *xn)
{
    float *re = s_lat;                    /* 49  */
    float *im = s_lat + FEA_NF;           /* 49  */
    float *mag = s_lat + 2 * FEA_NF;      /* 49  */

    for (int f = 0; f < FEA_NF; ++f)
        mag[f] = 0.0f;
    for (int d = 0; d < FEA_D; ++d) {
        fea_rfft96(xn + d, FEA_D, re, im);
#ifdef FEA_CAPTURE_STAGES
        if (d == 0) {
            CAP(fea_st_rfft_re, re, FEA_NF);
            CAP(fea_st_rfft_im, im, FEA_NF);
        }
#endif
        for (int f = 0; f < FEA_NF; ++f)
            mag[f] += sqrtf(re[f] * re[f] + im[f] * im[f]);
    }
    for (int f = 0; f < FEA_NF; ++f)
        mag[f] /= (float)FEA_D;
    CAP(fea_st_magmean, mag, FEA_NF);

    /* Conv1d(1 -> NB, k=5, pad=2) + AdaptiveAvgPool1d(1) */
    for (int c = 0; c < FEA_NB; ++c) {
        float acc = 0.0f;
        for (int f = 0; f < FEA_NF; ++f) {
            float z = B_GATE[c];
            for (int j = 0; j < 5; ++j) {
                int src = f + j - 2;
                if (src >= 0 && src < FEA_NF)
                    z += W_GATE[c * 5 + j] * mag[src];
            }
            acc += z;
        }
        s_gz[c] = acc / (float)FEA_NF;
    }
    fea_softmax(s_gz, s_g, FEA_NB);
    CAP(fea_st_gate_z, s_gz, FEA_NB);
    CAP(fea_st_gate, s_g, FEA_NB);
}

/*
 * In-place forward: x (FEA_L * FEA_D, time-major) is normalized and
 * consumed; the forecast (FEA_H * FEA_D, time-major) is returned in an
 * internal arena buffer valid until the next call.
 */
float *feather_forward_ip(float *x)
{
    /* half-RevIN normalize (in place - x is the arena input slot) */
    fea_instnorm_stats(x, FEA_L, FEA_D, s_mean, s_stdev);
    for (int t = 0; t < FEA_L; ++t)
        for (int d = 0; d < FEA_D; ++d)
            x[t * FEA_D + d] = (x[t * FEA_D + d] - s_mean[d]) / s_stdev[d];
    CAP(fea_st_mean, s_mean, FEA_D);
    CAP(fea_st_stdev, s_stdev, FEA_D);
    CAP(fea_st_xnorm, x, FEA_L * FEA_D);

    /* gate first, so each band folds into hidden immediately */
    freq_gate(x);

    /* input projection; x is free afterwards */
    fea_linear_t(x, s_B, W_INPROJ, B_INPROJ, FEA_L, FEA_D, FEA_D);
    CAP(fea_st_xproj, s_B, FEA_L * FEA_D);

    /* Bands fold into s_C in the order PyTorch stacks them, which fixes the
     * gate index: bands_list = [oP, oH, oM, oL] at num_bands=4. */

    /* POINT band (kernel 1) */
    fea_dwconv_same(s_B, s_D, W_CONVP, B_CONVP, FEA_L, FEA_D, 1);
    CAP(fea_st_fP, s_D, FEA_L * FEA_D);
    dtk(s_D, s_D);
    CAP(fea_st_oP, s_D, FEA_L * FEA_D);
    for (int i = 0; i < FEA_L * FEA_D; ++i)
        s_C[i] = s_g[0] * s_D[i];

    /* HIGH band (kernel 3) — exists only at num_bands == 4. PyTorch builds
     * conv_high solely in that case, so at 3 bands the weight array is absent
     * and the gate returns 3 weights, not 4. */
#if FEA_NB == 4
    fea_dwconv_same(s_B, s_D, W_CONVH, B_CONVH, FEA_L, FEA_D, 3);
    CAP(fea_st_fH, s_D, FEA_L * FEA_D);
    dtk(s_D, s_D);
    CAP(fea_st_oH, s_D, FEA_L * FEA_D);
    for (int i = 0; i < FEA_L * FEA_D; ++i)
        s_C[i] += s_g[1] * s_D[i];
#endif

    /* MID band (kernel 5). Gate index is 2 with a HIGH band, 1 without —
     * bands_list is [oP, oH, oM, oL] at 4 and [oP, oM, oL] at 3. */
    fea_dwconv_same(s_B, s_D, W_CONVM, B_CONVM, FEA_L, FEA_D, 5);
    CAP(fea_st_fM, s_D, FEA_L * FEA_D);
    dtk(s_D, s_D);
    CAP(fea_st_oM, s_D, FEA_L * FEA_D);
    for (int i = 0; i < FEA_L * FEA_D; ++i)
        s_C[i] += s_g[FEA_NB - 2] * s_D[i];

    /* LOW band (pooled seq parked in s_lat - free between DTKs) */
    fea_avgpool4(s_B, s_lat, FEA_L, FEA_D);
    CAP(fea_st_fLds, s_lat, FEA_LP * FEA_D);
    fea_interp_linear(s_lat, s_D, FEA_LP, FEA_L, FEA_D);
    CAP(fea_st_fL, s_D, FEA_L * FEA_D);
    dtk(s_D, s_D);
    CAP(fea_st_oL, s_D, FEA_L * FEA_D);
    for (int i = 0; i < FEA_L * FEA_D; ++i)
        s_C[i] += s_g[FEA_NB - 1] * s_D[i];
    CAP(fea_st_hidden, s_C, FEA_L * FEA_D);

    /* Residual MLP over the channel axis: h += W_up(GELU(W_down(h))).
     * FEATHer's only nonlinearity, and the one component the ablation shows
     * clearly earns its parameters. Bottleneck parks in s_lat, free here.
     * GELU is the exact erf form, matching torch.nn.functional.gelu's default
     * (approximate='none'); the tanh approximation differs by ~1e-3 and would
     * break the golden-vector tolerance. */
    fea_linear_t(s_C, s_lat, W_NLDOWN, B_NLDOWN, FEA_L, FEA_D, FEA_MLP_R);
    for (int i = 0; i < FEA_L * FEA_MLP_R; ++i)
        s_lat[i] = fea_gelu(s_lat[i]);
    for (int t = 0; t < FEA_L; ++t) {
        for (int d = 0; d < FEA_D; ++d) {
            float acc = B_NLUP[d];
            for (int r = 0; r < FEA_MLP_R; ++r)
                acc += W_NLUP[d * FEA_MLP_R + r] * s_lat[t * FEA_MLP_R + r];
            s_C[t * FEA_D + d] += acc;
        }
    }
    CAP(fea_st_mlp, s_C, FEA_L * FEA_D);

    /* SPK head: channel-major copy into s_B (free after LOW pool) */
    for (int d = 0; d < FEA_D; ++d)
        for (int t = 0; t < FEA_L; ++t)
            s_B[d * FEA_L + t] = s_C[t * FEA_D + d];

    /* residual + depthwise aggregation (out via s_lat, free here) */
    for (int d = 0; d < FEA_D; ++d)
        fea_dwconv_same(s_B + d * FEA_L, s_lat + d * FEA_L,
                        W_AGG + d * FEA_AGG_K, 0, FEA_L, 1, FEA_AGG_K);
    for (int i = 0; i < FEA_D * FEA_L; ++i)
        s_B[i] += s_lat[i];
    CAP(fea_st_hagg, s_B, FEA_D * FEA_L);

    /* phase-wise shared backbone -> yspk in s_D (free after bands) */
    for (int d = 0; d < FEA_D; ++d) {
        for (int ph = 0; ph < FEA_PERIOD; ++ph) {
            float v[FEA_N];
            for (int in = 0; in < FEA_N; ++in)
                v[in] = s_B[d * FEA_L + in * FEA_PERIOD + ph];
            for (int om = 0; om < FEA_M; ++om) {
                float acc = 0.0f;
                for (int in = 0; in < FEA_N; ++in)
                    acc += W_BB[om * FEA_N + in] * v[in];
                s_D[(om * FEA_PERIOD + ph) * FEA_D + d] = acc;
            }
        }
    }
    CAP(fea_st_yspk, s_D, FEA_H * FEA_D);

    /* denormalize in place -> s_D is the forecast */
    for (int t = 0; t < FEA_H; ++t)
        for (int d = 0; d < FEA_D; ++d)
            s_D[t * FEA_D + d] =
                s_D[t * FEA_D + d] * s_stdev[d] + s_mean[d];
    return s_D;
}

/* const-input convenience wrapper (tests, batch eval).
 *
 * Stages the input through s_D rather than a private buffer. s_D is not
 * touched until the POINT band, which runs after in_proj has consumed x, so
 * the aliasing is safe — and it is the same slot the forecast ends up in, so
 * feather_forward_ip's return value is still correct.
 *
 * This is not a micro-optimisation: at d_state 16 the arena reaches 14.0 KB,
 * and a separate 2.7 KB input copy pushed the F103RB's 20 KB SRAM over by
 * 1,239 bytes. Dropping it leaves ~1.4 KB of headroom. Keep the ordering in
 * mind before moving anything above the POINT band.
 */
void feather_forward(const float *x, float *y)
{
    memcpy(s_D, x, FEA_L * FEA_D * sizeof(float));
    const float *out = feather_forward_ip(s_D);
    memcpy(y, out, FEA_H * FEA_D * sizeof(float));
}
