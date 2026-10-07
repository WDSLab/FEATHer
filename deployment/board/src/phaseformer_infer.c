/*
 * PhaseFormer forward (eval: every dropout off), mirroring
 * baselines/PhaseFormer/PhaseFormer-main/models/PhaseFormer.py:
 *   RevIN (biased var, eps 1e-5, no affine) ->
 *   phase series: token p (p < period) carries x[n*P + p], n < n_x ->
 *   embedding Linear(n_x -> latent) + LayerNorm ->
 *   routing units (layers > 1: unit 0 no in-proj, last unit no out-proj;
 *   layers = 1: neither):
 *     z += LayerNorm(in_proj(phase series))           [in-proj units]
 *     x  = z (+ phase position embedding)
 *     routers (R x latent) attend over the P tokens  -> router buffer
 *     tokens attend over the router buffer           -> r
 *     z  = LN2(u + MLP(u)), u = LN1(x + r)           [MLP latent -> 4 latent -> latent, GELU]
 *     phase series = out_proj(z) (Linear latent -> n_x) [out-proj units]
 *   predictor Linear(latent -> n_y) -> sample t = j*P + p -> RevIN^-1.
 * Attention = TSLib AttentionLayer/FullAttention: q/k/v Linear(latent, latent),
 * heads split the latent, softmax(q.k / sqrt(head_dim)), out Linear(latent, latent).
 */
#include "phaseformer_infer.h"
#include "phaseformer_weights.h"      /* generated: golden/phaseformer_weights.h */

#include <math.h>

_Static_assert(PF_H == PF_H_GEN,
               "PF_H does not match the horizon the weights were exported at - re-run export_golden_phaseformer.py --pred_len");
_Static_assert(PF_L == PF_L_GEN,
               "PF_L does not match the input length the weights were exported at - re-run export_golden_phaseformer.py --seq_len");
/* upstream pads the input circularly / trims the output when P does not
 * divide L / H; every setting we run has P | L and P | H */
_Static_assert(PF_L % PF_P == 0 && PF_H % PF_P == 0,
               "period must divide L and H (the circular-padding path is not ported)");
_Static_assert(PF_DL % PF_NH == 0, "latent must split evenly over the heads");

#define PF_NX (PF_L / PF_P)        /* periods in   */
#define PF_NY (PF_H / PF_P)        /* periods out  */
#define PF_HD (PF_DL / PF_NH)      /* head dim     */
#define PF_DM (4 * PF_DL)          /* MLP hidden   */
#define PF_DD (PF_DL * PF_DL)
#define PF_MAXS (PF_P > PF_R ? PF_P : PF_R)

/* per routing-unit weight blocks (stacked over layers by the export) */
#define PF_ATT_W(base, li) (&(base)[(li) * PF_DD])
#define PF_ATT_B(base, li) (&(base)[(li) * PF_DL])

/* PER-CHANNEL WORKING SET - equal-effort rule (README).  PhaseFormer is
 * channel-independent ("without cross-channel fusion"), so one channel's
 * tokens are live at a time.  Buffers that are never live together share
 * storage (rule 3): the routers' query projection is dead before their
 * key/value projections are made.  Nothing is recomputed (rule 4); outputs
 * go straight to the caller's buffer (rule 2). */
#ifdef PF_CAPTURE_STAGES
float pf_st_mu[PF_D];
float pf_st_sigma[PF_D];
float pf_st_z0[PF_D * PF_P * PF_DL];
float pf_st_zl[PF_D * PF_P * PF_DL];
#endif

static float s_ps[PF_P * PF_NX];      /* phase series (input, then out-proj)   */
static float s_z[PF_P * PF_DL];       /* latent tokens                         */
static float s_x[PF_P * PF_DL];       /* routing-layer input (z + position)    */
static float s_k[PF_P * PF_DL];       /* token keys   (sender stage)           */
static float s_v[PF_P * PF_DL];       /* token values (sender stage)           */
static float s_rb[PF_R * PF_DL];      /* router buffer                         */
static float s_rq[PF_R * PF_DL];      /* router queries, then buffer keys      */
static float s_rv[PF_R * PF_DL];      /* buffer values                         */
static float s_sc[PF_MAXS];           /* one attention row                     */
static float s_t1[PF_DL], s_t2[PF_DL], s_t3[PF_DL];
static float s_hm[PF_DM];             /* MLP hidden, one token                 */

/* y = W x + b, W (nout, nin) row-major */
static void lin(const float *w, const float *b, const float *x, float *y, int nin, int nout)
{
    for (int o = 0; o < nout; ++o) {
        float acc = b[o];
        for (int i = 0; i < nin; ++i)
            acc += w[o * nin + i] * x[i];
        y[o] = acc;
    }
}

/* nn.LayerNorm(n), eps 1e-5, in place */
static void layernorm(float *x, const float *g, const float *b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; ++i)
        m += x[i];
    m /= (float)n;
    float v = 0.0f;
    for (int i = 0; i < n; ++i) {
        float c = x[i] - m;
        v += c * c;
    }
    v /= (float)n;
    const float r = sqrtf(v + 1e-5f);
    for (int i = 0; i < n; ++i)
        x[i] = (x[i] - m) / r * g[i] + b[i];
}

/* one query row against S key/value rows, all heads; out = concat of heads */
static void attend(const float *q, const float *k, const float *v, int S, float *out)
{
    const float scale = 1.0f / sqrtf((float)PF_HD);
    for (int h = 0; h < PF_NH; ++h) {
        const int o = h * PF_HD;
        float mx = -INFINITY;
        for (int s = 0; s < S; ++s) {
            float acc = 0.0f;
            for (int e = 0; e < PF_HD; ++e)
                acc += q[o + e] * k[s * PF_DL + o + e];
            s_sc[s] = scale * acc;
            if (s_sc[s] > mx) mx = s_sc[s];
        }
        float se = 0.0f;
        for (int s = 0; s < S; ++s) {
            s_sc[s] = expf(s_sc[s] - mx);
            se += s_sc[s];
        }
        for (int e = 0; e < PF_HD; ++e) {
            float acc = 0.0f;
            for (int s = 0; s < S; ++s)
                acc += (s_sc[s] / se) * v[s * PF_DL + o + e];
            out[o + e] = acc;
        }
    }
}

static void routing_layer(int li)
{
    /* x = z (+ phase position embedding) */
    for (int p = 0; p < PF_P; ++p)
        for (int i = 0; i < PF_DL; ++i)
            s_x[p * PF_DL + i] = s_z[p * PF_DL + i]
#if PF_PE
                + PF_POS[(li * PF_P + p) * PF_DL + i]
#endif
                ;

    /* stage 1 - routers aggregate the tokens */
    for (int r = 0; r < PF_R; ++r)
        lin(PF_ATT_W(PF_SQ_W, li), PF_ATT_B(PF_SQ_B, li),
            &PF_ROUTER[(li * PF_R + r) * PF_DL], &s_rq[r * PF_DL], PF_DL, PF_DL);
    for (int p = 0; p < PF_P; ++p) {
        lin(PF_ATT_W(PF_SK_W, li), PF_ATT_B(PF_SK_B, li), &s_x[p * PF_DL], &s_k[p * PF_DL], PF_DL, PF_DL);
        lin(PF_ATT_W(PF_SV_W, li), PF_ATT_B(PF_SV_B, li), &s_x[p * PF_DL], &s_v[p * PF_DL], PF_DL, PF_DL);
    }
    for (int r = 0; r < PF_R; ++r) {
        attend(&s_rq[r * PF_DL], s_k, s_v, PF_P, s_t1);
        lin(PF_ATT_W(PF_SO_W, li), PF_ATT_B(PF_SO_B, li), s_t1, &s_rb[r * PF_DL], PF_DL, PF_DL);
    }

    /* stage 2 - tokens read the router buffer (s_rq now holds its keys) */
    for (int r = 0; r < PF_R; ++r) {
        lin(PF_ATT_W(PF_RK_W, li), PF_ATT_B(PF_RK_B, li), &s_rb[r * PF_DL], &s_rq[r * PF_DL], PF_DL, PF_DL);
        lin(PF_ATT_W(PF_RV_W, li), PF_ATT_B(PF_RV_B, li), &s_rb[r * PF_DL], &s_rv[r * PF_DL], PF_DL, PF_DL);
    }
    for (int p = 0; p < PF_P; ++p) {
        const float *xp = &s_x[p * PF_DL];
        lin(PF_ATT_W(PF_RQ_W, li), PF_ATT_B(PF_RQ_B, li), xp, s_t1, PF_DL, PF_DL);
        attend(s_t1, s_rq, s_rv, PF_R, s_t2);
        lin(PF_ATT_W(PF_RO_W, li), PF_ATT_B(PF_RO_B, li), s_t2, s_t3, PF_DL, PF_DL);
        for (int i = 0; i < PF_DL; ++i)                       /* u = LN1(x + r) */
            s_t3[i] += xp[i];
        layernorm(s_t3, &PF_LN1_G[li * PF_DL], &PF_LN1_B[li * PF_DL], PF_DL);
        lin(&PF_M1_W[li * PF_DM * PF_DL], &PF_M1_B[li * PF_DM], s_t3, s_hm, PF_DL, PF_DM);
        for (int i = 0; i < PF_DM; ++i)                       /* exact GELU */
            s_hm[i] = 0.5f * s_hm[i] * (1.0f + erff(s_hm[i] * 0.70710678118654752f));
        lin(&PF_M2_W[li * PF_DL * PF_DM], &PF_M2_B[li * PF_DL], s_hm, s_t1, PF_DM, PF_DL);
        float *zp = &s_z[p * PF_DL];
        for (int i = 0; i < PF_DL; ++i)
            zp[i] = s_t3[i] + s_t1[i];
        layernorm(zp, &PF_LN2_G[li * PF_DL], &PF_LN2_B[li * PF_DL], PF_DL);
    }
}

void phaseformer_forward(const float *x, float *y)
{
    for (int d = 0; d < PF_D; ++d) {
        /* RevIN */
        float mu = 0.0f;
        for (int t = 0; t < PF_L; ++t)
            mu += x[t * PF_D + d];
        mu /= (float)PF_L;
        float v = 0.0f;
        for (int t = 0; t < PF_L; ++t) {
            float c = x[t * PF_D + d] - mu;
            v += c * c;
        }
        v /= (float)PF_L;
        const float sg = sqrtf(v + 1e-5f);
        for (int n = 0; n < PF_NX; ++n)
            for (int p = 0; p < PF_P; ++p)
                s_ps[p * PF_NX + n] = (x[(n * PF_P + p) * PF_D + d] - mu) / sg;

        /* embedding */
        for (int p = 0; p < PF_P; ++p) {
            lin(PF_EMB_W, PF_EMB_B, &s_ps[p * PF_NX], &s_z[p * PF_DL], PF_NX, PF_DL);
            layernorm(&s_z[p * PF_DL], PF_EMB_G, PF_EMB_BT, PF_DL);
        }
#ifdef PF_CAPTURE_STAGES
        pf_st_mu[d] = mu;
        pf_st_sigma[d] = sg;
        for (int i = 0; i < PF_P * PF_DL; ++i)
            pf_st_z0[d * PF_P * PF_DL + i] = s_z[i];
#endif

        for (int li = 0; li < PF_NL; ++li) {
#if PF_NL > 1
            if (li > 0) {           /* in-proj: z += LN(Linear(n_x -> latent)(phase series)) */
                for (int p = 0; p < PF_P; ++p) {
                    lin(&PF_IN_W[(li - 1) * PF_DL * PF_NX], &PF_IN_B[(li - 1) * PF_DL],
                        &s_ps[p * PF_NX], s_t1, PF_NX, PF_DL);
                    layernorm(s_t1, &PF_IN_G[(li - 1) * PF_DL], &PF_IN_BT[(li - 1) * PF_DL], PF_DL);
                    for (int i = 0; i < PF_DL; ++i)
                        s_z[p * PF_DL + i] += s_t1[i];
                }
            }
#endif
            routing_layer(li);
#if PF_NL > 1
            if (li < PF_NL - 1)     /* out-proj: phase series for the next unit */
                for (int p = 0; p < PF_P; ++p)
                    lin(&PF_OUT_W[li * PF_NX * PF_DL], &PF_OUT_B[li * PF_NX],
                        &s_z[p * PF_DL], &s_ps[p * PF_NX], PF_DL, PF_NX);
#endif
        }
#ifdef PF_CAPTURE_STAGES
        for (int i = 0; i < PF_P * PF_DL; ++i)
            pf_st_zl[d * PF_P * PF_DL + i] = s_z[i];
#endif

        /* predictor + reassembly + RevIN^-1 */
        for (int p = 0; p < PF_P; ++p)
            for (int j = 0; j < PF_NY; ++j) {
                float acc = PF_PRED_B[j];
                for (int i = 0; i < PF_DL; ++i)
                    acc += PF_PRED_W[j * PF_DL + i] * s_z[p * PF_DL + i];
                y[(j * PF_P + p) * PF_D + d] = acc * sg + mu;
            }
    }
}
