/*
 * MixLinear float32 inference. Transcribed from the bit-exact scalar
 * reference (deployment/board/_mixlinear_scalar.py). All buffers static
 * (no malloc); complex arithmetic kept as explicit re/im pairs.
 *
 * Weight layout (row-major, from export_golden_mixlinear.py):
 *   ML_CONV[25]          conv1d filter (shared 1->1)
 *   ML_TL1[4]/ML_TL2[4]  Linear(2,2) as [out*2 + in]
 *   ML_FL1_{RE,IM}[8]    complex Linear(4->2) as [out*4 + in]
 *   ML_FL2_{RE,IM}[8]    complex Linear(2->4) as [out*2 + in]
 *   ML_COS4/ML_SIN4[4]   cos/sin(2*pi*i/4)
 */
#include <string.h>

#include "mixlinear_infer.h"
/* Plain names, not "../golden/...": the x86 build passes -Igolden, and the
 * CubeIDE project flattens every header into Core/Inc, where a relative path
 * out of Core/Src does not resolve. Every other model's port already does it
 * this way. */
#include "mixlinear_weights.h"
#include "mixlinear_twiddle.h"

#define ML_ALPHA 0.5f
#define ML_PAD (ML_P / 2)             /* conv padding = 12 */

/* Output step s = k*P + p is emitted as (segment k, phase p).  That covers
 * every step only when the horizon is a whole number of periods, which holds
 * for every horizon we run (96/192/336/720 with P=24). */
_Static_assert(ML_H % ML_P == 0,
               "MixLinear port assumes ML_H is a multiple of ML_P");

/* PER-CHANNEL WORKING SET (2026-09-04, front end 2026-09-14) - equal-effort
 * rule, see deployment/board/README.md.  MixLinear is channel-independent:
 * the mean, the conv, both branches and the final mix all act on one channel,
 * so nothing needs to be held for all D at once.  Full-width copies are
 * test-build only.
 *
 * OUTPUT SIDE (2026-09-14).  Output step s = k*P + p depends only on phase
 * p's segment, so each phase is written straight into y as soon as both
 * branches are known for it - nothing of length H is held.  The time branch
 * also no longer evaluates the sq_y^2*P - H cells upstream computes and then
 * slices off ([:pred_len]).  Every value that IS computed keeps its terms and
 * their order, so the forecast is bit-identical; the arena loses the
 * (XT_N + XF_N)-float branch buffers (6.3 KB at H=720).  The per-phase
 * temporaries that grow with H (t1, the freq spectrum) are static so the
 * footprint table sees them; the fixed 4-float ones stay on the stack. */
static float ml_xc[ML_L];               /* this channel, centered -> conv+res */
static float ml_row[ML_L];              /* pre-conv copy of the same row      */
static float ml_t1[ML_SQX * ML_SQY];    /* time branch, first factor, 1 phase */
static float ml_zre[ML_SEGY], ml_zim[ML_SEGY];  /* freq spectrum, 1 phase     */

#ifdef ML_CAPTURE_STAGES
float mixlinear_st_mean[ML_D];
float mixlinear_st_xt[ML_XT_N * ML_D];
float mixlinear_st_xf[ML_XF_N * ML_D];
#endif

void mixlinear_forward(const float *x, float *y)
{
    int c, s, p, j, k, sx, o, o2, i, r, col;

    for (c = 0; c < ML_D; c++) {
        /* 1. mean over time of this channel */
        float m = 0.0f;
        for (s = 0; s < ML_L; s++) m += x[s * ML_D + c];
        m = m / (float)ML_L;
#ifdef ML_CAPTURE_STAGES
        mixlinear_st_mean[c] = m;
#endif

        /* 2. (x - mean) */
        for (s = 0; s < ML_L; s++)
            ml_xc[s] = x[s * ML_D + c] - m;

        /* 3. shared conv1d(1->1, same pad, k=25) + residual, in place */
        memcpy(ml_row, ml_xc, ML_L * sizeof(float));
        for (s = 0; s < ML_L; s++) {
            float acc = 0.0f;
            for (j = 0; j < ML_K; j++) {
                int ip = s + j - ML_PAD;
                if (ip >= 0 && ip < ML_L) acc += ML_CONV[j] * ml_row[ip];
            }
            ml_xc[s] = acc + ml_row[s];
        }

        /* 4/5/6. per period-phase: both branches, then emit that phase */
        for (p = 0; p < ML_P; p++) {
            float seg[ML_SEGX];
            for (sx = 0; sx < ML_SEGX; sx++)
                seg[sx] = ml_xc[sx * ML_P + p];       /* xr[c][p][sx] */

            /* ---- TIME branch: grid[r][col] = seg[r*2+col] ----
             * TLinear1: (sq_x -> sq_y) over the grid columns.
             * t1[r][o] = sum_col TL1[o][col] * grid[r][col],  r < sq_x */
            for (r = 0; r < ML_SQX; r++)
                for (o = 0; o < ML_SQY; o++) {
                    float a = 0.0f;
                    for (col = 0; col < ML_SQX; col++)
                        a += ML_TL1[o * ML_SQX + col] * seg[r * ML_SQX + col];
                    ml_t1[r * ML_SQY + o] = a;
                }

            /* ---- FREQ branch: len-4 FFT (real in) ---- */
            float Xre[ML_LPF], Xim[ML_LPF];
            for (j = 0; j < ML_LPF; j++) {
                float re = 0.0f, im = 0.0f;
                for (sx = 0; sx < ML_SEGX; sx++) {
                    int t = (j * sx) % ML_SEGX;
                    re += seg[sx] * ML_COS4[t];
                    im -= seg[sx] * ML_SIN4[t];
                }
                Xre[j] = re; Xim[j] = im;
            }
            /* FLinear1 complex (4->2) */
            float Y1re[2], Y1im[2];
            for (o = 0; o < 2; o++) {
                float re = 0.0f, im = 0.0f;
                for (j = 0; j < ML_LPF; j++) {
                    re += Xre[j] * ML_FL1_RE[o * ML_LPF + j] - Xim[j] * ML_FL1_IM[o * ML_LPF + j];
                    im += Xre[j] * ML_FL1_IM[o * ML_LPF + j] + Xim[j] * ML_FL1_RE[o * ML_LPF + j];
                }
                Y1re[o] = re; Y1im[o] = im;
            }
            /* FLinear2 complex (2->seg_y) */
            for (o = 0; o < ML_SEGY; o++) {
                float re = 0.0f, im = 0.0f;
                for (i = 0; i < 2; i++) {
                    re += Y1re[i] * ML_FL2_RE[o * 2 + i] - Y1im[i] * ML_FL2_IM[o * 2 + i];
                    im += Y1re[i] * ML_FL2_IM[o * 2 + i] + Y1im[i] * ML_FL2_RE[o * 2 + i];
                }
                ml_zre[o] = re; ml_zim[o] = im;
            }

            /* ---- emit this phase: steps s = k*P + p, k < seg_y = H/P ----
             * time: TLinear2 (sq_x -> sq_y) over the grid rows, cell
             *       k = o2*sq_y + o  (cells k >= seg_y overshoot the horizon
             *       and upstream slices them off, so they are not evaluated)
             * freq: iFFT real over seg_y points,
             *       xf[k] = (1/N) sum_j (Zre*cos - Zim*sin).
             *       The twiddle modulus is seg_y, NOT seg_x.  They coincide at
             *       H=96 (both 4), so the seg_x table was being used for both
             *       transforms and every other horizon came out wrong from the
             *       second output period onward.
             * mix:  y = x_t*alpha + mean + x_f*(1-alpha) */
            for (k = 0; k < ML_SEGY; k++) {
                o2 = k / ML_SQY;
                o = k % ML_SQY;
                float xt = 0.0f;
                for (r = 0; r < ML_SQX; r++)
                    xt += ML_TL2[o2 * ML_SQX + r] * ml_t1[r * ML_SQY + o];

                float a = 0.0f;
                for (j = 0; j < ML_SEGY; j++) {
                    int t = (j * k) % ML_SEGY;
                    a += ml_zre[j] * ML_COSY[t] - ml_zim[j] * ML_SINY[t];
                }
                float xf = a / (float)ML_SEGY;

                s = k * ML_P + p;
                y[s * ML_D + c] = xt * ML_ALPHA
                                + m
                                + xf * (1.0f - ML_ALPHA);
#ifdef ML_CAPTURE_STAGES
                mixlinear_st_xt[s * ML_D + c] = xt;
                mixlinear_st_xf[s * ML_D + c] = xf;
#endif
            }
        }
    }
}
