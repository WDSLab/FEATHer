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
#include "mixlinear_infer.h"
#include "../golden/mixlinear_weights.h"
#include "../golden/mixlinear_twiddle.h"

#define ML_ALPHA 0.5f
#define ML_PAD (ML_P / 2)             /* conv padding = 12 */

float mixlinear_st_mean[ML_D];
float mixlinear_st_xt[ML_H * ML_D];
float mixlinear_st_xf[ML_H * ML_D];

static float ml_xm[ML_D][ML_L];       /* channel-major (x - mean) */
static float ml_xc[ML_D][ML_L];       /* conv + residual          */

void mixlinear_forward(const float *x, float *y)
{
    int c, s, p, j, k, sx, o, o2, i, r, col;

    /* 1. per-channel mean over time */
    for (c = 0; c < ML_D; c++) {
        float acc = 0.0f;
        for (s = 0; s < ML_L; s++) acc += x[s * ML_D + c];
        mixlinear_st_mean[c] = acc / (float)ML_L;
    }

    /* 2. channel-major (x - mean) */
    for (c = 0; c < ML_D; c++)
        for (s = 0; s < ML_L; s++)
            ml_xm[c][s] = x[s * ML_D + c] - mixlinear_st_mean[c];

    /* 3. shared conv1d(1->1, same pad, k=25) + residual */
    for (c = 0; c < ML_D; c++) {
        for (s = 0; s < ML_L; s++) {
            float acc = 0.0f;
            for (j = 0; j < ML_K; j++) {
                int ip = s + j - ML_PAD;
                if (ip >= 0 && ip < ML_L) acc += ML_CONV[j] * ml_xm[c][ip];
            }
            ml_xc[c][s] = acc + ml_xm[c][s];
        }
    }

    /* 4/5. per (channel, period-phase): time + freq branch */
    for (c = 0; c < ML_D; c++) {
        for (p = 0; p < ML_P; p++) {
            float seg[ML_SEGX];
            for (sx = 0; sx < ML_SEGX; sx++)
                seg[sx] = ml_xc[c][sx * ML_P + p];    /* xr[c][p][sx] */

            /* ---- TIME branch: grid[r][col] = seg[r*2+col] ---- */
            /* t1[r][o] = sum_col TL1[o][col]*grid[r][col] */
            float t1[ML_SQ][ML_SQ];
            for (r = 0; r < ML_SQ; r++)
                for (o = 0; o < ML_SQ; o++) {
                    float a = 0.0f;
                    for (col = 0; col < ML_SQ; col++)
                        a += ML_TL1[o * ML_SQ + col] * seg[r * ML_SQ + col];
                    t1[r][o] = a;
                }
            /* t2[o][o2] = sum_r TL2[o2][r]*t1[r][o]; store kk=o2*2+o -> t=kk*P+p */
            for (o = 0; o < ML_SQ; o++)
                for (o2 = 0; o2 < ML_SQ; o2++) {
                    float a = 0.0f;
                    for (r = 0; r < ML_SQ; r++)
                        a += ML_TL2[o2 * ML_SQ + r] * t1[r][o];
                    int kk = o2 * ML_SQ + o;
                    mixlinear_st_xt[(kk * ML_P + p) * ML_D + c] = a;
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
            /* FLinear2 complex (2->4) */
            float Zre[ML_SEGY], Zim[ML_SEGY];
            for (o = 0; o < ML_SEGY; o++) {
                float re = 0.0f, im = 0.0f;
                for (i = 0; i < 2; i++) {
                    re += Y1re[i] * ML_FL2_RE[o * 2 + i] - Y1im[i] * ML_FL2_IM[o * 2 + i];
                    im += Y1re[i] * ML_FL2_IM[o * 2 + i] + Y1im[i] * ML_FL2_RE[o * 2 + i];
                }
                Zre[o] = re; Zim[o] = im;
            }
            /* iFFT real (len 4): xf[k] = (1/N) sum_j (Zre*cos - Zim*sin) */
            for (k = 0; k < ML_SEGY; k++) {
                float a = 0.0f;
                for (j = 0; j < ML_SEGY; j++) {
                    int t = (j * k) % ML_SEGX;
                    a += Zre[j] * ML_COS4[t] - Zim[j] * ML_SIN4[t];
                }
                mixlinear_st_xf[(k * ML_P + p) * ML_D + c] = a / (float)ML_SEGY;
            }
        }
    }

    /* 6. mix: y = x_t*alpha + mean + x_f*(1-alpha) */
    for (s = 0; s < ML_H; s++)
        for (c = 0; c < ML_D; c++)
            y[s * ML_D + c] = mixlinear_st_xt[s * ML_D + c] * ML_ALPHA
                            + mixlinear_st_mean[c]
                            + mixlinear_st_xf[s * ML_D + c] * (1.0f - ML_ALPHA);
}
