/*
 * MixLinear float32 inference, following the staged PyTorch reference of
 * export/export_golden_mixlinear.py. All buffers static
 * (no malloc); complex arithmetic kept as explicit re/im pairs.
 *
 * Weight layout (row-major, from export_golden_mixlinear.py):
 *   ML_CONV[25]          conv1d filter (shared 1->1)
 *   ML_TL1[4]/ML_TL2[4]  Linear(2,2) as [out*2 + in]
 *   ML_FL1_{RE,IM}[8]    complex Linear(4->2) as [out*4 + in]
 *   ML_FL2_{RE,IM}[8]    complex Linear(2->4) as [out*2 + in]
 * Both transforms are pocketfft's (src/mcu_pocket.c), the FFT PyTorch runs: fft of the real
 * segment = the real forward transform of seg_x, ifft = the complex backward transform of seg_y
 * with its 1 / seg_y scale.
 */
#include <string.h>

#include "mixlinear_infer.h"
#include "mcu_pt.h"
/* Plain names, not "../golden/...": the x86 build passes -Igolden, and the
 * CubeIDE project flattens every header into Core/Inc, where a relative path
 * out of Core/Src does not resolve. Every other model's port already does it
 * this way. */
#include "mixlinear_weights.h"

#ifndef ML_ALPHA                      /* mixlinear_geom.h writes it (candidate settings) */
#define ML_ALPHA 0.5f
#endif
#define ML_PAD (ML_P / 2)             /* conv padding = 12 */

/* Output step s = k*P + p is emitted as (segment k, phase p).  That covers
 * every step only when the horizon is a whole number of periods, which holds
 * for every horizon we run (96/192/336/720 with P=24). */
_Static_assert(ML_H % ML_P == 0,
               "MixLinear port assumes ML_H is a multiple of ML_P");

/* PER-CHANNEL WORKING SET - equal-effort rule (every port gets the same
 * memory tidy-up).  MixLinear is channel-independent:
 * the mean, the conv, both branches and the final mix all act on one channel,
 * so nothing needs to be held for all D at once.  Full-width copies are
 * test-build only.
 *
 * OUTPUT SIDE.  Output step s = k*P + p depends only on phase
 * p's segment, so each phase is written straight into y as soon as both
 * branches are known for it - nothing of length H is held.  Every value keeps its terms and
 * their order, so the forecast is bit-identical; the arena loses the
 * (XT_N + XF_N)-float branch buffers (6.3 KB at H=720).  The per-phase
 * temporaries that grow with H (t1, the freq spectrum) are static so the
 * footprint table sees them; the fixed 4-float ones stay on the stack. */
static float ml_xc[ML_L];               /* this channel, centered -> conv+res */
static float ml_row[ML_L];              /* pre-conv copy of the same row      */
static float ml_t1[ML_SQX * ML_SQY];    /* time branch, first factor, 1 phase */
#include "mixlinear_fft.h"
_Static_assert(ML_FFT_SEGX == ML_SEGX && ML_FFT_SEGY == ML_SEGY && ML_LPF <= ML_SEGX,
               "mixlinear_fft.h does not match the geometry - re-run export_golden_mixlinear.py");
/* FFT data / work, 1 phase: the real seg_x-point transform first, then the complex one over seg_y */
#define ML_Z_N (2 * ML_SEGY > ML_SEGX + 2 ? 2 * ML_SEGY : ML_SEGX + 2)
static float ml_z[ML_Z_N], ml_zo[ML_Z_N];

#ifdef ML_CAPTURE_STAGES
float mixlinear_st_mean[ML_D];
float mixlinear_st_xt[ML_XT_N * ML_D];
float mixlinear_st_xf[ML_XF_N * ML_D];
#endif

static float ml_mean[ML_D];   /* the means of all channels (torch.mean) */

static void mixlinear_run(const float *x, float *y, int d0, int d1, int ys, mcu_emit_fn emit)
{
    int c, s, p, j, k, sx, o, o2, i, r, col;

    for (c = d0; c < d1; c++) {
        float *yd = y ? ((ys == 1) ? y : y + c) : 0;   /* this channel's outputs */
        /* 1. mean over time = torch.mean's kernel (mcu_pt.c): all channels in one call, at the first */
        if (c == 0)
            pt_mean_t(x, ml_mean, ML_L, ML_D);
        const float m = ml_mean[c];
#ifdef ML_CAPTURE_STAGES
        mixlinear_st_mean[c] = m;
#endif

        /* 2. (x - mean) */
        for (s = 0; s < ML_L; s++)
            ml_xc[s] = x[s * ML_D + c] - m;

        /* 3. shared conv1d(1->1, same pad, k=25) + residual, in place */
        memcpy(ml_row, ml_xc, ML_L * sizeof(float));
        const float zero = mcu_rt(0.0f);          /* the padding: multiplied like any sample, as PyTorch does */
        for (s = 0; s < ML_L; s++) {
            float acc = 0.0f;
            for (j = 0; j < ML_K; j++) {
                int ip = s + j - ML_PAD;
                acc += ML_CONV[j] * ((ip >= 0 && ip < ML_L) ? ml_row[ip] : zero);
            }
            ml_xc[s] = acc + ml_row[s];
        }

        /* 4/5/6. per period-phase: both branches, then emit that phase */
        for (p = 0; p < ML_P; p++) {
            /* upstream zero-pads the segment axis to sq_x^2 for the time branch
             * (F.pad).  At P = 24 seg_x = 4 = 2^2 and there is no pad; at P = 48
             * seg_x = 2 < 4, so the pad is applied here too. */
            float seg[ML_SQX * ML_SQX];
            for (sx = 0; sx < ML_SEGX; sx++)
                seg[sx] = ml_xc[sx * ML_P + p];       /* xr[c][p][sx] */
            for (; sx < ML_SQX * ML_SQX; sx++)
                seg[sx] = 0.0f;

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

            /* ---- FREQ branch: seg_x-point FFT (real in), first lpf bins ---- */
            float Xre[ML_LPF], Xim[ML_LPF];
            /* transformed at ml_z + 1 (r0 r1 i1 r2 i2 ...): bin b >= 1 is the pair ml_z[2b], ml_z[2b+1];
             * the slot after the data holds the zero imaginary part of bin 0 and of the Nyquist bin
             * (a stored value, as in the spectrum torch returns).  Bins above seg_x / 2 are conjugates. */
            for (sx = 0; sx < ML_SEGX; sx++)
                ml_z[1 + sx] = seg[sx];
            ml_z[ML_SEGX + 1] = 0.0f;
            pk_rfft_forward(&ML_RFFT_F, ml_z + 1, ml_zo, 1.0f);
            for (j = 0; j < ML_LPF; j++) {
                const int b = (2 * j <= ML_SEGX) ? j : ML_SEGX - j;
                const float im = (b == 0) ? ml_z[ML_SEGX + 1] : ml_z[2 * b + 1];
                Xre[j] = (b == 0) ? ml_z[1] : ml_z[2 * b];
                Xim[j] = (b == j) ? im : -im;
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
                ml_z[2 * o] = re; ml_z[2 * o + 1] = im;
            }
            pk_cfft(&ML_IFFT, ml_z, ml_zo, 1.0f / (float)ML_SEGY);

            /* ---- emit this phase: steps s = k*P + p, k < seg_y = H/P ----
             * time: TLinear2 (sq_x -> sq_y) over the grid rows, cell
             *       k = o2*sq_y + o  (cells k >= seg_y overshoot the horizon: upstream
             *       computes them and slices them off, and so does this port)
             * freq: real part of the seg_y-point inverse FFT above
             * mix:  y = x_t*alpha + mean + x_f*(1-alpha) */
            for (k = 0; k < ML_SQY * ML_SQY; k++) {
                o2 = k / ML_SQY;
                o = k % ML_SQY;
                float xt = 0.0f;
                for (r = 0; r < ML_SQX; r++)
                    xt += ML_TL2[o2 * ML_SQX + r] * ml_t1[r * ML_SQY + o];
                if (k >= ML_SEGY) {               /* past the horizon: computed as upstream, then sliced off */
                    mcu_drop(xt);
                    continue;
                }

                float xf = ml_z[2 * k];

                s = k * ML_P + p;
                MCU_OUT(yd, ys, emit, c, s, xt * ML_ALPHA
                                + m
                                + xf * (1.0f - ML_ALPHA));
#ifdef ML_CAPTURE_STAGES
                mixlinear_st_xt[s * ML_D + c] = xt;
                mixlinear_st_xf[s * ML_D + c] = xf;
#endif
            }
        }
    }
}

/* Full buffer (caller owns H x D, time-major): the original API, same arithmetic. */
void mixlinear_forward(const float *x, float *y) { mixlinear_run(x, y, 0, ML_D, ML_D, 0); }

/* One channel d: its H forecasts to yc[0..H-1].  Lets the caller emit channel by
 * channel with an H-float buffer instead of holding H x D. */
void mixlinear_forward_ch(const float *x, int d, float *yc) { mixlinear_run(x, yc, d, d + 1, 1, 0); }

/* One channel d: each forecast value goes to emit(d, t, v) the moment it is final -
 * no H-float buffer (mcu_emit.h).  Same values as mixlinear_forward_ch, bit for bit. */
void mixlinear_forward_emit(const float *x, int d, mcu_emit_fn emit) { mixlinear_run(x, 0, d, d + 1, 1, emit); }
