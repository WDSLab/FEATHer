/*
 * Shared float32 primitives for the on-board models (FEATHer, SparseTSF,
 * FITS ...). Weight-free: everything is passed by pointer, so each
 * model's firmware image links only its own flash constants.
 *
 * x86 tests: -O2 -ffp-contract=off -fno-fast-math.  Board builds: the CubeIDE flags, the same for every
 * model.
 */
#ifndef MCU_KERNELS_H
#define MCU_KERNELS_H

/* Board builds: board_prep.py writes board_cfg.h (absent in the repo, so the x86 tests never see it). */
#if defined(__has_include)
#if __has_include("board_cfg.h")
#include "board_cfg.h"
#endif
#endif

/* depthwise conv over time, 'same' zero padding; x,y are (len, nch)
 * time-major interleaved; w is (nch, k); b may be NULL              */
void mcu_dwconv_same(const float *x, float *y, const float *w,
                     const float *b, int len, int nch, int k);

/* AvgPool1d(kernel=4, stride=4), (len, nch) -> (len/4, nch)         */
void mcu_avgpool4(const float *x, float *y, int len, int nch);

/* GELU, exact erf form: 0.5*x*(1+erf(x/sqrt(2))). This is what
 * torch.nn.functional.gelu computes by default (approximate='none') — the
 * cheaper tanh approximation is off by ~1e-3 near |x|~1 and would blow the
 * golden-vector tolerance (atol 1e-5). erff() is a libm call (the op
 * counter counts a libm call as 1). */
float mcu_gelu(float x);


#endif /* MCU_KERNELS_H */
