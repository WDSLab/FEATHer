/*
 * FITS float32 inference - companion model (estimate-footnote upgraded to
 * an optional measured row). Shared-upsampler benchmark default:
 * RevIN (unbiased var + 1e-5) -> rfft96 -> keep 24 low bins ->
 * complex Linear(24 -> 48) -> zero-pad to 97 bins -> irfft(192) ->
 * x2 energy compensation -> last 96 -> denorm.
 */
#ifndef FITS_INFER_H
#define FITS_INFER_H

#include "fea_kernels.h"

#define FITS_L 96
#define FITS_H 96
#define FITS_D 7
#define FITS_CUT 24                        /* kept low-freq bins       */
#define FITS_UP 48                         /* upsampled bins           */
#define FITS_TN (FITS_L + FITS_H)          /* irfft length = 192       */
#define FITS_TF (FITS_TN / 2 + 1)          /* padded spectrum bins = 97 */

/* x: (FITS_L, FITS_D) time-major -> y: (FITS_H, FITS_D) time-major */
void fits_forward(const float *x, float *y);

/* stage buffers for golden comparison */
extern float fits_st_mean[FITS_D];
extern float fits_st_sd[FITS_D];
extern float fits_st_xn[FITS_L * FITS_D];          /* time-major   */
extern float fits_st_up_re[FITS_D * FITS_UP];      /* channel-major */
extern float fits_st_up_im[FITS_D * FITS_UP];

#endif /* FITS_INFER_H */
