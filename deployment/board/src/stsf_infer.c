/*
 * SparseTSF (linear) forward, mirroring the upstream
 * baselines/SparseTSF/SparseTSF-main/models/SparseTSF.py exactly:
 *   mean-subtract -> shared 1x25 'same' conv + residual ->
 *   phase-wise shared linear n->m -> add mean back.
 */
#include "stsf_infer.h"
#include "stsf_weights.h"      /* generated: golden/stsf_weights.h */

/* The golden headers are regenerated per horizon.  Building this port
 * against weights exported for a DIFFERENT horizon would compile fine and
 * return a plausible, wrong forecast, so refuse it at compile time. */
_Static_assert(STSF_H == STSF_H_GEN,
               "STSF_H does not match the horizon the weights were exported at - re-run export_golden_*.py --pred_len");

/* PER-CHANNEL WORKING SET (2026-09-14) - equal-effort rule, see
 * deployment/board/README.md.  SparseTSF is channel-independent: the mean,
 * the conv and the phase-wise head all act on one channel, so nothing needs
 * to be held for all D at once.  The previous build kept the centered input
 * AND the conv+residual output for every channel (2*D*L floats, 21.5 KB at
 * D=28), which put this port over the 20 KB SRAM budget on BlastFurnace - an
 * artefact of the port, not of the model.  Same per-channel treatment as the
 * FITS / DiPE / MixLinear ports.  Full-width copies are test-build only. */
#ifdef STSF_CAPTURE_STAGES
float stsf_st_mean[STSF_D];
float stsf_st_xagg[STSF_D * STSF_L];
#endif

static float s_xc[STSF_L];    /* one channel: centered, then conv+residual */
static float s_cv[STSF_L];    /* conv output, one channel                  */

void stsf_forward(const float *x, float *y)
{
    for (int d = 0; d < STSF_D; ++d) {
        /* per-channel mean only (SparseTSF has no variance norm).  Same
         * accumulation order as fea_instnorm_stats, so the value is
         * bit-identical to the all-channel build. */
        float m = 0.0f;
        for (int t = 0; t < STSF_L; ++t)
            m += x[t * STSF_D + d];
        m /= (float)STSF_L;

        for (int t = 0; t < STSF_L; ++t)
            s_xc[t] = x[t * STSF_D + d] - m;

        /* one shared conv kernel + residual.  The conv reads s_xc and writes
         * s_cv, so the residual can then be folded back into s_xc in place. */
        fea_dwconv_same(s_xc, s_cv, STSF_W_CONV, 0, STSF_L, 1, STSF_K);
        for (int t = 0; t < STSF_L; ++t)
            s_xc[t] += s_cv[t];

#ifdef STSF_CAPTURE_STAGES
        stsf_st_mean[d] = m;
        for (int t = 0; t < STSF_L; ++t)
            stsf_st_xagg[d * STSF_L + t] = s_xc[t];
#endif

        /* phase-wise shared linear: v[n] -> out[m] per phase */
        for (int ph = 0; ph < STSF_P; ++ph) {
            float v[STSF_N];
            for (int in = 0; in < STSF_N; ++in)
                v[in] = s_xc[in * STSF_P + ph];
            for (int om = 0; om < STSF_M; ++om) {
                float acc = 0.0f;
                for (int in = 0; in < STSF_N; ++in)
                    acc += STSF_W_LIN[om * STSF_N + in] * v[in];
                y[(om * STSF_P + ph) * STSF_D + d] = acc + m;
            }
        }
    }
}
