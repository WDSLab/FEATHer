/* Calibration of the operation count (tools/op_count.py --calibrate): each wrapper runs one shared kernel of
 * mcu_kernels.c at a fixed size; op_count.py counts its float operations in the emulator and compares them with the
 * rule count of the matching torch op.  Globals (not static) so the compiler cannot treat the buffers as zeros. */
#include <math.h>
#include "mcu_kernels.h"
#include "mcu_pt.h"

float CX[96 * 16], CY[96 * 16], CW[8 * 16], CB[16];

void c_conv3_b(void)  { mcu_dwconv_same(CX, CY, CW, CB, 96, 7, 3); }
void c_conv5(void)    { mcu_dwconv_same(CX, CY, CW, 0, 96, 7, 5); }
void c_pool(void)     { mcu_avgpool4(CX, CY, 96, 7); }
void c_softmax(void)  { pt_softmax(CX, CY, 1, 37); }
void c_gelu(void)     { for (int i = 0; i < 100; ++i) CY[i] = mcu_gelu(CX[i]); }
void c_norm(void)                                    /* RevIN statistics of (96, 7), over time */
{
    pt_mean_t(CX, CY, 96, 7);
    pt_var(CX, CY + 7, 7, 1, 96, 7, 0);
    for (int d = 0; d < 7; ++d) CY[7 + d] = sqrtf(CY[7 + d] + 1e-5f);
}
