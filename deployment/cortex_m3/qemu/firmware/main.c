/*
 * Cortex-M3 inference firmware skeleton for FEATHer / baselines.
 *
 * This file is the WSL-side entry point that drives one inference
 * pass under QEMU emulation of the LM3S6965EVB. It is intentionally
 * minimal:
 *
 *   1. Initialize the DWT cycle counter so we can measure wall-clock
 *      cycles deterministically inside the emulator.
 *   2. Run the inference function emitted by the codegen step.
 *   3. Print the cycle count and the high-water-mark of the
 *      activation arena via semihosting (BKPT 0xAB instruction).
 *
 * To build:
 *     make ELF
 * To run under QEMU:
 *     qemu-system-arm -M lm3s6965evb -kernel firmware.elf \
 *                     -nographic -semihosting
 *
 * The weights.h / arena.h headers are produced by
 *   python -m deployment.cortex_m3.qemu.codegen ...
 * and live in the gen/ subdirectory next to this file.
 */
#include <stdint.h>
#include <stdio.h>

#include "gen/weights.h"
#include "gen/arena.h"

/* ---- DWT cycle counter on Cortex-M3 ---- */
#define DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DWT_CTRL_CYCCNTENA (1u << 0)
#define DEMCR_TRCENA (1u << 24)

static void enable_dwt_cyccnt(void)
{
    DEMCR |= DEMCR_TRCENA;
    DWT_CYCCNT = 0;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

/* ---- activation arena ---- */
static int8_t arena[ARENA_BYTES] __attribute__((aligned(8)));

/* ---- placeholder inference function ----
 *
 * Each baseline emits its own implementation of feather_inference()
 * that consumes weights.h symbols + the arena and produces a
 * forecast in the second half of the arena. For the skeleton we
 * implement a no-op so the build pipeline can be validated end-to-
 * end before the model-specific kernels are dropped in.
 */
__attribute__((noinline))
int feather_inference(int8_t *arena_, int arena_bytes_)
{
    /* TODO: replace with the model-specific CMSIS-NN kernel sequence
     *
     * For FEATHer the sequence is:
     *   1. instance-norm pass over the input region of the arena.
     *   2. four parallel depthwise convolutions (k=1,3,5,pool/up)
     *      writing into adjacent arena slots.
     *   3. shared DTK: Wm in (D->S), depthwise temporal conv,
     *      W_out (S->D), per branch.
     *   4. FFT magnitude + Conv1D + softmax to get gate weights.
     *   5. weighted sum -> SPK reshape -> linear -> output region.
     *
     * Each step calls arm_depthwise_conv_s8, arm_fully_connected_s8,
     * arm_softmax_s8 from CMSIS-NN.
     */
    (void)arena_;
    (void)arena_bytes_;
    return 0;
}

int main(void)
{
    enable_dwt_cyccnt();

    DWT_CYCCNT = 0;
    int rc = feather_inference(arena, ARENA_BYTES);
    uint32_t cycles = DWT_CYCCNT;

    printf("cycles    = %lu\n", (unsigned long)cycles);
    printf("arena_max = %lu bytes\n", (unsigned long)ARENA_BYTES);
    printf("rc        = %d\n", rc);

    /* Exit cleanly via semihosting so QEMU returns. */
    asm volatile (
        "mov r0, #0x18\n"   /* angel_SWIreason_ReportException */
        "ldr r1, =0x20026\n" /* ADP_Stopped_ApplicationExit */
        "bkpt 0xAB\n"
    );
    return 0;
}
