/* Linker script for QEMU lm3s6965evb with 64 KB SRAM budget.
 * Duplicate this file as lm3s6965_32k.ld and lm3s6965_16k.ld with
 * the corresponding RAM length to test the three deployment budgets.
 */
MEMORY
{
    FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 256K
    SRAM  (rwx): ORIGIN = 0x20000000, LENGTH = 64K
}

ENTRY(Reset_Handler)

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        _etext = .;
    } > FLASH

    .data :
    {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > SRAM AT> FLASH

    .bss :
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > SRAM

    /* Stack at the top of SRAM; the arena lives in .bss. */
    _estack = ORIGIN(SRAM) + LENGTH(SRAM);
}
