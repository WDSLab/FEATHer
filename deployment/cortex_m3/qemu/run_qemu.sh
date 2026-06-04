#!/usr/bin/env bash
# Build the firmware for a given RAM budget and run it under QEMU.
# Captures the semihosting stdout into a results file.
#
# Usage:
#     ./run_qemu.sh FEATHer ETTh1 96 64
#                   ^model  ^data ^H  ^RAM_KB
#
# Run from WSL after running the codegen step on the same checkpoint.

set -e
MODEL=${1:-FEATHer}
DATA=${2:-ETTh1}
HOR=${3:-96}
RAM_KB=${4:-64}

# Sanity: arm-none-eabi-gcc and qemu-system-arm must be installed.
command -v arm-none-eabi-gcc >/dev/null || { echo "arm-none-eabi-gcc missing"; exit 1; }
command -v qemu-system-arm >/dev/null || { echo "qemu-system-arm missing"; exit 1; }

FW_DIR="$(dirname "$0")/firmware"
cd "$FW_DIR"
make clean >/dev/null
make RAM_KB=${RAM_KB} firmware.elf

LOG="../results_${MODEL}_${DATA}_H${HOR}_${RAM_KB}k.log"
qemu-system-arm -M lm3s6965evb -kernel firmware.elf \
    -nographic -semihosting | tee "${LOG}"

echo "---"
echo "log saved to $(readlink -f "${LOG}")"
