# Layer 2 — QEMU validation pipeline (WSL workflow)

Cycle-accurate validation of the simulation-based estimator
(`deployment/cortex_m3/estimator.py`) by running an actual 8-bit
quantized FEATHer firmware on the QEMU emulation of the
LM3S6965EVB target.

This directory is intentionally *skeleton-only*. The actual build
+ run happens in WSL2 because the `arm-none-eabi-gcc` toolchain and
`qemu-system-arm` are best-supported on Linux. Everything in this
directory is platform-agnostic and can be edited from either host.

## One-time WSL setup

```bash
# inside WSL2 Ubuntu 22.04
sudo apt update
sudo apt install -y gcc-arm-none-eabi qemu-system-arm
arm-none-eabi-gcc --version   # should be 12.2.x or newer
qemu-system-arm --version
```

Optional but recommended: clone CMSIS-NN once and reuse:

```bash
git clone --depth 1 https://github.com/ARM-software/CMSIS-NN.git \
    ~/lib/CMSIS-NN
```

## Pipeline overview

```
  trained PyTorch model (.pth)
     │
     │  codegen.py        (Python; runnable on Windows)
     ▼
  weights.h + arena.h     (int8 quantized C arrays)
     │
     │  WSL: arm-none-eabi-gcc -Os -mthumb -mcpu=cortex-m3 ...
     ▼
  firmware.elf
     │
     │  WSL: qemu-system-arm -M lm3s6965evb -kernel firmware.elf \
     │        -nographic -semihosting -d cpu_reset
     ▼
  semihosted stdout:
      cycles    = 412785
      arena_max = 5832 bytes
      flash     = 8924 bytes
     │
     │  collect_results.py  (Python; runnable on Windows)
     ▼
  results/qemu_cycles.csv  (one row per model × dataset × horizon)
```

## Files in this directory

| File | Status | Purpose |
|---|---|---|
| `codegen.py` | TODO | quantize a checkpoint to int8 + emit `weights.h` / `arena.h` |
| `firmware/main.c` | TODO | inference loop with DWT cycle counter + semihosting print |
| `firmware/Makefile` | TODO | cross-compile rules + linker flags |
| `firmware/lm3s6965.ld` | TODO | linker script with 16/32/64 KB RAM variants |
| `run_qemu.sh` | TODO | drive QEMU on each ELF and capture stdout |
| `collect_results.py` | TODO | parse QEMU stdout, write `qemu_cycles.csv` |

## Why this is on a separate layer

The simulation estimator (Layer 1) in the parent directory already
covers every (model, dataset, horizon) cell and is what powers the
main Section IX tables. Layer 2 is a *calibration* pass that we
report alongside the estimator to bound its error — we only need to
run it for a small representative subset (FEATHer + DiPE-Linear +
SparseTSF + FITS) to land the result in the paper.

The split lets the manuscript move forward without waiting for the
WSL build environment, while keeping the validation pipeline in
the same commit history.
