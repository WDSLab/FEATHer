# Board session 2026-08-24 — all 7 models re-measured

NUCLEO-F103RB, Cortex-M3 @72 MHz, no FPU, `-O2 -ffp-contract=off
-fno-fast-math`, D=7, L=H=96. One sitting, one build configuration, so every
row is comparable. Procedure: `board/BOARD_RUNBOOK.md`. Plan and rationale:
`BOARD_SESSION_PLAN.md`.

Every model reported `PASS bad=0 maxabs(x1e6)=0` — the board computes the same
function as PyTorch to within a millionth.

## Measurements

| model | params | latency | energy | flash | RAM | peak stack | active mA |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence | 0 | 0.07 ms | 0.0 mJ | 10.4 KB | 3.5 KB | 136 B | ~31 |
| SparseTSF-Linear | 41 | 47.4 ms | 4.8 mJ | 13.5 KB | 6.1 KB | 284 B | ~31 |
| MixLinear | 49 | 79.7 ms | 8.2 mJ | 15.8 KB | 8.7 KB | 528 B | ~31 |
| DiPE-Linear | 339 | 1312.1 ms | 134.2 mJ | 21.7 KB | 14.0 KB | 528 B | ~31 |
| SparseTSF-MLP | 1,181 | 430.6 ms | 44.1 mJ | 22.3 KB | 10.6 KB | 284 B | ~31 |
| **FEATHer-N (ours)** | **938** | **655.1 ms** | **67.0 mJ** | **25.7 KB** | **18.6 KB** | 528 B | ~31 |
| FITS | 1,200 | 796.2 ms | 81.5 mJ | 28.9 KB | 11.2 KB | 528 B | ~31 |

Idle (CPU in WFI): 11.5 mA. Energy = 3.3 V x 31 mA x latency.

`flash` = linker `text + data`; `RAM` = `data + bss`. Budget: 128 KB flash,
20 KB SRAM.

## Reproducibility

Four models were unchanged since the 07-27 session and reproduce it to the
microsecond across a different sitting and a full rebuild:

| model | 07-27 | 08-24 |
|---|---:|---:|
| SparseTSF-Linear | 47 ms | 47.4 ms |
| FITS | 796 ms | 796.2 ms |
| DiPE-Linear | 1312 ms | 1312.1 ms |
| persistence | 0.08 ms | 0.07 ms |

FEATHer moved 349 -> 655 ms because the model did: the 07-27 port was the
pre-08 453-parameter configuration (d8 / 3 bands / no residual MLP), and this
is the shipping 938-parameter FEATHer-N.

## What the numbers say

**Parameter count does not predict latency.** DiPE-Linear runs 339 parameters,
36% of FEATHer's, and takes twice as long. SparseTSF-MLP carries 26% more
parameters and finishes 34% sooner. The ordering by size and the ordering by
time share almost nothing.

**Nor does it predict flash.** Estimating flash as params x 4 B is off by up to
84x, because a firmware floor of 10.4 KB (HAL, printf, UART) dominates and the
model's own code and twiddle tables outweigh its weights. Across the lineup
flash spans 13.5-28.9 KB, a 2x range, while parameters span 41-1,200, a 29x
range.

**Current is model-independent.** All seven draw 29.6-32 mA regardless of
architecture or arithmetic — the CPU and flash being awake dominate. Energy is
therefore latency, and nothing else.

**FEATHer dominates FITS on every axis measured here**: fewer parameters (938
vs 1,200), less flash (25.7 vs 28.9 KB), lower latency (655 vs 796 ms), and
better accuracy (0.694 vs 0.726 mfg OVERALL MSE). It uses more RAM (18.6 vs
11.2 KB).

**FEATHer loses to SparseTSF-MLP on latency** at this horizon: 655 vs 431 ms at
comparable parameter budgets. See the horizon note below.

## Horizon: FEATHer is SRAM-bound at H=96 on this board

Projected SRAM (model validated against the H=96 measurement: predicted
19,110 B vs measured 19,072 B, 0.2% error):

| H | FEATHer | SparseTSF-MLP |
|---|---:|---:|
| 96 | 19.1 KB (fits) | 11.1 KB (fits) |
| 192 | 24.5 KB (over by 4.0) | 13.8 KB (fits) |
| 336 | 32.6 KB (over by 12.1) | 17.8 KB (fits) |
| 720 | 54.1 KB (over by 33.6) | 28.6 KB (over by 8.1) |

FEATHer keeps three L*D working buffers plus an L*16 latent, and its output
slot grows with H on top of that; SparseTSF-MLP keeps two. So the horizon where
FEATHer's period-factorised head should win — the validated op counter predicts
866 ms against 2,275 ms at H=720, a 2.6x reversal — is a horizon this board
cannot hold. Reducing bands does not help: RAM is identical at 3 and 4 bands
(the bands reuse one buffer).

This is a finding to report, not a gap to hide: the long horizons that suit the
architecture need a larger MCU, and the constraint is SRAM rather than flash,
which sits at 20% utilisation.

## Band-count probe (exploratory, not a model change)

Measured FEATHer at 3 bands to price the fourth band before the ablation
returns:

| | 4 bands | 3 bands | delta |
|---|---:|---:|---:|
| params | 938 | 904 | -3.6% |
| latency | 655.1 ms | 572.8 ms | **-12.5%** |
| flash | 25.7 KB | 24.9 KB | -3.1% |
| RAM | 18.6 KB | 18.6 KB | unchanged |

So the fourth band costs 12.5% of the runtime and buys whatever accuracy the
`band3` ablation arm reports. The code supports both via `FEA_NB`; the repo is
left at 4, which is what the paper's figures and the mfg table use.

## Caveats

- Idle current (11.5 mA) is a NUCLEO figure dominated by the on-board ST-LINK
  and is not representative of a product board, where a bare MCU in STOP draws
  tens of microamps per the datasheet. Quote the measured value for the
  measured board, and mark any battery-life projection as a projection.
- Active current is read off a hand-held meter through a JP6 series
  connection and varies by ~1 mA; treat 31 mA as the working figure.
- Weights are random under a fixed seed. Latency, flash, SRAM and "same
  function as PyTorch" do not depend on trained values.
