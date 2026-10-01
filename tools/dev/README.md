# Fast dev loop (2026-08-26)

A screening protocol for FEATHer architecture work. It exists because the
shipping protocol costs ~10 min per run and that is too slow to iterate on the
model. **It ranks arms; it never produces a number for the paper.** Promote a
winner to the shipping protocol before believing its absolute MSE.

```bash
python tools/dev/run_dev.py --arms canon,redA,noinproj --data Steel --pred_len 96
python tools/dev/dev_summary.py --prefix dev --ref canon
python tools/dev/dev_cost.py --d_model 6 --explain      # params + peak SRAM
```

## Why it is fast

Profiled on Steel H=96 (RTX 4090):

| | share of an epoch |
|---|---|
| data loading | 6% |
| forward/backward of an **822-parameter** model | **94%** |

The model is not compute-bound, it is **CUDA-launch-bound**: 760 tiny batches
per epoch, each firing dozens of kernels (4 bands x DTK, rfft, interpolate).
Raising the batch cuts launches proportionally, with the same samples seen:

| batch | s/epoch (GPU) | s/epoch (CPU) |
|---|---|---|
| 32 (shipping) | 4.09 | 11.92 |
| 128 | 1.31 | 5.45 |
| 512 (dev) | **0.55** | 3.01 |

Running many jobs concurrently on one GPU was tried first and is a much weaker
lever — 6 concurrent processes gave only 2.3x net throughput, because a single
GPU serialises the launches anyway. Batch size is the real fix. Effective cost:
**~17 s/run**, ~35x faster than shipping.

## Calibration against the shipping protocol

Steel H=96, canonical FEATHer:

| protocol | MSE | seed std |
|---|---|---|
| shipping (bs 32, lr 1e-3, 3 seeds) | 0.4721 | 1.40% |
| dev (bs 512, lr 8e-3, 3 seeds) | **0.4696** | **0.35%** |

Absolute error 0.5%. A known delta reproduces in sign and rough size (`redA`:
-2.67% shipping at n=1, -1.45% dev at n=3). The dev noise floor is **4x lower**
because the larger batch removes gradient noise, so the loop resolves ~1%
deltas that the shipping protocol cannot separate from seed spread at n=3.

## lr is swept, not fixed

At bs=512 the loss surface has a cliff (Steel H96: lr 4e-3 -> 0.465,
lr 3e-3 -> 0.517), and different arms have different gradient scales, so a
single fixed lr would penalise an arm for the wrong reason. Each arm runs at
every lr in `--lrs` and the summary picks the best by **val**, never test.

## The bad basin — read this before trusting a delta

Some (arm, seed) pairs converge to a clearly separated worse optimum:
val ~0.49 / test ~0.51, against val ~0.43 / test ~0.47 for a good run. It is
**not undertraining** — 4x the epochs (200, patience 30) does not move it, and
the bad runs early-stop at epoch 12-67. It is seed-dependent: seed 2026 fails
for `gatestatic` and `s8`, seed 2025 for `dtkgelu`, while `canon`, `redA` and
`noinproj` never failed.

Suspected cause: the large batch removes the gradient noise that lets small
batches escape a bad basin — a standard large-batch pathology, i.e. an artifact
of this protocol rather than a property of the arm. **A shipping-protocol
cross-check of the failing arms is the way to settle it.**

Consequences for reading the table:
- `dev_summary.py` reports the **median** over seeds, not the mean.
- The `bad` column counts runs above the bad-basin val cutoff. **An arm with
  `bad > 0` has no usable delta** until it is re-run with more seeds or under
  the shipping protocol.

## Cost axes come with the accuracy

`dev_cost.py` reports parameters and peak Cortex-M3 SRAM for a config. The SRAM
model is transcribed from the shipping C port (`deployment/board/src/
feather_infer.{c,h}`), not estimated, and reproduces the board measurement
exactly (19,072 B predicted vs 19,072 B measured; `--validate`).

Every term in it is proportional to L or H — see `--explain`. That is why
FEATHer is RAM-last of the seven measured models and cannot reach H>=192 on a
20 KB part, and it is a property of the batch-window dataflow, not of any
component.

## Bug found while building this

`--mlp_rank` was forwarded to the worker only inside `if args.use_residual_mlp`,
but the shipping mfg config enables the residual MLP through
`_DATASET_OVERRIDES` instead of that flag. So `--mlp_rank N` alone was dropped
from the override dict *and* never forwarded, and the run silently trained the
worker default rank. The default is 16 = the canonical value, so **no past
result is affected**; rank experiments simply no-oped into duplicates of the
baseline. Fixed in `run_forecast.py` (tag `MLP-RANK-DROPPED`).
