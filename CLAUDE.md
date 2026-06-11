# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

**Status (2026-06-11):** Pre-sweep audit complete — the codebase is
ready for the server runs. Locked this week:
- **Evaluation protocol rewritten**: best-val-epoch selection with
  early stopping (patience=10, 50-epoch cap), test evaluated once on
  the selected model, checkpoint = that same model. The old worker did
  test-oracle selection with element-wise metric mixing — every
  pre-2026-06-11 result row (there were none in fcst_results.csv) is
  incompatible with the new protocol.
- **Traffic = full 862 channels** (was a legacy 162 slice in
  data_factory only; manuscript and edge estimator already said 862).
- **Sub-1K claim scoped to D≤14** (FEATHer params scale as D² via
  `in_proj`; see Known issues).
- **Ablation variants unblocked** (worker argparse choices excluded
  them) + `--ablation_axis` orchestrator dispatch.
- **run_hp_search.py added** (OFAT, val-only selection, 192 runs).
- **QEMU framing fixed**: instruction-level emulation, verified
  (memory fit, correctness, icount) vs modeled (latency, energy)
  separation in Sec IX, README, and firmware skeleton.
All 12 main models + 5 ablation axes smoke-tested end-to-end through
the new worker. 13-page PDF compiles cleanly. Sections VI-VIII remain
`\TODO{...}` skeletons waiting on sweep results.

**Next action (server):** `run_hp_search.py` (192 runs, freeze FEATHer
single config) → `run_forecast.py --save_model` (1,920 runs) →
`run_forecast.py --data SML --exp_tag main --save_model` (240 runs —
**required**: Sec VIII promises SML robustness heatmaps, and the
robustness worker loads `main` checkpoints; SML is not in the default
LTSF-8 sweep) → `run_robustness.py` (19,200 rows) → ablation
(decided scope: ETTh1+Weather+Electricity) → QEMU Layer 2 (WSL).
After the main sweep: `check_progress.py --exp_tag main` for the
cap-hit audit; after HP search: re-run FEATHer rows of
`edge_estimates.csv` if the chosen config differs from base.
Reject letter at `manuscript/notes/reject_mail.md`.

**Build:** from `manuscript/tex_workspace/`,
`pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj`.

## Environment

- **Conda env**: `feather` at `C:\Users\lee\.conda\envs\feather\` — Activate
  with `conda activate feather` before any commands. Has `torch 2.5.1+cu121`,
  `darts 0.37.1`, `reformer_pytorch 1.4.4` (for iTransformer's
  SelfAttention_Family).
- `mamba-ssm` is intentionally **not** installed (CUDA-toolchain build,
  out-of-scope for our edge-deployable benchmark).

## Repository structure

```
FEATHer/
├── run_forecast.py                 user-facing orchestrator (--check, resume, dispatch)
├── run_robustness.py               robustness orchestrator (loads --save_model checkpoints)
├── run_hp_search.py                FEATHer OFAT HP search (val-only selection → single config + R2 #23 sensitivity)
├── setup_baselines.sh              git-clone baseline upstream repos (idempotent)
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         worker (one model × dataset × pred_len × N seeds)
│       └── run_robustness.py       robustness worker (Gauss/miss/impulse/quant sweep)
├── baselines/                      top-level model directory
│   ├── _import_helper.py           sys.modules isolation for upstream imports
│   ├── __init__.py                 registry (12 main + 30 ablation) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
│   ├── FEATHer/                    our model (paper-faithful single config)
│   │   ├── FEATHer.py
│   │   └── ablation/{multiscale,dtk,gating,head,complexity}.py
│   ├── DLinear/wrapper.py          + DLinear-main/ (cure-lab/LTSF-Linear)
│   ├── PatchTST/wrapper.py         + PatchTST-main/ (yuqinie98)
│   ├── iTransformer/wrapper.py     + iTransformer-main/ (thuml)
│   ├── FITS/wrapper.py             + FITS-main/ (VEWOXIC)
│   ├── SparseTSF/wrapper.py        + SparseTSF-main/ (lss-1138)
│   ├── TimeMixer/wrapper.py        + TimeMixer-main/ (kwuking)
│   ├── TimesNet/wrapper.py         + TimesNet-main/ (thuml/Time-Series-Library)
│   ├── TQNet/wrapper.py            + TQNet-main/ (ACAT-SCUT)
│   ├── LMS_AutoTSF/wrapper.py      + LMS_AutoTSF-main/ (mribrahim/LMS-TSF)
│   ├── DiPE_Linear/wrapper.py      + DiPE_Linear-main/ (wintertee, DASFAA 2026)
│   └── MDMLP_EIA/wrapper.py        + MDMLP_EIA-main/ (zh1985csuccsu, AAAI 2026)
├── utils/
│   ├── seed.py                     set_seed() + parse_seed_list()
│   ├── data_factory.py             data_provider + data_select (lazy darts import)
│   ├── data_loader.py              Dataset classes (ETT_hour/minute, Custom, PEMS)
│   ├── losses.py                   spectral_separation_loss_scales (FEATHer-specific)
│   ├── metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
│   ├── noise.py                    4-axis corruption (gauss/miss/impulse/quant)
│   └── timefeatures.py
├── tools/
│   ├── audit/check_progress.py     read results CSV, print coverage + mean±std preview
│   └── paper/                      Phase 5 table generators
│       ├── main_table.py           5-seed mean±std table (md / latex / csv)
│       ├── wilcoxon.py             pairwise signed-rank vs FEATHer
│       ├── ablation_table.py       5 axes × 4 horizons aggregator
│       └── robust_summary.py       4-axis noise heatmaps + summary table
├── deployment/cortex_m3/           Layer-1 edge-cost estimator (this work, IoT-J Sec IX)
│   ├── op_costs.py                 Cortex-M3 reference profile + per-op FLOP counters
│   ├── estimator.py                forward-hook based peak RAM / arena / latency / energy
│   ├── run.py                      CLI sweep -> results/edge_estimates.csv
│   └── qemu/                       Layer-2 QEMU validation (skeleton; WSL workflow)
│       ├── README.md               build + run instructions
│       ├── codegen.py              PyTorch -> int8 + weights.h / arena.h
│       └── firmware/{main.c,Makefile,lm3s6965_64k.ld}
├── results/
│   ├── fcst_results.csv            main sweep (append-only, resume by orchestrator)
│   ├── robust_results.csv          robustness sweep
│   ├── edge_estimates.csv          384 cells from deployment/cortex_m3/run.py
│   └── checkpoints/<exp_tag>/      deterministic .pth from --save_model
├── data/                           .csv / .npy raw data (gitignored)
└── manuscript/                     reviews, drafts, notes (mostly gitignored)
    ├── tex_workspace/feather_iotj.tex  IoT-J source (selectively tracked)
    ├── tex_workspace/fig{1-5}*.tex     TikZ figures (tracked)
    └── drafts/*.md                     per-section drafts (tracked)
```

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Main sweep (Phase 4) ===
# Save checkpoints so robustness can reuse them.
python run_forecast.py --check                                           # show what's missing
python run_forecast.py --num_seeds 5 --num_epochs 50 --exp_tag main --save_model
python run_forecast.py --model FEATHer --save_model                      # one model across all datasets/horizons
python run_forecast.py --data ETTh1 --pred_len 96 --save_model           # one (data, horizon) across all models
python run_forecast.py --exclude TimesNet,MDMLP_EIA                      # skip heavy models for a fast pass
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke      # quick verify (no checkpoints)

# 2-GPU split: no DDP — `--gpu N` picks a device index. Run two
# processes with disjoint --exclude splits; sharing one results CSV is
# fine (one append per finished run, collisions practically impossible).
#   GPU0: python run_forecast.py --exp_tag main --save_model --gpu 0 \
#           --exclude TimesNet,MDMLP_EIA,iTransformer,PatchTST,TimeMixer,TQNet
#   GPU1: python run_forecast.py --exp_tag main --save_model --gpu 1 \
#           --exclude FEATHer,DLinear,DiPE_Linear,SparseTSF,FITS,LMS_AutoTSF

# SML is robustness-only: train its checkpoints under exp_tag=main
# (240 runs, see Status) but it stays OUT of the main accuracy table —
# the paper's main table is the 8 LTSF datasets; SML appears only in
# Sec VIII (its clean rows double as accuracy there).

# === Ablation sweep (after main; 30 FEATHer variants, 5 axes) ===
# DECIDED SCOPE (2026-06-11): ETTh1 + Weather + Electricity x 4 horizons
# x 5 seeds = 1,800 runs. Diversity (hourly-7ch / 10min-21ch / hourly-321ch)
# + answers R8 #6d (Electricity ablations promised but unreported in TPAMI).
python run_forecast.py --ablation_axis all --data ETTh1       --exp_tag ablation
python run_forecast.py --ablation_axis all --data Weather     --exp_tag ablation
python run_forecast.py --ablation_axis all --data Electricity --exp_tag ablation
# axes: ms (15) / gate (4) / dtk (4) / head (4) / complexity (3)
# single axis: python run_forecast.py --ablation_axis dtk --data ETTh1 --exp_tag ablation

# === FEATHer HP search (before Phase 4 — picks the single config) ===
# OFAT around the canonical config; selection by val_loss only.
# 16 configs × {ETTh1,ETTm1,Weather} × {96,720} × 2 seeds = 192 runs.
python run_hp_search.py --check
python run_hp_search.py                    # run all missing (resumable)
python run_hp_search.py --summary          # per-axis sensitivity + recommended config

# === Robustness sweep (Phase 4b) ===
# Requires `--save_model` to have populated results/checkpoints/<train_exp_tag>/.
python run_robustness.py --check
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --train_exp_tag main --fault_types gauss,miss   # subset
python run_robustness.py --model FEATHer --train_exp_tag main            # one model

# === Worker (rarely called directly; orchestrator dispatches) ===
python scripts/benchmarks/run_forecast.py \
    --model FEATHer --data ETTh1 --pred_len 96 \
    --seeds "2025,2026,2027,2028,2029" --exp_tag main --save_model

# === Progress audit ===
python tools/audit/check_progress.py                  # default results/fcst_results.csv
python tools/audit/check_progress.py --exp_tag main   # filter by experiment
python tools/audit/check_progress.py --model FEATHer  # filter by model

# === Paper artifact generators (Phase 5) ===
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer
python tools/paper/ablation_table.py --axis ms --format latex
python tools/paper/robust_summary.py --output_dir manuscript/figures/

# === Cortex-M3 edge cost estimator (Layer 1, simulation-based) ===
python -m deployment.cortex_m3.run --check                # show pending cells
python -m deployment.cortex_m3.run                         # full sweep -> results/edge_estimates.csv
python -m deployment.cortex_m3.run --model FEATHer --data ETTh1 --pred_len 96

# === Cortex-M3 QEMU validation (Layer 2, WSL) ===
# Inside WSL2 (Ubuntu 22.04 with arm-none-eabi-gcc + qemu-system-arm installed):
python -m deployment.cortex_m3.qemu.codegen --model FEATHer --data ETTh1 --pred_len 96 \
    --ckpt results/checkpoints/main/FEATHer_ETTh1_H96_L96_s2025.pth
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build ===
cd manuscript/tex_workspace
pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj
```

## Model registry (12 entries)

```python
from baselines import get_model, list_models, get_method_defaults, get_dataset_overrides
print(list_models())
# ['DLinear', 'DiPE_Linear', 'FEATHer', 'FITS', 'LMS_AutoTSF', 'MDMLP_EIA',
#  'PatchTST', 'SparseTSF', 'TQNet', 'TimeMixer', 'TimesNet', 'iTransformer']
```

| Model | Venue | Year | Params (ETTh1 paper config, D=7) |
|---|---|---|---|
| DiPE_Linear | DASFAA | 2026 | 267 |
| SparseTSF | ICML Oral | 2024 | 41 |
| FEATHer (ours) | (IoT-J) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | arXiv | 2024 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

→ 5 orders of magnitude (267 → 1.6M) across the lineup. **Two 2026 SOTA
papers (DiPE-Linear DASFAA, MDMLP-EIA AAAI)** address R4's "outdated
baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation (otherwise
   `utils.masking`, `layers.X`, etc. collide between baselines).
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures (`forward(x_enc, x_mark, x_dec,
   x_mark_dec)`, `forward(x, cycle_index)`, tuple returns) into our
   uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** (wzhwzhwzh0921/S-D-Mamba) — selective-scan CUDA kernels
  (`selective_scan_cuda`, `causal_conv1d_cuda`) have no Cortex-M / STM32
  port. Out of scope for edge-deployable comparison.
- **TimeCMA** (ChenxiLiu-HNU/TimeCMA, AAAI 2025) — depends on frozen GPT-2
  (~124M params) features, 5 orders of magnitude above our sub-1K budget.

Reviewer-defense paragraph (paper draft):
> "We deliberately exclude architectures incompatible with our deployment
> scope: Mamba-family models require custom CUDA kernels with no MCU port;
> LLM-augmented forecasters (e.g., TimeCMA) depend on frozen language
> model features that exceed our 1K-parameter budget by 5 orders of
> magnitude. These methods are cited in Related Work but lie outside our
> deployment-constrained benchmark."

## Training protocol — TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ✅ Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | LTSF benchmark convention (iTransformer/PatchTST conventions) |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; model selection = best-val epoch, test evaluated once on that model (matches every baseline's official protocol) |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025–2029 (5 seeds) | TFB / CF-JEPA convention |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ⚠️ Per-(method, dataset) from each repo's official ETT* / ECL / Traffic / Weather / Exchange script

| Axis | Source |
|---|---|
| `lr` | each paper's per-dataset script (1e-4 → 5e-2) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs (`d_model`, `d_ff`, `e_layers`, `n_heads`, `dropout`) | paper Table or official script |
| Data-frequency-tied (`cycle` for TQNet, `period_len` for SparseTSF) | dataset's native sampling rate |

Routing implementation:
- `baselines/__init__.py` holds `_METHOD_DEFAULTS` (canonical ETTh1 config
  per method) and `_DATASET_OVERRIDES` (per-(method, dataset) diff).
- Orchestrator merges defaults + overrides + any CLI `--lr` / `--loss`,
  then forwards `lr` / `loss` as top-level worker flags and the rest as
  `--model_overrides "key=value;key2=value2"`.
- Worker's `_apply_overrides()` parses with int/float/bool coercion and
  sets each key on `args` before model construction; wrapper's
  `getattr(args, "key", default)` picks up the dataset-specific value.

### FEATHer's protocol — single config across all 8 datasets

Deliberate paper-narrative choice. Reviewer-defense line:

> *"Each baseline uses its original-paper per-dataset hyperparameters.
> FEATHer uses a single configuration across all 8 datasets,
> demonstrating that the proposed design generalizes without
> dataset-specific tuning."*

## Architecture (FEATHer model)

The FEATHer model (`baselines/FEATHer/FEATHer.py`) has four components:

1. **Multi-band Decomposition** — depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). Configurable `num_bands ∈ {2,3,4}`.
2. **DenseTemporalKernel (DTK)** — project to latent space → depthwise
   causal conv → project back. Shared across all bands.
3. **FFTFrequencyGate** — FFT magnitude → Conv1d → softmax weights, one
   weight per band, adaptive per sample. Casts to float32 to avoid cuFFT
   half-precision restriction.
4. **SparsePeriodKernel (SPK)** — period-aware forecasting head. Reshapes
   by period phases, shared linear projection across periods, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss. Spectral
loss is FEATHer-specific; the worker skips it for other baselines.

Key training args (FEATHer-specific):

| Argument | Default | Description |
|---|---|---|
| `--d_state` | 8 | DTK latent dimension |
| `--kernel_size` | 7 | DTK conv kernel size |
| `--period` | 12 | SPK period (must divide both seq_len and pred_len) |
| `--num_bands` | 3 | Frequency bands (2, 3, or 4) |
| `--lambda_spec` | 0.01 | Spectral-separation loss weight |

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics are computed once on the best-val-epoch model (early stopping
patience=10); `val_loss` is that epoch's validation loss and `best_epoch`
its index. The saved checkpoint is the same best-val model.

`exp_tag` partitions experiments — e.g. `main`, `robustness_gauss`,
`ablation_dtk`, `smoke`. Use `--results_csv PATH` to redirect for
verification runs without contaminating the main CSV.

## Datasets

Loaded via `utils/data_factory.py`. Lazy darts import — only datasets that
actually use darts trigger the import.

| Dataset | Source | Horizon set |
|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic | darts library | [96, 192, 336, 720] |
| SML, Volatility | local CSV | [24, 48, 96, 192] |
| PEMS03/04/08/PEMS_BAY/METR | local CSV | [12, 24, 48, 96] |
| AirQuality, PM, nrel | local CSV | [96, 192, 336, 720] |

## Known issues / gotchas

- **AMP is forcibly disabled** in `scripts/benchmarks/run_forecast.py`.
  Re-enabling will break FITS (complex grads) and TimesNet (half-precision
  cuFFT on non-power-of-2 sizes).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import. Direct `from baselines.X.Y import Model` will leak
  `sys.modules` entries (`utils`, `layers`, `models`, etc.) across
  baselines and produce `ModuleNotFoundError: No module named
  'utils.masking'` or similar.
- **TQNet** receives `cycle_index = torch.zeros(B)` from the wrapper since
  our DataLoader does not expose per-sample cycle position. Sub-optimal vs
  paper but consistent across all batches.
- **R2 metric** can explode to extreme negatives (~-3e7 observed on
  Traffic) when channels are near-constant within a window (SST≈0).
  MSE/MAE/RMSE are unaffected; R2/CORR are supplementary only. The old
  `-1e9` placeholder bug is gone (single-pass test evaluation since
  2026-06-11). Triage later.
- **FEATHer params scale as D²** (`in_proj = Linear(D, D)`): 453 @ D=7
  (ETT), 866 @ D=14 (Volatility) — sub-1K holds only for D≤14. Weather
  1.4K, Electricity 115K, Traffic 776K. Paper scopes the sub-1K claim
  to edge-typical channel counts; Traffic/Electricity are framed as
  scalability stress tests. A channel-independent variant is a possible
  future ablation axis, not a pre-sweep change.

## Workflow phases

- **Phase 1** ✅ Multi-seed infrastructure (set_seed, registry, orchestrator,
  worker, audit dashboard).
- **Phase 2** ✅ Clone 11 baselines + write wrappers. 12-model lineup
  (FEATHer + 11), S-Mamba/TimeCMA dropped as out-of-scope. DiPE-Linear +
  MDMLP-EIA added 2026-06-02 to address R4 "outdated baselines".
- **Phase 3** ✅ Noise robustness infra — `utils/noise.py` (Gaussian /
  missing / impulse / quantization), `scripts/benchmarks/run_robustness.py`,
  top-level `run_robustness.py` orchestrator. **IoT-J selling point.**
- **Phase 4** (next) Main 5-seed sweep: 12 models × 8 datasets × 4
  horizons × 5 seeds = **1,920 runs**. GPU days. Run on server, not here.
- **Phase 4b** Robustness sweep: 12 models × **4 representative datasets**
  (ETTh1 / Weather / Electricity / SML — canonical / real-sensor /
  smart-grid / smart-home) × 4 horizons × 5 seeds × 20 conditions (1 clean
  + 5 gauss + 5 miss + 4 impulse + 5 quant) = **19,200 rows**. Inference
  only (loads `--save_model` checkpoints). The 4-dataset scope is set in
  `run_robustness.py:ROBUSTNESS_DATASETS` and keeps the sweep tractable
  vs. the 8-dataset main table.
- **Phase 5** Paper artifacts under `tools/paper/` — main table, robustness
  figure, statistical significance (Wilcoxon).

## Paper context

- Target venue: **IEEE Internet of Things Journal (IoT-J)**, IF ~8.
- Repositioning: stronger emphasis on Cortex-M3 deployment, sensor-noise
  robustness, missing-value handling. Lighter on theoretical claims (R8
  flagged Theorem 1/2/3-5 numbering inconsistency in TPAMI manuscript).
- Detailed reviewer-by-reviewer notes and excluded-baseline rationale in
  memory.
