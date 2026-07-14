# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

## Current status (2026-07-14c)

**Positioning (settled):** Fresh **JMS** submission (TPAMI reject → IoT-J →
JMS manufacturing pivot). Main table = **6 manufacturing datasets** (Steel ·
WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD), LTSF-8 demoted to a
generalization section, short-horizon PdM (CMAPSS/CMAPSS3/PMSM) **NOT in the
paper** (lr-banked shelf, `--group cmapss`). FEATHer's 453p > SparseTSF 41p is
justified by **ROBUSTNESS, not accuracy** (OFAT num_bands FLAT on clean).
Framing guard: NOT "more params = more robust" but "41p linear lacks headroom;
FEATHer's inductive bias buys graceful degradation at sub-1K". R3's
"long-term-on-edge is contrived" attack resolved via **A+C framing**: (A)
industrial edge NEEDS on-device long-horizon (air-gap/OT isolation, per-sensor
scale, outage autonomy — our datasets are the evidence), (C) edge = param
budget, horizon = task, orthogonal axes; "why 1K" → accuracy-per-param Pareto
frontier. 12-model baseline lineup unchanged.

**⚑ CODE AUDIT (2026-07-14c) — full-repo review done, findings in
`manuscript/notes/code_audit_2026-07-14.md`. ✅ THE CRITICAL BATCH IS FIXED
(same session, all validated end-to-end):**
- ✅ **P0-1 robustness override plumbing**: `run_robustness.py` dispatch now
  mirrors the training-time `get_method_defaults`+`get_dataset_overrides`
  merge → `--model_overrides`; the worker gained the flag +
  `_apply_overrides()`; `gate_logging.py` applies the same merge itself.
  Verified live: trained ETTh1 d16/k3/p8/B4 ckpt loads + clean MSE
  reproduces training-time test MSE exactly.
- ✅ **P0-2 corruption seeding**: salted `hash((ft,sev))` →
  `utils/noise.corruption_seed()` (deterministic index-based). Verified:
  re-run reproduces corrupt-MSE to 6 decimals across processes.
- ✅ **P0-3 robustness resume**: `load_robust_set` counts DISTINCT
  (fault,severity) pairs; worker skips conditions already in the CSV
  (per-condition resume — no duplicate rows on retry). Verified via
  truncate-and-refill test.
- ✅ **P1-1 spectral loss band order** fixed (`bands[S-1-s]` — POINT owns the
  highest slice, LOW the lowest). Synthetic check: correct placement → loss
  0.0, swapped → 2.0. Frozen LTSF OFAT rows kept (λ axis was flat).
- ✅ **P1-2 Weather split** → standard 0.7/0.1/0.2 (legacy hard-coded branch
  + dead PM/PEMS_BAY/METR branches deleted). ALL pre-fix Weather rows are
  invalid → purge + re-run (see server procedure below).
- ✅ **P1-3 RMSE** = sqrt(global MSE) in both workers (+ NaN-aware CORR/R2
  accumulation). Old rows: derive sqrt(MSE) at table time.
- ✅ **P2-1 worker exit code**: training worker now exits 1 if any seed run
  failed (kills the reformer-style silent-skip class). **P2-2 CSV header
  race**: atomic `open(mode="x")` create-with-header in both workers.
- **STILL OPEN (P2-3..8, P3)**: `check_duplicates.py --fix` robust-CSV guard,
  CLI-vs-override precedence for manual FEATHer runs, best_state=None
  degenerate row, mCE partial-reference silence, ablation_table cross-dataset
  pooling, --summary seed-coverage flags, prep interpolation lookahead
  (N2O/MetroPT3) + BlastFurnace winsorize bounds, misc stale docstrings.
  See the audit doc for details.

**▶ SERVER PROCEDURE after pulling the fix commit** (order matters):
```bash
git pull   # (at a natural pause; running workers finish on old code harmlessly)
# invalidated rows — resume refills them:
python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply
# λ-confound (see Evidence state): ALSO purge the λ OFAT slice + hp_base
# (the λ=0.01 anchor) so the λ axis refills under the corrected loss
# (4 tags × 8 datasets × 2H × 2seeds = 128 tiny FEATHer runs):
python tools/audit/purge_rows.py --csv results/hp_search.csv --exp_tag hp_lambda_spec_0.0,hp_lambda_spec_0.001,hp_lambda_spec_0.1,hp_base --apply
# then just re-run the same sweep commands — resume fills the gaps.
# Weather + lambda OFAT slices (hp_search) refill via run_hp_search.py;
# re-freeze the Weather row in _DATASET_OVERRIDES from --summary afterwards.
```

**Evidence state (verified from hp_search.csv)**: only **num_bands is
confirmed FLAT on clean** LTSF-8 (B2/B3/B4 mean MSE 0.4505/0.4418/0.4457).
Gate on/off clean effect UNTESTED (needs ablation sweep); ALL robustness
UNTESTED (sweep unrun). The robustness thesis = 1 confirmed fact + an unproven
bet → the robustness sweep is the #1 risk.
⚠ **λ-flat is CONFOUNDED** (user-flagged 2026-07-14): the OFAT "lambda_spec
flat" verdict was measured under the INVERTED spectral loss (P1-1) — small λ
plausibly = val-tuning suppressing a broken regularizer, not insensitivity.
λ's true effect under the corrected loss is UNMEASURED → re-run the
`hp_lambda_*` OFAT slice (~<100 tiny runs) when refilling Weather OFAT.
Rest of OFAT deliberately kept (λ≤0.01 vs L1 ~0.3-0.5 → rank flips unlikely;
mild provenance caveat only). Memory: project_lambda_flat_confound.md.
All audit-fix sites are tagged in-code — `grep -rn "AUDIT-FIX"` lists every
fixed location with its finding ID.

**Server state**: lr search running (`run_lr_search.py --ngpu 2`, 9 datasets
incl EWELD); LTSF baseline sweep resuming (12 models, 1,920). GPU box =
2×94.5GB, shared with CF-JEPA.

**Local git**: 2 commits LOCAL only, NOT pushed (dbf8179 robust_summary
metrics + 54e3407 gate_logging). Push policy: **ijaehun (personal) ONLY**,
not WDSLab, unless asked.

**▶ NEXT**
1. ✅ ~~Apply audit fixes~~ (done 2026-07-14c, committed with this block).
   Remaining audit items = P2-3..8 + P3 batch (see audit doc), lower urgency.
2. **Server**: run the purge + resume procedure above (Weather + FEATHer
   rows), then continue the chain: lr search finishes → `--summary` → MERGE
   lr into the EXISTING period entries in `_DATASET_OVERRIDES` (no dup keys)
   → mfg main sweep (`--group mfg --exp_tag main --save_model --ngpu 2`,
   1,440, makes robustness checkpoints) → robustness sweep →
   `robust_summary.py` + `gate_logging.py`.
3. **A+C Intro/Motivation draft** (`manuscript/drafts/`) — needs air-gap /
   OT-isolation citations; scope to genuinely-local settings.
4. **Still-missing analysis code**: component ablation UNDER corruption
   (gate-off/band-off variants through the robustness sweep) + Pareto plot
   generator (MSE-vs-params frontier).
5. Push dbf8179 + 54e3407 + the audit-fix commit to ijaehun when user OKs.
6. Raw-data cleanup under `data/`: DECIDED skip (raw = regen safety net; drop
   PMSM raw first if disk pressure).

### Decision history (compact — full blocks in `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-07-14.md`)

- **2026-07-14b**: `robust_summary.py` +4 metrics (μ_rel / AUC / mCE-vs-
  SparseTSF / degradation curve) + `gate_logging.py`, both synthetic-validated;
  R3 resolved (A+C, no short-term in paper); ROBUSTNESS_DATASETS
  GasTurbine→EWELD = [Steel, WindSCADA, EWELD, ETTh1, Weather].
- **2026-07-14**: mfg lineup LOCKED at 6 (EWELD in; GasTurbine fully removed —
  broken, no revival); freq-tied period HPs data-driven in
  `_DATASET_OVERRIDES`; SCINet added-then-reverted; TEP/ziya07/AI4I/GasSensor
  vetted+rejected; PyTSF-MfG = cite for mfg-scarcity gap.
- **2026-07-13c**: short-horizon PdM DROPPED from paper (kept as shelf);
  mfg 3→6 (+MetroPT3 +N2O +BlastFurnace-stress); Tier-2 loader prune
  (data_dict = 18 keys); push policy → ijaehun only.
- **2026-07-13/b**: strategy pivot — robustness (not accuracy) justifies
  params, protocol+citations in
  `manuscript/notes/robustness_justification_evidence.md`; server bugs fixed
  (reformer_pytorch import-time dep; --ngpu `torch.cuda.set_device`) —
  4 models were silently missing from CSVs, resume fills;
  `check_duplicates.py` added (fcst 769 rows, 0 dups).
- **2026-07-06/b**: OFAT 512 + combo 32 done → FEATHer LTSF 8 rows FROZEN into
  `_DATASET_OVERRIDES` (raw-verified); dataset precedent audit — all 6
  confirmed, "first to cast as edge LTSF" holds; wording guards in
  `manuscript/notes/dataset_precedent_audit.md`.
- **2026-07-03**: OFAT verdict — canonical mfg arch stays d8/k7/p12/B3
  (d16/p6 win val but break sub-1K); `--ngpu` dynamic queue + `--jobs_per_gpu`
  added; ⚠ FEATHer params scale with pred_len → sub-1K claim is H=96-scoped.
- **2026-07-02/b**: PMSM dropped from main (2Hz trivial persistence), rebuilt
  for PdM; PHM-2018 vetted+rejected; protocol sequencing locked (arch from
  LTSF OFAT, lr-only on mfg, symmetric).
- **2026-06-25/26**: JMS manufacturing pivot (professor's call, from IoT-J);
  TEP dropped, C-MAPSS added as separate short-horizon section (later
  shelved); Word submission format.
- **2026-06-11..18**: eval protocol rewritten (best-val-epoch, test-once);
  per-dataset FEATHer tuning replaces single-config narrative; baseline LTSF
  sweep launched; tex symbol/citation fixes.

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
├── run_hp_search.py                FEATHer OFAT HP search (LTSF generalization configs + R2 #23 sensitivity)
├── run_lr_search.py                per-(method,dataset) lr search for the MFG main table (Option B, all 12 models)
├── setup_baselines.sh              git-clone baseline upstream repos (idempotent) + pip deps
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         worker (one model × dataset × pred_len × N seeds)
│       └── run_robustness.py       robustness worker (Gauss/miss/impulse/quant sweep)
├── baselines/                      top-level model directory
│   ├── _import_helper.py           sys.modules isolation for upstream imports
│   ├── __init__.py                 registry (12 main + 30 ablation) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
│   ├── FEATHer/                    our model (FEATHer.py + ablation/{multiscale,dtk,gating,head,complexity}.py)
│   └── <11 baseline dirs>/wrapper.py + <Name>-main/ upstream clones (gitignored)
├── utils/
│   ├── seed.py                     set_seed() + parse_seed_list()
│   ├── data_factory.py             data_provider + data_select (lazy darts import; data_dict = 18 keys)
│   ├── data_loader.py              Dataset classes (ETT_hour/minute, Custom, CMAPSS)
│   ├── dispatch_queue.py           --ngpu dynamic job queue (one worker subprocess per GPU)
│   ├── losses.py                   spectral_separation_loss_scales (FEATHer-specific)
│   ├── metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
│   ├── noise.py                    4-axis corruption (gauss/miss/impulse/quant)
│   └── timefeatures.py
├── tools/
│   ├── prep_manufacturing.py       reproducible raw→data.csv for the 6 MFG datasets (tracked; data/ gitignored)
│   ├── audit/check_progress.py     read results CSV, print coverage + mean±std preview
│   ├── audit/check_duplicates.py   resume-key dup audit + --fix (fcst_results.csv schema ONLY — see audit P2-3)
│   └── paper/                      Phase 5 table/figure generators
│       ├── main_table.py           5-seed mean±std table + persistence column (md / latex / csv)
│       ├── wilcoxon.py             pairwise signed-rank vs FEATHer
│       ├── ablation_table.py       5 axes × 4 horizons aggregator (clean accuracy only)
│       ├── robust_summary.py       μ_rel / AUC / mCE-vs-SparseTSF / degradation curves + heatmaps
│       ├── gate_logging.py         FFT gate-under-corruption logging (per-band weights, CSV+PNG)
│       └── persistence_baseline.py copy-last baseline under the exact worker protocol
├── deployment/cortex_m3/           Layer-1 edge-cost estimator + qemu/ Layer-2 skeleton (WSL)
├── results/                        fcst_results.csv · robust_results.csv · edge_estimates.csv · checkpoints/<exp_tag>/
├── data/                           raw + processed data (gitignored — transfer to server manually)
└── manuscript/                     reviews, drafts, notes (gitignored; tex_workspace/ + drafts/*.md selectively tracked)
```

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos (+ reformer_pytorch dep) ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Manufacturing main table (default group; 6 datasets) ===
python run_forecast.py --check --exp_tag main             # 1,440 mfg runs (12×6×4×5)
python run_forecast.py --exp_tag main --save_model        # group=mfg by default; --save_model makes robustness ckpts
python run_forecast.py --group ltsf --exp_tag main --save_model   # LTSF-8 generalization (1,920)
python run_forecast.py --group cmapss --exp_tag main --save_model # PdM shelf (540; NOT in paper)
python run_forecast.py --group all  --exp_tag main --save_model   # mfg + ltsf = 3,360
python run_forecast.py --model FEATHer --save_model                # one model across group
python run_forecast.py --data ETTh1 --pred_len 96 --save_model     # one (data, horizon)
python run_forecast.py --exclude TimesNet,MDMLP_EIA                # skip heavy models
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke

# Multi-GPU: all four orchestrators take --ngpu N (dynamic queue, one worker
# subprocess per GPU; utils/dispatch_queue.py; no DDP) and --jobs_per_gpu M
# (ngpu×M workers, auto OMP/MKL pinning). ngpu=1 default = sequential.
python run_forecast.py --exp_tag main --save_model --ngpu 2 --jobs_per_gpu 3

# === lr search for the mfg main table (Option B; runs BEFORE the main sweep) ===
# 12 models × 5 lr × 9 datasets (6 mfg + 3 PdM shelf) × 2 horizons × 2 seeds = 2,160.
# Bypasses _DATASET_OVERRIDES (default arch) — period overrides only affect the main sweep.
python run_lr_search.py --check
python run_lr_search.py --ngpu 2            # resumable
python run_lr_search.py --summary           # best lr per (method,dataset) → MERGE into _DATASET_OVERRIDES
                                            # (merge into EXISTING period entries — no duplicate keys!)

# === FEATHer OFAT HP search (LTSF generalization only; DONE, rows frozen) ===
python run_hp_search.py --check / --summary / --validate
# Windows gotcha: run --summary with PYTHONUTF8=1 PYTHONIOENCODING=utf-8 (em-dash vs cp949)

# === Robustness sweep (after mfg-main checkpoints exist) ===
# ⚠ BLOCKED on audit P0 fixes (override forwarding etc.) — see
# manuscript/notes/code_audit_2026-07-14.md before launching.
python run_robustness.py --check            # 1,200 ckpt × 20 cond = 24,000 rows
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --model FEATHer --train_exp_tag main

# === Ablation sweep (5 axes × 30 FEATHer variants; scope extends to mfg) ===
python run_forecast.py --ablation_axis all --data ETTh1 --exp_tag ablation
# axes: ms (15) / gate (4) / dtk (4) / head (4) / complexity (3)

# === Progress audit ===
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py      # fcst_results.csv ONLY (not robust CSV)

# === Paper artifact generators (Phase 5) ===
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer
python tools/paper/ablation_table.py --axis ms --format latex
python tools/paper/robust_summary.py --output_dir manuscript/figures/
python tools/paper/gate_logging.py --data Steel --pred_len 96 --ckpt <path>

# === Data regen (data/ is gitignored) ===
python tools/prep_manufacturing.py            # all 6 mfg recipes
python tools/prep_manufacturing.py EWELD      # one (BlastFurnace needs openpyxl)

# === Cortex-M3 edge cost estimator / QEMU (Layer 1 / Layer 2, WSL) ===
python -m deployment.cortex_m3.run --check
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build (IoT-J tex now legacy; JMS = Word via drafts/*.md) ===
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
| FEATHer (ours) | (JMS) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | arXiv | 2024 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

(SCINet was briefly added 2026-07-13 then REMOVED — reviewer only named it as
prior multi-scale work, not a baseline request; Related-Work differentiation
instead.) → 5 orders of magnitude (267 → 1.6M). Two 2026 SOTA papers
(DiPE-Linear, MDMLP-EIA) address R4's "outdated baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation.
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures into uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** — selective-scan CUDA kernels have no Cortex-M port.
- **TimeCMA** (AAAI 2025) — frozen GPT-2 features, 5 orders of magnitude above budget.

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
| `seq_len` | 96 | LTSF benchmark convention |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; selection = best-val epoch, test evaluated once on that model |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025–2029 (5 seeds) | TFB / CF-JEPA convention |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ⚠️ Per-(method, dataset) overrides

| Axis | Source |
|---|---|
| `lr` | baselines: each paper's per-dataset script; mfg: `run_lr_search.py` (merge pending) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs | paper Table or official script; FEATHer LTSF-8 rows from OFAT (frozen) |
| Freq-tied (`cycle`, `period_len`, FEATHer `period`) | data-driven per mfg dataset (FFT-detected), same rule for all 3 period-models |

Routing: `_METHOD_DEFAULTS` + `_DATASET_OVERRIDES` in `baselines/__init__.py`;
orchestrator merges defaults + overrides + CLI, forwards `lr`/`loss` as worker
flags and the rest as `--model_overrides "k=v;k2=v2"`; worker's
`_apply_overrides()` sets each key on `args` before model construction.
FEATHer is tuned per-dataset like every baseline (symmetric fair-comparison
framing); canonical mfg arch = base d8/k7/p12/B3.

## Architecture (FEATHer model)

`baselines/FEATHer/FEATHer.py`, four components:

1. **Multi-band Decomposition** — depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). `num_bands ∈ {2,3,4}`.
2. **DenseTemporalKernel (DTK)** — project to latent → depthwise causal conv
   → project back. Shared across all bands.
3. **FFTFrequencyGate** — FFT magnitude → Conv1d → softmax weights, one per
   band, adaptive per sample. Casts to float32 (cuFFT half-precision).
4. **SparsePeriodKernel (SPK)** — period-aware head: reshape by period phases,
   shared linear projection, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss (FEATHer-only;
⚠ band-order inversion bug — audit P1-1 — fix pending).

Key FEATHer args: `--d_state 8` (DTK latent), `--kernel_size 7`, `--period 12`
(must divide seq_len AND pred_len), `--num_bands 3`, `--lambda_spec 0.01`.

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics computed once on the best-val-epoch model. `exp_tag` partitions
experiments (`main`, `robust`, `ablation_dtk`, `smoke`, `lr_*`, `hp_*`).
Use `--results_csv PATH` to redirect verification runs.

## Datasets

Loaded via `utils/data_factory.py` (lazy darts import; `data_dict` = 18 keys
after the 2026-07-13c Tier-2 prune).

| Dataset | Source | Horizon set |
|---|---|---|
| **Steel · WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD** (MFG **main table**; BlastFurnace D=28 = stress tier) | local CSV (`tools/prep_manufacturing.py`) | [96, 192, 336, 720] |
| CMAPSS / CMAPSS3 / PMSM (PdM **shelf**, not in paper; unit-aware `Dataset_CMAPSS`) | local CSV | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (generalization) | darts | [96, 192, 336, 720] |
| nrel (kept per user) | local CSV | [96, 192, 336, 720] |

Robustness set: `run_robustness.py:ROBUSTNESS_DATASETS` =
**[Steel, WindSCADA, EWELD, ETTh1, Weather]** (EWELD's 5,741 extreme-weather
event windows = the only non-synthetic distribution-shift test).
EWELD gotcha: strong daily cycle → MUST report a seasonal-naive baseline
(copy-last-day ~0.55). Each `data/<name>/` holds raw + cleaned `data.csv`;
regenerate with `python tools/prep_manufacturing.py [<name> ...]`
(BlastFurnace needs `openpyxl`). `data/` is gitignored — transfer manually.

## Known issues / gotchas

- **⚑ AUDIT (2026-07-14c)** — P0/P1/P2-1/2 batch FIXED + validated (see
  Current status). Still open from
  `manuscript/notes/code_audit_2026-07-14.md`: `check_duplicates.py --fix`
  must NOT be pointed at robust_results.csv (its key lacks fault/severity —
  it would delete 95% of the sweep); manual FEATHer sensitivity runs: CLI
  arch flags are silently beaten by `_DATASET_OVERRIDES` (sweeps unaffected);
  pre-fix Weather rows and stored-RMSE values in old CSVs are stale (purge /
  derive sqrt(MSE) respectively).
- **`--ngpu N` + upstream hardcoded `.cuda()` (fixed 2026-07-13)**: workers
  call `torch.cuda.set_device(args.gpu)` at startup. Do NOT sed the upstream
  `baselines/*/*-main/` files (gitignored, re-cloned).
- **`reformer_pytorch` is an import-time dep of 4 baselines** (iTransformer,
  LMS_AutoTSF, TimeMixer, TimesNet). Missing → those 4 die at import, write
  no CSV row, and look "not yet run" (failed workers currently exit 0 —
  audit P2-1). `setup_baselines.sh` installs `reformer_pytorch==1.4.4`.
- **AMP is forcibly disabled** in the worker. Re-enabling breaks FITS
  (complex grads) and TimesNet (half-precision cuFFT).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import (sys.modules collisions otherwise).
- **TQNet** receives `cycle_index = torch.zeros(B)` (our DataLoader has no
  per-sample cycle position) — consistent across all batches.
- **R2 metric** can explode to extreme negatives on near-constant channels
  (SST≈0); MSE/MAE/RMSE unaffected; R2/CORR supplementary only.
- **FEATHer params scale as D²** (`in_proj`): sub-1K holds for D≤14 only.
  Weather 1.4K / Electricity 115K / Traffic 776K / BlastFurnace(D=28) 1,986
  are framed as stress tests.
- **FEATHer params also scale with pred_len** (SPK backbone): quoted counts
  are H=96 values; at H=720 D≥11 exceeds 1K. Sub-1K claim must be scoped to
  H=96 in the manuscript (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1–3** ✅ infra (multi-seed, 12-model registry, 4-axis noise).
- **Phase 4** (running) main sweeps on server: mfg 1,440 + LTSF 1,920.
- **Phase 4b** (blocked on P0 fixes + mfg checkpoints) robustness sweep:
  12 models × 5 datasets × 4 horizons × 5 seeds × 20 conditions = 24,000 rows,
  inference-only from `--save_model` checkpoints.
- **Phase 5** paper artifacts under `tools/paper/` (+ still to write:
  ablation-under-corruption aggregation, Pareto plot).

## Paper context

- Target venue: **Journal of Manufacturing Systems (JMS)**, Elsevier/SME, Q1.
  Story: edge PdM/process-monitoring forecasting on machine-side MCUs;
  FEATHer = sub-1K Cortex-M3-deployable forecaster, robust to shop-floor
  sensor faults, validated across 6 industrial domains. Contributions:
  (1) sub-1K multiscale-frequency forecaster, (2) first edge/sub-1K mfg LTSF
  benchmark (6 domains), (3) sensor-fault robustness, (4) Cortex-M3
  deployment evidence.
- **Submission format: Word** (Elsevier "Your Paper Your Way"; no rigid
  template; Highlights 3–5 bullets ≤85 chars; Vancouver refs; CRediT +
  declarations). Draft in markdown under `manuscript/drafts/` → paste into
  Word. Order: Intro (A+C framing) → Datasets → Related → Method polish →
  results when sweeps land.
- Reuse map from `feather_iotj.tex`: Methodology + Theoretical Analysis
  ~as-is (trim theory); Intro/Related/Datasets need the manufacturing
  reframe; Results/Ablation/Robustness wait for sweep numbers.
- Wording guards: CMAPSS = "first to cast as a fixed-horizon multivariate
  sensor forecasting BENCHMARK" (not "first to forecast"); PMSM cite AIP
  Advances 16(4):045101; Steel all-channels departure stated explicitly;
  mfg-LTSF precedent = "first to cast as edge PdM LTSF"
  (`manuscript/notes/dataset_precedent_audit.md`).
- Reviewer-by-reviewer notes in memory; reject letter at
  `manuscript/notes/reject_mail.md`; robustness protocol citations at
  `manuscript/notes/robustness_justification_evidence.md`.
