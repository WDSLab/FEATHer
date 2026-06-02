# FEATHer

**FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting**

[![arXiv](https://img.shields.io/badge/arXiv-2601.11350-b31b1b.svg)](https://arxiv.org/abs/2601.11350)

---

## Overview

FEATHer is an ultra-lightweight model for long-term time series forecasting. It combines multi-scale frequency decomposition with a shared temporal kernel and period-aware sparse forecasting head.

### Key Features

- **Multi-scale Frequency Decomposition**: Separates input into frequency bands (Point, High, Mid, Low)
- **Shared Dense Temporal Kernel**: Efficient temporal mixing across all frequency bands
- **FFT-based Adaptive Gating**: Learns optimal weights for each frequency band
- **Sparse Period-aware Head**: Period-based sparse forecasting inspired by SparseTSF
- **Spectral Separation Loss**: Encourages frequency-specific learning in each band
- **Ultra-Lightweight**: Sub-1K parameters while maintaining competitive performance

---

## Project Structure

```
FEATHer/
├── run_forecast.py                 # User-facing orchestrator (--check, resume, dispatch)
├── run_robustness.py               # Robustness orchestrator (loads --save_model checkpoints)
├── setup_baselines.sh              # Idempotent git-clone of 11 upstream baselines
│
├── baselines/                      # Model registry (12 main + 30 FEATHer ablation)
│   ├── _import_helper.py           # sys.modules isolation for upstream imports
│   ├── __init__.py                 # get_model / list_models / list_ablation_models
│   ├── FEATHer/
│   │   ├── FEATHer.py              # Main model
│   │   └── ablation/
│   │       ├── multiscale.py       # 15 variants (P, H, M, L, PH, ..., PHML)
│   │       ├── gating.py           # 4 variants (none, uniform, softmax, fft)
│   │       ├── dtk.py              # 4 variants (none, mlp, shallow, full)
│   │       ├── head.py             # 4 variants (linear, mlp, conv, spk)
│   │       └── complexity.py       # 3 variants (half, full, double)
│   ├── DLinear/wrapper.py          # + DLinear-main/ (cure-lab/LTSF-Linear)
│   ├── PatchTST/wrapper.py         # + PatchTST-main/ (yuqinie98)
│   ├── iTransformer/wrapper.py     # + iTransformer-main/ (thuml)
│   ├── FITS/wrapper.py             # + FITS-main/ (VEWOXIC)
│   ├── SparseTSF/wrapper.py        # + SparseTSF-main/ (lss-1138)
│   ├── TimeMixer/wrapper.py        # + TimeMixer-main/ (kwuking)
│   ├── TimesNet/wrapper.py         # + TimesNet-main/ (thuml/Time-Series-Library)
│   ├── TQNet/wrapper.py            # + TQNet-main/ (ACAT-SCUT)
│   ├── LMS_AutoTSF/wrapper.py      # + LMS_AutoTSF-main/ (mribrahim)
│   ├── DiPE_Linear/wrapper.py      # + DiPE_Linear-main/ (wintertee, DASFAA 2026)
│   └── MDMLP_EIA/wrapper.py        # + MDMLP_EIA-main/ (zh1985csuccsu, AAAI 2026)
│
├── scripts/benchmarks/
│   ├── run_forecast.py             # Worker: one (model, data, pred_len) x N seeds
│   └── run_robustness.py           # Worker: corruption sweep over saved checkpoints
│
├── utils/
│   ├── data_factory.py             # Data provider (lazy darts import)
│   ├── data_loader.py              # Dataset classes (ETT, Custom, PEMS)
│   ├── losses.py                   # Spectral separation loss
│   ├── metrics.py                  # MSE / MAE / RMSE / CORR / R2
│   ├── noise.py                    # 4-axis corruption (gauss/miss/impulse/quant)
│   ├── seed.py                     # set_seed + parse_seed_list
│   └── timefeatures.py
│
├── tools/audit/check_progress.py   # Coverage matrix + mean+/-std preview
├── results/
│   ├── fcst_results.csv            # Main sweep (append-only, resume-by-CSV)
│   ├── robust_results.csv          # Robustness sweep
│   └── checkpoints/<exp_tag>/      # Deterministic .pth (from --save_model)
├── README.md
└── requirements.txt
```

---

## Installation

**Requirements**
- Python >= 3.9
- PyTorch >= 2.0 (CUDA 12.x recommended for the full sweep)
- `darts` for ETT/Weather/Exchange/Electricity/Traffic loaders
- `reformer_pytorch` (transitive dep of iTransformer's `SelfAttention_Family`)

```bash
# Install PyTorch first (match your CUDA version)
# https://pytorch.org/get-started/locally/

# Install dependencies
pip install -r requirements.txt

# Clone the 11 upstream baseline repos into baselines/<Name>/<Name>-main/
bash setup_baselines.sh           # all (idempotent)
bash setup_baselines.sh DLinear   # one
```

> **Note:** `mamba-ssm` is intentionally not installed. S-Mamba / TimeCMA
> are excluded from the benchmark because their CUDA kernels and LLM
> features are incompatible with the edge-deployable target (sub-1K
> parameters, Cortex-M3-class MCU).

---

## Quick Start

### Main forecasting sweep

```bash
# Smoke test (no checkpoints, separate CSV)
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke

# Show what is missing (resume by CSV — append-only, idempotent)
python run_forecast.py --check

# Full 5-seed sweep, save checkpoints for the robustness phase
python run_forecast.py --num_seeds 5 --num_epochs 50 --exp_tag main --save_model

# One model across all datasets/horizons
python run_forecast.py --model FEATHer --save_model
```

### Robustness sweep (loads checkpoints from --save_model)

```bash
python run_robustness.py --check
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --fault_types gauss,miss   # subset of axes
```

### Ablation Study — FEATHer variants are first-class models

The 30 FEATHer ablation variants are registered alongside the 12 baselines
and sweep through the same orchestrator, worker, CSV, and 5-seed protocol.

```bash
# One axis across ETTh1 / all horizons / 5 seeds
python run_forecast.py --model FEATHer_ms_PHML --data ETTh1 --exp_tag ablation_multiscale
python run_forecast.py --model FEATHer_gate_fft --data ETTh1 --exp_tag ablation_gating
python run_forecast.py --model FEATHer_dtk_full --data ETTh1 --exp_tag ablation_dtk
python run_forecast.py --model FEATHer_head_spk --data ETTh1 --exp_tag ablation_head
python run_forecast.py --model FEATHer_complexity_full --data ETTh1 --exp_tag ablation_complexity
```

Available variant names (30 total):

| Axis     | Variants                                                           |
|----------|--------------------------------------------------------------------|
| ms (15)  | P, H, M, L, PH, PM, PL, HM, HL, ML, PHM, PHL, PML, HML, PHML       |
| gate (4) | none, uniform, softmax, fft                                        |
| dtk (4)  | none, mlp, shallow, full                                           |
| head (4) | linear, mlp, conv, spk                                             |
| complexity (3) | half, full, double                                           |

`baselines.list_ablation_models()` returns the full list.

---

## Datasets

### Main benchmark (8 datasets via darts)

| Dataset     | Features | Frequency | Prediction Horizons |
|-------------|----------|-----------|---------------------|
| ETTh1       | 7        | Hourly    | 96, 192, 336, 720   |
| ETTh2       | 7        | Hourly    | 96, 192, 336, 720   |
| ETTm1       | 7        | 15-min    | 96, 192, 336, 720   |
| ETTm2       | 7        | 15-min    | 96, 192, 336, 720   |
| Weather     | 21       | 10-min    | 96, 192, 336, 720   |
| Electricity | 321      | Hourly    | 96, 192, 336, 720   |
| Traffic     | 862      | Hourly    | 96, 192, 336, 720   |
| Exchange    | 8        | Daily     | 96, 192, 336, 720   |

### Local CSV / NPY (opt-in via --data)

| Dataset                       | Horizons          |
|-------------------------------|-------------------|
| SML, Volatility               | 24, 48, 96, 192   |
| PEMS03/04/08, PEMS_BAY, METR  | 12, 24, 48, 96    |
| AirQuality, PM, nrel          | 96, 192, 336, 720 |

### Data Sources

- **ETT**: [Informer GitHub](https://github.com/zhouhaoyi/Informer2020) or `darts` (auto-download)
- **Weather**: [Autoformer GitHub](https://github.com/thuml/Autoformer) or `darts`
- **Electricity**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- **Traffic**: [California DOT PEMS](http://pems.dot.ca.gov/)
- **Exchange**: [Lai et al. GitHub](https://github.com/laiguokun/multivariate-time-series-data)
- **PEMS / METR-LA**: graph-traffic benchmarks (Li et al., 2018)
- **SML / Volatility / AirQuality / PM / nrel**: local CSVs under `data/`

darts loaders are imported lazily — non-darts datasets work without it.

---

## Model Usage

```python
from baselines.FEATHer.FEATHer import FEATHer

# Create model
model = FEATHer(
    seq_len=96,
    pred_len=96,
    d_model=7,        # number of features
    d_state=8,        # latent state dimension
    kernel_size=7,    # temporal kernel size
    period=24,        # period for sparse head
    num_bands=3,      # frequency bands (2, 3, or 4)
)

# Forward pass
import torch
x = torch.randn(32, 96, 7)  # (batch, seq_len, features)
y = model(x)                 # (batch, pred_len, features)
```

---

## Baseline lineup

| Model        | Venue           | Year | Params (ETTh1, D=7) |
|--------------|-----------------|------|---------------------|
| SparseTSF    | ICML Oral       | 2024 | 41                  |
| DiPE_Linear  | DASFAA          | 2026 | 267                 |
| **FEATHer**  | (IoT-J)         | 2026 | **453**             |
| FITS         | ICLR Spotlight  | 2024 | 1,200               |
| DLinear      | AAAI            | 2023 | 18,624              |
| PatchTST     | ICLR            | 2023 | 35,171              |
| TimeMixer    | ICLR            | 2024 | 75,497              |
| TQNet        | ICML            | 2025 | 86,192              |
| LMS_AutoTSF  | arXiv           | 2024 | 181,318             |
| TimesNet     | ICLR            | 2023 | 605,479             |
| iTransformer | ICLR            | 2024 | 841,568             |
| MDMLP_EIA    | AAAI            | 2026 | 1,632,832           |

Parameter counts span 5 orders of magnitude (41 → 1.6M). FEATHer sits in
the sub-1K regime alongside SparseTSF / DiPE-Linear / FITS.

## Benchmark protocol

| Axis                    | Value                                          |
|-------------------------|------------------------------------------------|
| Seeds                   | 5 (2025-2029)                                  |
| seq_len, batch_size     | 96, 32                                         |
| num_epochs              | 50                                             |
| Optimizer / scheduler   | Adam + CosineAnnealingLR                       |
| AMP                     | **off** (FITS complex grads + TimesNet cuFFT)  |
| Determinism             | `cudnn.deterministic=True`, `benchmark=False`  |
| Per-(method, dataset) HP| Each baseline's official paper / script values |
| FEATHer HP              | **Single config across all 8 datasets**        |

Per-(method, dataset) learning rates, depths, and architecture HPs are
pulled from each baseline's upstream `scripts/<dataset>.sh` and stored in
`baselines/__init__.py:_DATASET_OVERRIDES`. FEATHer deliberately uses a
single configuration across every dataset, demonstrating generalization
without dataset-specific tuning.

## Training Arguments

| Argument          | Default  | Description                                       |
|-------------------|----------|---------------------------------------------------|
| `--model`         | (all)    | Single model name, e.g. `FEATHer` or `FEATHer_ms_PHML` |
| `--data`          | (all)    | Dataset name (skip for full 8-dataset sweep)      |
| `--pred_len`      | 0        | 0 = all horizons for that dataset                 |
| `--seq_len`       | 96       | Input sequence length                             |
| `--num_seeds`     | 5        | Seeds 2025-2029 by default                        |
| `--num_epochs`    | 50       |                                                   |
| `--batch_size`    | 32       |                                                   |
| `--exp_tag`       | `main`   | Partitions experiments in the CSV                 |
| `--save_model`    | off      | Save checkpoint for robustness reuse              |
| `--exclude`       | (none)   | Comma-separated model names to skip               |
| `--results_csv`   | results/fcst_results.csv | Redirect for verification runs    |
| `--lr`, `--loss`  | (paper)  | Override the per-(method, dataset) defaults       |
| `--d_state`       | 8        | FEATHer DTK latent dimension                      |
| `--kernel_size`   | 7        | FEATHer DTK conv kernel                           |
| `--period`        | 12       | FEATHer SPK period (must divide seq_len & pred_len) |
| `--num_bands`     | 3        | FEATHer frequency bands (2, 3, or 4)              |
| `--lambda_spec`   | 0.01     | Spectral separation loss weight                   |

---

## Model Architecture

### Components

1. **Multi-band Decomposition**
   - POINT: kernel=1 (high-frequency details)
   - HIGH: kernel=3 (when num_bands=4)
   - MID: kernel=5 (when num_bands>=3)
   - LOW: avg pooling + interpolation (low-frequency trends)

2. **DenseTemporalKernel**
   - Input projection to latent space
   - Depthwise causal convolution
   - Output projection back to model dimension

3. **FFTFrequencyGate**
   - FFT magnitude spectrum computation
   - Conv1d for frequency feature extraction
   - Softmax-normalized weights per band

4. **SparsePeriodKernel**
   - Period-wise phase reorganization
   - Shared linear projection across periods
   - Temporal reconstruction

---

## On-Device Deployment

Edge deployment experiments were conducted on a physical Cortex-M3-class embedded platform:

| Setting | Value |
|---------|-------|
| **Target Board** | LM3S6965EVB (Stellaris) |
| **Processor** | ARM Cortex-M3 |
| **Compiler** | arm-none-eabi-gcc |
| **RAM Budgets** | 16KB / 32KB / 64KB |
| **Batch Size** | 1 (streaming edge usage) |

FEATHer achieves deployability under extreme memory constraints (16KB RAM on ETTh1) where most baseline models fail, demonstrating its suitability for resource-constrained edge devices.

> **Note**: The C implementation for MCU deployment is not included in this repository.

---

## Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| CORR | Correlation coefficient |
| R2 | R-squared score |

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{feather2025,
  title={FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting},
  author={Lee, Jaehoon and Lee, Seungwoo and Kim, Younghwi and Kim, Dohee and Sim, Sunghyun},
  journal={arXiv preprint arXiv:2601.11350},
  year={2025}
}
```

**Paper**: [https://arxiv.org/abs/2601.11350](https://arxiv.org/abs/2601.11350)

---

## License

MIT License
