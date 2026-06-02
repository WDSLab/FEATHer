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
├── models/
│   ├── base/
│   │   └── FEATHer.py              # Main model
│   └── ablation/
│       ├── multiscale.py           # Multi-scale ablation variants
│       ├── gating.py               # Gating ablation variants
│       ├── dtk.py                  # Dense Temporal Kernel variants
│       ├── head.py                 # Forecasting head variants
│       └── complexity.py           # Parameter complexity variants
│
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         # Worker: one (model, data, pred_len) x N seeds
│       └── run_robustness.py       # Worker: noise/missing/impulse/quant sweep
│
├── utils/
│   ├── data_factory.py             # Data provider (lazy darts import)
│   ├── data_loader.py              # Dataset classes
│   ├── losses.py                   # Spectral separation loss
│   ├── metrics.py                  # Evaluation metrics
│   ├── noise.py                    # 4-axis corruption (gauss/miss/impulse/quant)
│   ├── seed.py                     # set_seed + parse_seed_list
│   └── timefeatures.py             # Time feature extraction
│
├── README.md
└── requirements.txt
```

---

## Installation

**Requirements**
- Python >= 3.9
- PyTorch >= 2.0

```bash
# Install PyTorch first (based on your CUDA version)
# https://pytorch.org/get-started/locally/

# Install dependencies
pip install -r requirements.txt
```

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

### Benchmark Datasets

| Dataset | Features | Frequency | Prediction Horizons |
|---------|----------|-----------|---------------------|
| ETTh1 | 7 | Hourly | 96, 192, 336, 720 |
| ETTh2 | 7 | Hourly | 96, 192, 336, 720 |
| ETTm1 | 7 | 15-min | 96, 192, 336, 720 |
| ETTm2 | 7 | 15-min | 96, 192, 336, 720 |
| Weather | 21 | 10-min | 96, 192, 336, 720 |
| Electricity | 321 | Hourly | 96, 192, 336, 720 |
| Traffic | 862 | Hourly | 96, 192, 336, 720 |
| Exchange | 8 | Daily | 96, 192, 336, 720 |
| Solar-Energy | 137 | Hourly | 96, 192, 336, 720 |

### Data Sources

- **ETT**: [Informer GitHub](https://github.com/zhouhaoyi/Informer2020) or `darts` library
- **Weather**: [Autoformer GitHub](https://github.com/thuml/Autoformer) or `darts` library
- **Electricity**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
- **Traffic**: [California DOT PEMS](http://pems.dot.ca.gov/)
- **Exchange**: [Lai et al. GitHub](https://github.com/laiguokun/multivariate-time-series-data)
- **Solar-Energy**: [NREL](https://www.nrel.gov/grid/solar-power-data.html)

Most datasets are available through the `darts` library and will be automatically downloaded.

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

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | ETTh1 | Dataset name or "all" |
| `--pred_len` | 0 | Prediction horizon (0 = all horizons) |
| `--seq_len` | 96 | Input sequence length |
| `--d_state` | 8 | State dimension for DenseTemporalKernel |
| `--kernel_size` | 7 | Kernel size for DenseTemporalKernel |
| `--period` | 12 | Period for SparsePeriodKernel |
| `--num_bands` | 3 | Number of frequency bands (2, 3, or 4) |
| `--lambda_spec` | 0.01 | Weight for spectral separation loss |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--num_epochs` | 50 | Number of epochs |
| `--gpu` | 0 | GPU device ID |

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
