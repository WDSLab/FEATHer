# FEATHer

**FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting**

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
│   ├── train.py                    # Main training script
│   ├── train_ablation.py           # Ablation study training
│   └── train_hparam_search.py      # Hyperparameter search
│
├── utils/
│   ├── data_factory.py             # Data provider
│   ├── data_loader.py              # Dataset classes
│   ├── losses.py                   # Spectral separation loss
│   ├── metrics.py                  # Evaluation metrics
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

### Training

```bash
# Single dataset, single prediction horizon
python scripts/train.py --data ETTh1 --pred_len 96

# Single dataset, all prediction horizons (96, 192, 336, 720)
python scripts/train.py --data ETTh1

# All datasets
python scripts/train.py --data all

# With visualization
python scripts/train.py --data ETTh1 --pred_len 96 --save_plot
```

### Ablation Study

```bash
# Multi-scale decomposition ablation
python scripts/train_ablation.py --ablation multiscale --data ETTh1

# Gating mechanism ablation
python scripts/train_ablation.py --ablation gating --data ETTh1

# Available ablation types: multiscale, gating, dtk, head
```

### Hyperparameter Search

```bash
# Full grid search
python scripts/train_hparam_search.py --data ETTh1 --pred_len 96

# Distributed search with config range
python scripts/train_hparam_search.py --data all --config_start 0 --config_end 36 --results_dir results/run1
```

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
from models.base.FEATHer import FEATHer

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

```bibtex
@article{feather2025,
  title={FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting},
  author={},
  journal={},
  year={2025}
}
```

---

## License

MIT License
