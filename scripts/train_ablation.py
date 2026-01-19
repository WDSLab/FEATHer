# -*- coding: utf-8 -*-
"""
FEATHer Unified Ablation Training Script
========================================

Unified script for all ablation studies

Ablation Types:
- multiscale: Multi-scale decomposition (P, H, M, L, PH, PM, ..., PHML)
- gating: Gating mechanism (none, uniform, softmax, fft)
- dtk: Dense Temporal Kernel (none, mlp, shallow, full)
- head: Forecasting head (linear, mlp, conv, spk)

Usage:
    # Multiscale ablation
    python scripts/train_ablation.py --ablation multiscale --data ETTh1 --pred_len 96
    python scripts/train_ablation.py --ablation multiscale --data ETTh1 --variant PHML

    # Gating ablation
    python scripts/train_ablation.py --ablation gating --data ETTh1,ETTh2
    python scripts/train_ablation.py --ablation gating --data ETTh1 --variant fft

    # DTK ablation
    python scripts/train_ablation.py --ablation dtk --data Weather --pred_len 96

    # Head ablation
    python scripts/train_ablation.py --ablation head --data all

    # All ablations
    python scripts/train_ablation.py --ablation all --data ETTh1 --pred_len 96
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import argparse
import gc
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_factory
from utils import metrics

warnings.filterwarnings("ignore")


# =============================================================================
# Ablation Configuration
# =============================================================================

ABLATION_CONFIG = {
    'multiscale': {
        'variants': [
            'P', 'H', 'M', 'L',  # Single (4)
            'PH', 'PM', 'PL', 'HM', 'HL', 'ML',  # Dual (6)
            'PHM', 'PHL', 'PML', 'HML',  # Tri (4)
            'PHML',  # Full (1)
        ],
        'param_name': 'band_config',
        'folder_name': 'multiscale_ablation',
        'results_file': 'multiscale_ablation_results.csv',
    },
    'gating': {
        'variants': ['none', 'uniform', 'softmax', 'fft'],
        'param_name': 'gating_type',
        'folder_name': 'gating_ablation',
        'results_file': 'gating_ablation_results.csv',
    },
    'dtk': {
        'variants': ['none', 'mlp', 'shallow', 'full'],
        'param_name': 'dtk_type',
        'folder_name': 'dtk_ablation',
        'results_file': 'dtk_ablation_results.csv',
    },
    'head': {
        'variants': ['linear', 'mlp', 'conv', 'spk'],
        'param_name': 'head_type',
        'folder_name': 'head_ablation',
        'results_file': 'head_ablation_results.csv',
    },
}

PRED_LENS = {
    'standard': [96, 192, 336, 720],
    'short': [24, 48, 96, 192],
}

DATASET_PRED_LENS = {
    'ETTh1': 'standard',
    'ETTh2': 'standard',
    'ETTm1': 'standard',
    'ETTm2': 'standard',
    'Weather': 'standard',
    'Exchange': 'standard',
    'nrel': 'standard',
    'Electricity': 'standard',
    'Traffic': 'standard',
    'AirQuality': 'standard',
    'PM': 'standard',
    'Volatility': 'short',
    'SML': 'short',
}


# =============================================================================
# Model Factory
# =============================================================================

def get_ablation_model(ablation_type, variant, seq_len, pred_len, d_model, config):
    """Create ablation model based on type and variant"""

    if ablation_type == 'multiscale':
        from models.ablation.multiscale import FEATHer_Multiscale, get_variant_name
        model = FEATHer_Multiscale(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=config['d_state'],
            kernel_size=config['kernel_size'],
            period=config['period'],
            use_norm=True,
            band_config=variant,
        )

    elif ablation_type == 'gating':
        from models.ablation.gating import FEATHer_Gating, get_variant_name
        model = FEATHer_Gating(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=config['d_state'],
            kernel_size=config['kernel_size'],
            period=config['period'],
            num_bands=config['num_bands'],
            use_norm=True,
            gating_type=variant,
        )

    elif ablation_type == 'dtk':
        from models.ablation.dtk import FEATHer_DTK, get_variant_name
        model = FEATHer_DTK(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=config['d_state'],
            kernel_size=config['kernel_size'],
            period=config['period'],
            num_bands=config['num_bands'],
            use_norm=True,
            dtk_type=variant,
        )

    elif ablation_type == 'head':
        from models.ablation.head import FEATHer_Head, get_variant_name
        model = FEATHer_Head(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            d_state=config['d_state'],
            kernel_size=config['kernel_size'],
            period=config['period'],
            num_bands=config['num_bands'],
            use_norm=True,
            head_type=variant,
        )
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    return model, get_variant_name


def get_variant_name_for_type(ablation_type, variant):
    """Get variant name function for ablation type"""
    if ablation_type == 'multiscale':
        from models.ablation.multiscale import get_variant_name
    elif ablation_type == 'gating':
        from models.ablation.gating import get_variant_name
    elif ablation_type == 'dtk':
        from models.ablation.dtk import get_variant_name
    elif ablation_type == 'head':
        from models.ablation.head import get_variant_name
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    return get_variant_name(variant)


# =============================================================================
# Helper Functions
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_config(config_str):
    """'p12_k6_d5_b4' -> {'period': 12, 'kernel_size': 6, 'd_state': 5, 'num_bands': 4}"""
    parts = config_str.split('_')
    return {
        'period': int(parts[0][1:]),
        'kernel_size': int(parts[1][1:]),
        'd_state': int(parts[2][1:]),
        'num_bands': int(parts[3][1:]),
    }


def load_best_configs(csv_path):
    """Load best_configs.csv and convert to dictionary"""
    df = pd.read_csv(csv_path)
    configs = {}

    for _, row in df.iterrows():
        data = row['data']
        pred_len = int(row['pred_len'])
        config = parse_config(row['best_config'])
        configs[(data, pred_len)] = config

    return configs


def get_pred_lens(data_name):
    pred_type = DATASET_PRED_LENS.get(data_name, 'standard')
    return PRED_LENS[pred_type]


# =============================================================================
# Save Functions
# =============================================================================

def save_results(ablation_type, now, test_metric, data_name, pred_len, seq_len,
                 best_epoch, train_loss_hist, test_loss_hist, variant, config,
                 iteration, n_params, results_dir='results'):

    ablation_cfg = ABLATION_CONFIG[ablation_type]
    folder_name = ablation_cfg['folder_name']
    param_name = ablation_cfg['param_name']
    results_file = ablation_cfg['results_file']

    folder_path = f'{results_dir}/{folder_name}/{data_name}/{pred_len}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestamp = f'{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    variant_name = get_variant_name_for_type(ablation_type, variant)
    base_filename = f'{variant_name}_{data_name}_{pred_len}_iter{iteration}_{timestamp}'
    file_path = os.path.join(folder_path, base_filename)

    # Build result dict
    result_csv = {
        'ablation_type': ablation_type,
        'variant': variant_name,
        param_name: variant,
        'data': data_name,
        'pred_len': pred_len,
        'seq_len': seq_len,
        'iteration': iteration,
        'MSE': test_metric[0],
        'MAE': test_metric[1],
        'RMSE': test_metric[2],
        'CORR': test_metric[3],
        'R2': test_metric[4],
        'best_epoch': best_epoch,
        'n_params': n_params,
        'period': config['period'],
        'kernel_size': config['kernel_size'],
        'd_state': config['d_state'],
        'timestamp': timestamp,
    }

    # Add num_bands for non-multiscale ablations
    if ablation_type != 'multiscale':
        result_csv['num_bands'] = config['num_bands']

    pd.DataFrame([result_csv]).to_csv(file_path + '_result.csv', index=False)

    # Append to aggregated results CSV
    csv_path = f'{results_dir}/{results_file}'

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([result_csv])], ignore_index=True)
    else:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        df = pd.DataFrame([result_csv])

    df.to_csv(csv_path, index=False)

    return file_path


# =============================================================================
# Training Function
# =============================================================================

def train_single(args, ablation_type, variant, config, data_name, pred_len, device, iteration):

    now = datetime.now()
    num_metrics = 5  # MSE, MAE, RMSE, CORR, R2

    # Load data
    df, freq, embed = data_factory.data_select(data_name, args.root_path)
    n_features = df.shape[1] - 1

    train_data, train_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, 'train'
    )
    test_data, test_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, 'test'
    )

    # Build model
    model, _ = get_ablation_model(ablation_type, variant, args.seq_len, pred_len, n_features, config)
    model = model.float().to(device)

    n_params = count_parameters(model)
    variant_name = model.get_variant_name()

    print(f"    Model: {variant_name} | Params: {n_params:,}")

    # Training setup
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    is_cuda = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

    # Metric tracking
    test_metric = [10**9, 10**9, 10**9, -10**9, -10**9]
    train_loss_hist = []
    test_loss_hist = []
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=is_cuda):
                output = model(batch_x[:, :, :n_features])
                output = output[:, -pred_len:]
                target = batch_y[:, :, :n_features][:, -pred_len:]
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_hist.append(avg_train_loss)
        sched.step()

        # Test
        model.eval()
        sum_vector = np.zeros(num_metrics, dtype=np.float64)
        cnt = 0

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                output = model(batch_x[:, :, :n_features])

                y_true = batch_y[:, :, :n_features][:, -pred_len:].detach().cpu().numpy()
                y_pred = output[:, -pred_len:].detach().cpu().numpy()
                bs = y_true.shape[0]

                metric = np.array(metrics.metric(y_pred, y_true), dtype=np.float64)
                sum_vector += metric * bs
                cnt += bs

        epoch_metric = sum_vector / cnt
        test_loss_hist.append(epoch_metric[0])

        # Update best
        prev_best_mse = test_metric[0]
        test_metric = metrics.best(test_metric, epoch_metric)

        if test_metric[0] < prev_best_mse:
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:3d}/{args.num_epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Test MSE: {epoch_metric[0]:.4f} | "
                  f"Best: {test_metric[0]:.4f}")

    # Save results
    save_results(
        ablation_type, now, test_metric, data_name, pred_len, args.seq_len, best_epoch,
        train_loss_hist, test_loss_hist, variant, config, iteration, n_params,
        results_dir=args.results_dir
    )

    print(f"    -> Best MSE: {test_metric[0]:.4f}, MAE: {test_metric[1]:.4f}")

    return test_metric


# =============================================================================
# Main
# =============================================================================

def load_completed_experiments(results_path, param_name):
    """Load completed experiments"""
    if not os.path.exists(results_path):
        return set()

    df = pd.read_csv(results_path)
    completed = set()
    for _, row in df.iterrows():
        key = (row[param_name], row['data'], row['pred_len'], row['iteration'])
        completed.add(key)

    return completed


def run_ablation(args, ablation_type):
    """Run a single ablation type"""

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load best configs
    best_configs = load_best_configs(args.best_configs_path)
    print(f"Loaded {len(best_configs)} best configs from {args.best_configs_path}")

    ablation_cfg = ABLATION_CONFIG[ablation_type]
    param_name = ablation_cfg['param_name']
    results_file = ablation_cfg['results_file']

    # Dataset selection
    if args.data == 'all':
        d_list = list(DATASET_PRED_LENS.keys())
    else:
        d_list = [d.strip() for d in args.data.split(',')]

    # Variant selection
    if args.variant:
        variants = [args.variant]
    else:
        variants = ablation_cfg['variants']

    # Check completed experiments
    results_path = os.path.join(args.results_dir, results_file)
    completed = load_completed_experiments(results_path, param_name)

    # Calculate progress
    total_expected = 0
    total_remaining = 0

    for data_name in d_list:
        pred_lens = get_pred_lens(data_name)
        for pred_len in pred_lens:
            if args.pred_len is not None and pred_len != args.pred_len:
                continue

            if (data_name, pred_len) not in best_configs:
                continue
            config = best_configs[(data_name, pred_len)]
            for variant in variants:
                for iteration in range(args.iterations):
                    total_expected += 1
                    exp_key = (variant, data_name, pred_len, iteration)
                    if exp_key not in completed:
                        total_remaining += 1

    print(f"\n{'='*60}")
    print(f"{ablation_type.upper()} Ablation Study")
    print(f"{'='*60}")
    print(f"Variants: {variants}")
    print(f"Datasets: {d_list}")
    print(f"Pred_len: {args.pred_len if args.pred_len else 'all'}")
    print(f"Iterations: {args.iterations}")
    print(f"Expected: {total_expected}, Completed: {len(completed)}, Remaining: {total_remaining}")
    print(f"{'='*60}\n")

    if total_remaining == 0:
        print("All experiments completed!")
        return

    # Run experiments
    for data_name in d_list:

        print(f"\n{'='*60}")
        print(f"[Dataset] {data_name}")
        print(f"{'='*60}")

        pred_lens = get_pred_lens(data_name)

        for pred_len in pred_lens:

            if args.pred_len is not None and pred_len != args.pred_len:
                continue

            if (data_name, pred_len) not in best_configs:
                print(f"  pred_len={pred_len}: No best config found, skipping")
                continue

            config = best_configs[(data_name, pred_len)]

            # Period check
            if pred_len % config['period'] != 0:
                print(f"  pred_len={pred_len}: Not divisible by period={config['period']}, skipping")
                continue

            print(f"\n  pred_len={pred_len} | p={config['period']}, k={config['kernel_size']}, d={config['d_state']}")

            for variant in variants:

                variant_name = get_variant_name_for_type(ablation_type, variant)

                for iteration in range(args.iterations):

                    exp_key = (variant, data_name, pred_len, iteration)
                    if exp_key in completed:
                        print(f"    {variant_name} iter={iteration} - SKIPPED (done)")
                        continue

                    print(f"\n    [{variant_name}] iter={iteration+1}/{args.iterations}")

                    try:
                        best_metric = train_single(
                            args, ablation_type, variant, config, data_name, pred_len, device, iteration
                        )
                    except Exception as e:
                        print(f"    -> Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                    torch.cuda.empty_cache()
                    gc.collect()

    print(f"\n{'='*60}")
    print(f"{ablation_type.upper()} Ablation Complete!")
    print(f"Results: {args.results_dir}/{results_file}")
    print(f"{'='*60}")


def main(args):

    if args.ablation == 'all':
        ablation_types = ['multiscale', 'gating', 'dtk', 'head']
    else:
        ablation_types = [args.ablation]

    for ablation_type in ablation_types:
        run_ablation(args, ablation_type)


def parse_args():
    parser = argparse.ArgumentParser(description='FEATHer Unified Ablation Training')

    # Ablation type
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['multiscale', 'gating', 'dtk', 'head', 'all'],
                        help='Ablation type to run')

    # Data
    parser.add_argument('--root_path', type=str, default='data/', help='Root path of data')
    parser.add_argument('--data', type=str, default='ETTh1', help='Dataset name or "all"')
    parser.add_argument('--features', type=str, default='M', help='Features type')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=96, help='Label length')
    parser.add_argument('--pred_len', type=int, default=None,
                        help='Specific pred_len to run (e.g., 96). If None, run all.')

    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')

    # Device
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    # Paths
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--best_configs_path', type=str,
                        default='results/hparam_results/hparam_analysis/best_configs.csv',
                        help='Path to best_configs.csv')

    # Variant
    parser.add_argument('--variant', type=str, default=None,
                        help='Specific variant to run (depends on ablation type)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
