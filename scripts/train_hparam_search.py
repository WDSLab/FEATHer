# -*- coding: utf-8 -*-
"""
FEATHer Hyperparameter Search (Grid / Random)
==============================================

Hyperparameter search for the base model.
Uses models/base/FEATHer.py.

Search parameters:
    - period: [6, 12, 24, 48]
    - kernel_size: [6, 8, 10]
    - d_state: [5, 10, 15]
    - num_bands: [2, 3, 4]

Total combinations: 4 x 3 x 3 x 3 = 108
Random search: 50 samples (default)

Execution order: Dataset -> Config -> pred_len (complete one dataset before moving to the next)

Usage:
    # Grid search (all 108 configs)
    python scripts/train_hparam_search.py --data all

    # Random search (50 samples)
    python scripts/train_hparam_search.py --data all --random --num_samples 50

    # Random search with range specification
    python scripts/train_hparam_search.py --data all --random --num_samples 50 --config_start 0 --config_end 10
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import argparse
import gc
import random
from datetime import datetime
from itertools import product

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base.FEATHer import FEATHer
from utils import data_factory
from utils import metrics
from utils import losses

warnings.filterwarnings("ignore")


# =============================================================================
# Hyperparameter Search Space
# =============================================================================

HPARAM_SPACE = {
    'period': [6, 12, 24, 48],
    'kernel_size': [6, 8, 10],
    'd_state': [5, 10, 15],
    'num_bands': [2, 3, 4],
}

# =============================================================================
# Dataset Configuration
# =============================================================================

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

ALL_DATASETS = list(DATASET_PRED_LENS.keys())


# =============================================================================
# Helper Functions
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pred_lens(data_name):
    pred_type = DATASET_PRED_LENS.get(data_name, 'standard')
    return PRED_LENS[pred_type]


def generate_hparam_configs(use_random=False, num_samples=50, seed=42):
    """Generate hyperparameter configurations"""
    configs = []
    for period, kernel_size, d_state, num_bands in product(
        HPARAM_SPACE['period'],
        HPARAM_SPACE['kernel_size'],
        HPARAM_SPACE['d_state'],
        HPARAM_SPACE['num_bands']
    ):
        config_name = f'p{period}_k{kernel_size}_d{d_state}_b{num_bands}'
        config = {
            'period': period,
            'kernel_size': kernel_size,
            'd_state': d_state,
            'num_bands': num_bands,
        }
        configs.append((config_name, config))
    
    total = len(configs)  # 4 x 3 x 3 x 3 = 108 configs
    
    if use_random and num_samples < total:
        random.seed(seed)
        configs = random.sample(configs, num_samples)
        print(f"[Random Search] Sampled {num_samples} configs from {total} (seed={seed})")
    else:
        print(f"[Grid Search] Using all {total} configs")
    
    return configs


# =============================================================================
# Save Functions
# =============================================================================

def save_plots(file_path, train_loss_hist, test_loss_hist, prediction, actual,
               config_name, data_name, pred_len):
    import matplotlib.pyplot as plt
    
    # Loss curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_loss_hist) + 1)
    ax.plot(epochs, train_loss_hist, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, test_loss_hist, 'r-', label='Test MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{config_name} - {data_name} (pred_len={pred_len})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path + '_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Prediction vs Actual
    if prediction is not None and actual is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(actual[0, :, 0], 'b-', label='Actual', linewidth=2)
        ax.plot(prediction[0, :, 0], 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{config_name} - {data_name} (pred_len={pred_len})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_path + '_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()


def save_results(now, test_metric, data_name, pred_len, seq_len, best_epoch,
                 train_loss_hist, test_loss_hist, config_name, config, 
                 iteration, n_params, results_dir='results/hparam_resutls', prediction=None, actual=None):
    
    folder_path = f'{results_dir}/hparam_search/{data_name}/{pred_len}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    timestamp = f'{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    base_filename = f'{config_name}_{data_name}_{pred_len}_iter{iteration}_{timestamp}'
    file_path = os.path.join(folder_path, base_filename)
    
    # test_metric: [MSE, MAE, RMSE, CORR, R2]
    result_csv = {
        'config': config_name,
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
        'num_bands': config['num_bands'],
        'timestamp': timestamp,
    }
    
    pd.DataFrame([result_csv]).to_csv(file_path + '_result.csv', index=False)
    
    save_plots(file_path, train_loss_hist, test_loss_hist, prediction, actual,
               config_name, data_name, pred_len)
    
    csv_path = f'{results_dir}/hparam_search_results.csv'
    
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

def train_single(args, config_name, config, data_name, pred_len, device, iteration):
    
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
    model = FEATHer(
        seq_len=args.seq_len,
        pred_len=pred_len,
        d_model=n_features,
        d_state=config['d_state'],
        kernel_size=config['kernel_size'],
        period=config['period'],
        use_norm=True,
        num_bands=config['num_bands'],
        use_topk_gate=False,  # fixed
        topk=3,  # fixed
    ).float().to(device)
    
    n_params = count_parameters(model)
    
    # Training setup
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    is_cuda = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)
    
    # Metric tracking
    test_metric = [10**9, 10**9, 10**9, -10**9, -10**9]  # MSE, MAE, RMSE, CORR, R2
    train_loss_hist = []
    test_loss_hist = []
    best_epoch = 0
    best_prediction = None
    best_actual = None
    
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        total_train_mse = 0
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=is_cuda):
                output = model(batch_x[:, :, :n_features])
                output = output[:, -pred_len:]
                target = batch_y[:, :, :n_features][:, -pred_len:]
                
                loss = criterion(output, target)
                
                with torch.no_grad():
                    train_mse = F.mse_loss(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            total_train_mse += train_mse.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mse = total_train_mse / len(train_loader)
        
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
        test_loss_hist.append(epoch_metric[0])  # MSE
        
        # Update best (based on MSE)
        prev_best_mse = test_metric[0]
        test_metric = metrics.best(test_metric, epoch_metric)
        
        if test_metric[0] < prev_best_mse:
            best_epoch = epoch
            best_prediction = y_pred
            best_actual = y_true
        
        print(f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
            f"lr: {sched.get_last_lr()[0]:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train MSE: {avg_train_mse:.4f} | "
            f"Test MSE: {epoch_metric[0]:.4f} | "
            f"Best MSE: {test_metric[0]:.4f}")

    # Save results
    save_results(
        now, test_metric, data_name, pred_len, args.seq_len, best_epoch,
        train_loss_hist, test_loss_hist, config_name, config, iteration, n_params,
        results_dir=args.results_dir, prediction=best_prediction, actual=best_actual
    )
    
    print(f"  -> Best MSE: {test_metric[0]:.4f}, MAE: {test_metric[1]:.4f}, "
          f"RMSE: {test_metric[2]:.4f}, CORR: {test_metric[3]:.4f}, R2: {test_metric[4]:.4f}")
    
    return test_metric


# =============================================================================
# Main
# =============================================================================

def load_completed_experiments(results_path):
    if not os.path.exists(results_path):
        return set()
    
    df = pd.read_csv(results_path)
    completed = set()
    for _, row in df.iterrows():
        key = (row['config'], row['data'], row['pred_len'], row['iteration'])
        completed.add(key)
    
    return completed


def main(args):
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.data == 'all':
        d_list = ALL_DATASETS
    else:
        # Support for comma-separated multiple datasets
        d_list = [d.strip() for d in args.data.split(',')]
    
    # Generate configs (random or grid)
    all_configs = generate_hparam_configs(
        use_random=args.random,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Config range specification
    if args.config_start is not None and args.config_end is not None:
        all_configs = all_configs[args.config_start:args.config_end]
        print(f"Running configs {args.config_start} to {args.config_end}")
    
    total_configs = len(all_configs)
    
    results_path = os.path.join(args.results_dir, 'hparam_search_results.csv')
    completed = load_completed_experiments(results_path)
    
    remaining_configs = set()
    total_expected = 0
    total_remaining = 0
    
    for config_name, config in all_configs:
        period = config['period']
        for data_name in d_list:
            pred_lens = get_pred_lens(data_name)
            valid_pred_lens = [p for p in pred_lens if p % period == 0]
            for pred_len in valid_pred_lens:
                for iteration in range(args.iterations):
                    total_expected += 1
                    exp_key = (config_name, data_name, pred_len, iteration)
                    if exp_key not in completed:
                        total_remaining += 1
                        remaining_configs.add(config_name)
    
    completed_configs = set(c[0] for c in all_configs) - remaining_configs
    
    search_type = "Random Search" if args.random else "Grid Search"
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter {search_type}")
    print(f"{'='*60}")
    print(f"Total configs: {total_configs}")
    print(f"Datasets: {d_list}")
    print(f"Iterations: {args.iterations}")
    if args.random:
        print(f"Random seed: {args.seed}")
    print(f"{'='*60}")
    print(f"\n[Progress Check]")
    print(f"  Expected experiments: {total_expected}")
    print(f"  Completed experiments: {len(completed)}")
    print(f"  Remaining experiments: {total_remaining}")
    print(f"  Progress: {len(completed)/total_expected*100:.1f}%" if total_expected > 0 else "  Progress: 0%")
    print(f"{'='*60}\n")
    
    if total_remaining == 0:
        print("All experiments completed!")
        return
    
    # =========================================================================
    # Execution order: Dataset -> Config -> pred_len
    # (Complete all configs for one dataset before moving to the next)
    # =========================================================================
    
    for data_idx, data_name in enumerate(d_list):
        
        print(f"\n{'='*60}")
        print(f"[Dataset {data_idx + 1}/{len(d_list)}] {data_name}")
        print(f"{'='*60}")
        
        for config_idx, (config_name, config) in enumerate(all_configs):
            
            print(f"\n  [Config {config_idx + 1}/{total_configs}] {config_name}")
            print(f"    period={config['period']}, kernel_size={config['kernel_size']}, "
                  f"d_state={config['d_state']}, num_bands={config['num_bands']}")
            
            pred_lens = get_pred_lens(data_name)
            
            if args.pred_len > 0:
                pred_lens = [args.pred_len]
            
            period = config['period']
            valid_pred_lens = [p for p in pred_lens if p % period == 0]
            
            if len(valid_pred_lens) < len(pred_lens):
                skipped = [p for p in pred_lens if p % period != 0]
                print(f"    Skipping pred_lens {skipped} (not divisible by period={period})")
            
            for pred_len in valid_pred_lens:
                
                for iteration in range(args.iterations):
                    
                    exp_key = (config_name, data_name, pred_len, iteration)
                    if exp_key in completed:
                        print(f"    pred_len={pred_len} iter={iteration} - SKIPPED (already done)")
                        continue
                    
                    print(f"\n    pred_len={pred_len} iter={iteration+1}/{args.iterations}")
                    
                    try:
                        best_metric = train_single(
                            args, config_name, config, data_name, pred_len, device, iteration
                        )
                        
                    except Exception as e:
                        print(f"    -> Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(f"\n{'='*60}")
        print(f"[Dataset {data_name}] Complete!")
        print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"{search_type} Complete!")
    print(f"Results saved to: {args.results_dir}/hparam_results/hparam_search_results.csv")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='FEATHer Hyperparameter Search')
    
    parser.add_argument('--root_path', type=str, default='data/', help='Root path of data')
    parser.add_argument('--data', type=str, default='ETTh1', help='Dataset name or "all"')
    parser.add_argument('--features', type=str, default='M', help='Features type')
    
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=0, help='Prediction length (0=auto)')
    parser.add_argument('--label_len', type=int, default=96, help='Label length')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--results_dir', type=str, default='results/hparam_results', help='Results directory')
    
    # Config range
    parser.add_argument('--config_start', type=int, default=None, help='Start config index')
    parser.add_argument('--config_end', type=int, default=None, help='End config index')
    
    # Random search options
    parser.add_argument('--random', action='store_true', help='Use random search (default: grid search)')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of random samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)