# -*- coding: utf-8 -*-
"""
FEATHer Training Script
========================

Basic training script for FEATHer model.
Uses models/base/FEATHer.py (clean version without ablation flags)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse
import gc
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base.FEATHer import FEATHer
from utils import data_factory
from utils import metrics
from utils import losses

warnings.filterwarnings("ignore")


# =============================================================================
# Save Functions
# =============================================================================

def save_plots(file_path, train_loss, test_loss, prediction, actual, 
               model_name, data_name, pred_len):
    """Save training curves and prediction plots"""
    
    # 1. Loss curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, test_loss, 'r-', label='Test MSE', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} - {data_name} (pred_len={pred_len})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_path + '_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction vs Actual (first sample)
    if prediction is not None and actual is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(actual[0, :, 0], 'b-', label='Actual', linewidth=2)
        ax.plot(prediction[0, :, 0], 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{model_name} - {data_name} (pred_len={pred_len})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(file_path + '_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()


def save_results(now, model, val_metric, test_metric, actual, prediction, 
                 model_name, data_name, pred_len, seq_len, epoch, 
                 train_loss, test_loss, save_model=False, save_plot=False):
    """Save experiment results"""
    
    # Path: results/{model_name}/{data}/{pred_len}/
    folder_path = f'results/{model_name}/{data_name}/{pred_len}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    timestamp = f'{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
    base_filename = f'{model_name}_{data_name}_{pred_len}_{seq_len}_{timestamp}'
    sub_folder = os.path.join(folder_path, base_filename)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    
    file_path = os.path.join(sub_folder, base_filename)
    
    # 1. Save numpy results
    results = np.array([val_metric, test_metric, actual, prediction, epoch, train_loss, test_loss], dtype=object)
    np.save(file_path + '.npy', results)
    
    # 2. Save training history CSV
    history_csv = {
        'epoch': list(range(len(train_loss))),
        'train_loss': train_loss,
        'test_MSE': test_loss,
    }
    pd.DataFrame(history_csv).to_csv(file_path + '_history.csv', index=False)
    
    # 3. Save final result CSV
    result_csv = {
        'model': [model_name],
        'data': [data_name],
        'pred_len': [pred_len],
        'seq_len': [seq_len],
        'MSE': [test_metric[0]],
        'MAE': [test_metric[1]],
        'RMSE': [test_metric[2]],
        'CORR': [test_metric[3]],
        'R2': [test_metric[4]],
        'best_epoch': [epoch],
    }
    pd.DataFrame(result_csv).to_csv(file_path + '_result.csv', index=False)
    
    # 4. Save model weights (optional)
    if save_model:
        torch.save(model.state_dict(), file_path + '.pth')
    
    # 5. Save plots (optional)
    if save_plot:
        save_plots(file_path, train_loss, test_loss, prediction, actual,
                   model_name, data_name, pred_len)
    
    # 6. Append to aggregated results CSV
    csv_path = 'results/all_results.csv'
    new_row = {
        'model': model_name,
        'data': data_name,
        'pred_len': pred_len,
        'seq_len': seq_len,
        'MSE': test_metric[0],
        'MAE': test_metric[1],
        'RMSE': test_metric[2],
        'CORR': test_metric[3],
        'R2': test_metric[4],
        'best_epoch': epoch,
        'timestamp': timestamp
    }
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        if not os.path.exists('results'):
            os.makedirs('results')
        df = pd.DataFrame([new_row])
    
    df.to_csv(csv_path, index=False)


# =============================================================================
# Training Function
# =============================================================================

def train(args):
    """Main training function"""
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prediction lengths by dataset type
    pred_lst = [96, 192, 336, 720]
    s_pred_lst = [24, 48, 96, 192]
    st_pred_lst = [12, 24, 48, 96]
    
    # Dataset list
    if args.data == 'all':
        d_list = ['AirQuality', 'Volatility', 'SML', 'PM', 'Weather', 'Exchange', 'ETTh1', 'ETTh2']
    else:
        d_list = [args.data]
    
    # Training loop
    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")
        
        for data_name in d_list:
            
            # Select prediction lengths based on dataset
            if data_name in ['SML', 'Volatility', 'astd']:
                pred_lens = s_pred_lst
            elif data_name in ['PEMS_BAY', 'METR', 'PEMS04', 'PEMS08', 'PEMS03']:
                pred_lens = st_pred_lst
            else:
                pred_lens = pred_lst
            
            # Override if specific pred_len is given
            if args.pred_len > 0:
                pred_lens = [args.pred_len]
            
            for pred_len in pred_lens:
                try:
                    train_single(args, data_name, pred_len, device)
                except Exception as e:
                    print(f"Error: {data_name} / {pred_len}")
                    print(f"{e}")
                    continue
                
                # Memory cleanup
                torch.cuda.empty_cache()
                gc.collect()


def train_single(args, data_name, pred_len, device):
    """Train single configuration"""
    
    now = datetime.now()
    model_name = 'FEATHer'
    num_metrics = 5  # MSE, MAE, RMSE, CORR, R2
    
    print(f"\n{'-'*60}")
    print(f"Data: {data_name} | Pred_len: {pred_len} | Seq_len: {args.seq_len}")
    print(f"{'-'*60}")
    
    # Load data
    df, freq, embed = data_factory.data_select(data_name, args.root_path)
    n_features = df.shape[1] - 1
    
    train_data, train_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, 'train',
        starting_percent=args.starting_percent, percent=args.percent
    )
    test_data, test_loader = data_factory.data_provider(
        args.root_path, data_name, args.features, args.batch_size,
        args.seq_len, args.label_len, pred_len, 'test',
        starting_percent=args.starting_percent, percent=args.percent
    )
    
    # Build model
    model = FEATHer(
        seq_len=args.seq_len,
        pred_len=pred_len,
        d_model=n_features,
        d_state=args.d_state,
        kernel_size=args.kernel_size,
        use_norm=True,
        period=args.period,
        num_bands=args.num_bands,
        use_topk_gate=args.use_topk_gate,
        topk=args.topk,
    ).float().to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {model_size:,} ({model_size / 1e6:.2f}M)")
    
    # Training setup
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    is_cuda = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)
    
    # Metrics initialization
    val_metric = [1e9, 1e9, 1e9, -1e9, -1e9]   # MSE, MAE, RMSE, COR, R2
    test_metric = [1e9, 1e9, 1e9, -1e9, -1e9]
    train_loss_hist = []
    test_loss_hist = []
    best_epoch = 0
    best_prediction = None
    best_actual = None
    
    # Training loop
    for epoch in range(args.num_epochs):
        
        # ===================== Train =====================
        model.train()
        total_train_loss = 0
        total_train_mse = 0
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=is_cuda):
                output, H = model(batch_x[:, :, :n_features], return_components=True)
                output = output[:, -pred_len:]
                target = batch_y[:, :, :n_features][:, -pred_len:]
                
                # Main prediction loss
                main_loss = criterion(output, target)
                
                # Spectral separation loss
                spec_loss = losses.spectral_separation_loss_scales(H)
                
                # Total loss
                loss = main_loss + args.lambda_spec * spec_loss
                
                # Train MSE (for logging)
                with torch.no_grad():
                    train_mse = F.mse_loss(output, target)
            
            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss detected, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            
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
        
        scheduler.step()
        
        # ===================== Test (batch-wise weighted average) =====================
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
        
        # Update best metrics (based on MSE)
        prev_best_mse = test_metric[0]
        test_metric = metrics.best(test_metric, epoch_metric)
        
        if epoch_metric[0] < prev_best_mse:
            best_epoch = epoch
            best_prediction = y_pred
            best_actual = y_true
        
        # Log progress (every epoch)
        print(f"  Epoch {epoch+1:3d}/{args.num_epochs} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train MSE: {avg_train_mse:.4f} | "
              f"Test MSE: {epoch_metric[0]:.4f} | "
              f"Best MSE: {test_metric[0]:.4f}")
    
    # Save results
    save_results(
        now, model, val_metric, test_metric, 
        best_actual, best_prediction,
        model_name, data_name, pred_len, args.seq_len, best_epoch,
        train_loss_hist, test_loss_hist, args.save_model, args.save_plot
    )
    
    # Print final results
    print(f"\n  -> Best MSE: {test_metric[0]:.4f}, MAE: {test_metric[1]:.4f}, "
          f"RMSE: {test_metric[2]:.4f}, CORR: {test_metric[3]:.4f}, R2: {test_metric[4]:.4f}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='FEATHer Training')
    
    # Data
    parser.add_argument('--root_path', type=str, default='data/', help='Root path of data')
    parser.add_argument('--data', type=str, default='ETTh1', help='Dataset name or "all"')
    parser.add_argument('--features', type=str, default='M', help='Features type: M, S, MS')
    
    # Model
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=0, help='Prediction length (0=auto)')
    parser.add_argument('--label_len', type=int, default=96, help='Label length')
    parser.add_argument('--d_state', type=int, default=8, help='State dimension')
    parser.add_argument('--kernel_size', type=int, default=7, help='Kernel size for DenseTemporalKernel')
    parser.add_argument('--period', type=int, default=12, help='Period for sparse head')
    parser.add_argument('--num_bands', type=int, default=3, help='Number of frequency bands (2, 3, or 4)')
    parser.add_argument('--use_topk_gate', action='store_true', help='Use top-k sparse gate')
    parser.add_argument('--topk', type=int, default=2, help='Top-k for sparse gate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lambda_spec', type=float, default=0.01, help='Spectral loss weight')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    
    # Data split
    parser.add_argument('--starting_percent', type=int, default=0, help='Starting percent')
    parser.add_argument('--percent', type=int, default=100, help='Data percent')
    
    # System
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_model', action='store_true', help='Save model weights')
    parser.add_argument('--save_plot', action='store_true', help='Save loss curve and prediction plots')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)