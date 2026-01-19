import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# metrics
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr_mean(pred, true):
    # pred, true: (B, T, F)
    x, y = pred, true
    m = np.isfinite(x) & np.isfinite(y)              # Valid value mask
    cnt = m.sum(axis=1)                              # (B, F)
    valid = cnt >= 2

    # Masked mean
    x_mean = np.divide((x*m).sum(axis=1), cnt, where=valid)[:, None, :]
    y_mean = np.divide((y*m).sum(axis=1), cnt, where=valid)[:, None, :]

    # Centering
    xc = np.where(m, x - x_mean, 0.0)
    yc = np.where(m, y - y_mean, 0.0)

    # Standard deviation & covariance (sample, n-1)
    denom = np.maximum(cnt - 1, 1)
    sx = np.sqrt(np.divide((xc**2).sum(axis=1), denom))
    sy = np.sqrt(np.divide((yc**2).sum(axis=1), denom))
    cov = np.divide((xc*yc).sum(axis=1), denom)

    # Correlation coefficient
    denom_r = sx * sy
    ok = valid & (denom_r > 0)
    r = np.full_like(cov, np.nan, dtype=float)
    r[ok] = cov[ok] / denom_r[ok]

    return np.nanmean(r)   # Average all (B,F) pairs with equal weight


def r2_mean(pred, true):
    """
    pred, true: shape (B, T, F)
    - NaN/inf values are masked
    - All (B,F) pairs are averaged with equal weight
    - sklearn rules applied when SST==0:
        * SSE==0 -> R2 = 1.0
        * SSE>0  -> R2 = 0.0
    """
    x, y = pred, true
    m = np.isfinite(x) & np.isfinite(y)     # Valid value mask
    cnt = m.sum(axis=1)                     # (B, F)
    valid = cnt >= 2

    # y mean (masked)
    # Use minimum of 1 for denominator to avoid cnt=0, calculate only valid locations with where=valid
    y_mean = np.divide((y * m).sum(axis=1), np.maximum(cnt, 1), where=valid)[:, None, :]

    # SSE, SST (masked)
    sse = ((y - x) ** 2 * m).sum(axis=1)           # (B, F)
    sst = ((y - y_mean) ** 2 * m).sum(axis=1)      # (B, F)

    r2 = np.full_like(sse, np.nan, dtype=float)

    # Standard case: SST > 0
    mask_pos = valid & (sst > 0)
    r2[mask_pos] = 1.0 - sse[mask_pos] / sst[mask_pos]

    # Constant target: SST == 0 -> sklearn rule
    mask_zero = valid & (sst == 0)
    if np.any(mask_zero):
        r2[mask_zero] = np.where(sse[mask_zero] == 0.0, 1.0, 0.0)

    # Others (invalid) remain as NaN -> aggregated with nanmean
    return np.nanmean(r2)

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true)) * 100

def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true)) * 100


def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    corr = corr_mean(pred, true)
    r2 = r2_mean(pred, true)

    return mse, mae, rmse, corr, r2


# metric update
def best(best_metric, metric):
    # MSE, MAE, RMSE, COR, R2
    if best_metric[0] > metric[0]: best_metric[0] = metric[0]
    if best_metric[1] > metric[1]: best_metric[1] = metric[1]
    if best_metric[2] > metric[2]: best_metric[2] = metric[2]
    if best_metric[3] < metric[3]: best_metric[3] = metric[3]
    if best_metric[4] < metric[4]: best_metric[4] = metric[4]
    
    return best_metric

# prediction plot
def plot(pred, true, data):
    idx = [0, pred.shape[0] * 1 // 4, pred.shape[0] * 1 // 2, pred.shape[0] * 3 // 4]
    labels = ['Batch 0%', 'Batch 25%', 'Batch 50%', 'Batch 75%']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.suptitle(f"{data}", fontsize=18, fontweight='bold', y=1.02)

    for ax, i, label in zip(axs.flat, idx, labels):
        ax.plot(pred[i, :, -1], label='Prediction', color='crimson', linewidth=2)
        ax.plot(true[i, :, -1], label='Actual', color='navy', linestyle='--', linewidth=2)
        ax.set_title(label, fontsize=14, fontweight='semibold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=10)
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_concat_pred_grid(pred, true, model_name, data, seq_len, pred_len, total_len=864, var_idx=-1, save=False, now=None, change=False, epoch=10):
    """
    pred, true : np.ndarray, shape (B, S, V)
      - B: batch (window start moves 1 step back, no shuffle)
      - S: sequence length (usually same as pred_len)
      - V: variables

    pred_len  : Prediction length to use per window (equal to or less than S)
    total_len : Total length to concatenate and display in each subplot
    var_idx   : Variable index to visualize (default: last variable)
    """

    B, S, V = pred.shape
    assert (0 <= var_idx < V) or (var_idx == -1), "var_idx is out of variable range."
    if var_idx == -1:
        var_idx = V - 1

    use_len = min(pred_len, S)     # Actual length to use from each segment
    step_batches = pred_len        # Batch moves 1 step -> jumping by pred_len gives continuous segments
    needed_chunks = int(np.ceil(total_len / use_len))

    # To create needed_chunks segments starting from this start_batch,
    # the maximum allowed start batch must not exceed the value below
    max_start_batch = max(0, B - (needed_chunks - 1) * step_batches - 1)

    # Start batch candidates at 0%, 25%, 50%, 75% points (clamped to safe range)
    raw_starts = [0.0, 0.25, 0.5, 0.75]
    start_batches = [int(round(p * max_start_batch)) for p in raw_starts]

    # Figure setup (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{data}\n"
                 f"(var_idx={var_idx}, pred_len={pred_len}, total_len={total_len})",
                 fontsize=16, fontweight='bold', y=0.98)

    for ax, start_b, pct in zip(axes, start_batches, [0, 25, 50, 75]):
        # Collect segments to concatenate into one continuous sequence
        seqs_p, seqs_t = [], []
        b = start_b
        for _ in range(needed_chunks):
            if b >= B:
                break
            seqs_p.append(pred[b, :use_len, var_idx])
            seqs_t.append(true[b, :use_len, var_idx])
            b += step_batches

        if len(seqs_p) == 0:
            ax.text(0.5, 0.5, "No data to display due to insufficient data.",
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f"Batch {pct}% (start={start_b}) — empty", fontsize=12, pad=8)
            ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
            continue

        concat_pred = np.concatenate(seqs_p, axis=0)[:total_len]
        concat_true = np.concatenate(seqs_t, axis=0)[:total_len]

        # Visualization (line width/transparency/grid/spine cleanup)
        ax.plot(concat_true, label='Actual', linestyle='--', linewidth=1.8, alpha=0.9)
        ax.plot(concat_pred, label='Prediction', linewidth=2.2, alpha=0.95)

        # Clean style
        ax.set_title(f"Batch {pct}% (start={start_b})", fontsize=13, pad=8, fontweight='semibold')
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
        ax.tick_params(axis='both', labelsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        # Simple auxiliary info: Display RMSE/NRMSE (based on max actual value) (optional)
        rmse = float(np.sqrt(np.mean((concat_true - concat_pred) ** 2)))
        ymax = float(np.max(np.abs(concat_true))) if np.max(np.abs(concat_true)) != 0 else 1.0
        nrmse = (rmse / ymax) * 100.0
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9,
                  title=f"RMSE={rmse:.4f} | NRMSE={nrmse:.2f}%")

    axes[-1].set_xlabel('Time step', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save and change:
        plt.savefig(f'models/{model_name}/results/{data}/{pred_len}/{model_name}_{data}_{pred_len}_{seq_len}_{now.month}{now.day}{now.hour}{now.minute}/viz.png', bbox_inches='tight')
    if epoch%10 == 0:
        plt.show()
    plt.close(fig)

def save(now, model, val_metric, test_metric, actual, prediction, model_name, data, pred_len, seq_len, epoch, train_loss, test_loss, save=False):
    """Save training results and optionally model weights."""
    results = np.array([val_metric, test_metric, None, None, epoch, train_loss, test_loss], dtype=object)
    
    folder_path = f'models/{model_name}/results/{data}/{pred_len}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    base_filename = f'{model_name}_{data}_{pred_len}_{seq_len}_{now.month}{now.day}{now.hour}{now.minute}'
    folder_path = os.path.join(folder_path, base_filename)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    
    
    file_path = os.path.join(folder_path, base_filename)
    np.save(file_path + '.npy', results)
    if save:
        torch.save(model.state_dict(), file_path + '.pth')
    

