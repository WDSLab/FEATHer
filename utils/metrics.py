"""Test metrics: MSE, MAE, RMSE, CORR, R2 (the columns of the results CSV).

pred, true: (B, T, F) numpy arrays.  NaN target cells (imputed cells of the gap-filled industrial sets) are left out,
so every metric is computed on observed values only."""
import numpy as np


def corr_mean(pred, true):
    # pred, true: (B, T, F)
    x, y = pred, true
    m = np.isfinite(x) & np.isfinite(y)
    cnt = m.sum(axis=1)                              # (B, F)
    valid = cnt >= 2

    # Masked mean.  out=zeros: np.divide with `where=` leaves unselected entries
    # uninitialized.  Zero out with np.where, not `x * m`: `nan * 0 == nan`, so one
    # masked target would make the whole (B,F) pair NaN and silently drop it from
    # the closing nanmean (a biased subset of windows on the gap-filled sets).
    xz = np.where(m, x, 0.0)
    yz = np.where(m, y, 0.0)
    x_mean = np.divide(xz.sum(axis=1), cnt,
                       out=np.zeros(cnt.shape, dtype=float), where=valid)[:, None, :]
    y_mean = np.divide(yz.sum(axis=1), cnt,
                       out=np.zeros(cnt.shape, dtype=float), where=valid)[:, None, :]

    xc = np.where(m, x - x_mean, 0.0)
    yc = np.where(m, y - y_mean, 0.0)

    # Standard deviation & covariance (sample, n-1)
    denom = np.maximum(cnt - 1, 1)
    sx = np.sqrt(np.divide((xc**2).sum(axis=1), denom))
    sy = np.sqrt(np.divide((yc**2).sum(axis=1), denom))
    cov = np.divide((xc*yc).sum(axis=1), denom)

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
    m = np.isfinite(x) & np.isfinite(y)
    cnt = m.sum(axis=1)                     # (B, F)
    valid = cnt >= 2

    # Masked y mean; out=zeros and np.where for the same reasons as in corr_mean.
    y_mean = np.divide(np.where(m, y, 0.0).sum(axis=1), np.maximum(cnt, 1),
                       out=np.zeros(cnt.shape, dtype=float), where=valid)[:, None, :]

    sse = np.where(m, (y - x) ** 2, 0.0).sum(axis=1)       # (B, F)
    sst = np.where(m, (y - y_mean) ** 2, 0.0).sum(axis=1)  # (B, F)

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


class TestScores:
    """Accumulates the test metrics over batches; result() -> [MSE, MAE, RMSE, CORR, R2].

    MSE / MAE are one mean over all observed cells of the test split, not a mean of per-batch means (batches of the
    gap-filled sets hold different numbers of observed cells).  CORR / R2 (per-window values) are batch-size-weighted
    means of the per-batch values.
    """

    def __init__(self):
        self.se = self.ae = 0.0
        self.n = 0
        self.cr_sum = np.zeros(2, dtype=np.float64)   # [CORR, R2]
        self.cr_cnt = np.zeros(2, dtype=np.float64)

    def update(self, pred, true):
        observed = np.isfinite(true)
        if not np.isfinite(pred[observed]).all():      # a NaN forecast must not be dropped like a NaN target
            raise RuntimeError("non-finite prediction on observed test cells - degenerate best-val model")
        diff = pred - true                             # NaN where the target was imputed
        fin = np.isfinite(diff)
        self.se += float(np.sum(diff[fin] ** 2))
        self.ae += float(np.sum(np.abs(diff[fin])))
        self.n += int(fin.sum())
        bs = true.shape[0]
        for i, val in enumerate((corr_mean(pred, true), r2_mean(pred, true))):
            if np.isfinite(val):
                self.cr_sum[i] += val * bs
                self.cr_cnt[i] += bs

    def result(self):
        if self.n == 0:                                # not a fake perfect score of 0
            raise RuntimeError("no observed test cells across all batches - degenerate run")
        mse, mae = self.se / self.n, self.ae / self.n
        return np.array([mse, mae, float(np.sqrt(mse)),
                         self.cr_sum[0] / max(self.cr_cnt[0], 1.0),
                         self.cr_sum[1] / max(self.cr_cnt[1], 1.0)], dtype=np.float64)
