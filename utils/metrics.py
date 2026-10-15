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


class TestScores:
    """Accumulates the test metrics over batches; result() -> [MSE, MAE, RMSE, CORR, R2].

    MSE / MAE are one mean over all observed cells of the test split, not a mean of per-batch means (batches of the
    gap-filled sets hold different numbers of observed cells).  R2 = 1 - MSE / Var(y), Var pooled over the same
    observed cells (the standard forecasting R2; = tools/paper/compute_r2.py R2_global).  A per-window R2 divides by
    the within-window variance, which collapses on near-constant channels and gave ~-1e10 on MetroPT3.  CORR
    (per-window values) is the batch-size-weighted mean of the per-batch values.
    """

    def __init__(self):
        self.se = self.ae = 0.0
        self.n = 0
        self.y_sum = self.y_sq = 0.0                  # observed targets, for Var(y)
        self.cr_sum = 0.0                             # CORR
        self.cr_cnt = 0.0

    def update(self, pred, true):
        observed = np.isfinite(true)
        if not np.isfinite(pred[observed]).all():      # a NaN forecast must not be dropped like a NaN target
            raise RuntimeError("non-finite prediction on observed test cells - degenerate best-val model")
        diff = pred - true                             # NaN where the target was imputed
        fin = np.isfinite(diff)
        self.se += float(np.sum(diff[fin] ** 2))
        self.ae += float(np.sum(np.abs(diff[fin])))
        self.n += int(fin.sum())
        y = true[fin].astype(np.float64)
        self.y_sum += float(y.sum())
        self.y_sq += float((y ** 2).sum())
        val = corr_mean(pred, true)
        if np.isfinite(val):
            self.cr_sum += val * true.shape[0]
            self.cr_cnt += true.shape[0]

    def result(self):
        if self.n == 0:                                # not a fake perfect score of 0
            raise RuntimeError("no observed test cells across all batches - degenerate run")
        mse, mae = self.se / self.n, self.ae / self.n
        mean = self.y_sum / self.n
        var = self.y_sq / self.n - mean * mean
        r2 = 1.0 - mse / var if var > 0 else np.nan
        return np.array([mse, mae, float(np.sqrt(mse)), self.cr_sum / max(self.cr_cnt, 1.0), r2], dtype=np.float64)
