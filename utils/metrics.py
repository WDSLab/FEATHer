"""Test MSE / MAE over the observed cells: NaN target cells (imputed cells of the gap-filled industrial sets) are left
out.  One mean over all observed cells of the test split, not a mean of per-batch means."""
import numpy as np


class TestScores:
    """update(pred, true) per batch ((B, T, F) arrays); result() -> [MSE, MAE]."""

    def __init__(self):
        self.se = self.ae = 0.0
        self.n = 0

    def update(self, pred, true):
        observed = np.isfinite(true)
        if not np.isfinite(pred[observed]).all():      # a NaN forecast must not be dropped like a NaN target
            raise RuntimeError("non-finite prediction on observed test cells - degenerate best-val model")
        diff = pred[observed] - true[observed]
        self.se += float(np.sum(diff ** 2))
        self.ae += float(np.sum(np.abs(diff)))
        self.n += int(observed.sum())

    def result(self):
        if self.n == 0:                                # not a fake perfect score of 0
            raise RuntimeError("no observed test cells across all batches - degenerate run")
        return np.array([self.se / self.n, self.ae / self.n], dtype=np.float64)
