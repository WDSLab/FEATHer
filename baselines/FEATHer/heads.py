"""Forecasting head of FEATHer: (B, L, D) -> (B, H, D).  The ablation's dense head is in ablation.py."""
import torch
import torch.nn as nn


class SparsePeriodKernel(nn.Module):
    """SparseTSF-style head: smoothing conv, then one linear map shared by every phase of a segment.

    The series (after a residual depthwise conv of width 2*(period//2)+1) is cut into segments of length w; the
    values at the same position of each segment (n of them) are mapped to the m output segments by one
    Linear(n, m), shared by all w positions and all channels.
    seg: segment length w (10-13 size hyperparameter); 0 = the period (frozen model), 1 = a dense L -> H map.
    agg_shared: one smoothing kernel for all channels (as in SparseTSF) instead of one per channel.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int, seg: int = 0,
                 agg_shared: bool = False):
        super().__init__()
        w = int(seg) or period
        assert seq_len % w == 0 and pred_len % w == 0, "the segment length must divide seq_len and pred_len"
        self.seq_len, self.pred_len, self.d_model = seq_len, pred_len, d_model
        self.period = w                               # segment length used by forward()
        self.n, self.m = seq_len // w, pred_len // w
        k = 2 * (period // 2) + 1                     # the smoothing width keeps following the period
        self.agg_shared = bool(agg_shared)
        ch = 1 if self.agg_shared else d_model
        self.agg_conv = nn.Conv1d(ch, ch, kernel_size=k, padding=k // 2, groups=ch, bias=False)
        self.backbone = nn.Linear(self.n, self.m, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        w, n, m = self.period, self.n, self.m
        h_t = h.permute(0, 2, 1)
        if self.agg_shared:
            h_agg = h_t + self.agg_conv(h_t.reshape(B * D, 1, L)).reshape(B, D, L)
        else:
            h_agg = h_t + self.agg_conv(h_t)
        x = h_agg.reshape(B * D, L).view(B * D, n, w).permute(0, 2, 1).reshape(-1, n)   # (B*D*w, n)
        y = self.backbone(x).view(B * D, w, m).permute(0, 2, 1).reshape(B * D, self.pred_len)
        return y.view(B, D, self.pred_len).permute(0, 2, 1)

