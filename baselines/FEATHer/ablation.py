"""FEATHer ablation variants that swap one part (configs/experiments/ablation.yaml).

The ablations that only change a value (band count, mixing width, residual MLP off, RevIN off) are arguments of
FEATHer itself.  The classes here replace the gate or the head; each is the frozen model with that one part changed.
"""
import torch
import torch.nn as nn

from baselines.FEATHer.FEATHer import FEATHer


class FEATHerUniformGate(FEATHer):
    """gateuni: every band weighted 1/S (the spectrum gate is still built, then ignored)."""

    def _gate(self, x):
        return x.new_full((x.size(0), self.num_bands), 1.0 / self.num_bands)


class FEATHerNoGate(FEATHer):
    """gatenone: the bands are summed without weights; no gate is built."""

    def _make_gate(self, seq_len, num_bands):
        return None

    def _gate(self, x):
        return None


class StaticGate(nn.Module):
    """Input-independent learned band mixture softmax(theta), zero-initialised (uniform start); no FFT."""

    def __init__(self, num_bands: int):
        super().__init__()
        self.num_bands = num_bands
        self.band_logits = nn.Parameter(torch.zeros(num_bands))

    def forward(self, x):
        return torch.softmax(self.band_logits, dim=0).unsqueeze(0).expand(x.size(0), self.num_bands)


class FEATHerStaticGate(FEATHer):
    """gatestatic: the gate is a learned constant mixture instead of reading the spectrum."""

    def _make_gate(self, seq_len, num_bands):
        return StaticGate(num_bands)


class DenseLinearHead(nn.Module):
    """One Linear(L -> H) per channel (DLinear's head).  Not capacity-matched: ~L*H params vs the SPK head's few."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, h):
        return self.linear(h.permute(0, 2, 1)).permute(0, 2, 1)


class FEATHerLinearHead(FEATHer):
    """headlin: the SPK head is replaced by a dense linear head."""

    def _make_head(self, seq_len, pred_len, d, period, head_seg, agg_shared):
        return DenseLinearHead(seq_len, pred_len)
