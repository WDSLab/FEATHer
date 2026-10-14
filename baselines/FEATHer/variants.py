"""FEATHer variants for the control arms (configs/experiments/control.yaml).

Both are built as the frozen model first and changed afterwards, so the shared modules get the frozen model's
initialisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.FEATHer.FEATHer import FEATHer


class FEATHerNoFront(FEATHer):
    """Control "FEATHer_frontnone": no band bank and no gate - the normalised input goes straight to the mixer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_point = self.conv_mid = self.freq_gate = None

    def _bands(self, x):
        return [x]

    def _gate(self, x):
        return None


class FEATHerMixSep(FEATHer):
    """Control "FEATHer_mixsep": the residual MLP runs per channel (1 -> R -> 1), so nothing crosses channels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.use_residual_mlp and self.channel_mode == "mix", "mix_sep replaces the shared residual MLP"
        R = self.nl_down.out_features
        D = self.nl_down.in_features
        del self.nl_down, self.nl_up
        self.sep_down_w = nn.Parameter(torch.empty(D, R).uniform_(-1.0, 1.0))   # nn.Linear init, fan_in 1
        self.sep_down_b = nn.Parameter(torch.empty(D, R).uniform_(-1.0, 1.0))
        b = float(R) ** -0.5                                                     # fan_in R
        self.sep_up_w = nn.Parameter(torch.empty(D, R).uniform_(-b, b))
        self.sep_up_b = nn.Parameter(torch.empty(D).uniform_(-b, b))

    def _mixer(self, h):
        u = F.gelu(h.unsqueeze(-1) * self.sep_down_w + self.sep_down_b)          # (B, L, D, R)
        return (u * self.sep_up_w).sum(-1) + self.sep_up_b
