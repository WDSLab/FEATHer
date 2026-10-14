"""Band gate of FEATHer: how much of each frequency band goes into the sum."""
import torch
import torch.nn as nn


class FFTFrequencyGate(nn.Module):
    """Band weights from the input spectrum.

    The channel-mean power spectrum is cut into num_bands equal-width slices; the log energy share of each slice goes
    through Linear(S, S) and a softmax.  Zero-initialised, so training starts at the uniform mixture 1/S.  Slice order
    follows utils/losses.py: slices run low -> high and band s (0 = POINT, the fastest) owns slice S-1-s.
    """

    def __init__(self, seq_len: int, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands
        n_freq = seq_len // 2 + 1
        step = n_freq // num_bands
        self._bounds = [(s * step, n_freq if s == num_bands - 1 else (s + 1) * step) for s in range(num_bands)]
        self.proj = nn.Linear(num_bands, num_bands)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, num_bands) softmax weights."""
        mag = torch.abs(torch.fft.rfft(x.float(), dim=1))
        P = (mag ** 2).mean(dim=2)                                                     # (B, Lf)
        total = P.sum(dim=1, keepdim=True) + 1e-6
        ratios = torch.stack([P[:, lo:hi].sum(dim=1) for (lo, hi) in self._bounds], dim=1) / total
        feats = torch.log(ratios.flip(dims=[1]) + 1e-6)                                # band order: POINT first
        feats = feats - feats.mean(dim=1, keepdim=True)                                # centred (conditioning only)
        return torch.softmax(self.proj(feats), dim=-1)
