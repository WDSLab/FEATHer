# -*- coding: utf-8 -*-
"""
FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster

Components:
- FFT-based Frequency Gate (FFTFrequencyGate)
- Multi-band decomposition via 1D Conv (POINT / HIGH / MID / LOW)
- DenseTemporalKernel (depthwise temporal mixing + in/out projection)
- SparsePeriodKernel (period-aware sparse forecasting head)

Band configurations (num_bands):
    2: [POINT, LOW]
    3: [POINT, MID, LOW]
    4: [POINT, HIGH, MID, LOW]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_topk_gate(w: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Top-k sparsified gate (optional).

    Args:
        w: (B, C) gate weights
        k: number of active bands

    Returns:
        (B, C) sparse gate (only top-k non-zero, renormalized)
    """
    B, C = w.shape
    if k >= C:
        return w

    topk_vals, topk_idx = torch.topk(w, k=k, dim=-1)
    mask = torch.zeros_like(w)
    mask.scatter_(1, topk_idx, 1.0)

    w_sparse = w * mask
    denom = w_sparse.sum(dim=-1, keepdim=True) + 1e-8
    w_sparse = w_sparse / denom
    return w_sparse


class FFTFrequencyGate(nn.Module):
    """
    FFT-based frequency gate for adaptive band weighting.

    Two modes:

    "global" (default, canonical ??byte-identical to every frozen checkpoint)
        magnitude spectrum -> Conv1d -> AdaptiveAvgPool1d(1) -> softmax.

        ??MEASURED DEFECT (2026-08-14): this gate is provably BLIND to
        spectral SHAPE. Averaging a convolution over the whole frequency
        axis collapses to

            mean_i( sum_k w_k * mag[i+k] )  ~=  (sum_k w_k) * mean(mag)

        so every band logit is an affine function of ONE scalar (the mean
        spectral magnitude); the band index only changes a multiplier and a
        bias, and frequency POSITION is discarded. Since the input is
        instance-normalized, Parseval pins mean(mag) to near-constant, so
        the weights barely move at all. Probe: two inputs with identical
        energy but opposite spectra (2-cycle vs 40-cycle sinusoid) get
        gate weights equal to 4 decimals; across 512 real ETTh1 windows the
        per-band weight std is 0.0003-0.0115. The only shape sensitivity
        left is the zero-padding edge effect on the first/last few bins.
        This explains three separately-observed flat results: gate on/off
        makes no difference, num_bands is flat, and lambda_spec is flat ??
        the "adaptive" gate was already a fixed weight vector.

    "banded" (fix)
        Slice the power spectrum into num_bands equal-width contiguous
        frequency ranges, take each range's ENERGY RATIO (scale-invariant),
        and map the log-ratios to logits with a small linear layer. The gate
        then measures actual spectral composition and responds to shape.
        Band-to-slice assignment mirrors utils/losses.py exactly: slices run
        low->high and band s (0 = POINT, highest-frequency) owns slice
        S-1-s, so the gate, the decomposition, and the spectral-separation
        regularizer all use one convention. Params: S^2 + S (20 for S=4)
        vs the conv's 5S + S (24) ??slightly FEWER.

    Args:
        seq_len: Input sequence length
        num_bands: Number of frequency bands (2, 3, or 4)
        kernel_size: Kernel size for Conv1d ("global" mode only)
        gate_mode: "global" (canonical) or "banded" (shape-aware fix)
    """

    def __init__(self, seq_len: int, num_bands: int = 3, kernel_size: int = 5,
                 gate_mode: str = "global", gate_zero_init: bool = False,
                 gate_bounds: str = "linear"):
        super().__init__()
        assert gate_mode in ("global", "banded", "banded_sensor", "static", "static_rng"), gate_mode
        self.seq_len = seq_len
        self.num_bands = num_bands
        self.gate_mode = gate_mode

        if gate_mode in ("static", "static_rng"):
            # Input-INDEPENDENT learned band mixture: w = softmax(theta).
            # Motivated by measurement, not simplification: a TRAINED "global"
            # gate on ETTh1 returns [0.1496, 0.51055, 0.33985] for 2-, 20- and
            # 44-cycle inputs of equal energy ??identical to 5 decimals ??and
            # moves by only +-0.02 across 3,584 real test windows, while
            # sitting far from uniform (POINT band 0.012 vs 0.333). So the
            # gate's contribution is the LEARNED MIXTURE, not adaptivity; the
            # rfft that produces it is dead weight at inference. This arm keeps
            # the mixture and drops the FFT (S params instead of 6S, and no
            # transform / twiddle table on the MCU). Zero-init = uniform start.
            if gate_mode == "static_rng":
                # Same fixed mixture, but draw (and discard) the Linear(S, S)
                # that "banded" builds, so every module constructed after the
                # gate (the SPK head) gets the banded model's initialisation:
                # a difference in results is then the gate, not the init.
                nn.Linear(num_bands, num_bands)
            self.band_logits = nn.Parameter(torch.zeros(num_bands))
        elif gate_mode in ("banded", "banded_sensor"):
            # Precompute the equal-width rfft slices (low -> high), same rule
            # as utils/losses.spectral_separation_loss_scales.
            self._bounds_ps = None
            n_freq = seq_len // 2 + 1
            step = n_freq // num_bands
            self._bounds = [
                (s * step, n_freq if s == num_bands - 1 else (s + 1) * step)
                for s in range(num_bands)
            ]
            if gate_bounds == "octave":
                # (2026-09-15) Octave slices instead of equal width. With L=96 and
                # S=3 the edges are 13 and 25: the LOW slice (bins 0-12) is what the
                # AvgPool4 LOW branch can represent (24 samples -> Nyquist 12), and
                # the rest doubles (13-24, 25-48). Equal width put 80-87% of the
                # energy in the lowest slice, leaving the gate little to read.
                half = seq_len // 2
                edges = ([0] + [(half >> (num_bands - k)) + 1 for k in range(1, num_bands)]
                         + [n_freq])
                self._bounds = [(edges[s], edges[s + 1]) for s in range(num_bands)]
            elif ";" in str(gate_bounds):
                # Per-sensor edges "e1,e2;e1,e2;..." (one group per sensor), from
                # gate_bounds="tercile_sensor" resolved on the TRAIN split. Sensors
                # differ a lot in where their energy sits (Steel bin 1 holds 74% for
                # one sensor and 24% for another), so one edge pair per dataset gives
                # some sensors a nearly empty slice.
                self._bounds_ps = []
                for g in str(gate_bounds).split(";"):
                    e = [0] + [int(v) for v in g.split(",")] + [n_freq]
                    assert len(e) == num_bands + 1 and all(
                        a < b for a, b in zip(e, e[1:])), g
                    self._bounds_ps.append([(e[s], e[s + 1]) for s in range(num_bands)])
                self._bounds = self._bounds_ps[0]
            elif gate_bounds != "linear":
                # Explicit inner edges "e1,e2". gate_bounds="tercile" is turned into
                # such edges from the TRAIN split by the worker (resolve_gate_bounds).
                try:
                    inner = [int(v) for v in str(gate_bounds).split(",")]
                except ValueError:
                    raise ValueError(
                        "gate_bounds %r: use linear, octave or explicit edges 'e1,e2' "
                        "('tercile' has to be resolved by the worker first)" % (gate_bounds,))
                edges = [0] + inner + [n_freq]
                assert len(edges) == num_bands + 1 and all(
                    a < b for a, b in zip(edges, edges[1:])), gate_bounds
                self._bounds = [(edges[s], edges[s + 1]) for s in range(num_bands)]
            self.proj = nn.Linear(num_bands, num_bands)
            # Zero-init => the gate starts EXACTLY uniform (1/S per band),
            # i.e. at the fixed-weight behaviour the "global" gate is stuck
            # at, and learns to deviate from there. Two reasons: (a) random
            # init saturates the softmax immediately (log-ratios of a nearly
            # empty band reach log(1e-6) ~ -14, giving one-hot gates before
            # any training), (b) it makes the comparison controlled ??any
            # gain over gate_uniform is attributable to LEARNED adaptivity.
            # A zero-init linear still trains: dL/dW = dL/dz * feats^T != 0.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            self.conv = nn.Conv1d(
                in_channels=1,
                out_channels=num_bands,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            # MEASURED DEFECT (2026-08-27): with the default random init the
            # logits are proportional to the (large) mean spectral magnitude, so
            # the softmax is SATURATED before a single step is taken. On EWELD
            # the initial band mixture is e.g. [0.0002, 0.9954, 0.0002, 0.0042]
            # — max/min 5,077 — and a band at 2e-4 receives almost no gradient,
            # i.e. it starts as a dead branch. Across seeds the ratio tracks the
            # outcome exactly: the two seeds that fail to train on EWELD start
            # at 5,077x and 1,795x, the one that succeeds at 17x, and the failing
            # runs end up worse on TRAIN loss too (0.293 vs 0.278) — an
            # optimisation failure fixed at initialisation, not overfitting.
            # zero_init makes every band start at exactly 1/S (the same uniform
            # start gate_mode="banded" gets), which is what raised the EWELD
            # success rate from 33% to 83%. A zero-init conv still trains
            # (dL/dW = dL/dz * feats^T != 0). OFF by default so canonical stays
            # byte-identical.
            if gate_zero_init:
                nn.init.zeros_(self.conv.weight)
                nn.init.zeros_(self.conv.bias)

    def _slice_mask(self, P):
        """(S, Lf, D) 0/1 mask of the per-sensor slices, cached per device."""
        D = P.shape[2]
        m = getattr(self, "_mask", None)
        if (m is None or m.shape[2] != D or m.device != P.device
                or m.dtype != P.dtype):
            assert len(self._bounds_ps) == D, (
                "gate_bounds carries %d sensors, input has %d"
                % (len(self._bounds_ps), D))
            m = P.new_zeros(self.num_bands, P.shape[1], D)
            for d, bounds in enumerate(self._bounds_ps):
                for s, (lo, hi) in enumerate(bounds):
                    m[s, lo:hi, d] = 1.0
            self._mask = m
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor

        Returns:
            (B, num_bands) softmax-normalized gate weights
        """
        B, L, D = x.shape

        if self.gate_mode in ("static", "static_rng"):
            # No transform at all ??the mixture does not depend on x.
            w = torch.softmax(self.band_logits, dim=0)
            return w.unsqueeze(0).expand(B, self.num_bands)

        # Convert to float32 to avoid cuFFT errors in autocast(fp16)
        x32 = x.float()
        Freq = torch.fft.rfft(x32, dim=1)
        mag = torch.abs(Freq)

        if self.gate_mode == "banded_sensor":
            # Per-sensor variant (2026-09-15): the same energy-ratio gate, but
            # every sensor reads its OWN spectrum instead of the sensor mean, so
            # the gate is (B, D, S). proj is shared across sensors, i.e. the same
            # parameters and initialisation as "banded". No cross-sensor coupling:
            # on an MCU sensor d's gate is computed while sensor d is processed.
            P = mag ** 2                                     # (B, Lf, D)
            total = P.sum(dim=1, keepdim=True) + 1e-6        # (B, 1, D)
            if self._bounds_ps is not None:
                ratios = torch.einsum(
                    "bfd,sfd->bsd", P, self._slice_mask(P)) / total
            else:
                ratios = torch.stack(
                    [P[:, lo:hi].sum(dim=1) for (lo, hi) in self._bounds], dim=1
                ) / total                                    # (B, S, D)
            feats = torch.log(ratios.flip(dims=[1]) + 1e-6)
            feats = feats - feats.mean(dim=1, keepdim=True)
            self._feats = feats.transpose(1, 2)              # (B, D, S), reused by gate_mix
            return torch.softmax(self.proj(self._feats), dim=-1)

        if self.gate_mode == "banded":
            # Channel-mean POWER spectrum (mirrors utils/losses._power_spectrum)
            P = (mag ** 2).mean(dim=2)                       # (B, Lf)
            total = P.sum(dim=1, keepdim=True) + 1e-6
            ratios = torch.stack(
                [P[:, lo:hi].sum(dim=1) for (lo, hi) in self._bounds], dim=1
            ) / total                                        # (B, S), low -> high
            # band s owns slice S-1-s  ->  reverse to band order (POINT first)
            feats = torch.log(ratios.flip(dims=[1]) + 1e-6)
            # Center the log-ratios. Softmax is shift-invariant and a per-band
            # constant is absorbed by proj's bias, so this leaves the function
            # class unchanged ??it only keeps the linear layer's inputs from
            # sitting around -14 (conditioning).
            feats = feats - feats.mean(dim=1, keepdim=True)
            return torch.softmax(self.proj(feats), dim=-1)

        # Channel mean -> (B, L_f, 1) -> (B, 1, L_f)
        mag_mean = mag.mean(dim=2, keepdim=True).permute(0, 2, 1)

        # Conv1d + Global Pooling
        z = self.conv(mag_mean)
        z = self.pool(z).squeeze(-1)

        w = torch.softmax(z, dim=-1)
        return w


def _movavg(h_t: torch.Tensor, w: int) -> torch.Tensor:
    """Centered moving average of width w over the time axis. h_t: (B, D, L)."""
    pad = w // 2
    return F.avg_pool1d(F.pad(h_t, (pad, pad), mode="replicate"),
                        kernel_size=w, stride=1)


def split_bands_diff(h_t: torch.Tensor, num_bands: int):
    """Laplacian-pyramid band split: successive smoothings, then DIFFERENCES.

    The canonical split runs the input through learned convs of increasing width
    and uses the SMOOTHED signals themselves as bands. Those all retain the slow
    component, so the branches carry overlapping content (two moving averages of
    width 3 and 9 correlate 0.99 on a slow+fast mixture) and `lambda_spec` was
    added to push them apart from the outside.

    Taking differences instead makes the split structural:
        band_0 (POINT) = x        - smooth(w1)      fastest
        band_i         = smooth(wi) - smooth(w(i+1))
        band_S-1 (LOW) = smooth(w_{S-1})            slowest
    The bands are disjoint by construction and sum EXACTLY back to the input,
    and the split costs zero parameters. Band order matches the canonical
    convention (index 0 = POINT = highest frequency).
    """
    if num_bands == 1:
        return [h_t]
    widths = [4 ** i + 1 for i in range(1, num_bands)]      # 5, 17, 65, ...
    smooths = [_movavg(h_t, w) for w in widths]
    bands = [h_t - smooths[0]]
    for i in range(len(smooths) - 1):
        bands.append(smooths[i] - smooths[i + 1])
    bands.append(smooths[-1])
    return bands


def _fft_bounds(n_freq: int, num_bands: int, scale: str):
    """Slice boundaries over rfft bins, low -> high.

    "linear" splits the axis into equal-width slices. That mirrors
    utils/losses.spectral_separation_loss_scales, but it is a poor split for
    real series: measured on training windows (instance-normalized, L=96), the
    lowest quarter of the spectrum holds 81.7% of the energy on ETTh1 and 87.4%
    on Steel, while the top two slices hold 8.1% and 4.3%. A four-way equal
    split is therefore one real band plus three nearly empty ones.

    "octave" halves the bin index at each step, which is the standard remedy
    (the same reasoning behind wavelet / mel spacing). It balances the same
    windows to 64/17/10/8 (ETTh1) and 71/17/8/4 (Steel).
    """
    if scale == "octave":
        cuts, b = [], n_freq
        for _ in range(num_bands - 1):
            b = max(1, b // 2)
            cuts.append(b)
        edges = sorted(set([0] + cuts + [n_freq]))
        # a degenerate ladder (tiny n_freq) can yield fewer edges than asked
        while len(edges) - 1 < num_bands:
            edges.insert(1, min(edges[1] + 1, n_freq))
            edges = sorted(set(edges))
        return [(edges[i], edges[i + 1]) for i in range(num_bands)]
    step = n_freq // num_bands
    return [(s * step, n_freq if s == num_bands - 1 else (s + 1) * step)
            for s in range(num_bands)]


def split_bands_fft(h_t: torch.Tensor, num_bands: int, scale: str = "linear"):
    """Exact rFFT band split: slice the spectrum, transform each slice back.

    This is the "matched frequency band" control the ablation was missing: real,
    orthogonal, non-overlapping frequency bands that sum back to the input, with
    zero learned parameters. Without it a flat multi-band result can always be
    answered with "your bands were not really frequency bands".

    Slice assignment mirrors utils/losses.spectral_separation_loss_scales and
    FFTFrequencyGate(gate_mode="banded"): slices run low->high and band s owns
    slice S-1-s, so band 0 (POINT) is the highest-frequency slice.
    """
    if num_bands == 1:
        return [h_t]
    L = h_t.shape[-1]
    X = torch.fft.rfft(h_t.float(), dim=-1)
    n_freq = X.shape[-1]
    bounds = _fft_bounds(n_freq, num_bands, scale)
    out = []
    for s in range(num_bands):                       # band order: POINT first
        lo, hi = bounds[num_bands - 1 - s]           # its slice, low->high
        Xs = torch.zeros_like(X)
        Xs[..., lo:hi] = X[..., lo:hi]
        out.append(torch.fft.irfft(Xs, n=L, dim=-1).to(h_t.dtype))
    return out


class ZeroSumConv1d(nn.Conv1d):
    """Depthwise band conv whose taps are forced to sum to zero (2026-09-18).

    The kernel size alone does not decide which frequencies a branch takes — a
    k=3 kernel [1/3, 1/3, 1/3] is a low-pass — and the band check found the
    trained POINT/MID branches still pass 10-23 % of DC with the order
    POINT > MID > LOW broken on 13-49 % of sensors.  Subtracting the tap mean
    makes the DC gain exactly 0, so a POINT/MID branch can only see change and
    the slow part is left to LOW by construction.  Same parameters, same
    initialisation; on the MCU the centred taps are simply what gets exported.
    """

    def centred_weight(self) -> torch.Tensor:
        return self.weight - self.weight.mean(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(x, self.centred_weight(), self.bias)


class BandNonlinearity(nn.Module):
    """Per-band nonlinearity applied AFTER the shared kernel, BEFORE the gate sum.

    Why here (2026-08-27). The band bank + gate block is provably inert as long
    as every stage from the input to the merge is linear: for fixed gate weights
        sum_b w_b * DTK(F_b(x))  ==  DTK( sum_b w_b * F_b(x) )
    so four branches collapse into one filter (measured on the real module:
    max|diff| 1.9e-09). The gate is nominally input-dependent, which would break
    that on its own ??but the canonical "global" gate returns IDENTICAL weights
    for a slow and a fast sinusoid to four decimals (measured live), so w is
    effectively a constant and the collapse stands. This is the single reason
    num_bands, gate on/off and lambda_spec all measured flat.

    A nonlinearity between the branch and the sum cannot be pulled out of it, so
    the branches stay distinct and the gate finally has something to select
    between.

    Modes:
        "none"  canonical, byte-identical.
        "gelu"  elementwise GELU on the band. Strong, but it half-rectifies a
                zero-mean band and thus distorts the waveform ??the most likely
                reason `dtk_act=gelu` (the same idea inside the DTK latent)
                measured +6.8% on Steel under the shipping protocol.
        "se"    content-dependent multiplicative GAIN, squeeze-excitation style:
                    band <- band * sigmoid(a_b * log(rms(band)) + c_b)
                The gain depends on the band's own energy, so the map is
                nonlinear in x (a product of two x-dependent terms), yet the
                waveform is only rescaled, never clipped. 2 params per band.
    """

    def __init__(self, num_bands: int, mode: str = "none"):
        super().__init__()
        assert mode in ("none", "gelu", "se"), mode
        self.mode = mode
        if mode == "se":
            # Zero-init a, c -> gain = sigmoid(0) = 0.5 for every band, i.e. a
            # constant halving that the downstream weights absorb. The arm
            # therefore STARTS at the canonical function class and has to learn
            # its way out, which keeps the comparison controlled.
            self.a = nn.Parameter(torch.zeros(num_bands))
            self.c = nn.Parameter(torch.zeros(num_bands))

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        """bands: (B, S, L, D) -> same shape."""
        if self.mode == "none":
            return bands
        if self.mode == "gelu":
            return F.gelu(bands)
        rms = bands.float().pow(2).mean(dim=(2, 3), keepdim=True).add(1e-8).sqrt()
        gain = torch.sigmoid(self.a.view(1, -1, 1, 1) * torch.log(rms)
                             + self.c.view(1, -1, 1, 1))
        return bands * gain.to(bands.dtype)


class DenseTemporalKernel(nn.Module):
    """
    Learned depthwise convolution kernel for dense temporal mixing.

    Projects input to latent state space, applies channel-wise convolution,
    then projects back to original dimension.

    Args:
        d_model: Model dimension
        d_state: Latent state dimension
        kernel_size: Convolution kernel size
    """

    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7,
                 act: str = "none"):
        super().__init__()
        assert act in ("none", "gelu", "relu", "tanh"), act
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        # DTK is LINEAR by default (in_proj -> depthwise conv -> out_proj, no
        # activation anywhere). That is not a detail: the four bands are fed
        # through this SAME module and then combined by a weighted SUM, so with
        # a linear DTK
        #     sum_b w_b * DTK(F_b(x))  ==  DTK( sum_b w_b * F_b(x) )
        # and the whole "multi-band decomposition + gating" block collapses to a
        # SINGLE linear filter. Verified numerically (2026-08-16): with the gate
        # held constant the block is additive/homogeneous to 2e-9. That is why
        # num_bands (0.6% spread over 27 configs), lambda_spec, and a genuinely
        # shape-aware gate (gate_mode=banded, +1.1%) are all flat ??there is
        # nothing for a gate to select between when the branches merge.
        # A nonlinearity here cannot be pulled out of the sum, so the bands stay
        # distinct. It costs ZERO parameters, so the sub-1K budget is untouched.
        self.act = act

        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        self.kernel = nn.Parameter(torch.randn(d_state, kernel_size) * 0.01)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, L, D) input tensor

        Returns:
            (B, L, D) output tensor
        """
        B, L, D = u.size()

        # Project to latent space: (B, L, D) -> (B, S, L)
        x = self.in_proj(u).permute(0, 2, 1)

        # Depthwise conv: groups = S
        k = self.kernel.unsqueeze(1)
        y = F.conv1d(x, k, padding=self.kernel_size - 1, groups=self.d_state)
        y = y[:, :, :L]  # Trim to length L

        # Optional nonlinearity in the LATENT state, before projecting back.
        # "none" is canonical and byte-identical.
        if self.act == "gelu":
            y = F.gelu(y)
        elif self.act == "relu":
            y = F.relu(y)
        elif self.act == "tanh":
            y = torch.tanh(y)

        # Project back: (B, S, L) -> (B, L, D)
        y = self.out_proj(y.permute(0, 2, 1))
        return y


class LatentMLP(nn.Module):
    """DTK control: latent projection with a nonlinearity, no temporal conv.

    in_proj(D -> S) -> GELU -> out_proj(S -> D), applied per time step. This is
    the "MLP-only" row of the earlier DTK ablation table. It differs from
    kernel_size=1 in both directions: no convolution at all, but a genuine
    nonlinearity, so the pair separates "the DTK's temporal mixing matters"
    from "the DTK's latent bottleneck matters".
    """

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.gelu(self.in_proj(u)))


class MLPHead(nn.Module):
    """Two-layer time-axis MLP ??the nonlinear control head for the SPK ablation.

    Linear(L -> 2L) -> GELU -> Linear(2L -> pred_len), applied per channel.
    Sits between DenseLinearHead (linear, dense) and SPK (linear,
    period-structured): if SPK loses to this one but ties DenseLinearHead, the
    gap is nonlinearity rather than the period structure.

    hidden_ratio 2 reproduces the "MLP Head" row of the earlier ablation table
    so the comparison carries over. Like DenseLinearHead this is far larger
    than SPK, so report parameters next to the accuracy delta.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 hidden_ratio: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        hidden = seq_len * hidden_ratio
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, pred_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h.permute(0, 2, 1)                 # (B, D, L)
        z = self.fc2(F.gelu(self.fc1(z)))      # (B, D, H)
        return z.permute(0, 2, 1)              # (B, H, D)


class ConvHead(nn.Module):
    """Depthwise-conv head ??the local-context control for the SPK ablation.

    Two depthwise Conv1d layers with a GELU between them, then a dense
    Linear(L -> pred_len). Reproduces the "Conv Head" row of the earlier
    ablation table: it keeps local temporal structure (like SPK's aggregation
    conv) but drops the period-phase factorisation.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int,
                 kernel_size: int = 3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=d_model)
        self.proj = nn.Linear(seq_len, pred_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h.permute(0, 2, 1)                 # (B, D, L)
        z = self.conv2(F.gelu(self.conv1(z)))
        return self.proj(z).permute(0, 2, 1)   # (B, H, D)


class DenseLinearHead(nn.Module):
    """Plain time-axis projection ??the control head for the SPK ablation.

    Maps (B, L, D) -> (B, pred_len, D) with one shared Linear(L -> pred_len)
    per channel, i.e. the head DLinear and most linear forecasters use. It is
    the natural "what does the period structure buy us?" control: everything
    upstream (bands, DTK, gate, residual MLP, channel mode) is untouched.

    NOTE this is NOT capacity-matched, and that asymmetry is the finding, not
    a flaw: at Steel/H96 SPK costs ~166 params (a depthwise aggregation conv
    plus an n->m backbone over period phases) while this dense head costs
    L*H + H ~= 9.3k. Swapping it in makes the model an order of magnitude
    larger, so report the parameter count alongside the accuracy delta ??
    SPK's claim is efficiency, and a near-tie here IS the supporting result.
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, D, L) -> (B, D, H) -> (B, H, D)
        return self.linear(h.permute(0, 2, 1)).permute(0, 2, 1)


class SparsePeriodKernel(nn.Module):
    """
    Period-aware sparse forecasting kernel (inspired by SparseTSF).

    Reorganizes input by period phases, applies shared linear projection
    across periods, then reconstructs the output sequence.

    Args:
        seq_len: Input sequence length (must be divisible by period)
        pred_len: Prediction length (must be divisible by period)
        d_model: Model dimension
        period: Period size (e.g., 24 for hourly data with daily patterns)
        hidden: 0 (default, byte-identical) keeps the linear cross-period
            backbone. >0 inserts one GELU hidden layer, n -> hidden -> m.

            Motivation (2026-08-26): every model in the sub-1K family is
            linear end to end (SparseTSF-Linear, MixLinear, FITS, DLinear, and
            DiPE whose softmax sits on an nn.Parameter and is therefore a
            constant at inference), the one exception being SparseTSF-MLP's
            ReLU head ??which beats its own linear version by ~3%. On our side
            every nonlinearity added so far has paid (residual MLP +2-9%, MLP
            head -4.64%), while no rearrangement of linear parts has ever moved
            the number, because a linear block collapses (2026-08-16). Once the
            band bank and DTK are replaced by a single conv (the `redA`
            configuration) the residual MLP is the model's ONLY nonlinearity
            and the head is purely linear, so this is the cheapest place left
            to add one: n*h + h + h*m + m params (n=m=4, h=8 -> 76).
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int,
                 hidden: int = 0):
        super().__init__()
        assert seq_len % period == 0, "seq_len must be divisible by period"
        assert pred_len % period == 0, "pred_len must be divisible by period"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.period = period

        self.n = seq_len // period   # input periods
        self.m = pred_len // period  # output periods

        # Sliding aggregation conv
        k = 2 * (period // 2) + 1
        self.agg_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=k, padding=k // 2,
            groups=d_model, bias=False
        )

        # Cross-period backbone. hidden=0 keeps the canonical single bias-free
        # linear map; hidden>0 makes it a 2-layer MLP over the period axis.
        self.hidden = int(hidden)
        if self.hidden > 0:
            self.backbone = nn.Sequential(
                nn.Linear(self.n, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, self.m),
            )
        else:
            self.backbone = nn.Linear(self.n, self.m, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, L, D) input tensor

        Returns:
            (B, pred_len, D) output tensor
        """
        B, L, D = h.size()
        w, n, m, H = self.period, self.n, self.m, self.pred_len

        # Residual + depthwise aggregation
        h_t = h.permute(0, 2, 1)
        h_agg = h_t + self.agg_conv(h_t)

        # Reshape for period-wise processing
        x_merged = h_agg.reshape(B * D, L)
        x_by_phase = x_merged.view(B * D, n, w).permute(0, 2, 1)

        # Shared backbone for each phase
        x_per_phase = x_by_phase.reshape(-1, n)
        y_per_phase = self.backbone(x_per_phase)
        y_by_phase = y_per_phase.view(B * D, w, m)

        # Reconstruct output
        y_reordered = y_by_phase.permute(0, 2, 1).reshape(B * D, H)
        y = y_reordered.view(B, D, H).permute(0, 2, 1)
        return y


class FEATHer(nn.Module):
    """
    FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster

    An ultra-lightweight model for long-term time series forecasting that
    combines multi-scale frequency decomposition with adaptive gating
    and period-aware sparse forecasting.

    Args:
        seq_len: Input sequence length
        pred_len: Prediction length
        d_model: Number of input features
        d_state: Latent state dimension for DenseTemporalKernel
        kernel_size: Kernel size for DenseTemporalKernel
        use_norm: Whether to use instance normalization
        period: Period for SparsePeriodKernel
        num_bands: Number of frequency bands (2, 3, or 4)
        use_topk_gate: Whether to use top-k sparse gating
        topk: Number of active bands when using top-k gating
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_state: int = 8,
        kernel_size: int = 7,
        use_norm: bool = True,
        period: int = 24,
        num_bands: int = 3,
        use_topk_gate: bool = False,
        topk: int = 2,
        use_residual_mlp: bool = False,
        mlp_rank: int = 16,
        use_gate: bool = True,
        use_robust_norm: bool = False,
        revin_mode: str = None,
        winsor_bound: float = 4.0,
        center_trim: float = 0.0,
        channel_mode: str = "mix",
        gate_mode: str = "global",
        dtk_act: str = "none",
        head_mode: str = "spk",
        dtk_mode: str = "full",
        point_kernel: int = 1,
        band_kernels: str = "",
        band_dilations: str = "",
        use_in_proj: bool = True,
        spk_hidden: int = 0,
        band_act: str = "none",
        band_mode: str = "conv",

        gate_zero_init: bool = False,
        mix_mode: str = "hidden",
        mix_pool: int = 1,
        mlp_tie: bool = False,
        gate_bounds: str = "linear",
        band_zero_sum: bool = False,
        gate_mix: str = "none",
        head2: str = "none",
        mix_listen: bool = False,
        mix_lag: str = "",
        mix_zero_init: bool = False,
        far_coarse: int = 0,
        near_len: int = 96,
    ):
        super().__init__()

        # 1 = no decomposition (POINT branch only); the ablation floor.
        assert num_bands in [1, 2, 3, 4], "num_bands must be 1, 2, 3, or 4"

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.period = period
        self.num_bands = num_bands
        self.use_topk_gate = use_topk_gate
        self.topk = topk
        self.use_residual_mlp = use_residual_mlp
        # Diagnostic toggle (2026-07-31): when False, replace the input-adaptive
        # FFT gate with fixed uniform band weights. The freq_gate module is
        # still constructed (state_dict keys unchanged) but bypassed in forward
        # ??used to isolate whether the gate's reaction to corrupted spectra is
        # the source of FEATHer's robustness fragility. ON by default (canonical).
        self.use_gate = use_gate

        # Instance-norm variant selector (2026-08-01/02). All variants add ZERO
        # parameters (statistics only) ??"standard" is byte-identical canonical.
        # Ignored entirely when use_norm is False (--no_revin).
        #
        #   "standard"  canonical RevIN: window mean/std, norm + denorm.
        #   "mad"       median / 1.4826*MAD (+meanAD fallback, +-10 clamp).
        #               KEPT ONLY as the negative-result ablation arm: under a
        #               mass zero-fill miss fault the median hits its 50%
        #               breakdown (flips to the fill value) -> Dw explodes
        #               (Steel miss@0.5 Dw 4.6 vs 1.6 standard), and the noisier
        #               order statistics also hurt clean MSE. Matches R2-IN
        #               (arXiv:2510.04667) evaluated only on natural outliers.
        #   "winsor"    winsorize the window to +-_WINSOR_BOUND, then the usual
        #               mean/std on the CLIPPED copy (bounded influence: a spike
        #               contributes at most the bound; zero-fill degrades
        #               gracefully like the mean ??no breakdown jump).
        #   "center"    winsorized-mean CENTERING ONLY ??no window scale at all
        #               (x_norm = clip(x)-mean; denorm adds the mean back).
        #               Rationale (2026-08-01 fault-axis fingerprint): gauss/
        #               impulse/miss all corrupt the window std, which multiplies
        #               errors in BOTH norm and denorm (quant, which leaves std
        #               alone, shows Dw=1.0). Freezing scale at the train-time
        #               Scaler removes the multiplicative path; window adaptation
        #               keeps only the level term nonstationary data needs.
        #   "last"      NLinear-style LEVEL ANCHOR: use the last observed value
        #               of the window, x[:, -1, :], as the level instead of the
        #               window mean; no window scale (denorm adds the anchor
        #               back). For a (near) random walk the window mean is a
        #               strictly worse level estimate than the last value, and
        #               the canonical mean anchor is then actively harmful.
        #               Measured on the test windows (2026-08-14, naive-forecast
        #               MSE, no training): the last-value anchor beats the mean
        #               anchor by 13-42% on Exchange at every horizon, and LOSES
        #               by 16-85% on ETTh1/h2/m1/m2/Weather. So this is a
        #               per-dataset, val-selected option ??never a new default.
        #   "last_std"  same anchor, but keep the canonical window-std scaling
        #               (x_norm = (x - x[:,-1]) / std). Separates "which level"
        #               from "scale on/off"; without it a "last" win could not
        #               be attributed to the anchor rather than to dropping the
        #               scale (which "center" already changes on its own).
        if revin_mode is None:
            revin_mode = "mad" if use_robust_norm else "standard"
        assert revin_mode in ("standard", "mad", "winsor", "center",
                              "last", "last_std"), revin_mode
        self.revin_mode = revin_mode
        self.use_robust_norm = use_robust_norm
        # Knobs for the winsor/center statistics (2026-08-02 Steel-residual
        # probes; defaults reproduce the first winsor/center arms exactly):
        #   winsor_bound  clip range in train-Scaler sigmas. The impulse fault
        #                 injects ~3-sigma spikes, so the default 4.0 leaves
        #                 them UNclipped (measured: winsor arm ~= standard Dw);
        #                 2.5 actually caps them.
        #   center_trim   symmetric trimmed-mean fraction for the center
        #                 estimate (0.05 = drop top+bottom 5% of the window
        #                 before averaging). Removes <=5%-rate spikes from the
        #                 center entirely; under mass zero-fill it degrades
        #                 gracefully like the mean (no median-style flip).
        self.winsor_bound = float(winsor_bound)
        self.center_trim = float(center_trim)

        # Channel handling (2026-08-02). "mix" = canonical (in_proj D->D and DTK
        # D->d_state->D both MIX channels; params grow ~D^2, and D=28 gets
        # compressed into an 8-dim latent -> the high-channel accuracy gap vs
        # DLinear). "ci_full" = channel-independent: the forward folds
        # (B,L,D)->(B*D,L,1) so every layer sees ONE channel (weights shared
        # across channels, no cross-channel info), rebuilding all sub-modules at
        # width 1. Effect: params become D-INDEPENDENT (BlastFurnace D=28
        # 2274->~970, sub-1K restored), matches the DLinear/SparseTSF channel
        # strategy, and a corrupted channel can't leak into others (robustness
        # bonus). Risk: low-channel data (Steel D=6) may benefit from mixing, so
        # this is an A/B, not a default. OFF ("mix") = canonical byte-identical.
        assert channel_mode in ("mix", "ci_full"), channel_mode
        self.channel_mode = channel_mode
        eff_d = 1 if channel_mode == "ci_full" else d_model
        self.eff_d = eff_d

        # Input projection = the LINEAR channel mixer. The residual MLP below is
        # a second, NONLINEAR channel mixer, so the two are potentially
        # redundant; use_in_proj=False drops this one while keeping eff_d=D, to
        # ask that question directly. (channel_mode="ci_full" cannot: it sets
        # eff_d=1 and so removes BOTH mixers at once.) True = canonical.
        self.use_in_proj = use_in_proj
        self.in_proj = nn.Linear(eff_d, eff_d) if use_in_proj else nn.Identity()

        # Optional nonlinear residual MLP (channel-axis low-rank bottleneck),
        # applied to the gated representation just before the forecasting head:
        #     h <- h + W_up(GELU(W_down(h))),  W_down: D->r, W_up: r->D
        # Adds ~2*D*r params (D=7,r=16 -> +247). OFF by default so the canonical
        # model and every frozen checkpoint are byte-identical; enabled only for
        # the capacity A/B experiment (2026-07-31). This is FEATHer's first
        # pointwise nonlinearity in the signal path (otherwise linear+gating).
        # Mixer variants (2026-09-14, cost-reduction screen). The defaults are
        # the shipping model and byte-identical:
        #   mix_mode="hidden"  canonical: h <- h + MLP(h) at every timestep.
        #   mix_mode="input"   the correction is computed from the normalized
        #                      INPUT, average-pooled over time by mix_pool, then
        #                      linearly interpolated back to L and added to h.
        #                      This is the variant that can drop the (L x D)
        #                      hidden state on the MCU: the input window is
        #                      stored anyway, the mixer only needs
        #                      (L/mix_pool x D), and every other stage can then
        #                      run one sensor at a time. mix_pool=1 separates
        #                      "input vs hidden" from the pooling itself;
        #                      mix_pool=L gives one correction per sensor.
        #   mix_mode="input_raw" (2026-09-16) the same as "input", but the
        #                      summary is built from the window BEFORE instance
        #                      normalization (the train-standardized input).
        #                      On the MCU the summary then needs only the raw
        #                      window, which is kept anyway, so every sensor is
        #                      normalized exactly once, inside its own pass.
        #   mlp_tie=True       W_up = W_down^T with its own bias: half the mixer
        #                      weights, the same number of multiplications.
        assert mix_mode in ("hidden", "input", "input_raw"), mix_mode
        assert mix_pool >= 1 and seq_len % mix_pool == 0, (
            "mix_pool must divide seq_len: %r / %r" % (seq_len, mix_pool))
        if (mix_mode != "hidden" or mix_pool != 1 or mlp_tie) and not use_residual_mlp:
            raise ValueError("mix_mode / mix_pool / mlp_tie need use_residual_mlp")
        if mix_mode == "hidden" and mix_pool != 1:
            raise ValueError("mix_pool > 1 is only defined for mix_mode='input'")
        self.mix_mode = mix_mode
        self.mix_pool = int(mix_pool)
        self.mlp_tie = bool(mlp_tie)
        if use_residual_mlp:
            self.nl_down = nn.Linear(eff_d, mlp_rank)
            self.nl_up = nn.Linear(mlp_rank, eff_d)
            if mlp_tie:
                # Built, then stripped to its bias: the RNG stream - and with
                # it the initialisation of every module constructed after this
                # one - stays identical to the untied model, so a difference
                # in results is the tie and not a re-drawn init.
                self.nl_up_bias = nn.Parameter(self.nl_up.bias.detach().clone())
                del self.nl_up

        # Multi-band 1D Conv decomposition
        # POINT: kernel=1, captures high-frequency details
        # point_kernel=1 (default) is the canonical POINT branch and keeps every
        # frozen checkpoint byte-identical. It is a per-channel SCALAR, not a
        # filter, which is why num_bands=1 + dtk_mode=none leaves the model with
        # no temporal mixing at all before the head (measured: +2.2% MSE). Widen
        # it to run the "one real filter is enough" configuration, where the
        # single branch replaces both the band bank and the DTK convolution.
        assert point_kernel >= 1 and point_kernel % 2 == 1, \
            "point_kernel must be a positive odd integer, got %r" % (point_kernel,)
        # How the bands are produced (2026-08-27). "conv" is canonical and
        # byte-identical: learned depthwise convs of increasing width, whose
        # outputs are used directly as bands. "diff" and "fft" replace them with
        # splits that are disjoint by construction and cost ZERO parameters ??
        # see split_bands_diff / split_bands_fft.
        assert band_mode in ("conv", "diff", "fft", "fftoct"), band_mode
        self.band_mode = band_mode

        self.point_kernel = point_kernel
        # The conv bank exists ONLY for band_mode="conv". The parameter-free
        # splits do not call it, and allocating it anyway would inflate the
        # reported parameter count by the very weights the split removes (72 at
        # D=6/4 bands) ??which is the whole point of those modes.
        # 팽창(dilation). 수용영역은 dilation*(k-1)+1 이므로, 커널을 넓히지 않고도
        # 넓게 볼 수 있다(TCN/WaveNet 방식). k=3,d=15 는 탭 3개로 수용영역 31.
        # 커널을 넓히면 파라미터가 탭 수만큼 늘지만 팽창은 공짜다.
        _bd = [int(v) for v in str(band_dilations).split(",") if str(v).strip()]             if band_dilations else None
        if _bd is not None:
            assert len(_bd) == num_bands - 1, (
                "band_dilations 는 num_bands-1 개여야 함: %r" % (band_dilations,))
            assert all(v >= 1 for v in _bd), "dilation 은 1 이상"
        d_point, d_high, d_mid = (
            (_bd[0], (_bd[1] if num_bands == 4 else 1), _bd[-1]) if _bd else (1, 1, 1))

        _conv_bank = band_mode == "conv"
        self.conv_point = nn.Conv1d(
            eff_d, eff_d, kernel_size=point_kernel, dilation=d_point,
            padding=d_point * (point_kernel // 2), groups=eff_d
        ) if _conv_bank else None

        # 밴드 커널 사다리 (2026-08-31). 원래는 POINT=point_kernel / HIGH=3 / MID=5 로
        # 박혀 있었는데, 그러면 L=96 에서 수용영역이 {1,3,5,~8} = 전부 좁은 쪽에 몰린다
        # (최대/최소 비 2.7배). 밴드끼리 구별이 안 되면 게이트가 고를 대상이 없다.
        # --band_kernels "3,7,15" 처럼 주면 사다리를 넓힐 수 있다(옥타브 간격 등).
        # 빈 문자열 = 기존 동작 그대로 (byte-identical).

        if band_kernels:
            _bk = [int(v) for v in str(band_kernels).split(",") if str(v).strip()]
            need = num_bands - 1          # LOW 는 풀링이라 conv 가 없다
            assert len(_bk) == need, (
                "band_kernels 는 num_bands-1 개여야 함 (LOW=풀링): "
                "num_bands=%d -> %d 개, 받은 값 %r" % (num_bands, need, band_kernels))
            assert all(k >= 1 and k % 2 == 1 for k in _bk),                 "band_kernels 는 홀수여야 함: %r" % (_bk,)
            # 순서: POINT(제일 좁음) -> HIGH -> MID(제일 넓음) -> LOW(풀링)
            k_point = _bk[0]
            k_high = _bk[1] if num_bands == 4 else None
            k_mid = _bk[-1] if num_bands >= 3 else None
            self.conv_point = nn.Conv1d(
                eff_d, eff_d, kernel_size=k_point, dilation=d_point,
                padding=d_point * (k_point // 2), groups=eff_d) if _conv_bank else None
            self.point_kernel = k_point
        else:
            k_high, k_mid = 3, 5

        if _conv_bank and num_bands == 4:
            self.conv_high = nn.Conv1d(
                eff_d, eff_d, kernel_size=k_high, dilation=d_high,
                padding=d_high * (k_high // 2), groups=eff_d
            )
        else:
            self.conv_high = None

        if _conv_bank and num_bands >= 3:
            self.conv_mid = nn.Conv1d(
                eff_d, eff_d, kernel_size=k_mid, dilation=d_mid,
                padding=d_mid * (k_mid // 2), groups=eff_d
            )
        else:
            self.conv_mid = None

        # band_zero_sum (2026-09-18): POINT/HIGH/MID see change only (DC gain 0).
        # Swapping the class after construction keeps the parameters and the RNG
        # stream, so a seed initialises exactly as without the switch.
        self.band_zero_sum = bool(band_zero_sum)
        if self.band_zero_sum:
            assert _conv_bank, "band_zero_sum needs band_mode='conv'"
            for conv in (self.conv_point, self.conv_high, self.conv_mid):
                if conv is not None:
                    conv.__class__ = ZeroSumConv1d

        # LOW: avg pooling (L -> L/4) then upsample
        self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)

        # Shared temporal kernel. dtk_mode="none" removes it outright ??each
        # band passes through untouched ??which is the real "is the DTK
        # earning its place?" control. (Setting kernel_size=1 only removes the
        # temporal mixing while keeping the latent projections, so it answers a
        # narrower question; both are worth running.)
        assert dtk_mode in ("full", "none", "mlp"), dtk_mode
        self.dtk_mode = dtk_mode
        if dtk_mode == "none":
            self.dense_kernel = nn.Identity()
        elif dtk_mode == "mlp":
            # Latent projection with a nonlinearity but NO temporal conv, i.e.
            # the "MLP-only" row of the earlier DTK table. Distinct from
            # kernel_size=1, which keeps the conv and stays linear.
            self.dense_kernel = LatentMLP(d_model=eff_d, d_state=d_state)
        else:
            self.dense_kernel = DenseTemporalKernel(
                d_model=eff_d,
                d_state=d_state,
                kernel_size=kernel_size,
                act=dtk_act,
            )

        # Per-band nonlinearity (2026-08-27). Sits between the shared kernel and
        # the gate sum ??the only place that can stop the band bank collapsing
        # into a single filter. "none" is canonical and byte-identical.
        self.band_nl = BandNonlinearity(num_bands=num_bands, mode=band_act)

        # Band synthesis. "global"/"banded"/"static" are FFT-gate flavours;
        # "none" and "softmax" are the ablation controls from the earlier
        # gating table and build no FFT gate at all:
        #   none     unweighted SUM of the bands (no gate, and note this is
        #            NOT the same as --gate_uniform, which averages)
        #   softmax  time-pooled input -> Linear(D -> S) -> softmax, i.e. an
        #            input-dependent gate that never looks at the spectrum
        assert gate_mode in ("global", "banded", "banded_sensor", "static", "static_rng", "none",
                             "none_rng", "softmax"), gate_mode
        self.gate_mode = gate_mode
        if gate_mode in ("none", "none_rng"):
            # none_rng (2026-09-16): the same unweighted sum, but draw (and discard)
            # the Linear(S, S) the banded gate builds, so the SPK head gets the
            # banded model's initialisation - as static_rng does. Plain "none"
            # consumes no RNG here and shifts every later init.
            if gate_mode == "none_rng":
                nn.Linear(num_bands, num_bands)
            self.gate_mode = "none"
            self.freq_gate = None
        elif gate_mode == "softmax":
            self.freq_gate = None
            self.softmax_gate = nn.Linear(eff_d, num_bands)
        else:
            self.freq_gate = FFTFrequencyGate(
                seq_len=seq_len,
                num_bands=num_bands,
                kernel_size=5,
                gate_mode=gate_mode,

                gate_zero_init=gate_zero_init,
                gate_bounds=gate_bounds,
            )

        # Forecasting head. "spk" = canonical period-aware sparse kernel;
        # "linear" swaps in a dense time projection and is the ablation control
        # (see DenseLinearHead ??it is deliberately not capacity-matched).
        assert head_mode in ("spk", "linear", "mlp", "conv"), head_mode
        self.head_mode = head_mode
        # Near/far head (2026-10-04, idea A).  far_coarse = r > 0: the SPK head
        # above only produces the first near_len steps (all bands, full
        # resolution); a second SPK head forecasts the WHOLE horizon from the
        # pooled LOW band alone at 1/r resolution, interpolated back to H, and the
        # near head is added on top.  Far-ahead fast detail is not forecastable,
        # so the fast bands never feed the far horizon.  0 = off, byte-identical.
        self.far_coarse = int(far_coarse)
        self.near_len = min(int(near_len), pred_len) if self.far_coarse else pred_len
        if self.far_coarse:
            assert head_mode == "spk" and not spk_hidden and head2 == "none", \
                "far_coarse extends the plain SPK head"
            assert band_mode == "conv" and self.far_coarse == 4, \
                "the coarse head reads pool_low (AvgPool1d k=4)"
            assert self.near_len % period == 0, (self.near_len, period)
        if head_mode == "linear":
            self.sparse_head = DenseLinearHead(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=eff_d,
            )
        elif head_mode == "mlp":
            self.sparse_head = MLPHead(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=eff_d,
            )
        elif head_mode == "conv":
            self.sparse_head = ConvHead(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=eff_d,
            )
        else:
            self.sparse_head = SparsePeriodKernel(
                seq_len=seq_len,
                pred_len=self.near_len,
                d_model=eff_d,
                period=period,
                hidden=spk_hidden,
            )

        # FFT-gate redesigns (2026-09-18).  The band gate is accuracy-neutral
        # because its choice is absorbed by the linear head, so these give the
        # spectrum a decision the head cannot absorb.  "static" = same modules
        # with the input dependence removed (the ablation control).  Built LAST
        # so every existing module keeps the candidate's initialisation.
        #   gate_mix (G2): alpha = 2*sigmoid(W feats + b) scales the mixer
        #       correction per sensor and window; feats = the band gate's
        #       centred log energy ratios.  Zero-init -> alpha = 1 = candidate.
        #   head2 (G4-lite): a second SPK readout of the SAME aggregated series
        #       at period2 = 2*period (or period/2), blended by
        #       g = sigmoid(W e + b), e = log energy share at the harmonics of
        #       each period (the band ratios cannot tell 24 from 48: both sit
        #       in the lowest slice).  Zero-init -> g = 0.5.
        assert gate_mix in ("none", "fft", "static"), gate_mix
        assert head2 in ("none", "fft", "static"), head2
        self.gate_mix = gate_mix
        self.head2 = head2
        if gate_mix != "none":
            assert use_residual_mlp, "gate_mix needs the residual MLP"
            if gate_mix == "fft":
                assert gate_mode == "banded_sensor", "gate_mix=fft reads the per-sensor gate features"
            self.mix_gain = nn.Linear(num_bands, 1)
            nn.init.zeros_(self.mix_gain.weight)
            nn.init.zeros_(self.mix_gain.bias)
            if gate_mix == "static":
                self.mix_gain.weight.requires_grad_(False)
        if head2 != "none":
            assert head_mode == "spk" and not spk_hidden, "head2 extends the linear SPK head"
            p2 = 2 * period
            if seq_len % p2 or pred_len % p2:
                p2 = period // 2
            assert p2 >= 1 and seq_len % p2 == 0 and pred_len % p2 == 0, (period, p2)
            self.period2 = p2
            self.backbone2 = nn.Linear(seq_len // p2, pred_len // p2, bias=False)
            self.head2_gate = nn.Linear(2, 1)
            nn.init.zeros_(self.head2_gate.weight)
            nn.init.zeros_(self.head2_gate.bias)
            if head2 == "static":
                self.head2_gate.weight.requires_grad_(False)
            lo, hi = sorted((period, p2))
            half = seq_len // 2
            b_short = set(range(seq_len // lo, half + 1, seq_len // lo))
            b_long = set(range(seq_len // hi, half + 1, seq_len // hi)) - b_short
            idx = {lo: sorted(b_short), hi: sorted(b_long)}
            # order = [head 1 (period), head 2 (period2)]
            self._h2_bins = [idx[period], idx[p2]]

        # Light-mixer fixes (2026-09-21), each aimed at a MEASURED limitation of
        # the D->R->D residual mixer (see tools/dev/{sensor_lag,mix_per_sensor}.py):
        #   mix_listen    (A) per-sensor scale on the correction, init 1: lets each
        #                 sensor decide how much to take (BF: 15/28 sensors hurt).
        #   mix_lag "k,d" (B) depthwise temporal conv over the R summary channels,
        #                 kernel k, dilation d, centred, identity-initialised: the
        #                 correction at t can use other sensors' state a few steps
        #                 away (EWELD lags 14-17, N2O 5-7, MetroPT3 2-3).
        #   mix_zero_init (C) zero the up-projection: training starts as the
        #                 mixer-off model and grows the mixing (the Steel bad
        #                 basin appears only with mixing: 4/20 vs 0/20).
        # All three start exactly at the candidate's function except C (starts at
        # the independent model).  Defaults off = unchanged.
        self.mix_listen = bool(mix_listen)
        self.mix_lag = str(mix_lag or "")
        if (self.mix_listen or self.mix_lag or mix_zero_init) and not use_residual_mlp:
            raise ValueError("mix_listen / mix_lag / mix_zero_init need the residual MLP")
        if self.mix_listen:
            self.listen = nn.Parameter(torch.ones(eff_d))
        if self.mix_lag:
            k, dil = (int(v) for v in self.mix_lag.split(","))
            assert k % 2 == 1 and dil >= 1, self.mix_lag
            self.lag_conv = nn.Conv1d(mlp_rank, mlp_rank, kernel_size=k, dilation=dil,
                                      padding=dil * (k // 2), groups=mlp_rank, bias=False)
            with torch.no_grad():
                self.lag_conv.weight.zero_()
                self.lag_conv.weight[:, 0, k // 2] = 1.0
        if mix_zero_init:
            assert not mlp_tie, "mix_zero_init zeroes nl_up"
            nn.init.zeros_(self.nl_up.weight)
            nn.init.zeros_(self.nl_up.bias)
        # Near/far coarse head: built LAST so every other module keeps its init.
        if self.far_coarse:
            r = self.far_coarse
            assert seq_len % r == 0 and pred_len % r == 0 and period % r == 0, (seq_len, pred_len, period)
            self.coarse_head = SparsePeriodKernel(
                seq_len=seq_len // r, pred_len=pred_len // r,
                d_model=eff_d, period=period // r)

    def _head2_readout(self, hidden: torch.Tensor) -> torch.Tensor:
        """The SPK readout at period2 on the head's own aggregated series."""
        sp = self.sparse_head
        h_t = hidden.permute(0, 2, 1)
        h_agg = h_t + sp.agg_conv(h_t)                       # (B, D, L)
        B, D, L = h_agg.shape
        w = self.period2
        n = L // w
        xp = h_agg.reshape(B * D, n, w).permute(0, 2, 1).reshape(-1, n)
        y = self.backbone2(xp).view(B * D, w, -1).permute(0, 2, 1)
        return y.reshape(B, D, self.pred_len).permute(0, 2, 1)

    def _head2_weight(self, x_norm: torch.Tensor) -> torch.Tensor:
        """(B, 1, D) weight of head 1; head 2 gets 1 - g."""
        if self.head2 == "static":
            return torch.sigmoid(self.head2_gate.bias).view(1, 1, 1)
        P = torch.fft.rfft(x_norm.float(), dim=1).abs() ** 2   # (B, Lf, D)
        tot = P.sum(dim=1) + 1e-6
        e = torch.stack([torch.log(P[:, b].sum(dim=1) / tot + 1e-6)
                         for b in self._h2_bins], dim=-1)       # (B, D, 2)
        return torch.sigmoid(self.head2_gate(e)).permute(0, 2, 1)

    def _near_far(self, y_near: torch.Tensor) -> torch.Tensor:
        """Coarse whole-horizon forecast from the pooled LOW band + near detail.

        The LOW band goes through the same residual mixer (shared weights), so
        the far horizon keeps the channel mixing; the coarse head runs at period
        P/r on L/r points and its H/r outputs are linearly interpolated to H
        (align_corners=False: coarse point j sits at the centre of steps
        [rj, rj + r), matching AvgPool's cells)."""
        s = self._low_ds
        if self.use_residual_mlp:
            s = s + self._mixer(s)
        yc = self.coarse_head(s)                                    # (B, H/r, D)
        yc = F.interpolate(yc.permute(0, 2, 1), size=self.pred_len,
                           mode="linear", align_corners=False).permute(0, 2, 1)
        return yc + F.pad(y_near, (0, 0, 0, self.pred_len - y_near.size(1)))

    def _mixer(self, v: torch.Tensor) -> torch.Tensor:
        """Residual MLP over the channel axis: W_up(GELU(W_down v))."""
        u = F.gelu(self.nl_down(v))
        if getattr(self, "mix_lag", ""):
            u = self.lag_conv(u.transpose(1, 2)).transpose(1, 2)
        if self.mlp_tie:
            return F.linear(u, self.nl_down.weight.t(), self.nl_up_bias)
        out = self.nl_up(u)
        if getattr(self, "mix_listen", False):
            out = out * self.listen
        return out

    def forward(self, x: torch.Tensor, return_components: bool = False):
        """
        Args:
            x: (B, L, D) input tensor
            return_components: If True, also return intermediate band outputs

        Returns:
            y: (B, pred_len, D) prediction
            H (optional): (num_bands, B, L, D) band components after DTK
        """
        B, L, D = x.size()
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # Channel-independent mode: fold channels into the batch so every layer
        # (built at width 1) sees one univariate series. Normalization below then
        # runs per (sample, channel), which is the correct instance norm anyway.
        ci = self.channel_mode == "ci_full"
        if ci:
            x = x.permute(0, 2, 1).reshape(B * D, L, 1)

        # Instance normalization.
        # DELIBERATE asymmetry (audit P3, decided 2026-07-14): `means` is
        # detached but `stdev` is not ??gradients flow through the scale
        # statistic only. This half-RevIN variant is what every frozen OFAT
        # HP row and all trained checkpoints were selected under; detaching
        # stdev as well would change the trained function and invalidate
        # them, for no expected gain. If the paper's method section ever
        # describes this as standard RevIN, reword the paper, not this code.
        if self.use_norm:
            mode = self.revin_mode
            if mode == "mad":
                # Negative-result ablation arm (see __init__). Median center +
                # MAD scale with the Iglewicz-Hoaglin meanAD fallback (MAD is
                # exactly 0 whenever >50% of the window shares one value) and a
                # +-10 clamp backstop (near-degenerate scales still reach ~1e2).
                means = x.median(1, keepdim=True).values.detach()
                ad = (x - means).abs()
                mad = 1.4826 * ad.median(1, keepdim=True).values
                meanad = 1.253 * ad.mean(1, keepdim=True)
                stdev = torch.where(mad > 1e-6, mad, meanad) + 1e-5
                x_norm = ((x - means) / stdev).clamp(-10.0, 10.0)
            elif mode in ("last", "last_std"):
                # Level anchor = last observed step (NLinear). Detached like
                # every other centering statistic here; shape (B,1,D) so the
                # shared denorm path below broadcasts unchanged.
                means = x[:, -1:, :].detach()
                if mode == "last_std":
                    stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
                    x_norm = (x - means) / stdev
                else:
                    stdev = None
                    x_norm = x - means
            elif mode in ("winsor", "center"):
                # Bounded-influence statistics: winsorize the window (data is
                # train-Scaler standardized, so +-4 = "beyond 4 train-sigmas is
                # not a plausible sensor value"), then plain mean(/std) on the
                # clipped copy. The model consumes the clipped window too, so a
                # fault spike enters the signal path capped at the bound.
                xw = x.clamp(-self.winsor_bound, self.winsor_bound)
                if self.center_trim > 0:
                    # Symmetric trimmed mean along time: sort, drop k lowest +
                    # k highest, average the rest (k = trim fraction * L).
                    k = int(round(self.center_trim * L))
                    xs, _ = torch.sort(xw, dim=1)
                    means = xs[:, k:L - k, :].mean(1, keepdim=True).detach()
                else:
                    means = xw.mean(1, keepdim=True).detach()
                if mode == "winsor":
                    stdev = torch.sqrt(xw.var(1, keepdim=True, unbiased=False) + 1e-5)
                    x_norm = (xw - means) / stdev
                else:  # "center": no window scale ??see __init__ rationale.
                    stdev = None
                    x_norm = xw - means
            else:  # "standard" ??canonical RevIN, byte-identical.
                means = x.mean(1, keepdim=True).detach()
                stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
                x_norm = (x - means) / stdev
        else:
            means, stdev = None, None
            x_norm = x

        # Input projection
        x_proj = self.in_proj(x_norm)
        h_t = x_proj.permute(0, 2, 1)  # (B, D, L)

        # Parameter-free band splits (2026-08-27). They replace the learned conv
        # bank entirely, so they short-circuit the per-branch block below.
        if self.band_mode != "conv":
            if self.band_mode == "diff":
                raw = split_bands_diff(h_t, self.num_bands)
            else:
                raw = split_bands_fft(
                    h_t, self.num_bands,
                    scale="octave" if self.band_mode == "fftoct" else "linear")
            bands_list = [self.dense_kernel(b.permute(0, 2, 1)) for b in raw]
            return self._synthesize(x_norm, bands_list, means, stdev, B, D, ci,
                                    return_components, x_raw=x)

        # Multi-band decomposition
        # POINT band
        fP = self.conv_point(h_t).permute(0, 2, 1)
        oP = self.dense_kernel(fP)

        # HIGH band (optional)
        oH = None
        if self.num_bands == 4 and self.conv_high is not None:
            fH = self.conv_high(h_t).permute(0, 2, 1)
            oH = self.dense_kernel(fH)

        # MID band (optional)
        oM = None
        if self.num_bands >= 3 and self.conv_mid is not None:
            fM = self.conv_mid(h_t).permute(0, 2, 1)
            oM = self.dense_kernel(fM)

        # LOW band
        fL_ds = self.pool_low(h_t).permute(0, 2, 1)
        if self.far_coarse:
            self._low_ds = fL_ds              # (B, L/4, D) read by _near_far
        fL = F.interpolate(
            fL_ds.permute(0, 2, 1),
            size=self.seq_len,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        oL = self.dense_kernel(fL)

        # Collect bands according to num_bands
        if self.num_bands == 1:
            # No decomposition at all ??the POINT branch alone. This is the
            # full multi-band ablation (2 still keeps a coarse/fine split).
            bands_list = [oP]
        elif self.num_bands == 2:
            bands_list = [oP, oL]
        elif self.num_bands == 3:
            bands_list = [oP, oM, oL]
        elif self.num_bands == 4:
            bands_list = [oP, oH, oM, oL]
        else:
            raise ValueError("num_bands must be 1, 2, 3, or 4")

        return self._synthesize(x_norm, bands_list, means, stdev, B, D, ci,
                                return_components, x_raw=x)

    def _synthesize(self, x_norm, bands_list, means, stdev, B, D, ci,
                    return_components, x_raw=None):
        """Gate the bands, refine, forecast, denormalize.

        Split out so the parameter-free band modes ("diff"/"fft") reuse the
        exact same synthesis path as the canonical conv bank ??the only thing
        that differs between band modes is how bands_list was produced.
        """
        bands = torch.stack(bands_list, dim=1)  # (B, num_bands, L, D)

        # Break the band-bank collapse (no-op when band_act="none").
        bands = self.band_nl(bands)

        # Band synthesis. --gate_uniform (use_gate=False) fixes the weights at
        # 1/S = an average; gate_mode "none" instead sums the bands unweighted,
        # and "softmax" gates on the pooled input without touching the spectrum.
        if not self.use_gate:
            g = x_norm.new_full((x_norm.size(0), self.num_bands), 1.0 / self.num_bands)
        elif self.gate_mode == "none":
            g = None
        elif self.gate_mode == "softmax":
            g = torch.softmax(self.softmax_gate(x_norm.mean(dim=1)), dim=-1)
        else:
            g = self.freq_gate(x_norm)  # (B, num_bands)
            if self.use_topk_gate:
                g = sparse_topk_gate(g, k=self.topk)

        if g is None:
            hidden = bands.sum(dim=1)  # (B, L, D)
        else:
            if g.dim() == 3:  # per-sensor gate (B, D, S)
                g_expanded = g.permute(0, 2, 1).unsqueeze(2)  # (B, S, 1, D)
            else:
                g_expanded = g.view(-1, self.num_bands, 1, 1)
            hidden = (g_expanded * bands).sum(dim=1)  # (B, L, D)

        # Optional nonlinear residual refinement (channel-axis bottleneck),
        # OFF by default. nn.Linear acts on the last (channel) axis, shared
        # across the L=seq_len time steps.
        if self.use_residual_mlp:
            if self.mix_mode in ("input", "input_raw"):
                src = x_raw if self.mix_mode == "input_raw" else x_norm
                if self.mix_pool > 1:
                    src = F.avg_pool1d(src.permute(0, 2, 1), self.mix_pool,
                                       self.mix_pool).permute(0, 2, 1)
                delta = self._mixer(src)
                if self.mix_pool > 1:
                    delta = F.interpolate(delta.permute(0, 2, 1),
                                          size=self.seq_len, mode="linear",
                                          align_corners=False).permute(0, 2, 1)
            else:
                delta = self._mixer(hidden)
            if self.gate_mix == "fft":
                delta = delta * 2 * torch.sigmoid(self.mix_gain(self.freq_gate._feats)).permute(0, 2, 1)
            elif self.gate_mix == "static":
                delta = delta * 2 * torch.sigmoid(self.mix_gain.bias).view(1, 1, 1)
            hidden = hidden + delta

        # Forecasting head
        y = self.sparse_head(hidden)  # (B, pred_len, D)  [near_len if far_coarse]
        if self.far_coarse:
            y = self._near_far(y)
        if self.head2 != "none":
            g = self._head2_weight(x_norm)
            y = g * y + (1 - g) * self._head2_readout(hidden)

        # Denormalization ("center" mode has no window scale ??additive only)
        if self.use_norm:
            if stdev is not None:
                y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        # Unfold channels back: (B*D, pred_len, 1) -> (B, pred_len, D)
        if ci:
            y = y.reshape(B, D, self.pred_len).permute(0, 2, 1)

        if return_components:
            H = torch.stack(bands_list, dim=0)  # (num_bands, B, L, D)
            return y, H

        return y


