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

    "global" (default, canonical — byte-identical to every frozen checkpoint)
        magnitude spectrum -> Conv1d -> AdaptiveAvgPool1d(1) -> softmax.

        ⚠ MEASURED DEFECT (2026-08-14): this gate is provably BLIND to
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
        makes no difference, num_bands is flat, and lambda_spec is flat —
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
        vs the conv's 5S + S (24) — slightly FEWER.

    Args:
        seq_len: Input sequence length
        num_bands: Number of frequency bands (2, 3, or 4)
        kernel_size: Kernel size for Conv1d ("global" mode only)
        gate_mode: "global" (canonical) or "banded" (shape-aware fix)
    """

    def __init__(self, seq_len: int, num_bands: int = 3, kernel_size: int = 5,
                 gate_mode: str = "global"):
        super().__init__()
        assert gate_mode in ("global", "banded"), gate_mode
        self.seq_len = seq_len
        self.num_bands = num_bands
        self.gate_mode = gate_mode

        if gate_mode == "banded":
            # Precompute the equal-width rfft slices (low -> high), same rule
            # as utils/losses.spectral_separation_loss_scales.
            n_freq = seq_len // 2 + 1
            step = n_freq // num_bands
            self._bounds = [
                (s * step, n_freq if s == num_bands - 1 else (s + 1) * step)
                for s in range(num_bands)
            ]
            self.proj = nn.Linear(num_bands, num_bands)
            # Zero-init => the gate starts EXACTLY uniform (1/S per band),
            # i.e. at the fixed-weight behaviour the "global" gate is stuck
            # at, and learns to deviate from there. Two reasons: (a) random
            # init saturates the softmax immediately (log-ratios of a nearly
            # empty band reach log(1e-6) ~ -14, giving one-hot gates before
            # any training), (b) it makes the comparison controlled — any
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor

        Returns:
            (B, num_bands) softmax-normalized gate weights
        """
        B, L, D = x.shape

        # Convert to float32 to avoid cuFFT errors in autocast(fp16)
        x32 = x.float()
        Freq = torch.fft.rfft(x32, dim=1)
        mag = torch.abs(Freq)

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
            # class unchanged — it only keeps the linear layer's inputs from
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

    def __init__(self, d_model: int, d_state: int, kernel_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size

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

        # Project back: (B, S, L) -> (B, L, D)
        y = self.out_proj(y.permute(0, 2, 1))
        return y


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
    """

    def __init__(self, seq_len: int, pred_len: int, d_model: int, period: int):
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

        # Cross-period backbone
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
    ):
        super().__init__()

        assert num_bands in [2, 3, 4], "num_bands must be 2, 3, or 4"

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
        # — used to isolate whether the gate's reaction to corrupted spectra is
        # the source of FEATHer's robustness fragility. ON by default (canonical).
        self.use_gate = use_gate

        # Instance-norm variant selector (2026-08-01/02). All variants add ZERO
        # parameters (statistics only) — "standard" is byte-identical canonical.
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
        #               gracefully like the mean — no breakdown jump).
        #   "center"    winsorized-mean CENTERING ONLY — no window scale at all
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
        #               per-dataset, val-selected option — never a new default.
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

        # Input projection
        self.in_proj = nn.Linear(eff_d, eff_d)

        # Optional nonlinear residual MLP (channel-axis low-rank bottleneck),
        # applied to the gated representation just before the forecasting head:
        #     h <- h + W_up(GELU(W_down(h))),  W_down: D->r, W_up: r->D
        # Adds ~2*D*r params (D=7,r=16 -> +247). OFF by default so the canonical
        # model and every frozen checkpoint are byte-identical; enabled only for
        # the capacity A/B experiment (2026-07-31). This is FEATHer's first
        # pointwise nonlinearity in the signal path (otherwise linear+gating).
        if use_residual_mlp:
            self.nl_down = nn.Linear(eff_d, mlp_rank)
            self.nl_up = nn.Linear(mlp_rank, eff_d)

        # Multi-band 1D Conv decomposition
        # POINT: kernel=1, captures high-frequency details
        self.conv_point = nn.Conv1d(
            eff_d, eff_d, kernel_size=1, padding=0, groups=eff_d
        )

        # HIGH: kernel=3 (only when num_bands == 4)
        if num_bands == 4:
            self.conv_high = nn.Conv1d(
                eff_d, eff_d, kernel_size=3, padding=1, groups=eff_d
            )
        else:
            self.conv_high = None

        # MID: kernel=5 (when num_bands >= 3)
        if num_bands >= 3:
            self.conv_mid = nn.Conv1d(
                eff_d, eff_d, kernel_size=5, padding=2, groups=eff_d
            )
        else:
            self.conv_mid = None

        # LOW: avg pooling (L -> L/4) then upsample
        self.pool_low = nn.AvgPool1d(kernel_size=4, stride=4)

        # Shared temporal kernel
        self.dense_kernel = DenseTemporalKernel(
            d_model=eff_d,
            d_state=d_state,
            kernel_size=kernel_size
        )

        # FFT-based frequency gate
        self.freq_gate = FFTFrequencyGate(
            seq_len=seq_len,
            num_bands=num_bands,
            kernel_size=5,
            gate_mode=gate_mode,
        )

        # Sparse period-aware forecasting head
        self.sparse_head = SparsePeriodKernel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=eff_d,
            period=period,
        )

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
        # detached but `stdev` is not — gradients flow through the scale
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
                else:  # "center": no window scale — see __init__ rationale.
                    stdev = None
                    x_norm = xw - means
            else:  # "standard" — canonical RevIN, byte-identical.
                means = x.mean(1, keepdim=True).detach()
                stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
                x_norm = (x - means) / stdev
        else:
            means, stdev = None, None
            x_norm = x

        # Input projection
        x_proj = self.in_proj(x_norm)
        h_t = x_proj.permute(0, 2, 1)  # (B, D, L)

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
        fL = F.interpolate(
            fL_ds.permute(0, 2, 1),
            size=self.seq_len,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1)
        oL = self.dense_kernel(fL)

        # Collect bands according to num_bands
        if self.num_bands == 2:
            bands_list = [oP, oL]
        elif self.num_bands == 3:
            bands_list = [oP, oM, oL]
        elif self.num_bands == 4:
            bands_list = [oP, oH, oM, oL]
        else:
            raise ValueError("num_bands must be 2, 3, or 4")

        bands = torch.stack(bands_list, dim=1)  # (B, num_bands, L, D)

        # FFT-based gate (or fixed uniform weights when disabled — diagnostic)
        if self.use_gate:
            g = self.freq_gate(x_norm)  # (B, num_bands)
            if self.use_topk_gate:
                g = sparse_topk_gate(g, k=self.topk)
        else:
            g = x_norm.new_full((x_norm.size(0), self.num_bands), 1.0 / self.num_bands)

        g_expanded = g.view(-1, self.num_bands, 1, 1)
        hidden = (g_expanded * bands).sum(dim=1)  # (B, L, D)

        # Optional nonlinear residual refinement (channel-axis bottleneck),
        # OFF by default. nn.Linear acts on the last (channel) axis, shared
        # across the L=seq_len time steps.
        if self.use_residual_mlp:
            hidden = hidden + self.nl_up(F.gelu(self.nl_down(hidden)))

        # Forecasting head
        y = self.sparse_head(hidden)  # (B, pred_len, D)

        # Denormalization ("center" mode has no window scale — additive only)
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
