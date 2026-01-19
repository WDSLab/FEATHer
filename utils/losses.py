import torch


def _power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """
    Compute power spectrum averaged over features.

    Args:
        x: (B, L, D) input tensor

    Returns:
        (B, Lf) average power per frequency bin
    """
    x32 = x.float()
    Xf = torch.fft.rfft(x32, dim=1)   # (B, Lf, D)
    P = (Xf.abs() ** 2).mean(dim=2)   # (B, Lf)
    return P


def spectral_separation_loss_scales(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Spectral separation loss for multi-scale decomposition.

    Encourages each scale to focus on its designated frequency band.

    Args:
        H: (S, B, L, D) tensor where:
           S = num_scales (e.g., 3 or 4)
           B = batch size
           L = seq_len
           D = feature dim
        eps: small constant for numerical stability

    Returns:
        Scalar loss value
    """
    S, B, L, D = H.shape

    # Compute power spectrum for each scale: (B, L, D) -> (B, Lf)
    P_list = []
    for s in range(S):
        P_s = _power_spectrum(H[s])
        P_list.append(P_s)

    # Divide frequency axis into S equal bands
    B0, Lf = P_list[0].shape
    step = Lf // S
    bands = []
    for s in range(S):
        start = s * step
        end = Lf if s == S - 1 else (s + 1) * step
        bands.append(slice(start, end))

    def band_ratio(P, band):
        """Compute ratio of energy in band to total energy."""
        band_e = P[:, band].sum(dim=1)       # (B,)
        total_e = P.sum(dim=1) + eps         # (B,)
        return band_e / total_e              # (B,)

    main_terms = []
    overlap_terms = []

    # For each scale s:
    #   - Maximize energy ratio in its own band (close to 1)
    #   - Minimize energy ratio in other bands (close to 0)
    for s in range(S):
        P_s = P_list[s]

        # Energy ratio in own frequency band
        r_main = band_ratio(P_s, bands[s])
        main_terms.append((1.0 - r_main) ** 2)

        # Energy ratio in other bands (should be minimal)
        for t in range(S):
            if t == s:
                continue
            r_other = band_ratio(P_s, bands[t])
            overlap_terms.append(r_other ** 2)

    main_term = torch.stack(main_terms, dim=0).mean()
    overlap_term = torch.stack(overlap_terms, dim=0).mean()

    return main_term + overlap_term
