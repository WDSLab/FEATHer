import torch


def _power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """(B, L, D) -> (B, Lf): power per rfft bin, averaged over channels."""
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
           S = number of bands (1-3)
           B = batch size
           L = seq_len
           D = channels
        eps: small constant for numerical stability

    Returns:
        Scalar loss value
    """
    S, B, L, D = H.shape

    # Single-band variants have nothing to separate: return 0 rather than
    # crash on torch.stack of an empty list.
    if S < 2:
        return H.new_zeros(())

    P_list = []
    for s in range(S):
        P_s = _power_spectrum(H[s])
        P_list.append(P_s)

    # Divide frequency axis into S equal bands.
    # bands[0] = lowest rfft bins ... bands[S-1] = highest bins.
    B0, Lf = P_list[0].shape
    step = Lf // S
    bands = []
    for s in range(S):
        start = s * step
        end = Lf if s == S - 1 else (s + 1) * step
        bands.append(slice(start, end))

    def band_ratio(P, band):
        """Energy in `band` / total energy, per sample."""
        band_e = P[:, band].sum(dim=1)       # (B,)
        total_e = P.sum(dim=1) + eps         # (B,)
        return band_e / total_e              # (B,)

    main_terms = []
    overlap_terms = []

    # For each scale s: push its energy ratio in its own band toward 1 and
    # in the other bands toward 0.
    # H is stacked in FEATHer.forward's order [POINT, (MID,) LOW]: scale 0 is
    # the HIGHEST-frequency band, scale S-1 the lowest (AvgPool-smoothed).
    # Band slices run low -> high along the rfft axis, so scale s owns
    # bands[S-1-s] (not bands[s]).
    for s in range(S):
        P_s = P_list[s]
        own = S - 1 - s

        r_main = band_ratio(P_s, bands[own])
        main_terms.append((1.0 - r_main) ** 2)

        for t in range(S):
            if t == own:
                continue
            r_other = band_ratio(P_s, bands[t])
            overlap_terms.append(r_other ** 2)

    main_term = torch.stack(main_terms, dim=0).mean()
    overlap_term = torch.stack(overlap_terms, dim=0).mean()

    return main_term + overlap_term
