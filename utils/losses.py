import torch


def spectral_separation_loss_scales(H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FEATHer's band-separation term.  H: (S, B, L, D), the S band outputs in FEATHer.forward's order [POINT, (MID,)
    LOW] (fastest first).  The rfft axis is cut into S equal slices, low -> high, so band s owns slice S-1-s.  Per band:
    (1 - its energy share in its own slice)^2 averaged over the bands, plus (its share in each other slice)^2 averaged
    over all (band, other slice) pairs.  One band has nothing to separate -> 0."""
    S = H.shape[0]
    if S < 2:
        return H.new_zeros(())
    P = [(torch.fft.rfft(H[s].float(), dim=1).abs() ** 2).mean(dim=2) for s in range(S)]   # per band: (B, Lf) power
    Lf = P[0].shape[1]
    step = Lf // S
    cuts = [slice(s * step, Lf if s == S - 1 else (s + 1) * step) for s in range(S)]
    share = lambda s, t: P[s][:, cuts[t]].sum(dim=1) / (P[s].sum(dim=1) + eps)
    main, overlap = [], []
    for s in range(S):
        own = S - 1 - s
        main.append((1.0 - share(s, own)) ** 2)
        overlap += [share(s, t) ** 2 for t in range(S) if t != own]
    return torch.stack(main).mean() + torch.stack(overlap).mean()
