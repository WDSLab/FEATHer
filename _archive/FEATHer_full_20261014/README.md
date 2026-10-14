# FEATHer.py with every research option (copy of 10-14, before the cleanup)

`baselines/FEATHer/` now keeps only what the frozen model, the ablation / control / redesign arms and the
size-matched comparison use.  This copy still has the rejected options (DTK, global / per-sensor / top-k gates,
robust RevIN variants, parameter-free band splits, band nonlinearity, input-side mixer, head2, near/far head,
shared-factor head, ...) for the archived scripts (`_archive/board_legacy/`) and for reproducing old dev runs.
