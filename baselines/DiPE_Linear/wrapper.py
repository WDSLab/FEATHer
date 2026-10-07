# -*- coding: utf-8 -*-
"""DiPE-Linear (DASFAA 2026) wrapper.

Reference:
    arXiv:2411.17257 — "Disentangled Parameter-Efficient Linear Model for
    Long-Term Time Series Forecasting" (wintertee/DiPE-Linear, DASFAA 2026).

Defaults from `DiPE_Linear-main/configs/models/DiPE/base.yaml` (lr=1e-3,
50 epochs, MSE loss) and `timeprophet/models/DiPE.py` constructor:
use_revin=True, use_time_w=True, use_freq_w=True, num_experts=1.

The DiPE module is imported under `timeprophet.models.DiPE` and exposes a
class `DiPE` (not `Model`). forward(x) returns (B, L, D) directly.

Requires `einops` (already in the feather env via the iTransformer install).
"""

import os
import torch
import torch.nn as nn

from baselines._import_helper import isolated_import


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "DiPE_Linear-main")


def build(args, n_features, seq_len, pred_len):
    mod = isolated_import(_REPO, "timeprophet.models.DiPE")
    DiPE = mod.DiPE
    num_experts = int(getattr(args, "dipe_num_experts", 1))
    loss_alpha = float(getattr(args, "dipe_loss_alpha", 0.0))
    model = DiPE(
        input_len=seq_len,
        output_len=pred_len,
        input_features=n_features,
        output_features=n_features,
        # base.yaml + paper §3: instance norm wrapper on (scripts/DiPE.sh turns it off for
        # Weather / Electricity)
        use_revin=bool(getattr(args, "dipe_use_revin", True)),
        # Static frequency / time weights — paper main variant
        use_time_w=getattr(args, "dipe_use_time_w", True),
        use_freq_w=getattr(args, "dipe_use_freq_w", True),
        # 2026-10-07: experts / loss_alpha per dataset as in scripts/DiPE.sh (was fixed 1 / 0.0)
        num_experts=num_experts,
        individual_f=getattr(args, "dipe_individual_f", False),
        individual_t=getattr(args, "dipe_individual_t", False),
        individual_c=getattr(args, "dipe_individual_c", False),
        loss_alpha=loss_alpha,
        t_loss="mse",
    )
    # The upstream recipe lives outside the module (PyTorch Lightning): the loss is
    # model.loss() = (1 - alpha) * time MSE + alpha * L1 on the weighted rFFT, and with
    # experts > 1 the TemperatureScaling callback anneals the router temperature 30 -> 1 over
    # epochs 0..10 and uses 1 at test (timeprophet/utils/callbacks.py:31-51).  The worker calls
    # these hooks; with the defaults (1 expert, alpha 0) none is attached, so the model trains
    # exactly as before.
    if loss_alpha > 0:
        def _train_loss(output, target, main_loss, m=model, a=loss_alpha):
            # observed-only scoring: imputed target cells (NaN) get zero error in the frequency
            # term too (filled with the prediction), matching the masked time loss
            y = torch.where(torch.isfinite(target), target, output.detach())
            freq = m.loss(y, output) - (1 - a) * torch.nn.functional.mse_loss(y, output)
            return (1 - a) * main_loss + freq
        model._train_loss = _train_loss
    if num_experts > 1:
        def _epoch_start(epoch, m=model, t_max=30.0, t_min=1.0, max_epoch=10):
            m.temperature = t_min if epoch > max_epoch else t_max - (t_max - t_min) * epoch / max_epoch
        def _test_start(m=model):
            m.temperature = 1.0
        model._epoch_start = _epoch_start
        model._test_start = _test_start
    return model
