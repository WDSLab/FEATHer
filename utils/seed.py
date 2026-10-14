# -*- coding: utf-8 -*-
"""
Reproducibility helpers.

Single source of truth for seeding so every train run (multi-seed sweeps,
ablations, baselines) sets the same set of RNGs in the same way.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set all RNGs to `seed` and return a torch.Generator seeded to match.

    The returned generator is intended for DataLoader so that batch
    shuffling order is also reproducible across runs.

    Args:
        seed: integer seed.
        deterministic: if True, force cuDNN deterministic algorithms.
            Slightly slower but required for byte-identical reproduction.

    Returns:
        torch.Generator seeded to `seed`. Pass to DataLoader(generator=...).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    g = torch.Generator()
    g.manual_seed(seed)
    return g
