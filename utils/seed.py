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


def parse_seed_list(seeds_arg: str, num_seeds: int, base_seed: int) -> list:
    """
    Build a list of seeds from CLI args.

    Priority:
        1. If `seeds_arg` is non-empty (e.g. "2025,2026,2027"), use it.
        2. Otherwise generate `num_seeds` consecutive seeds starting at
           `base_seed`.

    Args:
        seeds_arg: comma-separated seeds string, or "" to disable.
        num_seeds: number of seeds to generate when seeds_arg is empty.
        base_seed: starting seed for auto-generation.

    Returns:
        List of int seeds.
    """
    if seeds_arg and seeds_arg.strip():
        return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    return [base_seed + i for i in range(num_seeds)]
