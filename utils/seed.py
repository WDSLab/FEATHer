# -*- coding: utf-8 -*-
"""One place that seeds every random number generator of a run."""
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """Seed Python, NumPy and torch (CPU + CUDA); deterministic=True also fixes the cuDNN algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
