# -*- coding: utf-8 -*-
"""Model registry: which code builds each model, and where its hyperparameters live.

  baselines/<Folder>/wrapper.py   builds the model (upstream code in <Folder>-main/, fetched by setup_baselines.sh)
  configs/<Model>.yaml            the model's hyperparameters: default (lr, loss) and per dataset / per horizon

To add a model: write its wrapper, add one line to MODELS, write configs/<Model>.yaml.
"""
import os
from functools import lru_cache
from importlib import import_module

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# name: (folder, build function in <folder>/wrapper.py)
MODELS = {
    "FEATHer":            ("FEATHer", "build"),
    # lightweight baselines
    "SparseTSF":          ("SparseTSF", "build"),
    "SparseTSF_MLP":      ("SparseTSF", "build_mlp"),
    "FITS":               ("FITS", "build"),
    "MixLinear":          ("MixLinear", "build"),
    "CMoS":               ("CMoS", "build"),
    "TimeBase":           ("TimeBase", "build"),
    "PhaseFormer":        ("PhaseFormer", "build"),
}


def get_model(name, args, n_features, seq_len, pred_len):
    """Build the registered model <name> from the worker's args."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODELS))}")
    folder, function = MODELS[name]
    return getattr(import_module(f"baselines.{folder}.wrapper"), function)(args, n_features, seq_len, pred_len)


@lru_cache(maxsize=None)
def _config(name):
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        return {}
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


def get_method_defaults(name):
    """The model's default training settings (lr, loss)."""
    return dict(_config(name).get("default", {}))


def get_dataset_overrides(name, data, pred_len=None):
    """The model's hyperparameters for <data>; with pred_len, the "<data>@<H>" row is merged on top."""
    rows = _config(name).get("datasets", {})
    out = dict(rows.get(data, {}))
    if pred_len is not None:
        out.update(rows.get(f"{data}@{int(pred_len)}", {}))
    return out


def list_models():
    return sorted(MODELS)
