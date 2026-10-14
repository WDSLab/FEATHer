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

# name: (folder, build function in <folder>/wrapper.py, extra keyword arguments for that function)
MODELS = {
    "FEATHer":            ("FEATHer", "build", {}),
    # lightweight baselines
    "SparseTSF":          ("SparseTSF", "build", {}),
    "SparseTSF_MLP":      ("SparseTSF", "build_mlp", {}),
    "DLinear":            ("DLinear", "build", {}),
    "FITS":               ("FITS", "build", {}),
    "CycleNet":           ("CycleNet", "build", {}),
    "DiPE_Linear":        ("DiPE_Linear", "build", {}),
    "MixLinear":          ("MixLinear", "build", {}),
    "CMoS":               ("CMoS", "build", {}),
    "TimeBase":           ("TimeBase", "build", {}),
    "PhaseFormer":        ("PhaseFormer", "build", {}),
    "SAMformer":          ("SAMformer", "build", {}),
    # server-class reference models
    "PatchTST":           ("PatchTST", "build", {}),
    "iTransformer":       ("iTransformer", "build", {}),
    "TimeMixer":          ("TimeMixer", "build", {}),
    "TimesNet":           ("TimesNet", "build", {}),
    "TQNet":              ("TQNet", "build", {}),
    "LMS_AutoTSF":        ("LMS_AutoTSF", "build", {}),
    "MDMLP_EIA":          ("MDMLP_EIA", "build", {}),
    # control arms: SparseTSF + FEATHer's mixing bus only
    "SparseTSF_mix":      ("SparseTSF", "build_mix", {"model_type": "linear", "norm": "revin"}),
    "SparseTSF_mixM":     ("SparseTSF", "build_mix", {"model_type": "linear", "norm": "mean"}),
    "SparseTSF_MLP_mix":  ("SparseTSF", "build_mix", {"model_type": "mlp", "norm": "revin"}),
    "SparseTSF_MLP_mixM": ("SparseTSF", "build_mix", {"model_type": "mlp", "norm": "mean"}),
}
# models that use another model's configs/<name>.yaml (the control arms run with SparseTSF's settings)
SHARED_CONFIG = {"SparseTSF_mix": "SparseTSF", "SparseTSF_mixM": "SparseTSF",
                 "SparseTSF_MLP_mix": "SparseTSF", "SparseTSF_MLP_mixM": "SparseTSF"}


def get_model(name, args, n_features, seq_len, pred_len):
    """Build the registered model <name> from the worker's args."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODELS))}")
    folder, function, extra = MODELS[name]
    build = getattr(import_module(f"baselines.{folder}.wrapper"), function)
    return build(args, n_features, seq_len, pred_len, **extra)


@lru_cache(maxsize=None)
def _config(name):
    path = os.path.join(CONFIG_DIR, f"{SHARED_CONFIG.get(name, name)}.yaml")
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
