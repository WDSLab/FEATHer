"""Does a model's forecast for one sensor depend on the OTHER sensors' inputs?

Replace sensor 0's input waveform (not an offset - RevIN would cancel a
constant) and measure how much the forecasts of sensors 1..D-1 move.
0 (to float precision) = the model forecasts each sensor on its own.
Random init, no training, no data: this is a structural property.

    python tools/dev/sensor_mixing_probe.py
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from baselines import get_model, get_method_defaults, get_dataset_overrides  # noqa: E402

L, D, H = 96, 7, 96
MODELS = ["SparseTSF", "MixLinear", "DiPE_Linear", "SparseTSF_MLP", "FITS", "CycleNet",
          "DLinear", "FEATHer_CI", "FEATHer_mix", "iTransformer", "PatchTST"]


def build(name):
    reg = "FEATHer" if name.startswith("FEATHer") else name
    ns = argparse.Namespace()
    # FEATHer needs its constructor args; take them from the worker's parser defaults
    if reg == "FEATHer":
        sys.argv = ["x"]
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "worker", os.path.join(ROOT, "scripts", "benchmarks", "run_forecast.py"))
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)
        sys.argv = ["x", "--model", "FEATHer", "--data", "MetroPT3", "--pred_len", "96"]
        ns = w.parse_args()
        for k, v in get_method_defaults("FEATHer").items():
            setattr(ns, k, v)
        for k, v in get_dataset_overrides("FEATHer", "MetroPT3").items():
            setattr(ns, k, v)
        ns.channel_mode = "mix" if name == "FEATHer_mix" else "ci_full"
    else:
        for k, v in get_method_defaults(reg).items():
            setattr(ns, k, v)
        for k, v in get_dataset_overrides(reg, "MetroPT3").items():
            setattr(ns, k, v)
    return get_model(reg, ns, D, L, H).float().eval()   # float32: FITS/MixLinear hold complex64 buffers


def main():
    torch.manual_seed(0)
    t = torch.arange(L, dtype=torch.float32)
    x = torch.randn(4, L, D, dtype=torch.float32).cumsum(1)
    x2 = x.clone()
    x2[:, :, 0] = 3 * torch.sin(t / 5.0) + torch.randn(4, L, dtype=torch.float32)  # new waveform
    for name in MODELS:
        try:
            m = build(name)
            with torch.no_grad():
                for p in m.parameters():            # break zero/identity inits
                    p.add_(0.05 * torch.randn_like(p))
                y1, y2 = m(x), m(x2)
            y1 = y1[0] if isinstance(y1, tuple) else y1
            y2 = y2[0] if isinstance(y2, tuple) else y2
            own = (y1[..., 0] - y2[..., 0]).abs().max().item()
            other = (y1[..., 1:] - y2[..., 1:]).abs().max().item()
            print(f"{name:14s} sensor0 moved {own:9.2e}   others moved {other:9.2e}   "
                  f"-> {'MIXES' if other > 0 else 'independent'}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:14s} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
