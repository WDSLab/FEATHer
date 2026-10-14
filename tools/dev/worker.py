"""Load scripts/train.py as a module (its data loaders and helpers) - used by deployment/board/replay.py and
tools/dev/sensor_dropout.py."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_worker():
    path = os.path.join(REPO, "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("fcst_worker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
