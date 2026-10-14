# Lazy submodule loading (PEP 562): an eager `from . import ...` would make any
# `import utils.<x>` pull torch transitively. `from utils import data_factory`
# and `utils.losses.foo` both still work.
import importlib

_SUBMODULES = {
    "data_factory", "data_loader", "dispatch_queue", "losses", "metrics",
    "seed", "timefeatures",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
