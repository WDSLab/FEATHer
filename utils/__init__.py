# Lazy submodule loading (PEP 562).
#
# AUDIT-FIX (2026-07-14, P3): eager `from . import data_factory, losses, ...`
# made ANY `import utils.<x>` pull torch (and matplotlib) transitively, which
# defeated the point of the torch-free utils/severity.py that
# tools/paper/robust_summary.py depends on. Submodules now import on first
# attribute access; `from utils import data_factory` and `utils.losses.foo`
# both keep working unchanged.
import importlib

_SUBMODULES = {
    "data_factory", "data_loader", "dispatch_queue", "losses", "metrics",
    "noise", "seed", "severity", "timefeatures",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
