# -*- coding: utf-8 -*-
"""
Isolated import helper for baseline upstream repos.

Each official repository under `baselines/<Name>/<Name>-main/` uses its own
internal layout (e.g. `from layers.Embed import ...`). Importing two
baselines into the same Python process therefore collides on common module
names (`layers`, `models`, `model`, etc.).

`isolated_import(repo_root, modulepath)` pushes `repo_root` onto sys.path,
imports the requested module, then snapshots and removes any sys.modules
entries that the import introduced under shared top-level names — leaving
the path clean for the next baseline.

Use it inside each wrapper.py:

    from baselines._import_helper import isolated_import
    Model = isolated_import(REPO_ROOT, "models.DLinear").Model
"""

import importlib
import os
import sys


SHARED_TOPLEVEL = {
    "layers", "models", "model", "model_", "modules", "module",
    "data_provider", "exp", "utils", "storage", "scripts",
}


def isolated_import(repo_root, modulepath):
    """
    Import `modulepath` from `repo_root` with full sys.modules isolation.

    Many upstream repos contain shared top-level package names (`utils`,
    `layers`, `models`, ...). Importing one baseline pollutes sys.modules
    with that baseline's version of those names; a subsequent baseline that
    needs its own copy then resolves to the wrong package — sometimes
    silently, sometimes with cryptic ModuleNotFoundErrors like
    "No module named 'utils.masking'" (our `utils/` doesn't have masking;
    the baseline's does).

    Strategy:
      1. Snapshot sys.modules entries that could conflict, then evict them.
      2. Push `repo_root` to sys.path[0] and import normally — Python now
         resolves shared names against the upstream copies.
      3. Restore the snapshot, replacing the upstream copies. The returned
         model class still holds direct Python references to its internal
         layers (resolved at import time), so subsequent calls keep working.

    Args:
        repo_root: absolute path that contains `layers/`, `models/`, etc.
        modulepath: dotted Python path inside that root, e.g.
            "model.iTransformer".

    Returns:
        The imported module object.
    """
    # 1) Save & evict shared top-level modules
    saved = {}
    for name in list(sys.modules.keys()):
        top = name.split(".", 1)[0]
        if top in SHARED_TOPLEVEL:
            saved[name] = sys.modules.pop(name)

    sys.path.insert(0, repo_root)
    try:
        mod = importlib.import_module(modulepath)
    finally:
        try:
            sys.path.remove(repo_root)
        except ValueError:
            pass
        # 2) Drop upstream copies that may have been registered during this
        #    import, then restore the originals.
        for name in list(sys.modules.keys()):
            top = name.split(".", 1)[0]
            if top in SHARED_TOPLEVEL:
                sys.modules.pop(name, None)
        sys.modules.update(saved)
    return mod
