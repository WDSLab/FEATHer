# -*- coding: utf-8 -*-
"""Import a module of an upstream repository (baselines/<Name>/<Name>-main/) without name clashes.

The upstream repositories share top-level package names (layers, models, utils, ...).  isolated_import() takes those
names out of sys.modules, imports from the repository, then puts the previous ones back, so each baseline keeps its
own copies.  A new baseline with another shared top-level package needs that name in SHARED_TOPLEVEL.

    Model = isolated_import(REPO_ROOT, "models.DLinear").Model
"""
import importlib
import sys

SHARED_TOPLEVEL = {
    "layers", "models", "model", "model_", "modules", "module",
    "data_provider", "exp", "utils", "storage", "scripts",
}


def isolated_import(repo_root, modulepath):
    """The module `modulepath` imported from `repo_root` (sys.modules and sys.path restored afterwards)."""
    # 1) Save and evict the shared top-level modules
    saved = {}
    for name in list(sys.modules.keys()):
        top = name.split(".", 1)[0]
        if top in SHARED_TOPLEVEL:
            saved[name] = sys.modules.pop(name)

    sys.path.insert(0, repo_root)
    try:
        mod = importlib.import_module(modulepath)
    finally:
        sys.path.remove(repo_root)
        # 2) Drop the upstream copies registered during this import, then restore the originals
        for name in list(sys.modules.keys()):
            top = name.split(".", 1)[0]
            if top in SHARED_TOPLEVEL:
                sys.modules.pop(name, None)
        sys.modules.update(saved)
    return mod
