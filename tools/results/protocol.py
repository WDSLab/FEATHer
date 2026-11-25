"""Experiment protocol in code: datasets, the size-matched search, and what each
config_id means.

config_id                      meaning
  frozen                       FEATHer's configuration in configs/FEATHer.yaml (+ the dataset's period)
  cand:<DS> / cand:default     the model's author setting for LTSF dataset <DS> (every H) or its default, with
                               period-type keys taken from the dataset being trained
  sz<tier>:<trial>+<k>=<v>...  one tuning trial's setting (tools/results/tune.py; the picks: tuned.json)

resolve(model, group, data, H, config_id) -> (lr, loss, overrides) for the worker.
"""
import json
import os
import shutil
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import baselines as B   # noqa: E402

MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"]
HS = [96, 192, 336, 720]

# result names of FEATHer; all train the registry model "FEATHer"
FEATHER_CORE = "FEATHer_core"                   # the main model: every part on, size / period / lr / lambda / RevIN searched
FEATHER_CI = "FEATHer_CI"                       # LTSF only: the parts searched too

# period-type keys: on the industrial data they always come from the industrial dataset's own entry
PERIOD_KEYS = {"period", "sparsetsf_period_len", "mixlinear_period_len",
               "timebase_period_len", "phaseformer_period_len", "cmos_period", "fits_base_t"}

# ---- experiment definitions: configs/experiments/*.yaml (the YAML holds the comments) ----------------
def _experiments(name):
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "experiments", f"{name}.yaml"), encoding="utf-8"))


_size = _experiments("size")
SIZE_TIERS, SIZE_COMMON = _size["tiers"], _size["common"]
LTSF_TIERS = ("ltsf", "ltsf1k")                  # chip rule / 1K rule on the LTSF datasets
TUNE = _size["tune"]                             # trials / seed / budget / pruner / max_rejects of the Optuna studies
FINAL = _size["final"]                           # epochs / patience / seeds of the final runs
SIZE_ABLATION = _size.get("ablation") or {}     # arm -> {key: value} set on the main model's tuned pick
SIZE_ABLATION_PLACES = _size.get("ablation_places") or []   # "<tier>|<data>" the arms run on (empty = every place)
SIZE_SPACE = _size["models"]                     # model -> {key: size / segment / period / switch / dropout / mix / [list] / unused}
SIZE_MODELS = list(SIZE_SPACE)
CHOSEN_FILE = os.path.join(ROOT, "configs", "experiments", "chosen_mfg.json")


def tune_models(tier):
    """Models tuned in a tier: on LTSF FEATHer is FEATHer_CI (its parts searched, one study); on the industrial
    tiers FEATHer is FEATHer_core.  SparseTSF is one study with its head (linear / MLP) searched, every tier."""
    out = _size.get("tier_excluded", {}).get(tier, [])  # cannot meet the tier's cap at some H (size.yaml)
    if tier in LTSF_TIERS:
        return [m for m in SIZE_MODELS if m not in (FEATHER_CORE, *out)]
    return [m for m in SIZE_MODELS if m not in (FEATHER_CI, *out)]


def size_data(tier):
    """Datasets of a size tier: the industrial six, or the LTSF five (size.yaml ltsf_data)."""
    return _size["ltsf_data"] if tier in LTSF_TIERS else MFG


def tier_group(tier):
    """"ltsf" or "mfg": the result files and the settings a tier's runs resolve to."""
    return "ltsf" if tier in LTSF_TIERS else "mfg"


def size_config_id(tier, trial_id, knobs):
    """"sz<tier>:<trial id>+<key>=<value>..." from the trial's knobs on one dataset."""
    return f"sz{tier}:{trial_id}" + "".join(f"+{k}={knobs[k]}" for k in sorted(knobs))


def _parse_value(v):
    """config_id value -> bool / int / float / str."""
    if v in ("True", "False"):
        return v == "True"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


TUNED_FILE = os.path.join(ROOT, "configs", "experiments", "tuned.json")


def tuned():
    """"<tier>|<model>|<data>|<H>" -> {"trial": "t07", "knobs": {key: value, "lr": ...}}: the pick of every tuning
    study (lowest validation MSE; written by the tune runner).  Empty until tuned."""
    return _load(TUNED_FILE)


def tuned_pick(tier, model, data, H):
    """The pick of one study, or None while not tuned."""
    return tuned().get(f"{tier}|{model}|{data}|{H}")


def _load(path):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def chosen_mfg():
    """model -> config_id chosen on the industrial validation loss (written by the selection verdict)."""
    return _load(CHOSEN_FILE)


def registry_name(model):
    return "FEATHer" if model.startswith("FEATHer") else model


def _merge(model, data, H):
    m = registry_name(model)
    return {**B.get_method_defaults(m), **B.get_dataset_overrides(m, data, H)}


def resolve(model, group, data, H, config_id):
    """Worker settings for one cell: (lr, loss, overrides dict)."""
    m = registry_name(model)
    if config_id.startswith("sz"):
        # size-matched trial: the model's industrial setting with every searched key of the trial set, lr included
        lr, loss, ov = resolve(model, group, data, H, mfg_main_config_id(model))
        for kv in config_id.split(":", 1)[1].split("+")[1:]:
            k, v = kv.split("=", 1)
            if k == "lr":
                lr = _parse_value(v)
            else:
                ov[k] = _parse_value(v)
        return lr, loss, ov
    elif config_id == "frozen":
        s = _merge(model, data, H)
        if model in (FEATHER_CI, FEATHER_CORE) and group == "mfg":
            s["channel_mode"] = "ci_full"
    elif config_id.startswith("cand:"):
        # the model's setting for LTSF dataset <src> (every H); period-type keys always come from the dataset
        # being trained (industrial: FFT / daily-cycle rule; LTSF: that dataset's entry).  "cand:<src>+<k>=<v>..."
        # sets keys on top (the chip builds of deployment/board, tune_budget.firmware)
        src, *kv = config_id[5:].split("+")
        s = dict(B.get_method_defaults(m))
        if src != "default":
            s.update({k: v for k, v in B.get_dataset_overrides(m, src, H).items() if k not in PERIOD_KEYS})
        s.update({k: v for k, v in B.get_dataset_overrides(m, data).items() if k in PERIOD_KEYS})
        s.update({k: _parse_value(v) for k, v in (x.split("=", 1) for x in kv)})
    else:
        raise ValueError(config_id)
    lr = s.pop("lr", 1e-3)
    loss = s.pop("loss", "mse")
    return lr, loss, s


def train_budget():
    """(epochs, patience) of every run.py stage (size.yaml final; the tuning budget is tune.py's own)."""
    return int(FINAL["epochs"]), int(FINAL["patience"])


def hparams(model, group, data, H, config_id):
    """Every setting of one cell as a JSON line (deployment/board/replay/replay_weights.py rebuilds the worker from it)."""
    lr, loss, ov = resolve(model, group, data, H, config_id)
    run = {"model": registry_name(model), "lr": lr, "loss": loss, "settings": ov}
    return json.dumps(run, sort_keys=True, separators=(",", ":"))


def mfg_main_config_id(model):
    """config_id of the main industrial runs, or None while a model's setting is not chosen yet."""
    if model.startswith("FEATHer"):
        return "frozen"
    return chosen_mfg().get(model)


# ---- shared helpers for result files ---------------------------------------------------------------
# One row per run (scripts/train.py writes it, merge_incoming.py only appends):
#   stage   tune (one row per trained tuning trial, tools/results/tune.py) / final (tuned pick, size.yaml final seeds)
#           / ablation
#   tier    1k / chip / ltsf / ltsf1k (size-matched comparison)
#   trial   t01..t40 (the tuning trial)
#   arm     tune: the trial's outcome (done / pruned / diverged); ablation: the arm (size.yaml ablation); "-" otherwise
#   settings  the searched values the run used, "lr=... key=value ..." (model prefix dropped)
RESULT_FIELDS = ["stage", "tier", "trial", "arm", "model", "data", "H", "seed", "MSE", "MAE", "val_loss", "best_epoch",
                 "params", "host", "time", "settings"]
KEY = ["stage", "tier", "trial", "arm", "model", "data", "H", "seed", "settings"]   # one run; a change -> runs again
RESULT_FILES = ["mfg_tune.csv", "mfg_final.csv", "mfg_ablation.csv", "ltsf_tune.csv", "ltsf_final.csv"]
STAGE = {"final": "final", "ablation": "ablation"}  # role -> stage
_PREFIX = ("sparsetsf_", "mixlinear_", "cmos_", "timebase_", "phaseformer_", "fits_")


def short_key(k):
    """mixlinear_alpha -> alpha (each model's keys share one prefix; FEATHer keys have none)."""
    return next((k[len(p):] for p in _PREFIX if k.startswith(p)), k)


def settings_str(knobs):
    """"lr=0.005 alpha=0.1 lpf=4 period_len=6": lr first, then the keys in order, values written as str()."""
    ks = sorted(knobs, key=lambda k: (k != "lr", short_key(k)))
    names = [short_key(k) for k in ks]
    assert len(set(names)) == len(names), names
    return " ".join(f"{n}={knobs[k]}" for n, k in zip(names, ks))


def labels(role, config_id, model, data, arm="-"):
    """Stored fields of one planned cell: dict(stage, tier, trial, arm, settings)."""
    tier, rest = config_id[2:].split(":", 1)
    trial, *kv = rest.split("+")
    knobs = dict(x.split("=", 1) for x in kv)          # values kept as written (str of the trial's value)
    return dict(stage=STAGE[role], tier=tier, trial=trial, arm=arm, settings=settings_str(knobs))


def read_results(path):
    """One result file or shard as RESULT_FIELDS."""
    d = pd.read_csv(path, on_bad_lines="skip", dtype={"settings": str, "trial": str, "arm": str, "tier": str})
    if "exp_tag" in d or "config_id" in d:
        raise SystemExit(f"{path}: written by an older version of the code; not merged")
    return d


def write_csv_safely(df, path):
    """Write a managed result file atomically: temp file -> os.replace, previous version kept as <path>.bak."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    os.replace(tmp, path)
