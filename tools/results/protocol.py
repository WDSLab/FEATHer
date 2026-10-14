"""Experiment protocol in code (docs/EXPERIMENT_PROTOCOL.md): which settings exist, and what each config_id means.

config_id                      meaning
  frozen                       FEATHer frozen configuration (+ the dataset's period), every group
  official / official-exc      LTSF: the model's per-(dataset, H) setting in configs/<model>.yaml
                               (official scripts; -exc = DLinear ETTh2, re-selected after divergence)
  ref-default                  server-class reference models on the industrial data: defaults
  cand:<DS>                    industrial candidate = the model's LTSF setting for <DS> (every H),
                               with period-type keys replaced by the industrial dataset's own value
  cand:default                 industrial candidate = the default setting the authors present
  ctrl:<arm>                   control arms (SparseTSF + mixing bus, ...)
  sz<tier>:<trial>+<k>=<v>...  size-matched search trial (docs/size_trials.json) on one dataset

resolve(model, group, data, H, config_id) -> (lr, loss, overrides) for the worker.
"""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import baselines as B   # noqa: E402

MFG = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange", "Electricity", "Traffic"]
HS = [96, 192, 336, 720]
SEEDS = [2025, 2026, 2027]
SELECT_SEED = 2025

HEAVY = ["PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet", "iTransformer", "MDMLP_EIA"]
FEATHER = ["FEATHer_mix", "FEATHer_CI"]          # result names; both train the registry model "FEATHer"
LIGHT = ["SparseTSF", "SparseTSF_MLP", "DLinear", "CycleNet", "DiPE_Linear", "MixLinear", "FITS",
         "CMoS", "TimeBase", "PhaseFormer", "SAMformer"]
ON_CHIP = ["SparseTSF", "SparseTSF_MLP", "MixLinear", "DiPE_Linear", "FITS", "CMoS", "TimeBase", "PhaseFormer"]

# period-type keys: on the industrial data they always come from the industrial dataset's own entry
PERIOD_KEYS = {"period", "sparsetsf_period_len", "mixlinear_period_len", "cyclenet_cycle", "tqnet_cycle",
               "timebase_period_len", "phaseformer_period_len", "cmos_period", "fits_base_t"}

# Industrial candidates (docs/EXPERIMENT_PROTOCOL.md §5).  Distinct settings only; datasets whose setting
# equals another's are listed once.  No chip filter on candidates: every
# model is selected on accuracy alone, and a chosen setting that exceeds the chip in a cell is reported
# as "does not fit" in that cell, the same treatment as DLinear / CycleNet / SAMformer.
CANDIDATES = {
    "SparseTSF": ["ETTh1", "ETTh2"],                                   # lr 0.02 / 0.03
    "SparseTSF_MLP": ["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"],
    "DLinear": ["ETTh1", "ETTm1", "ETTm2", "Exchange", "Electricity", "Traffic"],
    "CycleNet": ["ETTh1"],                                             # one setting (L96 scripts)
    "DiPE_Linear": ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "default"],
    "MixLinear": ["ETTh1", "ETTh2", "Electricity", "Traffic"],
    "FITS": ["ETTh1", "ETTm1", "Weather"],
    "CMoS": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "default"],
    "TimeBase": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"],
    "PhaseFormer": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"],
    "SAMformer": ["ETTh1", "Weather", "default"],
}

# LTSF cells whose authors left no final setting (docs/EXPERIMENT_PROTOCOL.md §5.3): no script for the
# dataset, or only a sweep script.  These are chosen like the industrial data: the model's published
# settings (CANDIDATES) compete on that dataset's validation loss.  Everything else on LTSF = "official".
LTSF_NO_OFFICIAL = {
    ("SparseTSF", "Exchange"), ("SparseTSF_MLP", "Exchange"), ("CycleNet", "Exchange"),
    ("DiPE_Linear", "Exchange"), ("DiPE_Linear", "Traffic"),            # DiPE.sh has no Traffic run line
    ("MixLinear", "Exchange"), ("MixLinear", "ETTm1"), ("MixLinear", "ETTm2"), ("MixLinear", "Weather"),
    ("FITS", "Exchange"), ("FITS", "Electricity"), ("FITS", "Traffic"),
    ("CMoS", "Exchange"), ("TimeBase", "Exchange"), ("PhaseFormer", "Exchange"),
}

# ---- experiment definitions: configs/experiments/*.yaml (the YAML holds the comments) ----------------
def _experiments(name):
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "experiments", f"{name}.yaml"), encoding="utf-8"))


_abl = _experiments("ablation")
ABLATION_ARMS = {arm: (a["what"], a["flags"], set(a["replaces"])) for arm, a in _abl.items()}   # arm -> (what, flags, keys)
ABLATION_PLAN = {arm: a["horizons"] for arm, a in _abl.items()}
CONTROL_ARMS = {arm: (a["model"], a["flags"], a["host"]) for arm, a in _experiments("control").items()}
_size = _experiments("size")
SIZE_TIERS, SIZE_SELECT_HS = _size["tiers"], _size["select_horizons"]
SIZE_TRIALS, SIZE_DRAW_SEED, SIZE_COMMON = _size["trials"], _size["draw_seed"], _size["common"]
SIZE_SPACE = _size["models"]                     # model -> {key: size / segment / period / switch / dropout / mix / [list] / unused}
SIZE_MODELS = list(SIZE_SPACE)
SIZE_TRIALS_FILE = os.path.join(ROOT, "docs", "size_trials.json")
CHOSEN_SIZE_FILE = os.path.join(ROOT, "docs", "chosen_size.json")


def size_trials(tier, model):
    """Drawn trials of one model in one tier: [{"id", "author", "knobs": {data: {key: value, lr}}, "stored", ...}]."""
    return _load(SIZE_TRIALS_FILE).get(tier, {}).get(model, [])


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


def chosen_size():
    """"<tier>|<model>|<data>" -> trial id (written by size_verdict.py)."""
    return _load(CHOSEN_SIZE_FILE)


def control_config_id(arm):
    """"ctrl:<arm>@<host config>" or None while the host's industrial setting is not chosen."""
    host_cfg = mfg_main_config_id(CONTROL_ARMS[arm][2])
    return None if host_cfg is None else f"ctrl:{arm}@{host_cfg}"


def _ctrl(config_id):
    arm, host_cfg = config_id[5:].split("@", 1)
    return arm, host_cfg


def control_row_model(arm):
    """Model name written in the results rows (the FEATHer arm keeps the name "FEATHer" its existing rows use)."""
    return "FEATHer" if arm == "FEATHer_frontnone" else arm


def cli_flags(model, config_id):
    """Extra worker CLI flags for a cell (ablation / control arms); [] otherwise."""
    if config_id.startswith("abl:"):
        return list(ABLATION_ARMS[config_id[4:]][1])
    if config_id.startswith("ctrl:"):
        return list(CONTROL_ARMS[_ctrl(config_id)[0]][1])
    return []

# chosen settings live in docs/ (committed) so that the public repository reproduces the main runs
CHOSEN_FILE = os.path.join(ROOT, "docs", "chosen_mfg.json")
CHOSEN_LTSF_FILE = os.path.join(ROOT, "docs", "chosen_ltsf.json")


def _load(path):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def chosen_mfg():
    """model -> config_id chosen on the industrial validation loss (written by the selection verdict)."""
    return _load(CHOSEN_FILE)


def chosen_ltsf():
    """"model|data" -> config_id chosen on that LTSF dataset's validation loss (no-official cells only)."""
    return _load(CHOSEN_LTSF_FILE)


def registry_name(model, config_id=""):
    if config_id.startswith("ctrl:"):
        return CONTROL_ARMS[_ctrl(config_id)[0]][0]
    return "FEATHer" if model in FEATHER else model


def _merge(model, data, H, config_id=""):
    m = registry_name(model, config_id)
    return {**B.get_method_defaults(m), **B.get_dataset_overrides(m, data, H)}


def resolve(model, group, data, H, config_id):
    """Worker settings for one cell: (lr, loss, overrides dict)."""
    m = registry_name(model, config_id)
    if config_id.startswith("abl:"):
        # frozen industrial FEATHer-mix minus the keys the arm's flags replace
        s = _merge("FEATHer", data, H)
        for k in ABLATION_ARMS[config_id[4:]][2]:
            s.pop(k, None)
    elif config_id.startswith("sz"):
        # size-matched trial: the model's industrial setting with every searched key of the trial set, lr included
        lr, loss, ov = resolve(model, group, data, H, mfg_main_config_id(model))
        for kv in config_id.split(":", 1)[1].split("+")[1:]:
            k, v = kv.split("=", 1)
            if k == "lr":
                lr = _parse_value(v)
            else:
                ov[k] = _parse_value(v)
        return lr, loss, ov
    elif config_id.startswith("ctrl:"):
        # the host's setting (its periods / lr / widths), trained under the arm's registry model
        arm, host_cfg = _ctrl(config_id)
        lr, loss, ov = resolve(CONTROL_ARMS[arm][2], group, data, H, host_cfg)
        return lr, loss, ov
    elif config_id in ("frozen", "official", "official-exc", "ref-default"):
        s = _merge(model, data, H, config_id)
        if model == "FEATHer_CI" and group == "mfg":
            s["channel_mode"] = "ci_full"
        if model == "FEATHer_mix" and group == "ltsf":
            s["channel_mode"] = "mix"
    elif config_id.startswith("cand:"):
        # a candidate = the model's setting for LTSF dataset <src> (every H); period-type keys always come
        # from the dataset being trained (industrial: FFT / daily-cycle rule; LTSF: that dataset's entry)
        # size-matched variants: "cand:<src>+<key>=<value>[+...]" = candidate <src> with the size knob(s)
        # replaced (the 1K-parameter budget study); everything else stays the author setting
        src, _, knobs = config_id.split(":", 1)[1].partition("+")
        base = dict(B.get_method_defaults(m))
        if src != "default":
            base.update({k: v for k, v in B.get_dataset_overrides(m, src, H).items() if k not in PERIOD_KEYS})
        own = B.get_dataset_overrides(m, data)
        base.update({k: v for k, v in own.items() if k in PERIOD_KEYS})
        for kv in filter(None, knobs.split("+")):
            k, v = kv.split("=", 1)
            base[k] = _parse_value(v)
        s = base
    else:
        raise ValueError(config_id)
    lr = s.pop("lr", 1e-3)
    loss = s.pop("loss", "mse")
    return lr, loss, s


def hparams(model, group, data, H, config_id):
    """Every hyperparameter of one cell as a JSON line (stored with the result; a change -> the cell runs again)."""
    lr, loss, ov = resolve(model, group, data, H, config_id)
    run = {"model": registry_name(model, config_id), "lr": lr, "loss": loss, "settings": ov,
           "flags": cli_flags(model, config_id)}
    return json.dumps(run, sort_keys=True, separators=(",", ":"))


def ltsf_candidates(model, data):
    """Candidates for a no-official LTSF cell (empty list for an official cell)."""
    if (model, data) not in LTSF_NO_OFFICIAL:
        return []
    return ["cand:" + c for c in CANDIDATES[model]]


def ltsf_config_id(model, data):
    """config_id of the main LTSF runs, or None while a no-official cell is not chosen yet."""
    if model in FEATHER:
        return "frozen"
    if model == "DLinear" and data == "ETTh2":
        return "official-exc"
    cands = ltsf_candidates(model, data)
    if cands:
        return cands[0] if len(cands) == 1 else chosen_ltsf().get(f"{model}|{data}")
    return "official"


def mfg_main_config_id(model):
    """config_id of the main industrial runs, or None while a model's setting is not chosen yet."""
    if model in FEATHER:
        return "frozen"
    if model in HEAVY:
        return "ref-default"
    if len(CANDIDATES.get(model, [])) == 1:
        return "cand:" + CANDIDATES[model][0]
    return chosen_mfg().get(model)


# ---- shared helpers for result files ---------------------------------------------------------------
KEY = ["role", "config_id", "model", "data", "pred_len", "seq_len", "seed"]


def split_exp_tag(exp_tag):
    """"<role>|<config_id>" (run.py) -> (role, config_id); (None, None) for other tags."""
    parts = str(exp_tag).split("|", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def main_rows(df, group, seq_len=96, finite=True, roles=("main", "reference")):
    """The rows a results table should use: role main/reference, each model's protocol setting only
    (FEATHer frozen; LTSF official / chosen; industrial chosen / single candidate; reference defaults),
    optionally dropping failed (non-finite) runs.  Rows of settings that were not chosen stay in the
    raw file for provenance but never reach a table through this function."""
    d = df[df.role.isin(list(roles)) & (df.seq_len == seq_len)]
    if group == "mfg":
        want = {m: mfg_main_config_id(m) for m in d.model.unique()}
        keep = [want.get(m) == c for m, c in zip(d.model, d.config_id)]
    else:
        keep = [ltsf_config_id(m, x) == c for m, x, c in zip(d.model, d.data, d.config_id)]
    d = d[keep]
    if finite:
        d = d[d.MSE.notna() & d.val_loss.notna()]
    return d


def write_csv_safely(df, path):
    """Write a managed result file atomically: temp file -> os.replace, previous version kept as <path>.bak."""
    import shutil
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    os.replace(tmp, path)
