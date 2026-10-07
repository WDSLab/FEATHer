"""Experiment protocol in code (EXPERIMENT_PROTOCOL.md): which settings exist, and what each config_id means.

config_id                      meaning
  frozen                       FEATHer frozen configuration (+ the dataset's period), every group
  official / official-exc      LTSF: the model's per-(dataset, H) setting in baselines/__init__.py
                               (official scripts; -exc = DLinear ETTh2, re-selected after divergence)
  ref-default                  server-class reference models on the industrial data: defaults
  cand:<DS>                    industrial candidate = the model's LTSF setting for <DS> (every H),
                               with period-type keys replaced by the industrial dataset's own value
  cand:default                 industrial candidate = the default setting the authors present
  ctrl:<arm>                   control arms (SparseTSF + mixing bus, ...)

resolve(model, group, data, H, config_id) -> (lr, loss, overrides) for the worker.
"""
import json
import os
import sys

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

# Industrial candidates (EXPERIMENT_PROTOCOL.md §4).  Distinct settings only; datasets whose setting
# equals another's are listed once.  Chip filter (weights alone > 128 KB at D = 28, H = 720) removed:
# CMoS ETTm1 / ETTm2 / Traffic, PhaseFormer ETTh1 (wide H720) / Electricity / Traffic.
CANDIDATES = {
    "SparseTSF": ["ETTh1", "ETTh2"],                                   # lr 0.02 / 0.03
    "SparseTSF_MLP": ["ETTh1", "ETTh2", "Weather", "Electricity", "Traffic"],
    "DLinear": ["ETTh1", "ETTm1", "ETTm2", "Exchange", "Electricity", "Traffic"],
    "CycleNet": ["ETTh1"],                                             # one setting (L96 scripts)
    "DiPE_Linear": ["ETTh1", "ETTh2", "ETTm2", "Weather", "Electricity", "default"],
    "MixLinear": ["ETTh1", "ETTh2", "Electricity", "Traffic"],
    "FITS": ["ETTh1", "ETTm1", "Weather"],
    "CMoS": ["ETTh1", "ETTh2", "Weather", "Electricity", "default"],
    "TimeBase": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic"],
    "PhaseFormer": ["ETTh2", "ETTm1", "ETTm2", "Weather"],
    "SAMformer": ["ETTh1", "Weather", "default"],
}

# Industrial base values of period-type keys that have no row in baselines/__init__.py yet:
# CMoS period hint = the FFT period (<= 48) rule; FITS base_T = the daily cycle (same values as CycleNet).
_MFG_EXTRA = {}
for _d in MFG:
    _p = B.get_dataset_overrides("SparseTSF", _d).get("sparsetsf_period_len", 24)
    _MFG_EXTRA[("CMoS", _d)] = {"cmos_period": _p}
    _MFG_EXTRA[("FITS", _d)] = {"fits_base_t": B.get_dataset_overrides("CycleNet", _d)["cyclenet_cycle"]}

CHOSEN_FILE = os.path.join(ROOT, "results", "chosen_mfg.json")


def chosen_mfg():
    """model -> config_id chosen on the industrial validation loss (written by the selection verdict)."""
    if os.path.exists(CHOSEN_FILE):
        return json.load(open(CHOSEN_FILE, encoding="utf-8"))
    return {}


def registry_name(model):
    return "FEATHer" if model in FEATHER else model


def _merge(model, data, H):
    m = registry_name(model)
    return {**B.get_method_defaults(m), **B.get_dataset_overrides(m, data, H)}


def resolve(model, group, data, H, config_id):
    """Worker settings for one cell: (lr, loss, overrides dict)."""
    m = registry_name(model)
    if config_id in ("frozen", "official", "official-exc", "ref-default") or config_id.startswith("ctrl:"):
        s = _merge(model, data, H)
        if model == "FEATHer_CI" and group == "mfg":
            s["channel_mode"] = "ci_full"
        if model == "FEATHer_mix" and group == "ltsf":
            s["channel_mode"] = "mix"
    elif config_id.startswith("cand:"):
        assert group == "mfg", config_id
        src = config_id.split(":", 1)[1]
        base = dict(B.get_method_defaults(m))
        if src != "default":
            base.update({k: v for k, v in B.get_dataset_overrides(m, src, H).items() if k not in PERIOD_KEYS})
        own = {**B.get_dataset_overrides(m, data), **_MFG_EXTRA.get((m, data), {})}
        base.update({k: v for k, v in own.items() if k in PERIOD_KEYS})
        s = base
    else:
        raise ValueError(config_id)
    lr = s.pop("lr", 1e-3)
    loss = s.pop("loss", "mse")
    return lr, loss, s


def ltsf_config_id(model, data):
    if model in FEATHER:
        return "frozen"
    if model == "DLinear" and data == "ETTh2":
        return "official-exc"
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
