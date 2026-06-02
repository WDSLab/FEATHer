# -*- coding: utf-8 -*-
"""
Baseline registry.

Single entry point for instantiating any model in the benchmark, including
FEATHer itself. Each baseline subdirectory contains:

  baselines/<Name>/
  ├── __init__.py      — exports the model class (optional convenience)
  ├── wrapper.py       — adapter from official Model(configs) → unified
  │                       (seq_len, pred_len, d_model, ...) interface
  └── <Name>-main/     — cloned upstream repo (gitignored, fetched by
                          setup_baselines.sh)

To add a new baseline:
  1. Clone upstream into baselines/<Name>/<Name>-main/
  2. Write baselines/<Name>/wrapper.py with a `build(args, n_features,
     seq_len, pred_len)` function returning an nn.Module.
  3. Register the name in _BUILDERS below.
"""

from typing import Callable, Dict


def _build_feather(args, n_features, seq_len, pred_len):
    from baselines.FEATHer.FEATHer import FEATHer
    return FEATHer(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=n_features,
        d_state=args.d_state,
        kernel_size=args.kernel_size,
        use_norm=True,
        period=args.period,
        num_bands=args.num_bands,
        use_topk_gate=args.use_topk_gate,
        topk=args.topk,
    )


# Stubs — each becomes active once baselines/<Name>/wrapper.py is in place.
def _build_dlinear(args, n_features, seq_len, pred_len):
    from baselines.DLinear.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_patchtst(args, n_features, seq_len, pred_len):
    from baselines.PatchTST.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_itransformer(args, n_features, seq_len, pred_len):
    from baselines.iTransformer.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_fits(args, n_features, seq_len, pred_len):
    from baselines.FITS.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_sparsetsf(args, n_features, seq_len, pred_len):
    from baselines.SparseTSF.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_timemixer(args, n_features, seq_len, pred_len):
    from baselines.TimeMixer.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_timesnet(args, n_features, seq_len, pred_len):
    from baselines.TimesNet.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_tqnet(args, n_features, seq_len, pred_len):
    from baselines.TQNet.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_lms_autotsf(args, n_features, seq_len, pred_len):
    from baselines.LMS_AutoTSF.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_dipe_linear(args, n_features, seq_len, pred_len):
    from baselines.DiPE_Linear.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_mdmlp_eia(args, n_features, seq_len, pred_len):
    from baselines.MDMLP_EIA.wrapper import build
    return build(args, n_features, seq_len, pred_len)


# NOTE — deliberately excluded from the benchmark (cite in Related Work only):
#   S-Mamba: selective-scan CUDA kernels have no Cortex-M / STM32 port.
#   TimeCMA: frozen GPT-2 features blow past the sub-1K parameter budget by
#            5 orders of magnitude; LLM-augmented forecasters are out of
#            scope for edge-deployable comparison.
_BUILDERS: Dict[str, Callable] = {
    "FEATHer":      _build_feather,
    "DLinear":      _build_dlinear,
    "PatchTST":     _build_patchtst,
    "iTransformer": _build_itransformer,
    "FITS":         _build_fits,
    "SparseTSF":    _build_sparsetsf,
    "TimeMixer":    _build_timemixer,
    "TimesNet":     _build_timesnet,
    "TQNet":        _build_tqnet,
    "LMS_AutoTSF":  _build_lms_autotsf,
    # R4-requested 2025-2026 SOTA — addresses "outdated baselines" critique
    "DiPE_Linear":  _build_dipe_linear,    # DASFAA 2026
    "MDMLP_EIA":    _build_mdmlp_eia,      # AAAI 2026
}


# Per-method training defaults — sourced from each paper's official ETTh1
# script (or run.py default when the script does not override). Following
# TFB (Hu et al., PVLDB 2024) and CF-JEPA (sibling project), we keep each
# method's paper-recommended LR and loss instead of forcing a single value
# across the lineup — paper LRs span two orders of magnitude (1e-4 → 2e-2)
# so a unified LR would disadvantage some models by 10-20× the optimum.
#
#   ref scripts:
#     PatchTST    baselines/PatchTST/PatchTST-main/.../scripts/PatchTST/*.sh
#     iTransformer baselines/iTransformer/iTransformer-main/run.py (default)
#     TimesNet    baselines/TimesNet/TimesNet-main/run.py (default)
#     LMS_AutoTSF baselines/LMS_AutoTSF/LMS_AutoTSF-main/run.py (default)
#     FITS        baselines/FITS/FITS-main/scripts/FITS/*.sh
#     TQNet       baselines/TQNet/TQNet-main/scripts/TQNet/etth1.sh
#     DLinear     baselines/DLinear/DLinear-main/scripts/.../Linear/etth1.sh
#     TimeMixer   baselines/TimeMixer/TimeMixer-main/scripts/.../TimeMixer_ETTh1_unify.sh
#     SparseTSF   baselines/SparseTSF/SparseTSF-main/scripts/SparseTSF/linear/etth1.sh
#
# Override at the CLI with `--lr`/`--loss` for ablations.
_METHOD_DEFAULTS: Dict[str, Dict] = {
    "FEATHer":      {"lr": 1e-3, "loss": "l1"},
    "DLinear":      {"lr": 5e-3, "loss": "mse"},
    "PatchTST":     {"lr": 1e-4, "loss": "mse"},
    "iTransformer": {"lr": 1e-4, "loss": "mse"},
    "FITS":         {"lr": 5e-4, "loss": "mse"},
    "SparseTSF":    {"lr": 2e-2, "loss": "mse"},
    "TimeMixer":    {"lr": 1e-2, "loss": "mse"},
    "TimesNet":     {"lr": 1e-4, "loss": "mse"},
    "TQNet":        {"lr": 1e-3, "loss": "mse"},
    "LMS_AutoTSF":  {"lr": 1e-4, "loss": "mse"},
    # DiPE-Linear: base.yaml lr=1e-3, t_loss='mse', single-expert
    "DiPE_Linear":  {"lr": 1e-3, "loss": "mse"},
    # MDMLP-EIA: run.py default lr=1e-4
    "MDMLP_EIA":    {"lr": 1e-4, "loss": "mse"},
}


def get_model(name, args, n_features, seq_len, pred_len):
    """Construct a model by registered name.

    Args:
        name: registered model key (case-sensitive).
        args: argparse.Namespace carrying model + training hyperparameters.
        n_features: input feature dimension.
        seq_len, pred_len: temporal lengths.

    Returns:
        nn.Module ready to .to(device).
    """
    if name not in _BUILDERS:
        available = ", ".join(sorted(_BUILDERS.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _BUILDERS[name](args, n_features, seq_len, pred_len)


def get_method_defaults(name):
    """Per-paper training defaults (lr, loss) for `name`.

    Returns an empty dict if the model is unregistered or has no entry —
    callers should fall back to their own defaults in that case.
    """
    return dict(_METHOD_DEFAULTS.get(name, {}))


# -----------------------------------------------------------------------------
# Per-(method, dataset) hyperparameter overrides
# -----------------------------------------------------------------------------
#
# Each baseline tunes architecture HPs and learning rate per dataset in its
# original paper / official scripts. To compare fairly we adopt those exact
# values from each repo's ETT* / ECL / Traffic / Weather / Exchange scripts.
#
# Held UNIFORM across all (method, dataset) cells (training protocol axes):
#     seq_len = 96    batch_size = 32    num_epochs = 50    AMP = off
#
# Free per cell (paper-reported values):
#     lr, architecture HPs (d_model, d_ff, e_layers, n_heads, dropout,
#     cycle, period_len, ...)
#
# A missing (model, dataset) entry falls back to `_METHOD_DEFAULTS[model]`
# (ETTh1 canonical config); the orchestrator then layers any CLI `--lr` /
# `--loss` on top.
#
# Reviewer-defense framing:
#   "Each baseline uses its original-paper per-dataset hyperparameters.
#    FEATHer uses a single configuration across all 8 datasets,
#    demonstrating that the proposed design generalizes without
#    dataset-specific tuning."
#
# Sources extracted from each repo's official ETTh1 / ETTh2 / ETTm1 / ETTm2
# / Electricity / Exchange / Traffic / Weather script. SML / Volatility /
# PEMS fall back to the closest hourly-data config (ETTh1) since no
# upstream paper provides values for them.
# -----------------------------------------------------------------------------

_DATASET_OVERRIDES: Dict = {

    # ---- DLinear (cure-lab/LTSF-Linear) -----------------------------------
    # scripts/EXP-LongForecasting/Linear/{dataset}.sh
    ("DLinear", "ETTh1"):       {"lr": 5e-3},
    ("DLinear", "ETTh2"):       {"lr": 5e-2},
    ("DLinear", "ETTm1"):       {"lr": 1e-4},
    ("DLinear", "ETTm2"):       {"lr": 1e-3},
    ("DLinear", "Electricity"): {"lr": 1e-3},
    ("DLinear", "Exchange"):    {"lr": 5e-4},
    ("DLinear", "Traffic"):     {"lr": 5e-2},
    ("DLinear", "Weather"):     {"lr": 5e-3},  # weather.sh has no lr override

    # ---- PatchTST (yuqinie98/PatchTST) ------------------------------------
    # PatchTST_supervised/scripts/PatchTST/{dataset}.sh
    # ETTh* uses small d_model=16 / n_heads=4 / dropout=0.3 (paper Table 9);
    # the rest use d_model=128 / n_heads=16 / dropout=0.2.
    ("PatchTST", "ETTh1"):       {"patchtst_d_model": 16, "patchtst_d_ff": 128, "patchtst_n_heads": 4,
                                  "patchtst_dropout": 0.3, "patchtst_fc_dropout": 0.3, "lr": 1e-4},
    ("PatchTST", "ETTh2"):       {"patchtst_d_model": 16, "patchtst_d_ff": 128, "patchtst_n_heads": 4,
                                  "patchtst_dropout": 0.3, "patchtst_fc_dropout": 0.3, "lr": 1e-4},
    ("PatchTST", "ETTm1"):       {"patchtst_d_model": 128, "patchtst_d_ff": 256, "patchtst_n_heads": 16,
                                  "patchtst_dropout": 0.2, "patchtst_fc_dropout": 0.2, "lr": 1e-4},
    ("PatchTST", "ETTm2"):       {"patchtst_d_model": 128, "patchtst_d_ff": 256, "patchtst_n_heads": 16,
                                  "patchtst_dropout": 0.2, "patchtst_fc_dropout": 0.2, "lr": 1e-4},
    ("PatchTST", "Electricity"): {"patchtst_d_model": 128, "patchtst_d_ff": 256, "patchtst_n_heads": 16,
                                  "patchtst_dropout": 0.2, "patchtst_fc_dropout": 0.2, "lr": 1e-4},
    ("PatchTST", "Exchange"):    {"patchtst_d_model": 16, "patchtst_d_ff": 128, "patchtst_n_heads": 4,
                                  "patchtst_dropout": 0.3, "patchtst_fc_dropout": 0.3, "lr": 1e-4},
    ("PatchTST", "Traffic"):     {"patchtst_d_model": 128, "patchtst_d_ff": 256, "patchtst_n_heads": 16,
                                  "patchtst_dropout": 0.2, "patchtst_fc_dropout": 0.2, "lr": 1e-4},
    ("PatchTST", "Weather"):     {"patchtst_d_model": 128, "patchtst_d_ff": 256, "patchtst_n_heads": 16,
                                  "patchtst_dropout": 0.2, "patchtst_fc_dropout": 0.2, "lr": 1e-4},

    # ---- iTransformer (thuml/iTransformer) --------------------------------
    # scripts/multivariate_forecasting/{dataset}/iTransformer*.sh
    ("iTransformer", "ETTh1"):       {"itrans_d_model": 256, "itrans_d_ff": 256, "itrans_e_layers": 2, "lr": 1e-4},
    ("iTransformer", "ETTh2"):       {"itrans_d_model": 128, "itrans_d_ff": 128, "itrans_e_layers": 2, "lr": 1e-4},
    ("iTransformer", "ETTm1"):       {"itrans_d_model": 128, "itrans_d_ff": 128, "itrans_e_layers": 2, "lr": 1e-4},
    ("iTransformer", "ETTm2"):       {"itrans_d_model": 128, "itrans_d_ff": 128, "itrans_e_layers": 2, "lr": 1e-4},
    ("iTransformer", "Electricity"): {"itrans_d_model": 512, "itrans_d_ff": 512, "itrans_e_layers": 3, "lr": 5e-4},
    ("iTransformer", "Exchange"):    {"itrans_d_model": 128, "itrans_d_ff": 128, "itrans_e_layers": 2, "lr": 1e-4},
    ("iTransformer", "Traffic"):     {"itrans_d_model": 512, "itrans_d_ff": 512, "itrans_e_layers": 4, "lr": 1e-3},
    ("iTransformer", "Weather"):     {"itrans_d_model": 512, "itrans_d_ff": 512, "itrans_e_layers": 3, "lr": 1e-4},

    # ---- FITS (VEWOXIC/FITS) ---------------------------------------------
    # scripts/FITS/{dataset}.sh — uniform lr=5e-4; cut_freq derived per
    # (seq_len, pred_len) via H_order which we approximate with seq_len/4
    # in the wrapper. No per-dataset overrides needed.

    # ---- SparseTSF (lss-1138/SparseTSF) ----------------------------------
    # scripts/SparseTSF/linear/{dataset}.sh — period_len follows native
    # sampling: 24 for hourly (ETTh*, Electricity, Traffic), 4 for 15-min
    # / 10-min (ETTm*, Weather, Solar)
    ("SparseTSF", "ETTh1"):       {"sparsetsf_period_len": 24, "lr": 2e-2},
    ("SparseTSF", "ETTh2"):       {"sparsetsf_period_len": 24, "lr": 3e-2},
    ("SparseTSF", "ETTm1"):       {"sparsetsf_period_len": 4,  "lr": 2e-2},
    ("SparseTSF", "ETTm2"):       {"sparsetsf_period_len": 4,  "lr": 2e-2},
    ("SparseTSF", "Electricity"): {"sparsetsf_period_len": 24, "lr": 2e-2},
    ("SparseTSF", "Exchange"):    {"sparsetsf_period_len": 24, "lr": 2e-2},  # daily; period_len=1 also valid
    ("SparseTSF", "Traffic"):     {"sparsetsf_period_len": 24, "lr": 3e-2},
    ("SparseTSF", "Weather"):     {"sparsetsf_period_len": 4,  "lr": 2e-2},

    # ---- TimeMixer (kwuking/TimeMixer) -----------------------------------
    # scripts/long_term_forecast/*/TimeMixer_unify.sh
    ("TimeMixer", "ETTh1"):       {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 2, "lr": 1e-2},
    ("TimeMixer", "ETTh2"):       {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 2, "lr": 1e-2},
    ("TimeMixer", "ETTm1"):       {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 2, "lr": 1e-2},
    ("TimeMixer", "ETTm2"):       {"tm_d_model": 32, "tm_d_ff": 32, "tm_e_layers": 2, "lr": 1e-2},
    ("TimeMixer", "Electricity"): {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 3, "lr": 1e-2},
    ("TimeMixer", "Exchange"):    {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 2, "lr": 1e-2},
    ("TimeMixer", "Traffic"):     {"tm_d_model": 32, "tm_d_ff": 64, "tm_e_layers": 3, "lr": 1e-2},
    ("TimeMixer", "Weather"):     {"tm_d_model": 16, "tm_d_ff": 32, "tm_e_layers": 3, "lr": 1e-2},

    # ---- TimesNet (thuml/Time-Series-Library) ----------------------------
    # scripts/long_term_forecast/*/TimesNet*.sh — d_model swings 16 → 512
    ("TimesNet", "ETTh1"):       {"tn_d_model": 16,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "ETTh2"):       {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "ETTm1"):       {"tn_d_model": 64,  "tn_d_ff": 64,  "lr": 1e-4},
    ("TimesNet", "ETTm2"):       {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "Electricity"): {"tn_d_model": 256, "tn_d_ff": 512, "lr": 1e-4},
    ("TimesNet", "Exchange"):    {"tn_d_model": 64,  "tn_d_ff": 64,  "lr": 1e-4},
    ("TimesNet", "Traffic"):     {"tn_d_model": 512, "tn_d_ff": 512, "lr": 1e-4},
    ("TimesNet", "Weather"):     {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},

    # ---- TQNet (ACAT-SCUT/TQNet) -----------------------------------------
    # scripts/TQNet/{dataset}.sh — cycle is data-frequency-tied:
    # 24 hourly, 96 15-min, 144 10-min weather/solar, 168 weekly Traffic/ECL
    ("TQNet", "ETTh1"):       {"tqnet_cycle": 24,  "tqnet_dropout": 0.5, "lr": 1e-3},
    ("TQNet", "ETTh2"):       {"tqnet_cycle": 24,  "tqnet_dropout": 0.5, "lr": 1e-3},
    ("TQNet", "ETTm1"):       {"tqnet_cycle": 96,  "tqnet_dropout": 0.5, "lr": 1e-3},
    ("TQNet", "ETTm2"):       {"tqnet_cycle": 96,  "tqnet_dropout": 0.5, "lr": 1e-3},
    ("TQNet", "Electricity"): {"tqnet_cycle": 168,                       "lr": 3e-3},
    ("TQNet", "Exchange"):    {"tqnet_cycle": 24,  "tqnet_dropout": 0.5, "lr": 1e-3},
    ("TQNet", "Traffic"):     {"tqnet_cycle": 168,                       "lr": 3e-3},
    ("TQNet", "Weather"):     {"tqnet_cycle": 144, "tqnet_dropout": 0.5, "lr": 1e-3},

    # ---- LMS-AutoTSF (mribrahim/LMS-TSF) ---------------------------------
    # All ETT/ECL/Weather/Traffic scripts use channel_independence=0,
    # e_layers=2, and Time-Series-Library default lr=1e-4. No per-dataset
    # differentiation; entries omitted (falls back to _METHOD_DEFAULTS).

    # ---- DiPE-Linear (wintertee/DiPE-Linear, DASFAA 2026) ----------------
    # configs/base.yaml is the only training config in the upstream repo;
    # no per-dataset YAMLs published. Uniform lr=1e-3, t_loss='mse',
    # single-expert. Entries omitted (falls back to _METHOD_DEFAULTS).

    # ---- MDMLP-EIA (zh1985csuccsu/MDMLP-EIA, AAAI 2026) ------------------
    # run.py exposes one default config (lr=1e-4); upstream sweep scripts
    # publish dataset-specific d_model only via CLI flags at run time, not
    # via committed config files. Entries omitted (falls back to
    # _METHOD_DEFAULTS).
}


def get_dataset_overrides(name, data):
    """Per-dataset architecture/lr overrides for (name, data).

    Returns an empty dict when the (model, dataset) cell has no entry —
    the caller should layer this on top of `get_method_defaults(name)`.
    """
    return dict(_DATASET_OVERRIDES.get((name, data), {}))


def list_models():
    return sorted(_BUILDERS.keys())
