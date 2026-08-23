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
        use_norm=getattr(args, "use_norm", True),
        period=args.period,
        num_bands=args.num_bands,
        use_topk_gate=args.use_topk_gate,
        topk=args.topk,
        use_residual_mlp=getattr(args, "use_residual_mlp", False),
        mlp_rank=getattr(args, "mlp_rank", 16),
        use_gate=getattr(args, "use_gate", True),
        use_robust_norm=getattr(args, "use_robust_norm", False),
        revin_mode=getattr(args, "revin_mode", None),
        winsor_bound=getattr(args, "winsor_bound", None) or 4.0,
        center_trim=getattr(args, "center_trim", None) or 0.0,
        channel_mode=getattr(args, "channel_mode", None) or "mix",
        gate_mode=getattr(args, "gate_mode", None) or "global",
        dtk_act=getattr(args, "dtk_act", None) or "none",
        head_mode=getattr(args, "head_mode", None) or "spk",
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


# Edge-tier additions (2026-07-31) — sub-1K / small-deployable competitors.
# (Classic practitioner baselines LSTM/GRU/TCN + XGBoost were archived to
# _archive/classic_baselines/ on 2026-07-31 and deferred to a revision;
# restore from there or git history e22236f if a reviewer asks for them.)
def _build_mixlinear(args, n_features, seq_len, pred_len):
    from baselines.MixLinear.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_cyclenet(args, n_features, seq_len, pred_len):
    from baselines.CycleNet.wrapper import build
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
    # Edge-tier additions (2026-07-31) — sub-1K / small-deployable competitors
    "MixLinear":    _build_mixlinear,      # ICLR 2026, ~0.1K params (sub-1K rival)
    "CycleNet":     _build_cyclenet,       # NeurIPS 2024, learnable cycle
}


# -----------------------------------------------------------------------------
# FEATHer ablation
# -----------------------------------------------------------------------------
# The 30 `FEATHer_<axis>_<variant>` builders that used to live here (multiscale
# 15 / gating 4 / DTK 4 / head 4 / complexity 3) were archived on 2026-08-24 to
# _archive/ablation_v1/, together with baselines/FEATHer/ablation/ and
# tools/paper/ablation_table.py.
#
# They were written before the residual MLP (07-31) and the channel mode
# (08-05) existed and never forwarded either, so they built the pre-08
# architecture. Running them would have ablated a model that is not the one in
# the main table, and the deltas would have been meaningless.
#
# Ablation now runs as `run_forecast.py --ablation_arm <arm>`: one switch off
# per arm on the shipping model, using the flags the worker already exposes.
# See ABLATION_ARMS there. Head/DTK-variant ablations are the one thing that
# did not survive the move — they need the archived modules updated to accept
# channel_mode / use_residual_mlp before they would be comparable.


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
    # loss=mse (was l1): match train loss to the eval metric so all 14 models
    # are consistent (train=mse, eval=mse) — l1-trained FEATHer was penalized on
    # the MSE table (2026-07-30 finding). An l1 variant is a supplementary MAE run.
    "FEATHer":      {"lr": 1e-3, "loss": "mse"},
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
    # Edge-tier additions (2026-07-31) — published script lr (the argparse
    # default 1e-4 is a fallback every author script overrides): MixLinear
    # scripts use 0.03, CycleNet scripts use 0.005.
    "MixLinear":    {"lr": 3e-2, "loss": "mse"},
    "CycleNet":     {"lr": 5e-3, "loss": "mse"},
}


def get_model(name, args, n_features, seq_len, pred_len):
    """Construct a model by registered name.

    Args:
        name: registered model key (case-sensitive) from the main lineup.
        args: argparse.Namespace carrying model + training hyperparameters.
        n_features: input feature dimension.
        seq_len, pred_len: temporal lengths.

    Returns:
        nn.Module ready to .to(device).
    """
    if name in _BUILDERS:
        return _BUILDERS[name](args, n_features, seq_len, pred_len)
    available = ", ".join(sorted(_BUILDERS))
    raise KeyError(f"Unknown model '{name}'. Available: {available}")


def get_method_defaults(name):
    """Per-paper training defaults (lr, loss) for `name`.

    Returns an empty dict if the model is unregistered.
    """
    if name in _METHOD_DEFAULTS:
        return dict(_METHOD_DEFAULTS[name])
    return {}


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
# Fair-comparison framing (revised 2026-06-17 — per-dataset for ALL):
#   "Every method, including FEATHer, uses per-dataset hyperparameters
#    selected on the validation split (FEATHer's via run_hp_search.py),
#    ensuring an apples-to-apples comparison at each method's best
#    operating point."
# FEATHer ("FEATHer", <data>) rows are added below from
# `run_hp_search.py --summary`; until then FEATHer falls back to
# _METHOD_DEFAULTS (base config) on every dataset.
#
# Sources extracted from each repo's official ETTh1 / ETTh2 / ETTm1 / ETTm2
# / Electricity / Exchange / Traffic / Weather script. Datasets with no
# upstream paper values (the manufacturing sets) keep each method's paper
# default architecture; only lr is dataset-selected via run_lr_search.py.
# -----------------------------------------------------------------------------

# FEATHer FINAL canonical mfg config ("N" cell) — shared across all 6 mfg
# datasets (one-config protocol); merged with a per-dataset `period` below.
_FEATHER_MFG_N: Dict = {
    "channel_mode": "mix",
    "use_residual_mlp": True,
    "mlp_rank": 16,
    "d_state": 16,
    "kernel_size": 7,
    "num_bands": 4,
    "lambda_spec": 0.01,
}

# FEATHer FINAL canonical LTSF config ("CI" cell) — channel-INDEPENDENT +
# residual; per-dataset d_state/kernel/num_bands/period merged in below.
_FEATHER_LTSF_CI: Dict = {
    "channel_mode": "ci_full",
    "use_residual_mlp": True,
    "mlp_rank": 16,
    "lambda_spec": 0.01,
}

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

    # ---- FEATHer (this work) — per-dataset val winners from
    # run_hp_search.py --summary (OFAT 512 + --validate combo verdict 32,
    # 2026-07-22 RE-FREEZE under corrected spectral loss + standard Weather
    # split; supersedes the 2026-07-06 freeze). Each row = best-ranked among
    # {base, single-axis variants, validated combo}. The --validate step
    # REJECTED the raw OFAT stack for 4/8 sets (ETTh1/ETTh2/Exchange/Traffic
    # fell back to base + one axis), confirming the flat axes (kernel/λ, and
    # num_bands off-Traffic) are noise — stacking them lost to base.
    # d_state=16 now only on ETTm1 (D=7, sub-1K OK) and Electricity (already a
    # D²-stress set, 115K). H=720 SPK overflow is pre-existing (affects base
    # too) — sub-1K claim stays H=96-scoped. These feed the LTSF
    # *generalization* section; the MFG main-table canonical arch stays frozen
    # at base (d8/k7/p12/B3) — nothing in this LTSF OFAT contradicts it.
    # FEATHer LTSF = FINAL "CI" cell (2026-08-13, grid 27/27 done). LTSF favors
    # channel-INDEPENDENCE (ci_full) over the mfg channel-mix: CIN OVERALL 0.3588
    # ties FITS 0.3582 (4/6 datasets won), beats the base-L cell (0.3647). Per-
    # dataset tuning (symmetric w/ baselines that tune per-dataset); val-selected
    # d_state/kernel/num_bands below, period = per-dataset FFT. loss=mse, lr=1e-3
    # from _METHOD_DEFAULTS. Old L-cell (mix, no residual) OFAT configs replaced.
    ("FEATHer", "ETTh1"):    {**_FEATHER_LTSF_CI, "d_state": 16, "kernel_size": 3, "num_bands": 3, "period": 6},   # 0.4429
    ("FEATHer", "ETTh2"):    {**_FEATHER_LTSF_CI, "d_state": 16, "kernel_size": 3, "num_bands": 4, "period": 12},  # 0.3847
    ("FEATHer", "ETTm1"):    {**_FEATHER_LTSF_CI, "d_state": 8,  "kernel_size": 7, "num_bands": 4, "period": 6},   # 0.4087
    ("FEATHer", "ETTm2"):    {**_FEATHER_LTSF_CI, "d_state": 8,  "kernel_size": 3, "num_bands": 2, "period": 6},   # 0.2849
    ("FEATHer", "Weather"):  {**_FEATHER_LTSF_CI, "d_state": 16, "kernel_size": 7, "num_bands": 3, "period": 6},   # 0.2669
    ("FEATHer", "Exchange"): {**_FEATHER_LTSF_CI, "d_state": 16, "kernel_size": 7, "num_bands": 2, "period": 8},   # 0.3648
    ("FEATHer", "Electricity"): {"d_state": 16, "kernel_size": 9, "period": 24, "num_bands": 4, "lambda_spec": 0.0,   "lr": 0.005},
    ("FEATHer", "Traffic"):     {"d_state": 8,  "kernel_size": 7, "period": 12, "num_bands": 3, "lambda_spec": 0.01,  "lr": 0.005},

    # ---- Manufacturing freq-tied period/cycle (data-driven, 2026-07-13) ------
    # FFT-detected dominant period on each train split (detrended). Applied by
    # the SAME rule to all three period-models, each per its own constraint —
    # exactly mirroring the LTSF convention (SparseTSF small period_len vs TQNet
    # daily cycle on the same data):
    #   FEATHer/SparseTSF: period must divide seq_len(96)/pred_len -> usable
    #                      range <=48 (detected dominant period in [2,48]).
    #   TQNet:             cycle unconstrained -> the daily cycle (samples/day;
    #                      BlastFurnace hourly->24, MetroPT3 detected ~5h=298).
    # This is the data-FREQUENCY-TIED axis (NOT the frozen mfg arch d8/k7/B3);
    # it gives SparseTSF a fair period instead of a crippling hourly default.
    # NOTE: lr for these mfg cells is MERGED IN later from run_lr_search.py
    # --summary — EDIT these dicts to add "lr", do NOT paste duplicate keys.
    ("SparseTSF", "Steel"):        {"sparsetsf_period_len": 24},
    ("SparseTSF", "WindSCADA"):    {"sparsetsf_period_len": 48},
    ("SparseTSF", "BlastFurnace"): {"sparsetsf_period_len": 48},
    ("SparseTSF", "MetroPT3"):     {"sparsetsf_period_len": 24},
    ("SparseTSF", "N2O"):          {"sparsetsf_period_len": 24},
    ("TQNet", "Steel"):        {"tqnet_cycle": 96},
    ("TQNet", "WindSCADA"):    {"tqnet_cycle": 144},
    ("TQNet", "BlastFurnace"): {"tqnet_cycle": 24},
    ("TQNet", "MetroPT3"):     {"tqnet_cycle": 298},
    ("TQNet", "N2O"):          {"tqnet_cycle": 288},
    # FEATHer mfg = FINAL canonical "N" cell (2026-08-11, grid search 27/27 done,
    # mean-rank val winner = ds16_k7_B4): channel-mixing + nonlinear residual MLP,
    # d_state 16 (only axis the grid moved; d16>d8>d4), num_bands 4 (full 4-band
    # decomposition, best at d16), kernel 7. ONE config across all mfg (symmetric
    # one-config protocol); only `period` is per-dataset (FFT-detected). loss=mse,
    # lr=1e-3 come from _METHOD_DEFAULTS. All keys forward via --model_overrides.
    ("FEATHer", "Steel"):        {**_FEATHER_MFG_N, "period": 24},
    ("FEATHer", "WindSCADA"):    {**_FEATHER_MFG_N, "period": 48},
    ("FEATHer", "BlastFurnace"): {**_FEATHER_MFG_N, "period": 48},
    ("FEATHer", "MetroPT3"):     {**_FEATHER_MFG_N, "period": 24},
    ("FEATHer", "N2O"):          {**_FEATHER_MFG_N, "period": 24},
    # EWELD (15-min factory load, added 2026-07-14): strong daily cycle ->
    # FEATHer/SparseTSF period 24 (snap of the [2,48] dominant), TQNet cycle 96
    # (daily = 96 samples/day), same rule as Steel (also 15-min).
    ("SparseTSF", "EWELD"): {"sparsetsf_period_len": 24},
    ("TQNet", "EWELD"):     {"tqnet_cycle": 96},
    ("FEATHer", "EWELD"):   {**_FEATHER_MFG_N, "period": 24},
    # ---- Edge-tier additions (2026-07-31): freq-tied period, same FFT-detected
    #   rule as every period model. MixLinear period_len == SparseTSF (segment
    #   length, divisor of seq_len); CycleNet cycle == TQNet (unconstrained
    #   daily cycle). lr uses each model's published default (_METHOD_DEFAULTS).
    ("MixLinear", "Steel"):        {"mixlinear_period_len": 24},
    ("MixLinear", "WindSCADA"):    {"mixlinear_period_len": 48},
    ("MixLinear", "BlastFurnace"): {"mixlinear_period_len": 48},
    ("MixLinear", "N2O"):          {"mixlinear_period_len": 24},
    ("MixLinear", "MetroPT3"):     {"mixlinear_period_len": 24},
    ("MixLinear", "EWELD"):        {"mixlinear_period_len": 24},
    ("MixLinear", "ETTh1"):        {"mixlinear_period_len": 24},
    ("MixLinear", "ETTh2"):        {"mixlinear_period_len": 24},
    ("MixLinear", "ETTm1"):        {"mixlinear_period_len": 4},
    ("MixLinear", "ETTm2"):        {"mixlinear_period_len": 4},
    ("MixLinear", "Weather"):      {"mixlinear_period_len": 4},
    ("MixLinear", "Exchange"):     {"mixlinear_period_len": 24},
    ("MixLinear", "Electricity"):  {"mixlinear_period_len": 24},
    ("MixLinear", "Traffic"):      {"mixlinear_period_len": 24},
    ("CycleNet", "Steel"):         {"cyclenet_cycle": 96},
    ("CycleNet", "WindSCADA"):     {"cyclenet_cycle": 144},
    ("CycleNet", "BlastFurnace"):  {"cyclenet_cycle": 24},
    ("CycleNet", "N2O"):           {"cyclenet_cycle": 288},
    ("CycleNet", "MetroPT3"):      {"cyclenet_cycle": 298},
    ("CycleNet", "EWELD"):         {"cyclenet_cycle": 96},
    ("CycleNet", "ETTh1"):         {"cyclenet_cycle": 24},
    ("CycleNet", "ETTh2"):         {"cyclenet_cycle": 24},
    ("CycleNet", "ETTm1"):         {"cyclenet_cycle": 96},
    ("CycleNet", "ETTm2"):         {"cyclenet_cycle": 96},
    ("CycleNet", "Weather"):       {"cyclenet_cycle": 144},
    ("CycleNet", "Exchange"):      {"cyclenet_cycle": 24},
    ("CycleNet", "Electricity"):   {"cyclenet_cycle": 168},
    ("CycleNet", "Traffic"):       {"cyclenet_cycle": 168},
}


def get_dataset_overrides(name, data):
    """Per-dataset architecture/lr overrides for (name, data).

    Returns an empty dict when the (model, dataset) cell has no entry —
    the caller should layer this on top of `get_method_defaults(name)`.
    FEATHer ablation variants share FEATHer's single-config protocol, so
    they have no per-dataset overrides by design.
    """
    return dict(_DATASET_OVERRIDES.get((name, data), {}))


def list_models():
    """Main benchmark lineup — 12 models (FEATHer + 11 baselines)."""
    return sorted(_BUILDERS.keys())
