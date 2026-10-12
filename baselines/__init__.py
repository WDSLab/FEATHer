# -*- coding: utf-8 -*-
"""
Baseline registry.

Single entry point for instantiating any model in the benchmark, including
FEATHer itself. Each baseline subdirectory contains:

  baselines/<Name>/
  ?쒋?? __init__.py      ??exports the model class (optional convenience)
  ?쒋?? wrapper.py       ??adapter from official Model(configs) ??unified
  ??                      (seq_len, pred_len, d_model, ...) interface
  ?붴?? <Name>-main/     ??cloned upstream repo (gitignored, fetched by
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
        dtk_mode=getattr(args, "dtk_mode", None) or "full",
        point_kernel=getattr(args, "point_kernel", None) or 1,
        use_in_proj=not getattr(args, "no_in_proj", False),
        spk_hidden=getattr(args, "spk_hidden", None) or 0,
        band_act=getattr(args, "band_act", None) or "none",
        band_mode=getattr(args, "band_mode", None) or "conv",
        gate_zero_init=getattr(args, "gate_zero_init", False),
        band_kernels=getattr(args, "band_kernels", None) or "",
        band_dilations=getattr(args, "band_dilations", None) or "",
        mix_mode=getattr(args, "mix_mode", None) or "hidden",
        mix_pool=getattr(args, "mix_pool", None) or 1,
        mlp_tie=getattr(args, "mlp_tie", False),
        gate_bounds=getattr(args, "gate_bounds", None) or "linear",
        band_zero_sum=getattr(args, "band_zero_sum", False),
        gate_mix=getattr(args, "gate_mix", None) or "none",
        head2=getattr(args, "head2", None) or "none",
        mix_listen=getattr(args, "mix_listen", False),
        mix_lag=getattr(args, "mix_lag", None) or "",
        mix_zero_init=getattr(args, "mix_zero_init", False),
        far_coarse=getattr(args, "far_coarse", None) or 0,
        near_len=getattr(args, "near_len", None) or 96,
        factor_head=getattr(args, "factor_head", None) or 0,
        factor_src=getattr(args, "factor_src", None) or "hidden",
        front=getattr(args, "front", None) or "bands",
        mix_sep=bool(getattr(args, "mix_sep", 0) or 0),
        mix_stats=bool(getattr(args, "mix_stats", 0) or 0),
    )


# Stubs ??each becomes active once baselines/<Name>/wrapper.py is in place.
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


def _build_sparsetsf_mlp(args, n_features, seq_len, pred_len):
    """SparseTSF's MLP variant ??the same-size rival, benchmarked separately.

    SparseTSF ships two heads. `linear` is the paper's headline "1K-parameter"
    model (41 params at D=7/H=96) and is what the main entry runs. `mlp` adds a
    d_model=128 hidden layer, which costs ~30x more (4,535 params worst case
    across the manufacturing sets) and lands almost exactly on FEATHer's own
    footprint (4,522) while still fitting the MCU budget.

    That makes it the one baseline that meets us at our size rather than below
    it, so comparing only against `linear` would leave the obvious question ??
    "is FEATHer better than SparseTSF, or just bigger?" ??unanswered. Kept as a
    separate registry entry rather than a flag so both variants appear in the
    same table with their own parameter counts.
    """
    from baselines.SparseTSF.wrapper import build
    import copy
    a = copy.copy(args)
    a.sparsetsf_model_type = "mlp"
    return build(a, n_features, seq_len, pred_len)


# 2026-10-06 control arms (user): upstream SparseTSF (linear / mlp head) + FEATHer's
# channel-mixing bus only (D -> 4 -> D residual GELU MLP), with FEATHer's instance norm
# ("_mix") or SparseTSF's own window-mean removal ("_mixM").  Answers "what does FEATHer
# do better than SparseTSF plus a small channel mixer?" - plan and pre-declared rule in
# manuscript/drafts/_records/control_sparsetsf_mix_plan.md.  Recipe (lr, period) = SparseTSF's.
def _build_sparsetsf_mix(args, n_features, seq_len, pred_len):
    from baselines.SparseTSF.wrapper import build_mix
    return build_mix(args, n_features, seq_len, pred_len, "linear", "revin")


def _build_sparsetsf_mixm(args, n_features, seq_len, pred_len):
    from baselines.SparseTSF.wrapper import build_mix
    return build_mix(args, n_features, seq_len, pred_len, "linear", "mean")


def _build_sparsetsf_mlp_mix(args, n_features, seq_len, pred_len):
    from baselines.SparseTSF.wrapper import build_mix
    return build_mix(args, n_features, seq_len, pred_len, "mlp", "revin")


def _build_sparsetsf_mlp_mixm(args, n_features, seq_len, pred_len):
    from baselines.SparseTSF.wrapper import build_mix
    return build_mix(args, n_features, seq_len, pred_len, "mlp", "mean")


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


# Edge-tier additions (2026-07-31) ??sub-1K / small-deployable competitors.
# (Classic practitioner baselines LSTM/GRU/TCN + XGBoost were archived to
# _archive/classic_baselines/ on 2026-07-31 and deferred to a revision;
# restore from there or git history e22236f if a reviewer asks for them.)
def _build_mixlinear(args, n_features, seq_len, pred_len):
    from baselines.MixLinear.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_cyclenet(args, n_features, seq_len, pred_len):
    from baselines.CycleNet.wrapper import build
    return build(args, n_features, seq_len, pred_len)


# 2026-10-06 (user): CMoS ICML 2025, super-lightweight with an edge claim; the
# paper we cite for "lightweight models do not mix channel values".
def _build_cmos(args, n_features, seq_len, pred_len):
    from baselines.CMoS.wrapper import build
    return build(args, n_features, seq_len, pred_len)


# 2026-10-06 (user, after the baseline-completeness check): TimeBase ICML'25
# (smallest, edge claim), PhaseFormer ICLR'26 (newest ~1K), SAMformer ICML'24
# (peer-reviewed lightweight model that mixes channel values).
def _build_timebase(args, n_features, seq_len, pred_len):
    from baselines.TimeBase.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_phaseformer(args, n_features, seq_len, pred_len):
    from baselines.PhaseFormer.wrapper import build
    return build(args, n_features, seq_len, pred_len)


def _build_samformer(args, n_features, seq_len, pred_len):
    from baselines.SAMformer.wrapper import build
    return build(args, n_features, seq_len, pred_len)


# NOTE ??deliberately excluded from the benchmark (cite in Related Work only):
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
    "SparseTSF_MLP": _build_sparsetsf_mlp,  # same-size rival (~4.5K, MCU-fitting)
    # 10-06 control arms: SparseTSF + FEATHer's mixing bus only (not benchmark baselines)
    "SparseTSF_mix":      _build_sparsetsf_mix,
    "SparseTSF_mixM":     _build_sparsetsf_mixm,
    "SparseTSF_MLP_mix":  _build_sparsetsf_mlp_mix,
    "SparseTSF_MLP_mixM": _build_sparsetsf_mlp_mixm,
    "TimeMixer":    _build_timemixer,
    "TimesNet":     _build_timesnet,
    "TQNet":        _build_tqnet,
    "LMS_AutoTSF":  _build_lms_autotsf,
    # R4-requested 2025-2026 SOTA ??addresses "outdated baselines" critique
    "DiPE_Linear":  _build_dipe_linear,    # DASFAA 2026
    "MDMLP_EIA":    _build_mdmlp_eia,      # AAAI 2026
    # Edge-tier additions (2026-07-31) ??sub-1K / small-deployable competitors
    "MixLinear":    _build_mixlinear,      # ICLR 2026, ~0.1K params (sub-1K rival)
    "CycleNet":     _build_cyclenet,       # NeurIPS 2024, learnable cycle
    "CMoS":         _build_cmos,           # ICML 2025, chunk-wise correlations (added 10-06)
    "TimeBase":     _build_timebase,       # ICML 2025, minimalist basis (added 10-06)
    "PhaseFormer":  _build_phaseformer,    # ICLR 2026, phase routing (added 10-06)
    "SAMformer":    _build_samformer,      # ICML 2024, channel-wise attention + SAM (added 10-06)
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
# Ablation now runs as `run.py --stage ablation` (arms in tools/results/protocol.py): one switch off
# per arm on the shipping model, using the flags the worker already exposes.
# See ABLATION_ARMS there. Head/DTK-variant ablations are the one thing that
# did not survive the move ??they need the archived modules updated to accept
# channel_mode / use_residual_mlp before they would be comparable.


# Per-method training defaults ??sourced from each paper's official ETTh1
# script (or run.py default when the script does not override). Following
# TFB (Hu et al., PVLDB 2024) and CF-JEPA (sibling project), we keep each
# method's paper-recommended LR and loss instead of forcing a single value
# across the lineup ??paper LRs span two orders of magnitude (1e-4 ??2e-2)
# so a unified LR would disadvantage some models by 10-20횞 the optimum.
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
    # are consistent (train=mse, eval=mse) ??l1-trained FEATHer was penalized on
    # the MSE table (2026-07-30 finding). An l1 variant is a supplementary MAE run.
    "FEATHer":      {"lr": 1e-3, "loss": "mse"},
    "DLinear":      {"lr": 5e-3, "loss": "mse"},
    "PatchTST":     {"lr": 1e-4, "loss": "mse"},
    "iTransformer": {"lr": 1e-4, "loss": "mse"},
    "FITS":         {"lr": 5e-4, "loss": "mse"},
    "SparseTSF":    {"lr": 2e-2, "loss": "mse"},
    "SparseTSF_MLP": {"lr": 2e-2, "loss": "mse"},  # same recipe, mlp head
    "SparseTSF_mix":      {"lr": 2e-2, "loss": "mse"},   # 10-06 control arms: host recipe
    "SparseTSF_mixM":     {"lr": 2e-2, "loss": "mse"},
    "SparseTSF_MLP_mix":  {"lr": 2e-2, "loss": "mse"},
    "SparseTSF_MLP_mixM": {"lr": 2e-2, "loss": "mse"},
    "TimeMixer":    {"lr": 1e-2, "loss": "mse"},
    "TimesNet":     {"lr": 1e-4, "loss": "mse"},
    "TQNet":        {"lr": 1e-3, "loss": "mse"},
    "LMS_AutoTSF":  {"lr": 1e-4, "loss": "mse"},
    # DiPE-Linear: base.yaml lr=1e-3, t_loss='mse', single-expert
    "DiPE_Linear":  {"lr": 1e-3, "loss": "mse"},
    # MDMLP-EIA: run.py default lr=1e-4
    "MDMLP_EIA":    {"lr": 1e-4, "loss": "mse"},
    # Edge-tier additions (2026-07-31) ??published script lr (the argparse
    # default 1e-4 is a fallback every author script overrides): MixLinear
    # scripts use 0.03, CycleNet scripts use 0.005.
    "MixLinear":    {"lr": 3e-2, "loss": "mse"},
    "CycleNet":     {"lr": 5e-3, "loss": "mse"},
    # CMoS: model/CMoS/config.toml default lr 8e-4, MSELoss (Pipeline/OneMulEdit.py)
    "CMoS":         {"lr": 8e-4, "loss": "mse"},
    # ETTh1 official settings (same rule as SparseTSF): TimeBase scripts/etth1.sh
    # lr 1e-1; PhaseFormer run_etth1.py (H<=336 lightweight config) lr 1e-3;
    # SAMformer PyTorch trainer default lr 1e-3 (+ SAM rho 0.5 in the worker).
    "TimeBase":     {"lr": 1e-1, "loss": "mse"},
    "PhaseFormer":  {"lr": 1e-3, "loss": "mse"},
    "SAMformer":    {"lr": 1e-3, "loss": "mse"},
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
# Fair-comparison framing (revised 2026-06-17 ??per-dataset for ALL):
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

# FEATHer FINAL canonical mfg config ("N" cell) ??shared across all 6 mfg
# datasets (one-config protocol); merged with a per-dataset `period` below.
# FROZEN 2026-09-04 (docs/MODEL_FINAL.md).  Until now these dicts still held the
# pre-freeze 938p config, so `--model FEATHer` silently built
# the OLD model -- the shipping sweeps only got the frozen one because they
# were launched with explicit CLI flags.  The two variants differ in exactly
# one key (channel_mode); everything else is shared, and only `period` is
# per-dataset.  d_state / kernel_size are GONE: they fed the DTK, which is
# removed (the constructor still takes them, from the worker defaults).
# RE-FROZEN 2026-09-23: mlp_rank 16 -> 4 (mixing summary 16 -> 4 slots).
# Steel H96 452p -> 296p; chip 24 -> 43 sensors (recompute port feather2r).
# Freeze rule passed on the 72-run: mfg 0.6990 vs 0.6943 (+0.66 %), worst
# dataset WindSCADA +2.42 %.  Shipping sweeps: results/feather_{mfg,ltsf}_
# candR4*.csv (launched with explicit --mlp_rank 4, so they did not depend on
# this dict; it is updated so that `--model FEATHer` alone builds the same).
_FEATHER_FROZEN: Dict = {
    "no_in_proj": True,          # linear channel mixer removed
    "num_bands": 3,              # {conv k=3, conv k=5, AvgPool4+interp}
    "point_kernel": 3,           # keep THIS form, not --band_kernels (RNG stream)
    "dtk_mode": "none",          # shared temporal kernel removed
    "gate_mode": "banded",       # per-band spectral energy gate
    "use_residual_mlp": True,
    "mlp_rank": 4,               # was 16 until 2026-09-23
    "lambda_spec": 0.01,
}

_FEATHER_MFG_N: Dict = {**_FEATHER_FROZEN, "channel_mode": "mix"}

# FEATHer FINAL canonical LTSF config ("CI" cell) ??channel-INDEPENDENT +
# residual; per-dataset d_state/kernel/num_bands/period merged in below.
_FEATHER_LTSF_CI: Dict = {**_FEATHER_FROZEN, "channel_mode": "ci_full"}

_DATASET_OVERRIDES: Dict = {

    # ---- DLinear (cure-lab/LTSF-Linear) -----------------------------------
    # scripts/EXP-LongForecasting/Linear/{dataset}.sh
    ("DLinear", "ETTh1"):       {"lr": 5e-3},
    # DIVERGENCE-FIX (2026-08-31): the official script value is 5e-2, but it is
    # paired with lradj=type1 (lr halved EVERY epoch) and train_epochs=10, so
    # 5e-2 is a one-epoch kick that is below 3e-3 by epoch 5. Under our unified
    # CosineAnnealingLR(T_max=50) the same number stays at ~5e-2 for ten epochs
    # and the run diverges (best_epoch 0/0/2, MSE 1.47-1.65x the cross-model
    # median at H96/336/720). Re-selected on VALIDATION loss only, from the grid
    # {5e-4, 1e-3, 5e-3, 1e-2, 5e-2}: 5e-3 wins at every horizon (val mean
    # 0.3700 vs 0.4365 for 5e-2). Probe: results/probes/probe_dlinear_etth2.csv.
    ("DLinear", "ETTh2"):       {"lr": 5e-3},
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
    # scripts/FITS/{dataset}.sh ??uniform lr=5e-4; cut_freq derived per
    # (seq_len, pred_len) via H_order which we approximate with seq_len/4
    # in the wrapper. No per-dataset overrides needed.

    # ---- SparseTSF (lss-1138/SparseTSF) ----------------------------------
    # scripts/SparseTSF/linear/{dataset}.sh ??period_len follows native
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
    # scripts/long_term_forecast/*/TimesNet*.sh ??d_model swings 16 ??512
    ("TimesNet", "ETTh1"):       {"tn_d_model": 16,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "ETTh2"):       {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "ETTm1"):       {"tn_d_model": 64,  "tn_d_ff": 64,  "lr": 1e-4},
    ("TimesNet", "ETTm2"):       {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},
    ("TimesNet", "Electricity"): {"tn_d_model": 256, "tn_d_ff": 512, "lr": 1e-4},
    ("TimesNet", "Exchange"):    {"tn_d_model": 64,  "tn_d_ff": 64,  "lr": 1e-4},
    ("TimesNet", "Traffic"):     {"tn_d_model": 512, "tn_d_ff": 512, "lr": 1e-4},
    ("TimesNet", "Weather"):     {"tn_d_model": 32,  "tn_d_ff": 32,  "lr": 1e-4},

    # ---- TQNet (ACAT-SCUT/TQNet) -----------------------------------------
    # scripts/TQNet/{dataset}.sh ??cycle is data-frequency-tied:
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
    # lr 1e-3 everywhere (configs/models/DiPE/base.yaml). Per-dataset model flags
    # (num_experts, loss_alpha, use_revin) come from scripts/DiPE.sh -> see the
    # 2026-10-07 correction block below (this comment said "none published" until 10-07).

    # ---- MDMLP-EIA (zh1985csuccsu/MDMLP-EIA, AAAI 2026) ------------------
    # run.py exposes one default config (lr=1e-4); upstream sweep scripts
    # publish dataset-specific d_model only via CLI flags at run time, not
    # via committed config files. Entries omitted (falls back to
    # _METHOD_DEFAULTS).

    # ---- FEATHer (this work) ??per-dataset val winners from
    # run_hp_search.py --summary (OFAT 512 + --validate combo verdict 32,
    # 2026-07-22 RE-FREEZE under corrected spectral loss + standard Weather
    # split; supersedes the 2026-07-06 freeze). Each row = best-ranked among
    # {base, single-axis variants, validated combo}. The --validate step
    # REJECTED the raw OFAT stack for 4/8 sets (ETTh1/ETTh2/Exchange/Traffic
    # fell back to base + one axis), confirming the flat axes (kernel/貫, and
    # num_bands off-Traffic) are noise ??stacking them lost to base.
    # d_state=16 now only on ETTm1 (D=7, sub-1K OK) and Electricity (already a
    # D짼-stress set, 115K). H=720 SPK overflow is pre-existing (affects base
    # too) ??sub-1K claim stays H=96-scoped. These feed the LTSF
    # *generalization* section; the MFG main-table canonical arch stays frozen
    # at base (d8/k7/p12/B3) ??nothing in this LTSF OFAT contradicts it.
    # FEATHer LTSF = FINAL "CI" cell (2026-08-13, grid 27/27 done). LTSF favors
    # channel-INDEPENDENCE (ci_full) over the mfg channel-mix: CIN OVERALL 0.3588
    # ties FITS 0.3582 (4/6 datasets won), beats the base-L cell (0.3647). Per-
    # dataset tuning (symmetric w/ baselines that tune per-dataset); val-selected
    # d_state/kernel/num_bands below, period = per-dataset FFT. loss=mse, lr=1e-3
    # from _METHOD_DEFAULTS. Old L-cell (mix, no residual) OFAT configs replaced.
    # 2026-09-04: per-dataset ARCH tuning is GONE -- one frozen config on all 12
    # datasets, `period` (FFT-detected) the only per-dataset key.  docs/MODEL_FINAL.md
    # section 6: per-dataset val selection scored 0.2504 vs 0.2501 for the single
    # setting, i.e. the grid was inside noise, and the baselines DO tune per
    # dataset -- so dropping it is the conservative (unfavourable) choice and a
    # fairness argument we can state.
    ("FEATHer", "ETTh1"):    {**_FEATHER_LTSF_CI, "period": 6},
    ("FEATHer", "ETTh2"):    {**_FEATHER_LTSF_CI, "period": 12},
    ("FEATHer", "ETTm1"):    {**_FEATHER_LTSF_CI, "period": 6},
    ("FEATHer", "ETTm2"):    {**_FEATHER_LTSF_CI, "period": 6},
    ("FEATHer", "Weather"):  {**_FEATHER_LTSF_CI, "period": 6},
    ("FEATHer", "Exchange"): {**_FEATHER_LTSF_CI, "period": 8},
    # Electricity / Traffic keep their OFAT-era arch (they were never in the
    # CIN grid ??321 and 862 channels made the mix-mode search prohibitive),
    # but the CIN cell is merged in so they run the same shipping variant as
    # the other six. Without this they silently trained the old mix-mode
    # FEATHer-L while every other LTSF row was CIN. Parameter counts confirm
    # the pairing reproduces results/probes/probe_ltsf_highD.csv exactly
    # (Electricity 304/320/344/408, Traffic 226/290/386/642 at H=96..720).
    # Channel-independence is what keeps them sub-1K at D=862 at all.
    ("FEATHer", "Electricity"): {**_FEATHER_LTSF_CI, "period": 24, "lambda_spec": 0.0,
                                 "lr": 0.005},
    ("FEATHer", "Traffic"):     {**_FEATHER_LTSF_CI, "period": 12, "lr": 0.005},

    # ---- Manufacturing freq-tied period/cycle (data-driven, 2026-07-13) ------
    # FFT-detected dominant period on each train split (detrended). Applied by
    # the SAME rule to all three period-models, each per its own constraint ??
    # exactly mirroring the LTSF convention (SparseTSF small period_len vs TQNet
    # daily cycle on the same data):
    #   FEATHer/SparseTSF: period must divide seq_len(96)/pred_len -> usable
    #                      range <=48 (detected dominant period in [2,48]).
    #   TQNet:             cycle unconstrained -> the daily cycle (samples/day;
    #                      BlastFurnace hourly->24, MetroPT3 detected ~5h=298).
    # This is the data-FREQUENCY-TIED axis (NOT the frozen mfg arch d8/k7/B3);
    # it gives SparseTSF a fair period instead of a crippling hourly default.
    # NOTE: lr for these mfg cells is MERGED IN later from run_lr_search.py
    # --summary ??EDIT these dicts to add "lr", do NOT paste duplicate keys.
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
    # 2026-10-06: TimeBase / PhaseFormer period = the SparseTSF FFT period (same rule).
    ("TimeBase", "Steel"): {"timebase_period_len": 24},
    ("TimeBase", "WindSCADA"): {"timebase_period_len": 48},
    ("TimeBase", "BlastFurnace"): {"timebase_period_len": 48},
    ("TimeBase", "MetroPT3"): {"timebase_period_len": 24},
    ("TimeBase", "N2O"): {"timebase_period_len": 24},
    ("TimeBase", "EWELD"): {"timebase_period_len": 24},
    ("PhaseFormer", "Steel"): {"phaseformer_period_len": 24},
    ("PhaseFormer", "WindSCADA"): {"phaseformer_period_len": 48},
    ("PhaseFormer", "BlastFurnace"): {"phaseformer_period_len": 48},
    ("PhaseFormer", "MetroPT3"): {"phaseformer_period_len": 24},
    ("PhaseFormer", "N2O"): {"phaseformer_period_len": 24},
    ("PhaseFormer", "EWELD"): {"phaseformer_period_len": 24},
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


# Variants that must inherit another model's per-dataset overrides. SparseTSF's
# two heads share every dataset-level choice ??the FFT-detected period_len and
# the published lr ??and only differ in the head itself, so the MLP entry reads
# the linear entry's rows. Duplicating the 14 cells would let them drift apart,
# and a silent drift here means the two variants are no longer the controlled
# comparison the table claims.
_OVERRIDE_ALIASES = {"SparseTSF_MLP": "SparseTSF",
                     # 10-06 control arms share SparseTSF's per-dataset period / lr
                     "SparseTSF_mix": "SparseTSF", "SparseTSF_mixM": "SparseTSF",
                     "SparseTSF_MLP_mix": "SparseTSF", "SparseTSF_MLP_mixM": "SparseTSF"}


# ---- 2026-10-06 LTSF settings of the four added baselines (official scripts; L = 96 by protocol).
# CMoS / TimeBase tune per horizon -> (model, data, H) keys.  Exchange: SAMformer has an official
# setting (run_script.sh exchange_rate); CMoS / TimeBase / PhaseFormer have none -> chosen on validation
# since 10-07 (tools/results/protocol.py LTSF_NO_OFFICIAL); the rows below are their period entries.
_DATASET_OVERRIDES.update({
    ("CMoS", "ETTh1"): {"cmos_seg_size": 24, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 24, "lr": 8e-05},
    ("CMoS", "ETTh2"): {"cmos_seg_size": 8, "cmos_num_map": 2, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 24, "lr": 5e-05},
    ("CMoS", "ETTm1"): {"cmos_seg_size": 4, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 96, "lr": 5e-05},
    ("CMoS", "ETTm2"): {"cmos_seg_size": 2, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 96, "lr": 2e-05},
    ("CMoS", "Weather"): {"cmos_seg_size": 4, "cmos_num_map": 4, "cmos_kernel_size": 4, "cmos_conv_stride": 2, "cmos_dropout": 0.0008, "cmos_use_pi": False, "cmos_period": 144, "lr": 0.0008},
    ("CMoS", "Electricity"): {"cmos_seg_size": 8, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 168, "lr": 0.0008},
    ("CMoS", "Traffic"): {"cmos_seg_size": 4, "cmos_num_map": 8, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "cmos_dropout": 0.05, "cmos_use_pi": True, "cmos_period": 168, "lr": 0.0008},
    ("CMoS", "ETTh1", 96): {"cmos_seg_size": 24, "cmos_num_map": 2, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "lr": 5e-05},
    ("CMoS", "ETTh1", 192): {"cmos_seg_size": 24, "cmos_num_map": 4, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 0.0002},
    ("CMoS", "ETTh1", 336): {"cmos_seg_size": 24, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "lr": 8e-05},
    ("CMoS", "ETTh1", 720): {"cmos_seg_size": 24, "cmos_num_map": 2, "cmos_kernel_size": 24, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTh2", 96): {"cmos_seg_size": 4, "cmos_num_map": 8, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTh2", 192): {"cmos_seg_size": 4, "cmos_num_map": 8, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 5e-05},
    ("CMoS", "ETTh2", 336): {"cmos_seg_size": 4, "cmos_num_map": 4, "cmos_kernel_size": 12, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTh2", 720): {"cmos_seg_size": 4, "cmos_num_map": 2, "cmos_kernel_size": 12, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTm1", 96): {"cmos_seg_size": 2, "cmos_num_map": 8, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 5e-05},
    ("CMoS", "ETTm1", 192): {"cmos_seg_size": 2, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "lr": 0.0008},
    ("CMoS", "ETTm1", 336): {"cmos_seg_size": 2, "cmos_num_map": 8, "cmos_kernel_size": 24, "cmos_conv_stride": 4, "lr": 5e-05},
    ("CMoS", "ETTm1", 720): {"cmos_seg_size": 2, "cmos_num_map": 4, "cmos_kernel_size": 24, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTm2", 96): {"cmos_seg_size": 2, "cmos_num_map": 2, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTm2", 192): {"cmos_seg_size": 2, "cmos_num_map": 4, "cmos_kernel_size": 8, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTm2", 336): {"cmos_seg_size": 2, "cmos_num_map": 4, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "ETTm2", 720): {"cmos_seg_size": 2, "cmos_num_map": 2, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 2e-05},
    ("CMoS", "Weather", 96): {"cmos_seg_size": 4, "cmos_num_map": 4, "cmos_kernel_size": 4, "cmos_conv_stride": 4, "lr": 0.0005},
    ("TimeBase", "ETTh1", 96): {"timebase_orthogonal_weight": 0.16, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.1},
    ("TimeBase", "ETTh1", 192): {"timebase_orthogonal_weight": 0.16, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.4},
    ("TimeBase", "ETTh1", 336): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.4},
    ("TimeBase", "ETTh1", 720): {"timebase_orthogonal_weight": 0.12, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.05},
    ("TimeBase", "ETTh2", 96): {"timebase_orthogonal_weight": 0.2, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.2},
    ("TimeBase", "ETTh2", 192): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 24, "timebase_basis_num": 4, "lr": 0.06},
    ("TimeBase", "ETTh2", 336): {"timebase_orthogonal_weight": 0.12, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.4},
    ("TimeBase", "ETTh2", 720): {"timebase_orthogonal_weight": 0.12, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.4},
    ("TimeBase", "ETTm1", 96): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 18, "lr": 0.02},
    ("TimeBase", "ETTm1", 192): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 20, "lr": 0.02},
    ("TimeBase", "ETTm1", 336): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 4, "timebase_basis_num": 20, "lr": 0.02},
    ("TimeBase", "ETTm1", 720): {"timebase_orthogonal_weight": 0.12, "timebase_period_len": 6, "timebase_basis_num": 20, "lr": 0.01},
    ("TimeBase", "ETTm2", 96): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 20, "lr": 0.01},
    ("TimeBase", "ETTm2", 192): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 20, "lr": 0.01},
    ("TimeBase", "ETTm2", 336): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 20, "lr": 0.01},
    ("TimeBase", "ETTm2", 720): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 6, "timebase_basis_num": 20, "lr": 0.01},
    ("TimeBase", "Weather", 96): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 6, "lr": 0.02, "timebase_individual": 1},
    ("TimeBase", "Weather", 192): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 6, "lr": 0.02, "timebase_individual": 1},
    ("TimeBase", "Weather", 336): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 4, "timebase_basis_num": 6, "lr": 0.05, "timebase_individual": 1},
    ("TimeBase", "Weather", 720): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 4, "timebase_basis_num": 6, "lr": 0.05, "timebase_individual": 1},
    ("TimeBase", "Electricity", 96): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.02},
    ("TimeBase", "Electricity", 192): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.02},
    ("TimeBase", "Electricity", 336): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.02},
    ("TimeBase", "Electricity", 720): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.08},
    ("TimeBase", "Traffic", 96): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 6, "lr": 0.05},
    ("TimeBase", "Traffic", 192): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 8, "lr": 0.03},
    ("TimeBase", "Traffic", 336): {"timebase_orthogonal_weight": 0.08, "timebase_period_len": 24, "timebase_basis_num": 15, "lr": 0.03},
    ("TimeBase", "Traffic", 720): {"timebase_orthogonal_weight": 0.04, "timebase_period_len": 24, "timebase_basis_num": 8, "lr": 0.03},
    ("TimeBase", "Exchange"): {"timebase_period_len": 24},
    ("PhaseFormer", "ETTh1"): {"phaseformer_layers": 3, "phaseformer_latent_dim": 4, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "ETTh2"): {"phaseformer_layers": 1, "phaseformer_latent_dim": 8, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "ETTm1"): {"phaseformer_layers": 2, "phaseformer_latent_dim": 8, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "ETTm2"): {"phaseformer_layers": 2, "phaseformer_latent_dim": 8, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "Weather"): {"phaseformer_layers": 3, "phaseformer_latent_dim": 8, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "Electricity"): {"phaseformer_layers": 2, "phaseformer_latent_dim": 8, "phaseformer_routers": 8, "phaseformer_heads": 1, "phaseformer_pos_embed": True, "phaseformer_period_len": 24, "lr": 0.002},
    ("PhaseFormer", "Traffic"): {"phaseformer_layers": 2, "phaseformer_latent_dim": 32, "phaseformer_routers": 1, "phaseformer_heads": 8, "phaseformer_pos_embed": False, "phaseformer_period_len": 24, "lr": 0.001},
    ("PhaseFormer", "Exchange"): {"phaseformer_period_len": 24},
    ("SAMformer", "ETTh1"): {"samformer_rho": 0.7, "lr": 0.001},
    ("SAMformer", "ETTh2"): {"samformer_rho": 0.7, "lr": 0.001},
    ("SAMformer", "ETTm1"): {"samformer_rho": 0.7, "lr": 0.001},
    ("SAMformer", "ETTm2"): {"samformer_rho": 0.7, "lr": 0.001},
    ("SAMformer", "Weather"): {"samformer_rho": 0.7, "lr": 0.0001},
    ("SAMformer", "Electricity"): {"samformer_rho": 0.7, "lr": 0.0001},
    ("SAMformer", "Traffic"): {"samformer_rho": 0.7, "lr": 0.0001},
    ("SAMformer", "Exchange"): {"samformer_rho": 0.7, "lr": 0.001},
})


# ---- 2026-10-07 protocol corrections (docs/EXPERIMENT_PROTOCOL.md; official scripts re-read line by line,
# manuscript/drafts/_records/baseline_official_configs.md).  Rules used below:
#   * LTSF = each model's FINAL per-(dataset[, H]) script values; sweep-only or missing datasets
#     take the model's ETTh1 setting (period-type keys stay the dataset's own value).
#   * L = 96 scripts win where they exist (CycleNet Linear-Input-96: lr 0.01 everywhere).
#   * training-procedure choices (Huber loss, epochs, lradj, FITS train_mode) stay unified.
_MLP_LTSF = {"ETTh1": 2e-3, "ETTh2": 1e-3, "ETTm1": 1e-3, "ETTm2": 1e-3, "Weather": 5e-2,
             "Electricity": 2e-2, "Traffic": 1e-2, "Exchange": 2e-3}      # mlp/<ds>.sh; Exchange = ETTh1
_DIPE_LTSF = {  # scripts/DiPE.sh (num_experts, loss_alpha, use_revin); Traffic / Exchange = ETTh1
    "ETTh1": (1, 1.0, True), "ETTh2": (3, 0.9, True), "ETTm1": (1, 1.0, True), "ETTm2": (1, 0.9, True),
    "Weather": (4, 0.9, False), "Electricity": (4, 0.3, False), "Traffic": (1, 1.0, True),
    "Exchange": (1, 1.0, True)}
_MIX_LTSF = {   # *_best.sh: lr, (alpha, lpf) per H; ETTm1 / ETTm2 / Weather (sweep only) and Exchange = ETTh1
    "ETTh1": (3e-2, {96: (0.95, 1), 192: (0.95, 5), 336: (0.99, 2), 720: (0.99, 19)}),
    "ETTh2": (2e-2, {96: (0.5, 19), 192: (0.01, 19), 336: (0.01, 15), 720: (0.5, 19)}),
    "Electricity": (3e-2, {96: (0.5, 19), 192: (0.99, 15), 336: (0.99, 15), 720: (0.5, 15)}),
    "Traffic": (3e-2, {96: (0.5, 19), 192: (0.01, 15), 336: (0.5, 19), 720: (0.9, 19)})}
for _d in ("ETTm1", "ETTm2", "Weather", "Exchange"):
    _MIX_LTSF[_d] = _MIX_LTSF["ETTh1"]
_FITS_LTSF = {  # *_best.sh: (H_order per H, base_T, individual); Electricity / Traffic (sweep only),
    #             Exchange = ETTh1 values (base_T = the dataset's daily cycle where it has one)
    "ETTh1": ({96: 6, 192: 6, 336: 6, 720: 6}, 24, False), "ETTh2": ({96: 6, 192: 6, 336: 6, 720: 6}, 24, False),
    "ETTm1": ({96: 14, 192: 14, 336: 14, 720: 14}, 96, False),
    "ETTm2": ({96: 14, 192: 14, 336: 14, 720: 14}, 96, False),
    "Weather": ({96: 12, 192: 12, 336: 8, 720: 12}, 144, True),
    "Electricity": ({96: 6, 192: 6, 336: 6, 720: 6}, 24, False),
    "Traffic": ({96: 6, 192: 6, 336: 6, 720: 6}, 24, False),
    "Exchange": ({96: 6, 192: 6, 336: 6, 720: 6}, 24, False)}
# PhaseFormer original repo, run_<ds>.py get_best_config_for_horizon (via PhaseFormer_TSL lineage):
# (layers, latent, enc_hidden, pred_hidden, routers, heads, lr, attn_dropout); Exchange = ETTh1.
_PF = lambda L, lat, eh, ph, r, h, lr=1e-3, dr=0.1: dict(
    phaseformer_layers=L, phaseformer_latent_dim=lat, phaseformer_enc_hidden=eh, phaseformer_pred_hidden=ph,
    phaseformer_routers=r, phaseformer_heads=h, phaseformer_attn_dropout=dr, phaseformer_pos_embed=True, lr=lr)
_PF_LTSF = {
    "ETTh1": {96: _PF(3, 4, 16, 32, 8, 1), 192: _PF(3, 4, 16, 32, 8, 1), 336: _PF(3, 4, 16, 32, 8, 1),
              720: _PF(3, 32, 128, 256, 16, 2, lr=1.5e-4, dr=0.0)},
    "ETTh2": {96: _PF(1, 8, 32, 64, 8, 1), 192: _PF(1, 8, 32, 64, 8, 1), 336: _PF(1, 8, 32, 64, 8, 1),
              720: _PF(1, 4, 8, 8, 4, 1)},
    "ETTm1": {96: _PF(2, 8, 32, 64, 8, 1), 192: _PF(2, 8, 32, 64, 8, 1), 336: _PF(1, 8, 32, 64, 8, 1),
              720: _PF(2, 8, 32, 64, 8, 1)},
    "ETTm2": {96: _PF(2, 8, 32, 64, 8, 1), 192: _PF(1, 8, 32, 64, 8, 1), 336: _PF(1, 8, 32, 64, 8, 1),
              720: _PF(1, 8, 32, 64, 8, 1)},
    "Weather": {96: _PF(3, 8, 32, 64, 8, 1), 192: _PF(2, 8, 32, 64, 8, 1), 336: _PF(2, 8, 32, 64, 8, 1),
                720: _PF(2, 8, 32, 64, 8, 1)},
    "Electricity": {96: _PF(2, 8, 32, 64, 8, 1, lr=2e-3), 192: _PF(1, 128, 16, 32, 4, 4),
                    336: _PF(2, 8, 32, 64, 8, 1), 720: _PF(1, 128, 16, 32, 4, 4)},
    "Traffic": {96: _PF(2, 32, 64, 128, 1, 8), 192: _PF(1, 128, 16, 32, 4, 4), 336: _PF(1, 128, 16, 32, 4, 4),
                720: _PF(1, 128, 16, 32, 4, 4)}}
_PF_LTSF["Exchange"] = _PF_LTSF["ETTh1"]

_P10 = {}
for _d, _lr in _MLP_LTSF.items():
    _per = _DATASET_OVERRIDES.get(("SparseTSF", _d), {}).get("sparsetsf_period_len", 24)
    _P10[("SparseTSF_MLP", _d)] = {"sparsetsf_period_len": _per, "lr": _lr,
                                   **({"sparsetsf_d_model": 512} if _d == "Traffic" else {})}
for _key, _v in list(_DATASET_OVERRIDES.items()):        # MLP head: same mfg FFT period as SparseTSF
    if len(_key) == 2 and _key[0] == "SparseTSF" and _key[1] not in _MLP_LTSF:
        _P10[("SparseTSF_MLP", _key[1])] = {"sparsetsf_period_len": _v["sparsetsf_period_len"]}
_P10[("DLinear", "Weather")] = {"lr": 1e-4}                   # weather.sh passes no lr -> argparse 1e-4
_P10[("DLinear", "ETTm2", 336)] = {"lr": 1e-2}                # ettm2.sh:52
_P10[("DLinear", "ETTm2", 720)] = {"lr": 1e-2}                # ettm2.sh:66
for _d, (_e, _a, _rv) in _DIPE_LTSF.items():
    _P10[("DiPE_Linear", _d)] = {"dipe_num_experts": _e, "dipe_loss_alpha": _a, "dipe_use_revin": _rv}
for _d, (_lr, _byh) in _MIX_LTSF.items():
    _P10[("MixLinear", _d)] = {**_DATASET_OVERRIDES.get(("MixLinear", _d), {}), "lr": _lr}
    for _h, (_al, _lp) in _byh.items():
        _P10[("MixLinear", _d, _h)] = {"mixlinear_alpha": _al, "mixlinear_lpf": _lp}
for _d, (_ho, _bt, _ind) in _FITS_LTSF.items():
    _P10[("FITS", _d)] = {"fits_base_t": _bt, "fits_individual": _ind}
    for _h, _o in _ho.items():
        _P10[("FITS", _d, _h)] = {"fits_h_order": _o}
for _d, _byh in _PF_LTSF.items():
    _P10[("PhaseFormer", _d)] = {"phaseformer_period_len": 24}
    for _h, _cfg in _byh.items():
        _P10[("PhaseFormer", _d, _h)] = dict(_cfg)
# Exchange (no official setting in these repos) = the ETTh1 setting, dataset period 24
_P10[("CMoS", "Exchange")] = {**_DATASET_OVERRIDES[("CMoS", "ETTh1")], "cmos_period": 24}
for _h in (96, 192, 336, 720):
    _P10[("CMoS", "Exchange", _h)] = dict(_DATASET_OVERRIDES[("CMoS", "ETTh1", _h)])
    _P10[("TimeBase", "Exchange", _h)] = dict(_DATASET_OVERRIDES[("TimeBase", "ETTh1", _h)])
_DATASET_OVERRIDES.update(_P10)
_METHOD_DEFAULTS["CycleNet"] = {"lr": 1e-2, "loss": "mse"}    # Linear-Input-96 scripts (was 5e-3 = L336/720)
_OVERRIDE_ALIASES.pop("SparseTSF_MLP", None)                  # MLP head now has its own official lr rows


def get_dataset_overrides(name, data, pred_len=None):
    """Per-dataset architecture/lr overrides for (name, data).

    Returns an empty dict when the (model, dataset) cell has no entry ??
    the caller should layer this on top of `get_method_defaults(name)`.
    2026-10-06: a (name, data, pred_len) entry, if present, is merged on top
    (official scripts that tune per horizon: CMoS, TimeBase).  Callers that
    pass no pred_len get exactly the old behaviour.
    """
    if (name, data) not in _DATASET_OVERRIDES and name in _OVERRIDE_ALIASES:
        name = _OVERRIDE_ALIASES[name]
    out = dict(_DATASET_OVERRIDES.get((name, data), {}))
    if pred_len is not None:
        out.update(_DATASET_OVERRIDES.get((name, data, int(pred_len)), {}))
    return out


def list_models():
    """Main benchmark lineup ??FEATHer + the reproduced baselines."""
    return sorted(_BUILDERS.keys())



