# -*- coding: utf-8 -*-
"""
Forecasting orchestrator.

User-facing wrapper. Reads `results/fcst_results.csv`, figures out which
(exp_tag, model, dataset, pred_len, seed) combos are still missing, and
dispatches the missing ones to the worker (`scripts/benchmarks/run_forecast.py`)
one (model, dataset, pred_len) at a time. Re-running this script picks up
where it left off ??completed seeds are skipped.

Usage:
    python run_forecast.py --check                       # show status
    python run_forecast.py                               # run all missing
    python run_forecast.py --model FEATHer               # one model
    python run_forecast.py --data ETTh1 --pred_len 96    # one combo
    python run_forecast.py --num_seeds 5 --exp_tag main  # main 5-seed sweep
    python run_forecast.py --exclude TimesNet,S_Mamba    # skip heavy models
    python run_forecast.py --ngpu 2                      # dynamic 2-GPU queue

Component ablation (FEATHer only ??each arm removes one thing WE added, on the
shipping model; see ABLATION_ARMS below):
    python run_forecast.py --ablation_arm all --group mfg --num_seeds 1
                                                         # 6 arms x 24 = 144 runs
    python run_forecast.py --ablation_arm gateuni,band2 --group mfg
                                                         # subset
    python run_forecast.py --check --ablation_arm all --group mfg --num_seeds 1
                                                         # count first
    python tools/paper/ablation_summary.py --per_dataset # verdict (>=2% = signal)
Rows are tagged abl_<arm> and routed to results/ablation.csv, so the headline
CSV stays clean; the main sweep is the baseline and is never re-run.

Multi-GPU: `--ngpu N` runs a CF-JEPA-style dynamic queue ??one worker
subprocess per GPU (indices --gpu .. --gpu+N-1); a free GPU grabs the next
(model, dataset, pred_len) job, so fast models never leave a GPU idle.
Single specific GPU: `--ngpu 1 --gpu 0` pins everything to cuda:0.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (list_models,
                       get_method_defaults, get_dataset_overrides)
from utils.dispatch_queue import run_on_gpus


# -----------------------------------------------------------------------------
# Defaults ??adjust here if the main sweep changes
# -----------------------------------------------------------------------------

DEFAULT_MODELS = list_models()  # all registered

# Manufacturing datasets ??MAIN table (JMS pivot, 2026-06-25).
# 3 real continuous sensor streams, all D<=14 (sub-1K holds across the table).
# TEP dropped 2026-06-26 (anomaly-detection benchmark, not forecasting; D=50
# broke sub-1K). C-MAPSS replaces it but as a SEPARATE short-horizon section
# (CMAPSS_DATASETS) ??its run-to-failure trajectories are too short for the
# [96..720] long horizons (see Dataset_CMAPSS / CMAPSS_PREDS).
# PMSM dropped 2026-07-02: at native 2Hz the [96..720] horizons span only
# 48s-6min of slow rotor-thermal drift ??copy-last persistence already scores
# MSE 0.004 (target) / 0.117 (all-channel mean) at H=96, so the cells are
# trivially easy; a multi-session+downsample rebuild would only support short
# horizons anyway. Data + prep recipe kept on disk like TEP.
# BlastFurnace added 2026-07-13 (Mendeley 6d7jbc7tb5, Data in Brief 2025):
# 27 hourly process channels + Si (hot-metal silicon = product-quality target,
# asof-merged). D=28 -> above sub-1K, so it is the manufacturing STRESS/breadth
# tier (like Weather/Electricity in LTSF); the sub-1K claim stays on the D<=14
# core (Steel/WindSCADA/MetroPT3/N2O/EWELD). Forecasting-native, gapless, no
# LTSF precedent -> "first to cast as edge-PdM LTSF" holds.
# MetroPT3 added 2026-07-13 (UCI #791, metro-train APU compressor): 7 analog
# channels @ 1-min mean, D=7 -> sub-1K CORE. Persistence-audited non-trivial
# (0.80-1.14 std). Native anomaly-detection set cast as forecasting (TIME
# precedent). Adds an air-compressor PdM domain to the manufacturing lineup.
# N2O added 2026-07-13 (Avedore WWTP, Mendeley xmbxhscgpr): 14 continuous
# process channels @ 5-min mean, D=14 -> sub-1K core. Persistence-audited
# non-trivial (1.10-1.53 std). Forecasting-native. Adds a continuous
# industrial-process (bio-chemical) domain; frame as "industrial process
# monitoring" (it is a utility process, not discrete manufacturing).
# Manufacturing MAIN table ??FINAL 6 (locked 2026-07-14). All real, continuous,
# non-trivial, domain-diverse: Steel (energy), WindSCADA (wind, forecasting
# precedent), BlastFurnace (iron process, gapless), N2O (WWTP process),
# MetroPT3 (compressor machine-sensor), EWELD (factory load ??6-yr ANNUAL cycle
# showcases the multi-band head + labeled extreme-weather windows = real
# distribution-shift robustness). GasTurbine fully REMOVED 2026-07-14 (data +
# loader + prep deleted ??data-verified discontinuity: ~15% hours missing/year,
# synthetic index mixes non-adjacent real times = broken sliding window; rejected
# as BROKEN, not deferred-shelf). Vetted+rejected likewise: TEP/Machine-health/
# AI4I (synthetic), Gas-Sensor (mediocre lab). CMAPSS/CMAPSS3/PMSM = PdM shelf
# (good data, deferred section ??kept).
MFG_DATASETS = ["Steel", "WindSCADA", "BlastFurnace", "MetroPT3", "N2O", "EWELD"]
MFG_PREDS = [96, 192, 336, 720]

# 2026-07-27: UNIFIED dispatch-order priority across mfg AND LTSF ??cheap
# datasets run first, expensive ones last, so the fast datasets finish first
# and the slow tail doesn't block partial results. Same rule for every group
# (mfg / ltsf / all) so the two sweeps behave identically. Tier ~ dispatch
# cost ??rows 횞 channels (a rough wall-clock proxy). Lower = runs earlier.
# CMAPSS shelf datasets aren't listed ??default tier (5); they only run in the
# isolated `--group cmapss`, where they simply tie and keep (model, name) order.
_DATA_DISPATCH_ORDER = {
    # ---- cheap (small rows 횞 few channels) ----
    "Exchange":     0,   # 7.6k 횞 8
    "ETTh1":        1,   # 17k  횞 7
    "ETTh2":        1,   # 17k  횞 7
    "Steel":        2,   # 35k  횞 7
    "ETTm1":        3,   # 70k  횞 7
    "ETTm2":        3,   # 70k  횞 7
    "WindSCADA":    4,   # 52k  횞 15
    "BlastFurnace": 4,   # 30k  횞 29
    "Weather":      5,   # 53k  횞 21
    "EWELD":        6,   # 105k 횞 15
    # ---- expensive (big rows, or huge channel counts) ----
    "MetroPT3":     7,   # 306k 횞 8
    "N2O":          7,   # 210k 횞 15
    "Electricity":  8,   # 26k  횞 321  (channel-heavy)
    "Traffic":      9,   # 17k  횞 862  (channel-heaviest)
}

# 2026-07-31: MODEL dispatch priority ??the lightweight edge-tier (sub-1K to
# ~18K, our main comparison table) runs BEFORE the heavier ceiling models, so
# the main table fills first and the slow ceiling (Pareto reference) doesn't
# block partial results. Ordered by param count (H=96, D=7). Lower = earlier.
_MODEL_DISPATCH_ORDER = {name: i for i, name in enumerate([
    # edge tier (main table) ??lightest first
    "SparseTSF", "MixLinear", "DiPE_Linear", "FEATHer", "FITS", "CycleNet", "DLinear",
    # ceiling tier (Pareto reference) ??heaviest last
    "PatchTST", "TimeMixer", "TQNet", "LMS_AutoTSF", "TimesNet", "iTransformer", "MDMLP_EIA",
])}

# ARCHIVED 2026-07-13 ??short-horizon PdM section DROPPED from the paper.
# Rationale: it was the one experiment block that attached to neither the
# long-term "frequency forecaster" identity (it is short-horizon, where the
# multi-scale/FFT inductive bias is weakest) nor the robustness param-
# justification (its anchors are Steel/GT/Wind/ETTh1/Weather), AND it created a
# grid-asymmetry attack surface ("why short horizons only here?"). Dropping it
# unifies every dataset to [96..720] and keeps the identity sharp. The loader
# (Dataset_CMAPSS), prep recipes, and lr search (run_lr_search) are KEPT as a
# shelf so a reviewer-driven revision can restore the section immediately.
# Still reachable via `--group cmapss`; just out of the default + `all`.
#   CMAPSS = C-MAPSS FD001 쨌 CMAPSS3 = FD003 쨌 PMSM = Paderborn motor (69-sess)
CMAPSS_DATASETS = ["CMAPSS", "CMAPSS3", "PMSM"]   # shelf (archived from paper)
CMAPSS_PREDS = [24, 48, 96]

# Long-term horizon datasets ??now the GENERALIZATION section (demoted from
# main when the paper pivoted to manufacturing).
LTSF_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity",
                 "Traffic", "Exchange"]
LTSF_PREDS = [96, 192, 336, 720]

DEFAULT_BASE_SEED = 2025
DEFAULT_NUM_SEEDS = 5
DEFAULT_EXP_TAG = "main"
DEFAULT_SEQ_LEN = 96
RESULTS_CSV = "results/fcst_results.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pred_lens_for(dataset):
    if dataset in MFG_DATASETS:
        return MFG_PREDS
    if dataset in CMAPSS_DATASETS:
        return CMAPSS_PREDS
    return LTSF_PREDS


def load_done_set(csv_path):
    """Return set of (exp_tag, model, data, pred_len, seq_len, seed) tuples
    already present in the results CSV."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    need = ["exp_tag", "model", "data", "pred_len", "seq_len", "seed"]
    for col in need:
        if col not in df.columns:
            return set()
    return set(
        (str(r.exp_tag), str(r.model), str(r.data),
         int(r.pred_len), int(r.seq_len), int(r.seed))
        for r in df[need].itertuples(index=False)
    )


# Component ablation ??one of OUR components off per arm, on the SHIPPING model.
#
# Each arm patches the args that `dispatch()` forwards to the worker, so the
# run starts from whatever `_DATASET_OVERRIDES` freezes for the group (mfg =
# FEATHer-N: channel_mode=mix, residual MLP r16, d16/k7/B4, per-dataset FFT
# period) and turns off exactly one thing. That keeps every arm directly
# comparable to the main-table row, which supplies the baseline and is not
# re-run.
#
# This REPLACED the old `--ablation_axis` sweep over the 30
# `FEATHer_<axis>_<variant>` registry entries (archived 2026-08-24). Those
# builders predate the residual MLP and the channel mode, so they construct
# the pre-08 architecture: their numbers could not be compared against the
# shipping model, which is the whole point of an ablation.
#
# Verdict threshold: a 24-cell mean carries +-0.3~0.5% seed noise, so only a
# >=2% change is signal (2026-08-16 calibration). Promote those arms to 3 seeds.
# Scope: every arm removes something WE added. RevIN is deliberately not an arm
# ??it is Kim et al. 2022, used by most of the lineup, so removing it measures
# a borrowed component and justifies none of our parameters (and on the pre-08
# model removing it *improved* Steel clean MSE, which would read badly in a
# table that is supposed to defend the design). Capacity knobs (d_state,
# kernel width, band count as a size) are not components either; their
# sensitivity already comes from the 27-config grid in tools/search/.
# channel_mode is likewise absent: mix vs ci_full is not a component being
# removed, it is the axis that separates the two shipping variants (mfg
# FEATHer-N mixes, LTSF FEATHer-CIN does not). Putting it here would claim
# "removing channel mixing hurts" in a table while our own LTSF model runs the
# removed setting. It belongs in the model-family comparison, where the
# 08-05 ceiling scan already puts CIN at 0.7356 vs N 0.6974 on mfg.
# Entries are (tier, what-it-removes, arg patch). The tier splits the reported
# table in two, because the arms are not the same kind of claim:
#
#   "design"   FEATHer's own frequency machinery ??the components the method
#              section argues for. The 08-16 linear-collapse analysis predicts
#              most of these land near zero, and saying so first (with the
#              proof and the failed nonlinear repair) is the honest-analysis
#              contribution rather than a hole in the paper.
#   "standard" borrowed, off-the-shelf capacity. The residual MLP is an FFN-
#              style bottleneck with no novelty, bolted on in 07-31 as a
#              capacity A/B. It is NOT a contribution and must not be reported
#              as one ??but it is 247 of the shipping model's 938 parameters
#              and is likely the largest effect in the table, so omitting it
#              would leave a quarter of the model unexplained and would hide
#              the one part that clearly works.
# Axis coverage mirrors the four ablation tables of the earlier paper (scales /
# DTK / gating / SPK head) so the structure carries over, extended where a
# single point could not separate "this does nothing" from "the setting was
# already past the knee". The earlier NUMBERS are not reusable ??that code
# selected on test (see project-old-faster-test-peeking-0809) ??so every row
# is re-measured here under the val-selected protocol.
# Arms are patches against the FROZEN model (2026-09-04, MODEL_FINAL.md):
#   no_in_proj, num_bands=3, point_kernel=3, dtk_mode=none, gate_mode=banded,
#   residual MLP r16, lambda_spec=0.01, channel_mode mix (mfg) / ci_full (LTSF)
#
# 2026-09-08: three arms were dropped because they no longer change anything -
# `band3` (the model IS 3 bands), `dtknone` (the DTK is already gone) and
# `dtk1` (kernel_size only fed the DTK).  Each would have trained the
# unmodified model, logged it as abl_<arm>, and reported "0% effect", which
# reads as "this component does not earn its place" when it actually means
# "this component was never switched off".  Verified by building every arm and
# diffing its forward against the baseline; do that again after any change to
# the frozen config.
#   ! The banded gate is zero-initialised, so an UNTRAINED gate arm is exactly
#     uniform and looks identical to gateuni.  Perturb freq_gate.proj before
#     comparing or the check silently calls a live arm dead.
ABLATION_ARMS = {
    # --- temporal scales -------------------------------------------------
    # num_bands=1 keeps only the POINT branch = no decomposition at all.
    "band1":    ("design", "single-scale (1 band)",        {"num_bands": 1}),
    "band2":    ("design", "dual-scale (2 bands)",         {"num_bands": 2}),

    # --- gating ----------------------------------------------------------
    # "none" sums the bands unweighted; "uniform" averages them; "softmax"
    # gates on the pooled input without ever looking at the spectrum, which
    # is the arm that isolates what the FFT itself contributes.
    "gatenone": ("design", "w/o gating (unweighted sum)",  {"gate_mode": "none"}),
    "gateuni":  ("design", "uniform gating (1/S)",         {"gate_uniform": True}),
    "gatesm":   ("design", "softmax gating (no FFT)",      {"gate_mode": "softmax"}),
    # the strongest gate control (2026-09-23): keeps the LEARNED band mixture
    # but drops the transform, so it isolates the per-window adaptivity of the
    # FFT from the value of the mixture itself.
    "gatestatic": ("design", "static learned mixture (no FFT)", {"gate_mode": "static"}),

    # --- SPK head --------------------------------------------------------
    # The head cannot be switched off - without one the model emits no
    # forecast - so all three controls REPLACE it and all are far larger than
    # SPK.  That asymmetry is the finding: SPK claims parameter efficiency, so
    # a near-tie supports it.  The three separate "loses to nonlinearity" from
    # "loses to local context" from "loses to a dense map".
    "headmlp":  ("design", "SPK -> 2-layer MLP head",      {"head_mode": "mlp"}),
    "headconv": ("design", "SPK -> depthwise conv head",   {"head_mode": "conv"}),
    "headlin":  ("design", "SPK -> dense linear head",     {"head_mode": "linear"}),

    # --- normalization ---------------------------------------------------
    # RevIN is the one component measured to be CONDITIONALLY HARMFUL (Steel
    # improves without it, ETTh2 / EWELD collapse).  The paper reports that,
    # so the arm has to exist; it was documented in CLAUDE.md but never
    # actually defined here.
    "norevin":  ("standard", "RevIN off",                  {"no_revin": True}),

    # --- spectral-separation loss ----------------------------------------
    # Baseline 0.01.  Zero alone only shows "off"; an order up distinguishes
    # an inert term from a mistuned weight.  These change the LOSS, not the
    # architecture, so an init-time forward diff is expected to be zero.
    "spec0":    ("design", "spectral loss off (l=0)",      {"lambda_spec": 0.0}),
    "specHi":   ("design", "spectral loss l=0.1",          {"lambda_spec": 0.1}),

    # --- borrowed capacity (not a contribution) --------------------------
    "noresid":  ("standard", "residual MLP off",           {"no_residual_mlp": True}),

    # --- REMOVED components, put back ------------------------------------
    # Not ablations of the shipping model but the mirror question: we deleted
    # these, so does adding them back pay?  Reported separately - an ablation
    # table that mixes "off" and "back on" rows invites misreading.
    "readd_dtk":    ("readd", "DTK back (mlp variant)",    {"dtk_mode": "mlp"}),
    "readd_inproj": ("readd", "linear channel mixer back", {"no_in_proj": False}),
}

# arm patch key -> the _DATASET_OVERRIDES keys it must displace, where the
# names differ (the worker translates these switches into model arguments).
_ARM_ALIAS = {
    "gate_uniform":    {"use_gate"},
    "no_residual_mlp": {"use_residual_mlp", "mlp_rank"},
    "no_revin":        {"use_norm"},
}

ABLATION_CSV = "results/ablation.csv"


def enumerate_combos(args):
    """Generate (exp_tag, model, data, pred_len, seq_len, seed) tuples for
    the requested sweep."""

    if args.model:
        models = [args.model]
    elif args.exclude:
        excl = {s.strip() for s in args.exclude.split(",") if s.strip()}
        models = [m for m in DEFAULT_MODELS if m not in excl]
    else:
        models = DEFAULT_MODELS

    if args.data:
        datasets = [s.strip() for s in args.data.split(",") if s.strip()]
    elif args.group == "ltsf":
        datasets = LTSF_DATASETS      # generalization section
    elif args.group == "cmapss":
        datasets = CMAPSS_DATASETS    # separate short-horizon PdM section
    elif args.group == "all":
        # CMAPSS archived from the paper (2026-07-13); use --group cmapss to reach it.
        datasets = MFG_DATASETS + LTSF_DATASETS
    else:
        datasets = MFG_DATASETS       # default: manufacturing MAIN table

    if args.pred_len > 0:
        pred_lens_override = [args.pred_len]
    else:
        pred_lens_override = None

    seeds = [args.base_seed + i for i in range(args.num_seeds)]

    combos = []
    for m in models:
        for d in datasets:
            preds = pred_lens_override or pred_lens_for(d)
            for h in preds:
                for s in seeds:
                    combos.append((args.exp_tag, m, d, h, args.seq_len, s))
    return combos


def group_missing(missing):
    """Group missing combos by (model, data, pred_len) ??[seeds]. The
    worker takes one (model, data, pred_len) per call and iterates seeds
    internally."""
    groups = {}
    for exp_tag, m, d, h, sl, sd in missing:
        key = (exp_tag, m, d, h, sl)
        groups.setdefault(key, []).append(sd)
    return groups


def print_status(missing, total, csv_path=RESULTS_CSV):
    print("\n" + "=" * 70)
    print(f"  Forecast experiment status")
    print(f"  CSV: {csv_path}")
    print(f"  Total requested: {total}  |  Done: {total - len(missing)}  "
          f"|  Missing: {len(missing)}")
    print("=" * 70)
    if not missing:
        print("  All complete!")
        return

    groups = group_missing(missing)
    for (exp_tag, m, d, h, sl), seeds in sorted(groups.items()):
        seeds_str = ",".join(str(s) for s in sorted(seeds))
        print(f"  [{exp_tag:>12}] {m:>15} | {d:>12} | H={h:<4} L={sl} | "
              f"seeds: {seeds_str}")


def dispatch(missing, args):
    groups = group_missing(missing)
    # Sort order: small datasets first (_DATA_DISPATCH_ORDER), then within a
    # data tier the lightweight edge models before the heavy ceiling models
    # (_MODEL_DISPATCH_ORDER) so the main table fills first and the slow
    # ceiling횞big-data jobs land last without blocking partial results.
    keys_sorted = sorted(groups.keys(),
                         key=lambda k: (_DATA_DISPATCH_ORDER.get(k[2], 5),
                                        _MODEL_DISPATCH_ORDER.get(k[1], 999), k[2], k[3]))

    jobs = []
    for (exp_tag, m, d, h, sl) in keys_sorted:
        seeds = sorted(groups[(exp_tag, m, d, h, sl)])

        # Merge precedence: paper-default (method) ??per-dataset paper
        # override ??CLI flag. lr and loss are top-level worker flags;
        # everything else goes through --model_overrides.
        method_def  = get_method_defaults(m)
        dataset_ovr = get_dataset_overrides(m, d)
        merged = {**method_def, **dataset_ovr}

        lr_to_use   = args.lr   if args.lr   is not None else merged.get("lr",   1e-3)
        loss_to_use = args.loss if args.loss is not None else merged.get("loss", "mse")

        # AUDIT-FIX (2026-07-14, P2-4): the worker applies --model_overrides
        # AFTER parse_args, so an override key silently beats an explicit CLI
        # arch flag (e.g. `--d_state 4` on ETTh1 used to train d16 while the
        # CSV recorded nothing). Explicitly-set FEATHer arch flags (default
        # None = unset) are therefore DROPPED from the override dict and
        # forwarded as worker CLI flags instead ??CLI wins, sweeps unchanged.
        explicit_arch = {k: v for k, v in {
            "d_state":     args.d_state,
            "kernel_size": args.kernel_size,
            "period":      args.period,
            "num_bands":   args.num_bands,
            "lambda_spec": args.lambda_spec,
        }.items() if v is not None}

        # AUDIT-FIX (2026-08-24, ABL-OVERRIDE-PRECEDENCE): the switches added
        # after P2-4 (channel_mode / residual MLP / gate / RevIN modes) are
        # forwarded by the dedicated blocks below, but they ALSO live in
        # _DATASET_OVERRIDES (`_FEATHER_MFG_N` sets channel_mode=mix,
        # use_residual_mlp=True, mlp_rank=4). Since the worker applies
        # --model_overrides after parse_args, the override silently beat the
        # CLI flag ??an ablation like `--channel_mode ci_full` or
        # `--no_residual_mlp` would train the UNMODIFIED model and report
        # baseline numbers, reading as "this component contributes nothing".
        # Drop every switch the caller set explicitly so the CLI wins.
        cli_switches = set()
        if args.channel_mode:
            cli_switches.add("channel_mode")
        if args.gate_mode:
            cli_switches.add("gate_mode")
        if args.gate_bounds:
            cli_switches.add("gate_bounds")
        if args.dtk_act:
            cli_switches.add("dtk_act")
        if args.head_mode:
            cli_switches.add("head_mode")
        if args.dtk_mode:
            cli_switches.add("dtk_mode")
        if args.revin_mode:
            cli_switches.add("revin_mode")
        if args.mlp_rank is not None:
            cli_switches.add("mlp_rank")
        if args.no_revin:
            cli_switches.add("use_norm")
        if args.gate_uniform:
            cli_switches.add("use_gate")
        if args.no_residual_mlp:
            cli_switches.update({"use_residual_mlp", "mlp_rank"})
        if args.point_kernel is not None:
            cli_switches.add("point_kernel")
        if args.band_kernels is not None:
            cli_switches.add("band_kernels")
        if args.band_dilations is not None:
            cli_switches.add("band_dilations")
        if args.no_in_proj:
            cli_switches.add("no_in_proj")
        if args.mix_mode:
            cli_switches.add("mix_mode")
        if args.mix_pool is not None:
            cli_switches.add("mix_pool")
        if args.mlp_tie:
            cli_switches.add("mlp_tie")
        if args.band_zero_sum:
            cli_switches.add("band_zero_sum")
        if args.gate_mix:
            cli_switches.add("gate_mix")
        if args.head2:
            cli_switches.add("head2")
        for _k in ("mix_listen", "mix_lag", "mix_zero_init"):
            if getattr(args, _k):
                cli_switches.add(_k)
        # An ablation arm must beat _DATASET_OVERRIDES even when its patch sets
        # a FALSY value (readd_inproj sets no_in_proj=False, which no `if
        # args.x` test above can see).  The arm runner lists the registry keys
        # it is taking control of; without this the override wins and the arm
        # silently trains the unmodified model.
        cli_switches |= getattr(args, "_arm_strip", set())

        hp_overrides = {k: v for k, v in merged.items()
                        if k not in ("lr", "loss")
                        and k not in explicit_arch
                        and k not in cli_switches}
        overrides_str = ";".join(f"{k}={v}" for k, v in sorted(hp_overrides.items()))

        label = (f"{m} | {d} | H={h} | seeds={seeds} | lr={lr_to_use} | "
                 f"loss={loss_to_use}"
                 + (f" | overrides={overrides_str}" if overrides_str else ""))

        cmd = [
            sys.executable,
            "scripts/benchmarks/run_forecast.py",
            "--model", m,
            "--data", d,
            "--pred_len", str(h),
            "--seq_len", str(sl),
            "--seeds", ",".join(str(s) for s in seeds),
            "--exp_tag", exp_tag,
            "--results_csv", args.results_csv,
            "--num_epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--lr", str(lr_to_use),
            "--loss", loss_to_use,
        ]
        if args.train_frac is not None:
            cmd += ["--train_frac", str(args.train_frac),
                    "--train_frac_mode", args.train_frac_mode]
        no_save = {x.strip() for x in (args.no_save_data or "").split(",") if x.strip()}
        if args.save_model and d not in no_save:
            cmd += ["--save_model"]
        if args.dump_preds:
            cmd += ["--dump_preds", args.dump_preds,
                    "--dump_stride", str(args.dump_stride)]
        if overrides_str:
            cmd += ["--model_overrides", overrides_str]
        # FEATHer arch flags: forward ONLY the explicitly-set ones (P2-4).
        # Unset flags fall back to the worker's own defaults (d8/k7/p12/B3/
        # 貫0.01 ??identical values), still overridable per dataset above.
        for k, v in explicit_arch.items():
            cmd += [f"--{k}", str(v)]
        # Capacity A/B: forward only when enabled (canonical stays byte-identical).
        if args.use_residual_mlp:
            cmd += ["--use_residual_mlp"]
        # AUDIT-FIX (2026-08-26, MLP-RANK-DROPPED): --mlp_rank used to be
        # forwarded only INSIDE the branch above, i.e. only when the caller also
        # passed --use_residual_mlp. The shipping mfg config turns the residual
        # MLP on through _DATASET_OVERRIDES (_FEATHER_MFG_N) rather than through
        # that flag, so `--mlp_rank 8` alone was dropped from the override dict
        # (it is in cli_switches) AND never forwarded ??the run silently trained
        # the worker default rank. That default is 16, which equals the
        # canonical value, so no past result is affected; only rank experiments
        # were, and they no-oped into a duplicate of the baseline. Forward it
        # unconditionally: when unset (None) nothing changes.
        if args.mlp_rank is not None:
            cmd += ["--mlp_rank", str(args.mlp_rank)]
        if args.no_revin:
            cmd += ["--no_revin"]
        if args.gate_uniform:
            cmd += ["--gate_uniform"]
        if args.robust_revin:
            cmd += ["--robust_revin"]
        if args.revin_mode:
            cmd += ["--revin_mode", args.revin_mode]
        if args.winsor_bound is not None:
            cmd += ["--winsor_bound", str(args.winsor_bound)]
        if args.center_trim is not None:
            cmd += ["--center_trim", str(args.center_trim)]
        if args.channel_mode:
            cmd += ["--channel_mode", args.channel_mode]
        if args.gate_mode:
            cmd += ["--gate_mode", args.gate_mode]
        if args.gate_bounds:
            cmd += ["--gate_bounds", args.gate_bounds]
        if args.dtk_act:
            cmd += ["--dtk_act", args.dtk_act]
        if args.head_mode:
            cmd += ["--head_mode", args.head_mode]
        if args.dtk_mode:
            cmd += ["--dtk_mode", args.dtk_mode]
        if args.point_kernel is not None:
            cmd += ["--point_kernel", str(args.point_kernel)]
        if args.band_kernels is not None:
            cmd += ["--band_kernels", str(args.band_kernels)]
        if args.band_dilations is not None:
            cmd += ["--band_dilations", str(args.band_dilations)]
        if args.no_in_proj:
            cmd += ["--no_in_proj"]
        if args.mix_mode:
            cmd += ["--mix_mode", args.mix_mode]
        if args.mix_pool is not None:
            cmd += ["--mix_pool", str(args.mix_pool)]
        if args.mlp_tie:
            cmd += ["--mlp_tie"]
        if args.band_zero_sum:
            cmd += ["--band_zero_sum"]
        if args.gate_mix:
            cmd += ["--gate_mix", args.gate_mix]
        if args.head2:
            cmd += ["--head2", args.head2]
        if args.mix_listen:
            cmd += ["--mix_listen"]
        if args.mix_lag:
            cmd += ["--mix_lag", args.mix_lag]
        if args.mix_zero_init:
            cmd += ["--mix_zero_init"]
        if args.spk_hidden is not None:
            cmd += ["--spk_hidden", str(args.spk_hidden)]
        if args.band_act:
            cmd += ["--band_act", args.band_act]
        if args.band_mode:
            cmd += ["--band_mode", args.band_mode]

        if args.gate_zero_init:
            cmd += ["--gate_zero_init"]
        jobs.append((label, cmd))

    failed = run_on_gpus(jobs, args.ngpu, base_gpu=args.gpu,
                         jobs_per_gpu=args.jobs_per_gpu)
    if failed:
        print(f"\n[WARN] {len(failed)} group(s) exited non-zero; their runs "
              "stay missing ??re-run to retry.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Forecast experiment orchestrator")

    # What to run
    p.add_argument("--model",   type=str, default=None,
                   help="Single model; omit to run all registered.")
    p.add_argument("--ablation_arm", type=str, default=None,
                   help="Component ablation on the shipping FEATHer: "
                        + ", ".join(ABLATION_ARMS) + ", or 'all'. Each arm "
                        "turns off one component and tags its rows abl_<arm>; "
                        "results go to " + ABLATION_CSV + " unless "
                        "--results_csv says otherwise. The main sweep supplies "
                        "the baseline and is not re-run "
                        "(compare with tools/paper/ablation_summary.py).")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated models to skip.")
    p.add_argument("--data",    type=str, default=None,
                   help="Dataset, or a comma-separated list; omit to sweep a "
                        "--group. Useful for screening on the cheap datasets "
                        "first ??MetroPT3 alone runs ~2 h/cell vs ~3 min for "
                        "BlastFurnace.")
    p.add_argument("--group",   type=str, default="mfg",
                   choices=["mfg", "ltsf", "cmapss", "all"],
                   help="Dataset group when --data is omitted: mfg (6 "
                        "manufacturing, MAIN table [96..720]; default), cmapss "
                        "(C-MAPSS short-horizon PdM section [24,48,96]), ltsf "
                        "(8 LTSF generalization), or all.")
    p.add_argument("--pred_len", type=int, default=0,
                   help="Single horizon; 0 = use defaults per dataset.")
    p.add_argument("--seq_len",  type=int, default=DEFAULT_SEQ_LEN)

    # Seeds + grouping
    p.add_argument("--num_seeds",  type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--base_seed",  type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--exp_tag",    type=str, default=DEFAULT_EXP_TAG)

    # Per-worker overrides (all forwarded)
    p.add_argument("--num_epochs", type=int,   default=50)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early-stopping patience on val loss (epochs).")
    p.add_argument("--batch_size", type=int,   default=32)
    # `--lr` / `--loss` default to None so we can detect "user did not set
    # this" and fall back to each method's paper default
    # (baselines.get_method_defaults). Pass a value explicitly to override
    # across the whole sweep (ablations).
    p.add_argument("--lr",         type=float, default=None,
                   help="Override LR for all models. Defaults to each method's "
                        "paper-recommended value (see baselines.get_method_defaults).")
    p.add_argument("--loss",       type=str,   default=None, choices=["l1", "mse"],
                   help="Override loss. Defaults to each method's paper "
                        "loss (MSE for baselines, L1 for FEATHer).")
    # Data-efficiency / cold-start: train on only `train_frac` of the train
    # split (mode=recent -> most recent, the 'new line, limited history' case).
    # Forwarded to the worker. Use a distinct --exp_tag per fraction.
    p.add_argument("--train_frac", type=float, default=None,
                   help="fraction (0,1] of the train split (data-efficiency sweep)")
    p.add_argument("--train_frac_mode", default="recent", choices=["recent", "first"])
    # FEATHer arch flags default to None (= unset) so dispatch can tell an
    # explicit user value from the default: explicit values BEAT
    # _DATASET_OVERRIDES (audit P2-4); unset falls back to per-dataset
    # override, then the worker default (d8/k7/p12/B3/lambda 0.01).
    p.add_argument("--d_state",     type=int,   default=None)
    p.add_argument("--kernel_size", type=int,   default=None)
    p.add_argument("--period",      type=int,   default=None)
    p.add_argument("--num_bands",   type=int,   default=None)
    p.add_argument("--lambda_spec", type=float, default=None)
    # Capacity A/B (2026-07-31): forwarded to the worker only when set, so the
    # canonical sweep is byte-identical when omitted.
    p.add_argument("--use_residual_mlp", action="store_true")
    p.add_argument("--no_residual_mlp", action="store_true",
                   help="Ablate the residual MLP OFF even when the dataset "
                        "override enables it (FEATHer-N sets it by default).")
    p.add_argument("--mlp_rank",    type=int,   default=None)
    # Robustness diagnostics (2026-07-31): forwarded to worker only when set.
    p.add_argument("--no_revin",     action="store_true")
    p.add_argument("--gate_uniform", action="store_true")
    p.add_argument("--robust_revin", action="store_true")
    p.add_argument("--revin_mode", type=str, default=None,
                   choices=["standard", "mad", "winsor", "center",
                            "last", "last_std"])
    p.add_argument("--winsor_bound", type=float, default=None)
    p.add_argument("--center_trim",  type=float, default=None)
    p.add_argument("--channel_mode", type=str, default=None,
                   choices=["mix", "ci_full"])
    p.add_argument("--gate_mode", type=str, default=None,
                   choices=["global", "banded", "banded_sensor", "static", "static_rng", "none", "none_rng", "softmax"])
    p.add_argument("--gate_bounds", type=str, default=None,
                   help="banded gate slice edges: linear = equal width (canonical), "
                        "octave = LOW slice matched to the AvgPool4 branch, then doubling; "
                        "tercile = equal-energy edges from the TRAIN split, tercile_sensor = "
                        "one edge pair per sensor; or explicit 'e1,e2' / 'e1,e2;e1,e2;...'.")
    # DTK latent nonlinearity (2026-08-16): "none" = canonical (linear DTK,
    # which makes the multi-band block collapse to one filter).
    p.add_argument("--head_mode", type=str, default=None,
                   choices=["spk", "linear", "mlp", "conv"],
                   help="Forecasting head: spk (canonical period-aware kernel) "
                        "or linear / mlp (dense controls for the SPK ablation; "
                        "NOT capacity-matched, both are far larger).")
    p.add_argument("--point_kernel", type=int, default=None,
                   help="Width of the POINT branch conv (odd; default 1 = a "
                        "per-channel scalar). Widen it to make a single band "
                        "carry real temporal filtering, replacing both the "
                        "band bank and the DTK convolution.")
    p.add_argument("--band_kernels", type=str, default=None,
                   help="밴드 conv 커널 사다리, num_bands-1 개 (LOW=풀링). "
                        "예: '3,7,15'. 생략하면 기존 동작(byte-identical).")
    p.add_argument("--band_dilations", type=str, default=None,
                   help="밴드 conv 팽창률, num_bands-1 개. 수용영역=d*(k-1)+1 이라 "
                        "커널을 넓히지 않고 넓게 본다. 예: '1,4,15'")
    p.add_argument("--no_in_proj", action="store_true",
                   help="Drop the LINEAR channel mixer, keeping the nonlinear "
                        "residual MLP ??tests whether the two are redundant "
                        "(--channel_mode ci_full removes both at once).")
    p.add_argument("--mix_mode", type=str, default=None, choices=["hidden", "input", "input_raw"],
                   help="Residual-MLP input: hidden = gated bands at every "
                        "timestep (canonical); input = the normalized input, "
                        "time-pooled by --mix_pool, correction interpolated "
                        "back (removes the L x D hidden state on the MCU).")
    p.add_argument("--mix_pool", type=int, default=None,
                   help="Time pooling factor for --mix_mode input (divides seq_len).")
    p.add_argument("--mlp_tie", action="store_true",
                   help="Tie the residual MLP: W_up = W_down^T with its own bias.")
    p.add_argument("--band_zero_sum", action="store_true",
                   help="Force the POINT/MID band conv taps to sum to 0, so those "
                        "branches cannot pass DC (band separation by construction).")
    p.add_argument("--gate_mix", type=str, default=None, choices=["none", "fft", "static"],
                   help="G2 (2026-09-18): the spectrum scales the mixer correction per "
                        "sensor/window; static = same module, input-independent.")
    p.add_argument("--head2", type=str, default=None, choices=["none", "fft", "static"],
                   help="G4-lite (2026-09-18): second SPK readout at 2x (or 0.5x) period, "
                        "blended by period-harmonic energy; static = fixed blend.")
    p.add_argument("--mix_listen", action="store_true",
                   help="(A, 2026-09-21) per-sensor scale on the mixer correction, init 1.")
    p.add_argument("--mix_lag", type=str, default=None,
                   help="(B) 'k,dil': identity-initialised temporal conv on the mixer summary.")
    p.add_argument("--mix_zero_init", action="store_true",
                   help="(C) zero-init the mixer up-projection (start as the mixer-off model).")
    p.add_argument("--gate_zero_init", action="store_true",

                   help="Zero-init the global gate conv so every band starts at "

                        "1/S. The default random init saturates the softmax "

                        "before training (EWELD: max/min 5,077x) and leaves "

                        "bands dead; that tracks which seeds fail to train.")

    p.add_argument("--band_mode", type=str, default=None,
                   choices=["conv", "diff", "fft", "fftoct"],
                   help="How bands are produced. conv = canonical learned "
                        "depthwise convs (their outputs overlap: two moving "
                        "averages of width 3 and 9 correlate 0.99). diff = "
                        "Laplacian differences, fft = exact rFFT slices ??both "
                        "disjoint by construction, sum back to the input, and "
                        "cost zero parameters. 'fft' is the matched "
                        "frequency-band control the ablation was missing.")
    p.add_argument("--band_act", type=str, default=None,
                   choices=["none", "gelu", "se"],
                   help="Per-band nonlinearity between the shared kernel and "
                        "the gate sum. Without one the band bank provably "
                        "collapses to a single filter, which is why num_bands, "
                        "the gate and lambda_spec all measure flat. 'se' is a "
                        "content-dependent multiplicative gain (2 params/band) "
                        "that rescales rather than rectifies.")
    p.add_argument("--spk_hidden", type=int, default=None,
                   help="Width of a GELU hidden layer inside the SPK "
                        "cross-period backbone (0/unset = canonical linear "
                        "map). The head is otherwise the only purely linear "
                        "stage left once the band bank and DTK are collapsed.")
    p.add_argument("--dtk_mode", type=str, default=None,
                   choices=["full", "none", "mlp"],
                   help="none removes the shared temporal kernel entirely "
                        "(vs --kernel_size 1, which only drops its temporal "
                        "mixing).")
    p.add_argument("--dtk_act", type=str, default=None,
                   choices=["none", "gelu", "relu", "tanh"])

    # Output / system
    p.add_argument("--results_csv", type=str, default=RESULTS_CSV)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index (with --ngpu N: the FIRST of N indices)")
    p.add_argument("--ngpu",        type=int, default=1,
                   help="Number of GPUs; N>1 = dynamic job queue across "
                        "GPUs gpu..gpu+N-1 (CF-JEPA-style load balancing)")
    p.add_argument("--jobs_per_gpu", type=int, default=1,
                   help="Concurrent workers per GPU (oversubscription for "
                        "tiny host-bound models; OMP threads auto-pinned "
                        "to cpu_count/total_workers)")
    p.add_argument("--dump_preds", type=str, default=None,
                   help="Forward to the worker: write test predictions (.npz) "
                        "to this directory for downstream-KPI analysis.")
    p.add_argument("--dump_stride", type=int, default=1,
                   help="Keep every Nth test window in --dump_preds "
                        "(96 = one non-overlapping day on a 15-min grid).")
    p.add_argument("--save_model",  action="store_true",
                   help="Save checkpoints (forwarded to worker).")
    p.add_argument("--no_save_data", type=str, default="",
                   help="Comma-separated datasets to train+score normally "
                        "but skip --save_model for. Default '' saves every "
                        "dataset (~42GB). Pass 'Traffic' to skip its unused "
                        "checkpoints (~24GB of TimesNet weights; Traffic is "
                        "not a robustness dataset).")

    # Mode
    p.add_argument("--check", action="store_true",
                   help="Print status without running anything")

    args = p.parse_args()

    if not args.ablation_arm:
        run_sweep(args)
        return

    # --- component ablation: one sweep per arm ------------------------------
    arms = (list(ABLATION_ARMS) if args.ablation_arm == "all"
            else [a.strip() for a in args.ablation_arm.split(",") if a.strip()])
    unknown = [a for a in arms if a not in ABLATION_ARMS]
    if unknown:
        raise SystemExit(f"Unknown ablation arm(s) {unknown}; "
                         f"choose from {list(ABLATION_ARMS)} or 'all'")

    if args.model and args.model != "FEATHer":
        raise SystemExit("--ablation_arm ablates FEATHer's own components; "
                         f"it cannot be combined with --model {args.model}")
    args.model = "FEATHer"

    # Never let ablation rows land in the main results CSV ??the exp_tag would
    # keep them separate, but a stray arm row in the headline file is the kind
    # of thing nobody notices until a table looks wrong.
    if args.results_csv == RESULTS_CSV:
        args.results_csv = ABLATION_CSV
        print(f"[ablation] results -> {args.results_csv}")

    base_tag = args.exp_tag
    for arm in arms:
        tier, desc, patch = ABLATION_ARMS[arm]
        arm_args = argparse.Namespace(**vars(args))
        for k, v in patch.items():
            setattr(arm_args, k, v)
        # keys in _DATASET_OVERRIDES that this arm overrides (see cli_switches)
        arm_args._arm_strip = set()
        for k in patch:
            arm_args._arm_strip.add(k)
            arm_args._arm_strip |= _ARM_ALIAS.get(k, set())
        arm_args.exp_tag = (f"abl_{arm}" if base_tag == DEFAULT_EXP_TAG
                            else f"{base_tag}_{arm}")
        print("\n" + "=" * 70)
        print(f"ABLATION ARM  {arm}  [{tier}]  ({desc})   exp_tag={arm_args.exp_tag}")
        print("=" * 70, flush=True)
        run_sweep(arm_args)


def run_sweep(args):
    """Enumerate ??resume-filter ??dispatch one sweep configuration."""
    combos = enumerate_combos(args)
    done = load_done_set(args.results_csv)
    missing = [c for c in combos if c not in done]

    print_status(missing, total=len(combos), csv_path=args.results_csv)

    if args.check:
        return

    if not missing:
        print("\nNothing to run.")
        return

    print(f"\n>>> Dispatching {len(missing)} missing runs "
          f"({len(group_missing(missing))} worker invocations)...")
    dispatch(missing, args)

    print("\n>>> Final status:")
    done_final = load_done_set(args.results_csv)
    missing_final = [c for c in combos if c not in done_final]
    print_status(missing_final, total=len(combos), csv_path=args.results_csv)


if __name__ == "__main__":
    main()


