# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

**Status (2026-07-03) — OFAT 512 done; cross-dataset verdict = KEEP base
canonical; combo validation pending on server:**
- User brought back `run_hp_search.py --summary` (all 512 runs finished).
  Cross-dataset mean-rank aggregation (8 LTSF sets): **d_state 16 wins 7/8**
  (5.8 vs 8.5) and **period 6 wins consistently** (4.8 vs 12's 8.5), BUT both
  are blocked by the sub-1K budget — verified with real param counts:
  d16 @D=14/H96 = **1,146 ✗**; p6 @D=14/H720 = 2,638 (2× base). k/λ flat,
  **num_bands PERFECTLY flat (2→8.44 / 3→8.50 / 4→8.38) → B=3 stays; the tex
  ~485 "B=4 industrial default" sentence has NO LTSF support** (mfg ms-axis
  ablation gives the final word). lr=5e-3 sweeps 8/8 (lr comes from lr-search
  on mfg anyway). → **Canonical arch for mfg = base d8/k7/p12/B3 unchanged.**
  Sensitivity narrative for the paper: "d_state shows mild monotone gains but
  16 exceeds the 1K budget at D=14; within budget 8 is optimal" (R2 #23).
- **Per-dataset LTSF rows are NOT frozen yet**: the OFAT "recommended" combos
  were never trained (axes interact; ETTh1 changes 6 axes at once). Added
  **`run_hp_search.py --validate`** (combo runs, exp_tag=`hp_combo_<cfg>`,
  config encoded in tag → resume-safe) + `--summary` now prints a per-dataset
  verdict: final = best mean-rank among {base, single-axis variants, combo},
  so no dataset can end worse than an observed config. Offline-tested
  (synthetic CSV: adopt/fallback/no-pollution all pass). **Server next:
  `git pull` → `python run_hp_search.py --validate` (≤32 runs) → `--summary`
  → paste the final 8 FEATHer LTSF rows.** ② (FEATHer LTSF 160) waits on this;
  ③ baselines' lr search can start now.
- ⚠ Side-finding: FEATHer's param count grows with pred_len (SPK backbone
  n×m): quoted 402/677/866 are H=96 values; at H=720 base already exceeds 1K
  for D≥11 (GasTurbine 1,093 / WindSCADA 1,282). Check how the manuscript
  quotes params — sub-1K must be scoped to H=96 (or per-horizon counts shown)
  before the claim ships.

**Status (2026-07-02b) — experiment protocol sequencing locked (same day,
after the scope was closed):**
- **FEATHer's canonical architecture for the MFG tables comes from the LTSF
  OFAT** (the 512-run search on the server): aggregate `--summary` ACROSS the
  8 LTSF datasets → if the base config (d_state=8, k=7, P=12, B=3) sits in the
  flat region, keep it; if an axis value wins consistently (num_bands 3 vs 4
  is the live question — tex line ~485 claims "B=4 default for industrial"
  while code default is 3), update the canonical. Clean story: *architecture
  fixed on independent LTSF data before any manufacturing experiment; on the
  mfg datasets every method (FEATHer included) runs a fixed architecture +
  lr-only search — fully symmetric.*
- **Server order therefore**: (1) `run_hp_search.py --check` → if done,
  `--summary` → confirm/update canonical FEATHer config; (2) `run_lr_search.py`
  — baselines' 1,320 runs can start before (1) resolves; FEATHer's 120 lr rows
  should run AFTER the canonical config is fixed (lr interacts with arch);
  (3) paste both summaries into `_DATASET_OVERRIDES` (LTSF rows from OFAT,
  mfg rows from lr search); (4) main sweeps 720 + 540.
- **Ablation extended to manufacturing datasets — CONFIRMED** (the ms axis
  provides the B=2/3/4 evidence on mfg data as analysis, without breaking
  main-table symmetry; also settles the tex "B=4 industrial" sentence).
- **Robustness swap to manufacturing datasets — decided in principle**, code
  change pending (`run_robustness.py:ROBUSTNESS_DATASETS`); settles SML.
- Freq-tied HPs (SparseTSF `period_len`, TQNet `cycle`, FEATHer `period`) —
  if adjusted later, adjust ALL THREE by the same deterministic
  sampling-rate rule (no model-specific favors).
- Docs updated: `manuscript/drafts/intro_jms.md` + `manuscript/notes/
  PAPER_CONTEXT.md` now carry the final scope (intro Para 4 / C5 rewritten:
  two-tier benchmark, sub-1K everywhere, persistence rows).

**Status (2026-07-02) — PMSM dropped after task-fit audit; main table = 3
datasets:** Ran a quantitative task-fit audit on the actual local data (channel-
level naive-persistence baselines, timestamp continuity, per-split window
counts, physical horizon spans). Findings:
- **Steel ✅ / WindSCADA ✅ / C-MAPSS ✅** — sound as scoped (Steel's 729
  non-modal timestamps = the known benign 24:00 date artifact; WindSCADA zero
  gaps; C-MAPSS window counts reproduce exactly).
- **GasTurbine ⚠ kept with framing** — 8/11 channels have persistence MSE>1
  already at H=96 (all-channel mean 1.31→1.63); same profile as Exchange in
  LTSF, so defensible by convention. Frame the synthetic index honestly as an
  "operating-hours" axis; add a persistence baseline row to the table.
- **PMSM ✂ dropped from the MAIN table (2026-07-02)** — at native 2Hz,
  H=96..720 spans only 48s–6min of slow rotor-thermal drift; copy-last
  persistence = MSE **0.004** (target) / **0.117** (all-channel) at H=96 →
  trivially easy, indefensible against a naive-baseline row.
- **Short-horizon PdM section EXPANDED same day (user asked for more PdM
  datasets; all vetted against real data):** section = **CMAPSS (FD001) +
  CMAPSS3 (FD003) + PMSM (rebuilt)**, horizons [24,48,96], all via the
  unit-aware `Dataset_CMAPSS` loader:
  * **CMAPSS3** = C-MAPSS FD003 (raw already on disk): same single operating
    condition as FD001 but 2 fault modes + longer trajectories (med 220 /
    max 525) → ~3x FD001's H=96 windows (4365/447/1358). Same 14-channel
    set as FD001 (FD003's constants are a subset of FD001's drops) → D=14,
    target s11, directly comparable. **FD002/FD004 rejected**: 6 op-conditions
    driven by unobserved flight profile (unpredictable regime switches) + all
    21 sensors active → D=21 breaks sub-1K.
  * **PMSM rebuilt**: ALL 69 sessions, 30-s MEAN aggregation (not decimation —
    no aliasing; matches SCADA convention), `['unit', chans, pm]` schema,
    session-aware windows/split. 22,216 rows, D=12. Measured difficulty:
    all-channel persistence 1.12/1.40/1.50, target pm 0.16/0.35/0.62 at
    H=24/48/96 → non-trivial. Horizons = 12/24/48 min of motor-thermal
    forecasting under unknown future load.
  * Both smoke-verified (FEATHer/SparseTSF/iTransformer × H=24 train+score;
    models beat persistence even at 2 epochs).
  * **PHM 2018 ion mill etch (semiconductor) REJECTED after vetting the real
    data** (downloaded the 5.36GB tarball, audited tool 01_M01, 3.1M rows):
    (a) NOT continuous — 11,963 gaps >60s, max gap 220 h across a 460-day
    span; (b) only ~9 of 17 numeric channels are true continuous measurements
    (rest = setpoints with 4–121 unique values, monotonic usage counters,
    a NaN-carrying binary); (c) dynamics are batch/recipe-driven (3,776 runs,
    median 40 min) → forecasting = predicting the exogenous production
    schedule (same reason FD002/4 were rejected); (d) 60s-mean persistence
    ≈0.93–1.06 even at 24–96 min; (e) native task is TTF/anomaly (TEP-style
    mismatch). Download deleted (re-fetch: gdown id
    15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb). This closes the search — the PdM
    section stays at CMAPSS/CMAPSS3/PMSM.
- **Scope now: main long-horizon table = Steel/GasTurbine/WindSCADA (3) ×
  [96..720]; short-horizon PdM section = CMAPSS/CMAPSS3/PMSM (3) × [24,48,96].
  5 domain cards (steel energy, gas-turbine emissions, wind, aero-engine ×2
  fault regimes, electric drives), sub-1K everywhere.** Run counts: main
  **720** (12×3×4×5), PdM **540** (12×3×3×5), lr search **1,440** (12×5×6×2×2)
  — all verified via `--check`. Both tables get a naive persistence baseline
  row (cheap, defuses the "is this task trivial?" attack).
- **data/ cleanup (2026-07-02, user-directed):** DELETED LG_AC/LG_PAC_pool/
  LG_RAC_pool (LG air-conditioner telemetry from an unrelated project, zero
  code references), TEP (rejected; raw re-downloadable from the mirror in
  `prep_manufacturing.py`, recipe still tracked), and PM (obsolete IoT-J-era
  air-quality set). KEPT: the 6 active sets + their raws, **nrel (solar —
  user explicitly keeps it)**, and SML (may still serve the LTSF robustness
  section; scope decision pending).

**Status (2026-06-26) — TEP dropped, C-MAPSS added (data validation):**
Validated the manufacturing datasets against top-tier usage (web search, not
deep-research). Finding: none are established *long-term forecasting* (LTSF)
benchmarks; native tasks are energy regression (Steel), non-continuous emission
regression (GasTurbine), **anomaly detection (TEP)**, wind-power forecasting
(WindSCADA, the one true forecasting precedent), short-horizon temp estimation
(PMSM). Acted on the two weakest:
- **TEP DROPPED** — anomaly-detection benchmark (task mismatch) *and* D=50 broke
  the sub-1K claim. **Main table is now 4 datasets** (Steel/GasTurbine/WindSCADA/
  PMSM), all D≤14 → **sub-1K now holds across the entire main table**.
- **C-MAPSS ADDED but as a SEPARATE short-horizon PdM section, NOT the main
  table.** NASA C-MAPSS FD001 (de-facto prognostics benchmark, 100 engines, 14
  informative sensors → D=14, FEATHer=818p). Verified from the data: engine
  trajectories median ~200 / max 543 cycles, so the [96..720] long horizons are
  structurally impossible (816 rows needed). Uses its own horizons **[24,48,96]**
  with a dedicated engine-aware loader `Dataset_CMAPSS` (windows never cross an
  engine boundary; train/val/test split BY ENGINE 70/10/20 → no leakage; zero
  time-marks like PEMS). Group `cmapss`; smoke-verified end-to-end (FEATHer 818p,
  iTransformer 823K, SparseTSF 29 all train+score; window counts 5800/818/2113
  match the engine-aware enumeration). Raw mirror:
  github.com/cyrilli/TurboEngine_Dataset_NASA/CMAPSSData → `data/CMAPSS/raw/`.
- Run counts now: main sweep **960** (12×4×4×5); C-MAPSS section **180**
  (12×1×3×5); lr search drops TEP, adds C-MAPSS at [24,96].
- **DATASET SCOPE FINALIZED (2026-07-01): 4 main + C-MAPSS, no more additions.**
  Vetted every extra candidate against real data and all failed the
  continuous-multivariate-LTSF bar:
  * Bosch CNC Machining (UCI #752) — downloaded + opened an .h5: `vibration_data`
    (268288, **3**) = only 3 accelerometer axes, one ~134 s machining op per file
    (segmented), good/bad **classification** labels → not LTSF (TEP-style trap).
  * SCANIA Component X — features are histograms/counters (aggregated), not a
    continuous per-entity sensor stream → poor LTSF fit.
  * Engie "La Haute Borne" wind SCADA — clean + continuous + downloadable, BUT a
    2nd wind farm = **domain-redundant** with Kelmarsh (weakens the "5
    heterogeneous domains" claim). Rejected on redundancy, not viability.
  * 3W (Petrobras oil wells) — anomaly/event-segmented, D=5 (thin). HVAC/building
    sets — not manufacturing. Structural wall: genuinely continuous + multivariate
    (D≤14) + long manufacturing sensor streams are rare; the existing 4 are
    well-chosen precisely because they ARE continuous monitoring streams.
- **No temporal downsampling** — all datasets kept at native sampling. Only
  channel selection (drop accounting/constant/categorical cols) and, for
  WindSCADA (Turbine 1 / 2017) and PMSM (longest single session), subset
  selection to get one clean continuous series. `h5py` installed into the
  `feather` env (to inspect Bosch .h5).

**Status (2026-06-25) — ⚑ MANUFACTURING PIVOT (professor's call, supersedes
the IoT-J plan below):** Target venue changed from **IEEE IoT-J → Journal of
Manufacturing Systems (JMS)**, Elsevier/SME, Q1. The paper is repositioned
around manufacturing/predictive-maintenance forecasting.
- **Main table** (updated 2026-06-26: TEP dropped → **4 datasets**; C-MAPSS is a
  separate short-horizon section, see the 2026-06-26 block above). The old
  LTSF-8 is demoted to a "generalization" section. Datasets, all wired + QC'd +
  smoke-verified (raw + processed `data.csv` both under `data/<name>/`;
  reproducible recipes in `tools/prep_manufacturing.py`):
  Steel (UCI #851, KR steel, D=6, 402p), GasTurbine (UCI #551, D=11, 677p),
  ~~TEP (process sim, D=50, 4.5Kp — breaks sub-1K)~~ **dropped**, WindSCADA
  (Kelmarsh, D=14, 866p), PMSM (Paderborn motor, D=12, 738p). All 4 real.
- **Orchestrator**: `run_forecast.py` default sweep group is now `mfg`
  (`--group {mfg,ltsf,cmapss,all}`); `python run_forecast.py --exp_tag main
  --save_model` → 960 runs (12×4×4×5). C-MAPSS via `--group cmapss` (180).
- **HP protocol = Option B (lr-only)**: baselines have no paper HPs for these
  datasets, so `run_lr_search.py` selects lr per (method,dataset) on val
  (arch fixed at paper default), 1,200 runs (5 datasets incl. C-MAPSS ×
  12 × 5 lr × 2 horizons × 2 seeds) → `--summary` → paste into
  `baselines/__init__.py` `_DATASET_OVERRIDES`. The old FEATHer
  `run_hp_search.py` (OFAT on LTSF-8) is superseded for the main table.
- **Server run order**: `run_lr_search.py` → `--summary` → paste overrides →
  `run_forecast.py --exp_tag main --save_model` → robustness/ablation.
- **Pending (decide later / test on server)**: extend robustness + ablation
  to manufacturing datasets (the PdM noise story makes mfg robustness ~core);
  freq-tied HPs (SparseTSF `period_len`, TQNet `cycle`) left at hourly default
  — **user chose to just test this on the server** and tune if needed; edge
  estimator FEATHer rows recompute.
- **NOT committed** (user commits everything together later). **`data/` is
  gitignored** → committing does NOT ship data to the server; transfer the 5
  raw sets (or data.csv) + run `tools/prep_manufacturing.py`. The in-flight
  server runs (512-run FEATHer search + 1,760 LTSF baseline sweep) are left
  running — the LTSF baselines still serve the generalization section.
- Memory: `memory/project_jms_manufacturing_pivot.md` has the full detail.

---
**Status (2026-06-11):** Pre-sweep audit complete — the codebase is
ready for the server runs. Locked this week:
- **Evaluation protocol rewritten**: best-val-epoch selection with
  early stopping (patience=10, 50-epoch cap), test evaluated once on
  the selected model, checkpoint = that same model. The old worker did
  test-oracle selection with element-wise metric mixing — every
  pre-2026-06-11 result row (there were none in fcst_results.csv) is
  incompatible with the new protocol.
- **Traffic = full 862 channels** (was a legacy 162 slice in
  data_factory only; manuscript and edge estimator already said 862).
- **Sub-1K claim scoped to D≤14** (FEATHer params scale as D² via
  `in_proj`; see Known issues).
- **Ablation variants unblocked** (worker argparse choices excluded
  them) + `--ablation_axis` orchestrator dispatch.
- **run_hp_search.py added** (OFAT, val-only selection, 192 runs).
- **QEMU framing fixed**: instruction-level emulation, verified
  (memory fit, correctness, icount) vs modeled (latency, energy)
  separation in Sec IX, README, and firmware skeleton.
All 12 main models + 5 ablation axes smoke-tested end-to-end through
the new worker. 13-page PDF compiles cleanly. Sections VI-VIII remain
`\TODO{...}` skeletons waiting on sweep results.

**In progress (2026-06-17):** Step 1 — `run_hp_search.py` now scoped to
**per-dataset FEATHer tuning** (decided 2026-06-17): all 8 main-table
datasets × 16 OFAT configs × 2 horizons × 2 seeds = **512 runs** (was
192 / single-config). FEATHer is tuned per-dataset like every baseline
to maximize main-table numbers; the old "single configuration across all
8 datasets" selling point is **dropped**. Resume-safe — the ~10 already-
done ETTh1/ETTm1/Weather runs carry over. No `fcst_results.csv` yet;
`results/checkpoints/` still empty. On return: `run_hp_search.py
--summary` → paste the per-dataset winners into
`baselines/__init__.py` `_DATASET_OVERRIDES` (FEATHer rows) → proceed to
main sweep below. The rescope (+ the `--no_save_data` checkpoint flag) is
committed + pushed to both remotes; user is re-launching the 512-run
search on the server (resume picks up the ~10 done runs) and will report
results.

Checkpoint policy (decided 2026-06-17): main sweep saves **all 8 datasets**
(~42GB; TimesNet-on-Traffic alone ≈ 24GB). `--no_save_data` exists as an
opt-in to skip Traffic's unused `.pth` (~16GB) if disk gets tight, but
default is save-all.

**Update (2026-06-18):** Baseline main sweep launched on the server in
**parallel** with the HP search — `run_forecast.py --exclude FEATHer
--num_seeds 5 --num_epochs 50 --exp_tag main --save_model` (1,760 runs;
baselines need no HP search, their per-dataset paper HPs are already in
`_DATASET_OVERRIDES`). FEATHer's 160 main rows stay blocked on the HP
search → `_DATASET_OVERRIDES` paste. **Gotcha hit:** the 11 `*-main/`
upstream baseline repos are gitignored (`baselines/*/*-main/`); a fresh
server checkout fails non-FEATHer runs with `No module named 'models'`
until `bash setup_baselines.sh` clones them. Done on the server.

**Manuscript revisions (2026-06-18, done — single-config TODO closed):**
edited `feather_iotj.tex` + `feather.bib` (NOT yet committed; plan to
commit with the main-table numbers once the sweep lands):
- **single-config narrative removed** (4 spots: protocol para, Conclusion,
  Limitations(i), Future-directions) → "per-dataset selection like every
  baseline" symmetric fair-comparison framing.
- **R6#5/R8#5 symbol fix**: uppercase `H` was overloaded (gate-fused L×D
  feature *and* scalar horizon) → fused feature renamed `H_g` /
  residual-smoothed `H_{agg}`, aligned with Algorithm 1's `h`/`h_agg`,
  notation table rows added. (R6#1d L_f→B transition was already resolved.)
- **R1#6**: added BLS-AttnTCN (Su et al., ESWA 2026) + BLS-QLSTM (Su et
  al., HSSC 2025) bib entries (real metadata via Crossref) + one Related-
  Work paragraph framing them as out-of-edge-scope hybrids.
- **R8#1** (tone down "broadly SOTA"): already resolved 2026-06-04, no-op.
- Build verified: 13-page PDF, 3 clean passes; only undefined refs are
  `tab:main-mse`/`tab:main-mae` (the pending main tables).

**Next action (server) — SUPERSEDED by the 2026-06-25 manufacturing pivot
above.** The manufacturing main-table pipeline is: `run_lr_search.py` (1,200)
→ `--summary` → paste lr into `_DATASET_OVERRIDES` → `run_forecast.py
--exp_tag main --save_model` (1,200, default group=mfg) → `run_robustness.py`
(extend to mfg datasets — pending) → ablation (pending mfg scope) → QEMU
Layer 2 (WSL). The LTSF-8 pipeline below still runs for the generalization
section. After the main sweep: `check_progress.py --exp_tag main`; re-run
FEATHer rows of `edge_estimates.csv` for the manufacturing datasets.

*(historical IoT-J plan, kept for the generalization-section runs):*
`run_hp_search.py` (512 runs) → `run_forecast.py --group ltsf --save_model`
(1,920) → `run_forecast.py --data SML --exp_tag main --save_model` (240) →
`run_robustness.py` → ablation (ETTh1+Weather+Electricity) → QEMU.
Reject letter at `manuscript/notes/reject_mail.md`.

**Build:** from `manuscript/tex_workspace/`,
`pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj`.

## Environment

- **Conda env**: `feather` at `C:\Users\lee\.conda\envs\feather\` — Activate
  with `conda activate feather` before any commands. Has `torch 2.5.1+cu121`,
  `darts 0.37.1`, `reformer_pytorch 1.4.4` (for iTransformer's
  SelfAttention_Family).
- `mamba-ssm` is intentionally **not** installed (CUDA-toolchain build,
  out-of-scope for our edge-deployable benchmark).

## Repository structure

```
FEATHer/
├── run_forecast.py                 user-facing orchestrator (--check, resume, dispatch)
├── run_robustness.py               robustness orchestrator (loads --save_model checkpoints)
├── run_hp_search.py                FEATHer OFAT HP search (LTSF generalization configs + R2 #23 sensitivity)
├── run_lr_search.py                per-(method,dataset) lr search for the MFG main table (Option B, all 12 models)
├── setup_baselines.sh              git-clone baseline upstream repos (idempotent)
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         worker (one model × dataset × pred_len × N seeds)
│       └── run_robustness.py       robustness worker (Gauss/miss/impulse/quant sweep)
├── baselines/                      top-level model directory
│   ├── _import_helper.py           sys.modules isolation for upstream imports
│   ├── __init__.py                 registry (12 main + 30 ablation) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
│   ├── FEATHer/                    our model (paper-faithful single config)
│   │   ├── FEATHer.py
│   │   └── ablation/{multiscale,dtk,gating,head,complexity}.py
│   ├── DLinear/wrapper.py          + DLinear-main/ (cure-lab/LTSF-Linear)
│   ├── PatchTST/wrapper.py         + PatchTST-main/ (yuqinie98)
│   ├── iTransformer/wrapper.py     + iTransformer-main/ (thuml)
│   ├── FITS/wrapper.py             + FITS-main/ (VEWOXIC)
│   ├── SparseTSF/wrapper.py        + SparseTSF-main/ (lss-1138)
│   ├── TimeMixer/wrapper.py        + TimeMixer-main/ (kwuking)
│   ├── TimesNet/wrapper.py         + TimesNet-main/ (thuml/Time-Series-Library)
│   ├── TQNet/wrapper.py            + TQNet-main/ (ACAT-SCUT)
│   ├── LMS_AutoTSF/wrapper.py      + LMS_AutoTSF-main/ (mribrahim/LMS-TSF)
│   ├── DiPE_Linear/wrapper.py      + DiPE_Linear-main/ (wintertee, DASFAA 2026)
│   └── MDMLP_EIA/wrapper.py        + MDMLP_EIA-main/ (zh1985csuccsu, AAAI 2026)
├── utils/
│   ├── seed.py                     set_seed() + parse_seed_list()
│   ├── data_factory.py             data_provider + data_select (lazy darts import)
│   ├── data_loader.py              Dataset classes (ETT_hour/minute, Custom, PEMS)
│   ├── losses.py                   spectral_separation_loss_scales (FEATHer-specific)
│   ├── metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
│   ├── noise.py                    4-axis corruption (gauss/miss/impulse/quant)
│   └── timefeatures.py
├── tools/
│   ├── prep_manufacturing.py       reproducible raw→data.csv for the 5 MFG datasets (tracked; data/ is gitignored)
│   ├── audit/check_progress.py     read results CSV, print coverage + mean±std preview
│   └── paper/                      Phase 5 table generators
│       ├── main_table.py           5-seed mean±std table (md / latex / csv)
│       ├── wilcoxon.py             pairwise signed-rank vs FEATHer
│       ├── ablation_table.py       5 axes × 4 horizons aggregator
│       └── robust_summary.py       4-axis noise heatmaps + summary table
├── deployment/cortex_m3/           Layer-1 edge-cost estimator (this work, IoT-J Sec IX)
│   ├── op_costs.py                 Cortex-M3 reference profile + per-op FLOP counters
│   ├── estimator.py                forward-hook based peak RAM / arena / latency / energy
│   ├── run.py                      CLI sweep -> results/edge_estimates.csv
│   └── qemu/                       Layer-2 QEMU validation (skeleton; WSL workflow)
│       ├── README.md               build + run instructions
│       ├── codegen.py              PyTorch -> int8 + weights.h / arena.h
│       └── firmware/{main.c,Makefile,lm3s6965_64k.ld}
├── results/
│   ├── fcst_results.csv            main sweep (append-only, resume by orchestrator)
│   ├── robust_results.csv          robustness sweep
│   ├── edge_estimates.csv          384 cells from deployment/cortex_m3/run.py
│   └── checkpoints/<exp_tag>/      deterministic .pth from --save_model
├── data/                           .csv / .npy raw data (gitignored)
└── manuscript/                     reviews, drafts, notes (mostly gitignored)
    ├── tex_workspace/feather_iotj.tex  IoT-J source (selectively tracked)
    ├── tex_workspace/fig{1-5}*.tex     TikZ figures (tracked)
    └── drafts/*.md                     per-section drafts (tracked)
```

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Main sweep (Phase 4) ===
# Save checkpoints so robustness can reuse them. --save_model saves ALL 8
# datasets by default (~42GB; TimesNet-on-Traffic alone is ~24GB). If disk
# gets tight, --no_save_data "Traffic" skips Traffic's .pth (still trained
# + scored into the CSV; Traffic isn't a robustness dataset) → ~16GB.
python run_forecast.py --check                                           # show what's missing
python run_forecast.py --num_seeds 5 --num_epochs 50 --exp_tag main --save_model
python run_forecast.py --model FEATHer --save_model                      # one model across all datasets/horizons
python run_forecast.py --data ETTh1 --pred_len 96 --save_model           # one (data, horizon) across all models
python run_forecast.py --exclude TimesNet,MDMLP_EIA                      # skip heavy models for a fast pass
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke      # quick verify (no checkpoints)

# Multi-GPU (2026-07-03): all four orchestrators (run_forecast / run_hp_search
# / run_lr_search / run_robustness) take `--ngpu N` — a CF-JEPA-style DYNAMIC
# job queue (utils/dispatch_queue.py): one worker subprocess per GPU (indices
# --gpu .. --gpu+N-1); a free GPU grabs the next job, so fast models never
# idle a GPU. No torchrun/DDP/file locks — the orchestrator owns an in-memory
# queue. ngpu=1 (default) = old sequential behavior.
#   python run_forecast.py --exp_tag main --save_model --ngpu 2
# Manual alternative (two processes, disjoint --exclude splits, one per
# --gpu) still works; sharing one results CSV is fine either way (one
# append per finished run).

# SML is robustness-only: train its checkpoints under exp_tag=main
# (240 runs, see Status) but it stays OUT of the main accuracy table —
# the paper's main table is the 8 LTSF datasets; SML appears only in
# Sec VIII (its clean rows double as accuracy there).

# === Ablation sweep (after main; 30 FEATHer variants, 5 axes) ===
# DECIDED SCOPE (2026-06-11): ETTh1 + Weather + Electricity x 4 horizons
# x 5 seeds = 1,800 runs. Diversity (hourly-7ch / 10min-21ch / hourly-321ch)
# + answers R8 #6d (Electricity ablations promised but unreported in TPAMI).
python run_forecast.py --ablation_axis all --data ETTh1       --exp_tag ablation
python run_forecast.py --ablation_axis all --data Weather     --exp_tag ablation
python run_forecast.py --ablation_axis all --data Electricity --exp_tag ablation
# axes: ms (15) / gate (4) / dtk (4) / head (4) / complexity (3)
# single axis: python run_forecast.py --ablation_axis dtk --data ETTh1 --exp_tag ablation

# === Manufacturing main table (JMS pivot) ===
# Default sweep group is `mfg` (3 datasets: TEP dropped 2026-06-26, PMSM 2026-07-02).
python run_forecast.py --check --exp_tag main             # 720 mfg runs (12x3x4x5)
python run_forecast.py --exp_tag main --save_model        # group=mfg by default
python run_forecast.py --group cmapss --exp_tag main --save_model # short-horizon PdM: CMAPSS+CMAPSS3+PMSM (540)
python run_forecast.py --group ltsf --exp_tag main --save_model   # generalization section
python run_forecast.py --group all  --exp_tag main --save_model   # mfg + cmapss + ltsf

# === lr search for the manufacturing main table (Option B, before the sweep) ===
# Per-(method, dataset) lr selected on val; arch fixed at each method's paper
# default. 12 models × 5 lr × 6 datasets × 2 horizons × 2 seeds = 1,440
# (Steel/GasTurbine/WindSCADA at {96,720}; CMAPSS/CMAPSS3/PMSM at {24,96}).
python run_lr_search.py --check
python run_lr_search.py                     # run all missing (resumable)
python run_lr_search.py --summary           # best lr per (method,dataset) + _DATASET_OVERRIDES lines

# === FEATHer OFAT HP search (LTSF generalization section / sensitivity only) ===
# Superseded for the MAIN table by run_lr_search.py; still used for the LTSF-8
# generalization configs + the R2 #23 sensitivity curves.
python run_hp_search.py --check
python run_hp_search.py
python run_hp_search.py --summary     # per-dataset OFAT recommendation
python run_hp_search.py --validate    # run the combined recommendations (≤32 runs)
python run_hp_search.py --summary     # final verdict: combo adopted, or best
                                      #   observed single-change config as fallback

# === Robustness sweep (Phase 4b) ===
# Requires `--save_model` to have populated results/checkpoints/<train_exp_tag>/.
python run_robustness.py --check
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --train_exp_tag main --fault_types gauss,miss   # subset
python run_robustness.py --model FEATHer --train_exp_tag main            # one model

# === Worker (rarely called directly; orchestrator dispatches) ===
python scripts/benchmarks/run_forecast.py \
    --model FEATHer --data ETTh1 --pred_len 96 \
    --seeds "2025,2026,2027,2028,2029" --exp_tag main --save_model

# === Progress audit ===
python tools/audit/check_progress.py                  # default results/fcst_results.csv
python tools/audit/check_progress.py --exp_tag main   # filter by experiment
python tools/audit/check_progress.py --model FEATHer  # filter by model

# === Paper artifact generators (Phase 5) ===
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer
python tools/paper/ablation_table.py --axis ms --format latex
python tools/paper/robust_summary.py --output_dir manuscript/figures/

# === Cortex-M3 edge cost estimator (Layer 1, simulation-based) ===
python -m deployment.cortex_m3.run --check                # show pending cells
python -m deployment.cortex_m3.run                         # full sweep -> results/edge_estimates.csv
python -m deployment.cortex_m3.run --model FEATHer --data ETTh1 --pred_len 96

# === Cortex-M3 QEMU validation (Layer 2, WSL) ===
# Inside WSL2 (Ubuntu 22.04 with arm-none-eabi-gcc + qemu-system-arm installed):
python -m deployment.cortex_m3.qemu.codegen --model FEATHer --data ETTh1 --pred_len 96 \
    --ckpt results/checkpoints/main/FEATHer_ETTh1_H96_L96_s2025.pth
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build ===
cd manuscript/tex_workspace
pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj
```

## Model registry (12 entries)

```python
from baselines import get_model, list_models, get_method_defaults, get_dataset_overrides
print(list_models())
# ['DLinear', 'DiPE_Linear', 'FEATHer', 'FITS', 'LMS_AutoTSF', 'MDMLP_EIA',
#  'PatchTST', 'SparseTSF', 'TQNet', 'TimeMixer', 'TimesNet', 'iTransformer']
```

| Model | Venue | Year | Params (ETTh1 paper config, D=7) |
|---|---|---|---|
| DiPE_Linear | DASFAA | 2026 | 267 |
| SparseTSF | ICML Oral | 2024 | 41 |
| FEATHer (ours) | (IoT-J) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | arXiv | 2024 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

→ 5 orders of magnitude (267 → 1.6M) across the lineup. **Two 2026 SOTA
papers (DiPE-Linear DASFAA, MDMLP-EIA AAAI)** address R4's "outdated
baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation (otherwise
   `utils.masking`, `layers.X`, etc. collide between baselines).
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures (`forward(x_enc, x_mark, x_dec,
   x_mark_dec)`, `forward(x, cycle_index)`, tuple returns) into our
   uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** (wzhwzhwzh0921/S-D-Mamba) — selective-scan CUDA kernels
  (`selective_scan_cuda`, `causal_conv1d_cuda`) have no Cortex-M / STM32
  port. Out of scope for edge-deployable comparison.
- **TimeCMA** (ChenxiLiu-HNU/TimeCMA, AAAI 2025) — depends on frozen GPT-2
  (~124M params) features, 5 orders of magnitude above our sub-1K budget.

Reviewer-defense paragraph (paper draft):
> "We deliberately exclude architectures incompatible with our deployment
> scope: Mamba-family models require custom CUDA kernels with no MCU port;
> LLM-augmented forecasters (e.g., TimeCMA) depend on frozen language
> model features that exceed our 1K-parameter budget by 5 orders of
> magnitude. These methods are cited in Related Work but lie outside our
> deployment-constrained benchmark."

## Training protocol — TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ✅ Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | LTSF benchmark convention (iTransformer/PatchTST conventions) |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; model selection = best-val epoch, test evaluated once on that model (matches every baseline's official protocol) |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025–2029 (5 seeds) | TFB / CF-JEPA convention |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ⚠️ Per-(method, dataset) from each repo's official ETT* / ECL / Traffic / Weather / Exchange script

| Axis | Source |
|---|---|
| `lr` | each paper's per-dataset script (1e-4 → 5e-2) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs (`d_model`, `d_ff`, `e_layers`, `n_heads`, `dropout`) | paper Table or official script |
| Data-frequency-tied (`cycle` for TQNet, `period_len` for SparseTSF) | dataset's native sampling rate |

Routing implementation:
- `baselines/__init__.py` holds `_METHOD_DEFAULTS` (canonical ETTh1 config
  per method) and `_DATASET_OVERRIDES` (per-(method, dataset) diff).
- Orchestrator merges defaults + overrides + any CLI `--lr` / `--loss`,
  then forwards `lr` / `loss` as top-level worker flags and the rest as
  `--model_overrides "key=value;key2=value2"`.
- Worker's `_apply_overrides()` parses with int/float/bool coercion and
  sets each key on `args` before model construction; wrapper's
  `getattr(args, "key", default)` picks up the dataset-specific value.

### FEATHer's protocol — per-dataset config (revised 2026-06-17)

**Superseded:** FEATHer originally used a single config across all 8
datasets ("generalizes without tuning" narrative). Decided 2026-06-17 to
tune FEATHer **per-dataset like every baseline** so the main table shows
its best numbers — a single untuned config risked losing cells to
per-dataset-tuned baselines. `run_hp_search.py --summary` emits the
per-dataset winners; they go into `_DATASET_OVERRIDES` as `("FEATHer",
<data>)` rows. The single-config sensitivity data still falls out of the
search for free (the `hp_base` rows) if ever wanted as a side analysis.

Fair-comparison framing (now symmetric):

> *"Every method, including FEATHer, uses per-dataset hyperparameters
> selected on the validation split, ensuring an apples-to-apples
> comparison at each method's best operating point."*

## Architecture (FEATHer model)

The FEATHer model (`baselines/FEATHer/FEATHer.py`) has four components:

1. **Multi-band Decomposition** — depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). Configurable `num_bands ∈ {2,3,4}`.
2. **DenseTemporalKernel (DTK)** — project to latent space → depthwise
   causal conv → project back. Shared across all bands.
3. **FFTFrequencyGate** — FFT magnitude → Conv1d → softmax weights, one
   weight per band, adaptive per sample. Casts to float32 to avoid cuFFT
   half-precision restriction.
4. **SparsePeriodKernel (SPK)** — period-aware forecasting head. Reshapes
   by period phases, shared linear projection across periods, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss. Spectral
loss is FEATHer-specific; the worker skips it for other baselines.

Key training args (FEATHer-specific):

| Argument | Default | Description |
|---|---|---|
| `--d_state` | 8 | DTK latent dimension |
| `--kernel_size` | 7 | DTK conv kernel size |
| `--period` | 12 | SPK period (must divide both seq_len and pred_len) |
| `--num_bands` | 3 | Frequency bands (2, 3, or 4) |
| `--lambda_spec` | 0.01 | Spectral-separation loss weight |

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics are computed once on the best-val-epoch model (early stopping
patience=10); `val_loss` is that epoch's validation loss and `best_epoch`
its index. The saved checkpoint is the same best-val model.

`exp_tag` partitions experiments — e.g. `main`, `robustness_gauss`,
`ablation_dtk`, `smoke`. Use `--results_csv PATH` to redirect for
verification runs without contaminating the main CSV.

## Datasets

Loaded via `utils/data_factory.py`. Lazy darts import — only datasets that
actually use darts trigger the import.

| Dataset | Source | Horizon set |
|---|---|---|
| **Steel / GasTurbine / WindSCADA** (MFG, **main table**; PMSM dropped 2026-07-02) | local CSV (see `tools/prep_manufacturing.py`) | [96, 192, 336, 720] |
| **CMAPSS / CMAPSS3 / PMSM** (short-horizon PdM section: C-MAPSS FD001, FD003, Paderborn motor multi-session) | local CSV (unit-aware `Dataset_CMAPSS`) | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (now generalization) | darts library | [96, 192, 336, 720] |
| SML, Volatility | local CSV | [24, 48, 96, 192] |
| PEMS03/04/08/PEMS_BAY/METR | local CSV | [12, 24, 48, 96] |
| AirQuality, PM, nrel | local CSV | [96, 192, 336, 720] |

Manufacturing datasets (JMS pivot): each `data/<name>/` holds the raw source
+ the cleaned `data.csv` that `data_factory.py` loads. Regenerate any
`data.csv` from raw with `python tools/prep_manufacturing.py [<name> ...]`.
`data/` is gitignored — transfer raw (or data.csv) to the server manually.

## Known issues / gotchas

- **AMP is forcibly disabled** in `scripts/benchmarks/run_forecast.py`.
  Re-enabling will break FITS (complex grads) and TimesNet (half-precision
  cuFFT on non-power-of-2 sizes).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import. Direct `from baselines.X.Y import Model` will leak
  `sys.modules` entries (`utils`, `layers`, `models`, etc.) across
  baselines and produce `ModuleNotFoundError: No module named
  'utils.masking'` or similar.
- **TQNet** receives `cycle_index = torch.zeros(B)` from the wrapper since
  our DataLoader does not expose per-sample cycle position. Sub-optimal vs
  paper but consistent across all batches.
- **R2 metric** can explode to extreme negatives (~-3e7 observed on
  Traffic) when channels are near-constant within a window (SST≈0).
  MSE/MAE/RMSE are unaffected; R2/CORR are supplementary only. The old
  `-1e9` placeholder bug is gone (single-pass test evaluation since
  2026-06-11). Triage later.
- **FEATHer params scale as D²** (`in_proj = Linear(D, D)`): 453 @ D=7
  (ETT), 866 @ D=14 (Volatility) — sub-1K holds only for D≤14. Weather
  1.4K, Electricity 115K, Traffic 776K. Paper scopes the sub-1K claim
  to edge-typical channel counts; Traffic/Electricity are framed as
  scalability stress tests. A channel-independent variant is a possible
  future ablation axis, not a pre-sweep change.
- **FEATHer params also scale with pred_len** (SPK `backbone = Linear(n, m)`,
  n=seq/P, m=H/P): all quoted counts (402/677/738/866) are **H=96** values.
  At H=720 with the base config, D≥11 exceeds 1K (GasTurbine 1,093 / PMSM
  1,154 / WindSCADA 1,282). The manuscript's sub-1K claim must be scoped to
  H=96 or report per-horizon counts (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1** ✅ Multi-seed infrastructure (set_seed, registry, orchestrator,
  worker, audit dashboard).
- **Phase 2** ✅ Clone 11 baselines + write wrappers. 12-model lineup
  (FEATHer + 11), S-Mamba/TimeCMA dropped as out-of-scope. DiPE-Linear +
  MDMLP-EIA added 2026-06-02 to address R4 "outdated baselines".
- **Phase 3** ✅ Noise robustness infra — `utils/noise.py` (Gaussian /
  missing / impulse / quantization), `scripts/benchmarks/run_robustness.py`,
  top-level `run_robustness.py` orchestrator. **IoT-J selling point.**
- **Phase 4** (next) Main 5-seed sweep: 12 models × 8 datasets × 4
  horizons × 5 seeds = **1,920 runs**. GPU days. Run on server, not here.
- **Phase 4b** Robustness sweep: 12 models × **4 representative datasets**
  (ETTh1 / Weather / Electricity / SML — canonical / real-sensor /
  smart-grid / smart-home) × 4 horizons × 5 seeds × 20 conditions (1 clean
  + 5 gauss + 5 miss + 4 impulse + 5 quant) = **19,200 rows**. Inference
  only (loads `--save_model` checkpoints). The 4-dataset scope is set in
  `run_robustness.py:ROBUSTNESS_DATASETS` and keeps the sweep tractable
  vs. the 8-dataset main table.
- **Phase 5** Paper artifacts under `tools/paper/` — main table, robustness
  figure, statistical significance (Wilcoxon).

## Paper context

- Target venue: **Journal of Manufacturing Systems (JMS)**, Elsevier/SME, Q1
  (changed 2026-06-25 from IEEE IoT-J — see the manufacturing-pivot status at
  the top). Manuscript needs a major rewrite toward manufacturing/predictive-
  maintenance framing; `feather_iotj.tex` IoT-J framing is now largely
  obsolete. Manufacturing-LTSF precedent caveat: Steel/GasTurbine/PMSM are
  established as datasets but not as long-term-forecasting benchmarks
  (WindSCADA forecasting is) — frame as "first to cast these as edge PdM LTSF".
- **Submission format (decided 2026-06-25): Word** (user prefers Word over
  LaTeX). Elsevier "Your Paper Your Way" → NO rigid template needed at initial
  submission; write a plain single-column Word doc following the JMS Guide for
  Authors (sections, structured content, Highlights 3–5 bullets ≤85 chars,
  Vancouver-numbered refs via Mendeley/Zotero, CRediT + competing-interest +
  data-availability declarations). The generic Elsevier "Research Article
  template and guidance.docx" is COMPAG-branded (wrong journal) — not used.
- **Manuscript writing plan**: existing `feather_iotj.tex` is structurally
  near-complete — reuse map: **Methodology (Sec III) + Theoretical Analysis
  (Sec IV) reusable ~as-is** (trim theory for an applied venue); **Intro,
  Related Work, Datasets need manufacturing/PdM reframe**; Results/Ablation/
  Robustness wait for sweep numbers (placeholders). Drafting medium: write
  markdown in `manuscript/drafts/` → paste into Word. Order: Intro → Datasets
  → Related → Method/Theory/Setup polish → results when sweeps land.
- **Proposed story (pending final user OK)**: edge predictive-maintenance /
  process-monitoring forecasting that runs on machine-side MCUs; FEATHer =
  sub-1K-param Cortex-M3-deployable forecaster, robust to shop-floor sensor
  faults, validated across 5 manufacturing domains. Contributions: (1) sub-1K
  multiscale-frequency forecaster for edge PdM, (2) 5 heterogeneous mfg sensor
  domains, (3) robustness to sensor faults, (4) Cortex-M3 deployment evidence.
- Repositioning: stronger emphasis on Cortex-M3 deployment, sensor-noise
  robustness, missing-value handling. Lighter on theoretical claims (R8
  flagged Theorem 1/2/3-5 numbering inconsistency in TPAMI manuscript).
- Detailed reviewer-by-reviewer notes and excluded-baseline rationale in
  memory.
