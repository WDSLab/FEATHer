# Introduction — JMS draft (manufacturing reframe)

> Working draft for the Journal of Manufacturing Systems submission. Reframes
> the IoT-J Introduction (`feather_iotj.tex` §I) around manufacturing systems /
> predictive maintenance. Citations are `[key]` placeholders matching
> `feather.bib`; resolve in the Word reference manager. Numbers tagged `‹TBD›`
> wait on the manufacturing sweep. Technical contributions (C1–C4) are
> unchanged from the IoT-J version; C5–C7 are reframed for manufacturing.

## Para 1 — Manufacturing motivation + the move to the edge

Time-series forecasting underpins core functions of modern manufacturing
systems: predictive maintenance, process and condition monitoring, energy
management, production scheduling, and early warning of quality or safety
anomalies. As smart-manufacturing and Industry-4.0 architectures move from
centralized cloud analytics toward cyber-physical operation, these forecasting
functions are increasingly required to run **directly on the machine** — on
programmable logic controllers (PLCs), embedded microcontrollers, and
industrial-IoT sensor nodes attached to production equipment — to meet
real-time control deadlines, reduce dependence on plant-floor connectivity,
and keep operational data on-premises. Such edge platforms typically offer
kilobyte-scale weight budgets, single-digit-megahertz cores, and
millisecond-level response deadlines, while operating under the nonstationary
conditions characteristic of the shop floor: sensor noise, intermittent
communication losses, and quantization-induced distortion. Large
Transformer-based architectures, deep convolutional encoders, and even
moderately sized state-space models are impractical in this regime. This
motivates forecasting designs that retain accuracy over long horizons under
**extreme parameter budgets** and that remain reliable under realistic sensor
faults.

## Para 2 — Gaps in lightweight forecasters for manufacturing

Recent research has produced a range of lightweight forecasting architectures —
DLinear [Zeng2023DLinear], TiDE [Das2024TiDE], TSMixer [Chen2023TSMixer],
FITS [Xu2024FITS], CycleNet [Lin2024CycleNet], SparseTSF [Lin2024SparseTSF],
and very recently DiPE-Linear [Wintertee2026DiPELinear] and
MDMLP-EIA [Zh2026MDMLPEIA] — demonstrating the value of linear decomposition,
shallow temporal mixing, and period-aware projections. Three challenges
nevertheless remain for long-horizon forecasting of manufacturing signals.
**First**, many designs rely on a *single* temporal scale or a fixed periodic
structure, which is insufficient for the hierarchical patterns of industrial
equipment signals, where rapid fluctuations (e.g., load transients), mid-range
transitions (process regime changes), and long-term drifts (seasonal demand,
tool/component wear) coexist. **Second**, lightweight architectures often lack
an explicit mechanism for structured frequency decomposition, forcing
heterogeneous temporal components into a single representational pathway and
inducing cross-frequency interference. **Third**, despite the "lightweight"
label, many models still operate in the 10³–10⁵ parameter range, exceeding the
budgets of tightly constrained machine-side controllers. Compounding these, the
lightweight-forecasting literature is validated almost entirely on generic LTSF
benchmarks (electricity, weather, traffic); whether these models transfer to
heterogeneous manufacturing sensor streams — and whether they survive realistic
shop-floor sensor degradation — remains largely unexamined.

## Para 3 — FEATHer

To address these gaps, we present **FEATHer**, a deployable forecasting system
for accurate long-horizon prediction under severe resource constraints. FEATHer
organizes representations explicitly across temporal scales and fuses them
adaptively from the spectral characteristics of the input. It comprises four
lightweight components (Fig. 1): a **multiscale temporal decomposition** that
splits the instance-normalized input into point-, high-, mid-, and
low-frequency pathways via depthwise 1-D convolutions; a shared **Dense
Temporal Kernel (DTK)** that mixes time information within each band through a
projection–depthwise–projection sequence, without recurrence or self-attention;
a **frequency-aware branch gating** module that fuses pathways using softmax
weights computed from the FFT magnitude of the normalized input; and a
**Sparse-Period Kernel (SPK)** that reconstructs long-horizon outputs through
period-aligned reshaping and a shared linear projection.

## Para 4 — Manufacturing validation, robustness, edge deployment

These choices place FEATHer in a sub-1K-parameter regime across the entire
benchmark — ‹402›–‹866› parameters on every dataset evaluated — with deployment
validated on Cortex-M3 hardware under 16–64 KB RAM budgets in streaming
batch-one inference (§Edge). We evaluate FEATHer across **five heterogeneous
industrial domains organized in two evaluation settings that match the
structure of the underlying signals**. *Long-horizon forecasting* (96–720
steps) is evaluated on three continuous process-monitoring streams —
steel-plant energy [Sathishkumar2021Steel], combined-cycle gas-turbine
emission [Kaya2019GasTurbine], and wind-turbine SCADA [Plumley2022Kelmarsh].
*Short-horizon predictive-maintenance forecasting* (24–96 steps) is evaluated
on unit-segmented degradation data, where trajectories are structurally too
short for long horizons: the NASA C-MAPSS turbofan prognostics benchmark
(subsets FD001 and FD003, two fault regimes) [Saxena2008CMAPSS] and a
multi-session electric-motor (PMSM) drive stream [Kirchgassner2021PMSM],
with windows never crossing a unit boundary and train/validation/test splits
by unit. All tables include a naive-persistence baseline, making the
difficulty of each forecasting task explicit. Beyond clean-input accuracy, we
assess robustness against four classes of shop-floor sensor faults — additive
Gaussian noise, missing values, impulse outliers, and quantization
distortion — across ‹TBD› corruption conditions, an evaluation axis
underrepresented in the forecasting literature and central to field-deployed
predictive maintenance (§Robustness).

## Contributions

- **(C1)** An ultra-lightweight **multiscale decomposition** that separates
  input dynamics into four frequency-aligned pathways, enabling band-specialized
  temporal modeling under extreme parameter budgets while suppressing
  cross-frequency interference.
- **(C2)** A shared **Dense Temporal Kernel (DTK)** that captures temporal
  dependencies without recurrence or self-attention and prevents parameter
  growth in multi-branch designs.
- **(C3)** A **frequency-aware gating** mechanism that adaptively fuses the
  multiscale pathways from the FFT magnitude of the input, providing structured
  adaptation to nonstationary dynamics with negligible parameter overhead.
- **(C4)** A **Sparse-Period Kernel (SPK)** for phase-aligned long-horizon
  reconstruction that captures periodic/seasonal structure with parameter count
  exactly nm = (L/P)(H/P) (Theorem ‹thm:spk-minimal›), without increasing depth.
- **(C5, reframed)** A two-tier benchmark on **five heterogeneous industrial
  domains** — long-horizon forecasting on three continuous process-monitoring
  streams and short-horizon predictive-maintenance forecasting on two
  unit-segmented degradation datasets — against eleven recent baselines
  spanning five orders of magnitude in parameter count (SparseTSF/DiPE-Linear
  at ≤10³ to MDMLP-EIA at 1.6×10⁶), with per-(method, dataset)
  validation-selected learning rates under fixed official architectures, five
  random seeds, naive-persistence baselines, and pairwise Wilcoxon signed-rank
  tests on every comparison (§Results). To our knowledge this is the first
  study to cast these manufacturing sensor datasets as an edge
  predictive-maintenance forecasting benchmark.
- **(C6, reframed)** A **shop-floor sensor-fault robustness** evaluation: ‹TBD›
  corruption conditions across four fault axes on manufacturing datasets, with
  trained checkpoints reused at inference so the result reflects the deployed
  model exactly (§Robustness).
- **(C7)** A Cortex-M3 **on-device measurement protocol** reporting peak RAM,
  flash usage, activation arena, and per-sample latency under 16–64 KB RAM
  budgets, making the deployability claim verifiable rather than abstract
  (§Edge).

## Paper organization

‹standard "the remainder is organized as follows" paragraph — fill once the
final section list for the JMS version is fixed (theory section likely
trimmed for the applied venue).›

---
### Reframe notes (not for the manuscript)
- Opening narrowed from "manufacturing, logistics, transportation" → manufacturing-centric.
- **Dataset scope FINAL (2026-07-02)**: main long-horizon = Steel + GasTurbine
  + WindSCADA [96..720]; short-horizon PdM section = C-MAPSS FD001 + FD003 +
  PMSM (69 sessions, 30-s mean, unit-aware) [24,48,96]. TEP dropped (anomaly
  benchmark, D=50 breaks sub-1K); PMSM moved out of the long-horizon table
  (native-2Hz horizons trivial vs copy-last) into the PdM section. Sub-1K now
  holds on EVERY dataset → Para 4 upgraded from "configuration-dependent"
  to a clean sub-1K claim scoped to this benchmark.
- HP protocol (state in §Setup): all 12 methods use official/canonical fixed
  architectures; per-(method, dataset) learning rate selected on validation
  from a shared 5-point grid. FEATHer's canonical architecture is fixed on the
  standard LTSF benchmarks (OFAT) BEFORE any manufacturing experiment — no
  manufacturing data touches architecture choices.
- Naive persistence baseline row in both tables (defuses "is the task
  trivial?"; GasTurbine framed on an operating-hours axis).
- Removed the IoT-J label from C6 → "shop-floor sensor-fault robustness".
- Robustness condition count left ‹TBD› (was 19,200 for the LTSF 4-dataset
  scope; manufacturing robustness dataset swap decided in principle, count
  pending code change).
- Bib keys needed: Sathishkumar2021Steel, Kaya2019GasTurbine,
  Plumley2022Kelmarsh, Kirchgassner2021PMSM, Saxena2008CMAPSS (dataset-origin
  papers; Downs1993TEP no longer needed unless TEP is cited as excluded).
