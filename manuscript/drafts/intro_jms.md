# Introduction — JMS draft (manufacturing reframe)

> Working draft for the Journal of Manufacturing Systems submission. Reframes
> the IoT-J Introduction (`feather_iotj.tex` §I) around manufacturing systems /
> predictive maintenance. Citations are `[key]` placeholders matching
> `feather.bib`; resolve in the Word reference manager. Numbers tagged `‹TBD›`
> wait on the manufacturing sweep.
>
> **English paragraph + Korean mirror** (paragraph-level). Paste the English
> into Word; the `>` Korean lines are for internal review, drop them from the
> submission. Rewritten 2026-07-16 to the current framing: manufacturing-6
> datasets, A+C motivation with the high-rate "bridge" (motivation only, we
> evaluate the slow-process regime), robustness stated conservatively as one
> reliability axis (central-pillar strength deferred to sweep results).

## Para 1 — Manufacturing motivation, why long horizons on the edge

Time-series forecasting underpins core functions of modern manufacturing
systems: predictive maintenance, process and condition monitoring, energy
management, and early warning of quality or safety anomalies. As Industry-4.0
architectures move from centralized cloud analytics toward cyber-physical
operation, these forecasting functions are increasingly required to run
directly on the machine, on programmable logic controllers (PLCs), embedded
microcontrollers, and industrial-IoT sensor nodes attached to production
equipment, in order to meet real-time control deadlines, reduce dependence on
plant-floor connectivity, and keep operational data on-premises [‹cite›].

> 시계열 예측은 현대 제조 시스템의 핵심 기능을 떠받친다: 예지보전, 공정 및
> 상태 모니터링, 에너지 관리, 품질·안전 이상의 조기 경보. Industry-4.0
> 아키텍처가 중앙집중식 클라우드 분석에서 사이버-물리 운영으로 이동하면서,
> 이러한 예측 기능은 실시간 제어 마감을 맞추고, 현장 연결성 의존을 줄이며,
> 운영 데이터를 온프레미스로 유지하기 위해 점점 더 기계 위에서 직접,
> 즉 PLC·임베디드 마이크로컨트롤러·산업용 IoT 센서 노드에서 실행될 것을
> 요구받는다.

A defining feature of this setting is that forecasting horizons are counted in
sampling steps, and the required step count is set jointly by the sensing rate
and the operational lead time. At one extreme, high-rate monitoring of
transient mechanical events samples at kilohertz-to-megahertz rates, where even
a sub-second decision window spans hundreds to thousands of steps, and the fast
dynamics that must be predicted cannot be removed by decimation without
aliasing away the very event of interest [‹HR-SHM cite›]. At the other extreme,
slow process monitoring samples every few seconds to minutes, where operational
lead times (component-replacement windows, scheduling horizons, thermal and
wear drift) again push the forecast to hundreds of steps ahead. Both extremes
make many-step, long-horizon forecasting a native requirement of on-device
manufacturing intelligence, not an artifact of resolution choice. This work
targets the slow-process regime, for which public long-horizon benchmarks
exist; the high-rate regime motivates the same many-step problem at the
opposite end of the sensing spectrum.

> 이 설정의 결정적 특징은 예측 지평(horizon)이 샘플링 스텝 수로 측정되며, 필요한
> 스텝 수가 센싱 속도와 운영 리드타임에 의해 함께 결정된다는 점이다. 한쪽
> 극단에서, 과도(transient) 기계 이벤트의 고속 모니터링은 킬로헤르츠~메가헤르츠로
> 샘플링하며, 1초 이내의 결정 구간조차 수백~수천 스텝에 걸치고, 예측해야 하는
> 빠른 동역학은 다운샘플링으로 제거하면 관심 대상 이벤트 자체가 에일리어싱되어
> 사라진다. 다른 쪽 극단에서, 느린 공정 모니터링은 수 초~수 분마다 샘플링하며,
> 운영 리드타임(부품 교체 구간, 스케줄링 지평, 열·마모 드리프트)이 예측을 다시
> 수백 스텝 앞으로 밀어낸다. 두 극단 모두 다스텝·장기 예측을 온디바이스 제조
> 지능의 본질적 요구사항으로 만들며, 이는 해상도 선택의 부산물이 아니다. 본
> 연구는 공개 장기 벤치마크가 존재하는 느린-공정 영역을 대상으로 하며, 고속
> 영역은 센싱 스펙트럼의 반대쪽 끝에서 동일한 다스텝 문제를 동기화한다.

Such edge platforms typically offer kilobyte-scale weight budgets,
single-digit-megahertz cores, and millisecond-level response deadlines, while
operating under the nonstationary conditions characteristic of the shop floor:
sensor noise, intermittent communication losses, and quantization-induced
distortion. Large Transformer-based architectures, deep convolutional encoders,
and even moderately sized state-space models are impractical in this regime.
This motivates forecasting designs that retain accuracy over long horizons
under extreme parameter budgets and that remain reliable under realistic sensor
faults.

> 이러한 엣지 플랫폼은 일반적으로 킬로바이트급 가중치 예산, 단일 자릿수
> 메가헤르츠 코어, 밀리초 수준의 응답 마감을 제공하며, 동시에 현장 특유의
> 비정상(nonstationary) 조건, 즉 센서 잡음·간헐적 통신 손실·양자화로 인한 왜곡
> 아래에서 동작한다. 대형 Transformer 기반 아키텍처, 깊은 합성곱 인코더,
> 심지어 중간 규모의 상태공간 모델조차 이 영역에서는 비현실적이다. 이는 극단적
> 파라미터 예산 아래에서 장기 지평에 걸쳐 정확도를 유지하면서, 현실적인 센서
> 결함 아래에서도 신뢰성을 유지하는 예측 설계를 요구한다.

## Para 2 — Gaps in lightweight forecasters for manufacturing

Recent research has produced a range of lightweight forecasting architectures,
including DLinear [Zeng2023DLinear], TiDE [Das2024TiDE], TSMixer
[Chen2023TSMixer], FITS [Xu2024FITS], CycleNet [Lin2024CycleNet], SparseTSF
[Lin2024SparseTSF], and very recently DiPE-Linear [Wintertee2026DiPELinear] and
MDMLP-EIA [Zh2026MDMLPEIA], demonstrating the value of linear decomposition,
shallow temporal mixing, and period-aware projections. Three challenges
nevertheless remain for long-horizon forecasting of manufacturing signals.
First, many designs rely on a single temporal scale or a fixed periodic
structure, which is insufficient for the hierarchical patterns of industrial
equipment signals, where rapid fluctuations (load transients), mid-range
transitions (process regime changes), and long-term drifts (seasonal demand,
tool and component wear) coexist. Second, lightweight architectures often lack
an explicit mechanism for structured frequency decomposition, forcing
heterogeneous temporal components into a single representational pathway and
inducing cross-frequency interference. Third, despite the "lightweight" label,
many models still operate in the 10^3 to 10^5 parameter range, exceeding the
budgets of tightly constrained machine-side controllers. Compounding these, the
lightweight-forecasting literature is validated almost entirely on generic LTSF
benchmarks (electricity, weather, traffic); whether these models transfer to
heterogeneous manufacturing sensor streams, and whether they survive realistic
shop-floor sensor degradation, remains largely unexamined.

> 최근 연구는 다양한 경량 예측 아키텍처를 내놓았다: DLinear, TiDE, TSMixer,
> FITS, CycleNet, SparseTSF, 그리고 아주 최근의 DiPE-Linear와 MDMLP-EIA는
> 선형 분해, 얕은 시간 혼합, 주기-인식 투영의 가치를 보여준다. 그럼에도
> 제조 신호의 장기 예측에는 세 가지 과제가 남는다. 첫째, 많은 설계가 단일
> 시간 스케일이나 고정된 주기 구조에 의존하는데, 이는 빠른 변동(부하 과도),
> 중간 범위 전이(공정 국면 변화), 장기 드리프트(계절 수요, 공구·부품 마모)가
> 공존하는 산업 장비 신호의 계층적 패턴에는 불충분하다. 둘째, 경량
> 아키텍처는 구조적 주파수 분해를 위한 명시적 메커니즘이 없는 경우가 많아,
> 이질적 시간 성분을 단일 표현 경로로 강제하고 교차-주파수 간섭을 유발한다.
> 셋째, "경량"이라는 이름표에도 불구하고 많은 모델이 여전히 10^3~10^5
> 파라미터 범위에서 동작하여, 빠듯하게 제약된 기계-측 컨트롤러의 예산을
> 초과한다. 여기에 더해, 경량-예측 문헌은 거의 전적으로 일반 LTSF
> 벤치마크(전력·기상·교통)에서만 검증되며, 이 모델들이 이질적인 제조 센서
> 스트림으로 전이되는지, 그리고 현실적인 현장 센서 열화를 견디는지는 대체로
> 검증되지 않은 채 남아 있다.

## Para 3 — FEATHer

To address these gaps, we present FEATHer, a deployable forecasting system for
accurate long-horizon prediction under severe resource constraints. FEATHer
organizes representations explicitly across temporal scales and fuses them
adaptively from the spectral characteristics of the input. It comprises four
lightweight components (Fig. 1): a multiscale temporal decomposition that
splits the instance-normalized input into point-, high-, mid-, and
low-frequency pathways via depthwise 1-D convolutions; a shared Dense Temporal
Kernel (DTK) that mixes time information within each band through a
projection-depthwise-projection sequence, without recurrence or self-attention;
a frequency-aware branch gating module that fuses pathways using softmax weights
computed from the FFT magnitude of the normalized input; and a Sparse-Period
Kernel (SPK) that reconstructs long-horizon outputs through period-aligned
reshaping and a shared linear projection.

> 이러한 한계를 해결하기 위해 우리는 심각한 자원 제약 아래에서 정확한 장기
> 예측을 수행하는 배포 가능 예측 시스템 FEATHer를 제안한다. FEATHer는 표현을
> 시간 스케일에 걸쳐 명시적으로 구성하고, 입력의 스펙트럼 특성으로부터
> 적응적으로 융합한다. 이는 네 개의 경량 구성요소로 이루어진다(Fig. 1):
> 인스턴스-정규화된 입력을 깊이별(depthwise) 1-D 합성곱으로 점(point)·고·중·
> 저주파 경로로 분리하는 다중스케일 시간 분해; 재귀나 셀프-어텐션 없이
> 투영-깊이별-투영 순서로 각 대역 내부의 시간 정보를 혼합하는 공유 Dense
> Temporal Kernel(DTK); 정규화된 입력의 FFT 크기로부터 계산된 softmax
> 가중치로 경로를 융합하는 주파수-인식 분기 게이팅 모듈; 그리고 주기-정렬
> 재구성과 공유 선형 투영으로 장기 출력을 복원하는 Sparse-Period
> Kernel(SPK).

## Para 4 — Manufacturing validation, robustness, edge deployment

These choices place FEATHer in a sub-1K-parameter regime at the H=96 horizon on
the five edge-scale (D<=14) manufacturing datasets, ‹402›-‹866› parameters,
with blast-furnace process (D=28) retained as a declared higher-dimensional
stress case; deployment is validated on Cortex-M3 hardware under 16-64 KB RAM
budgets in streaming batch-one inference (§Edge). We evaluate FEATHer on six heterogeneous
industrial domains cast as an edge long-horizon forecasting benchmark: steel-
plant energy [Sathishkumar2021Steel], wind-turbine SCADA [Plumley2022Kelmarsh],
blast-furnace process signals [‹BlastFurnace›], industrial N2O emissions
[‹N2O›], metro compressor sensors [‹MetroPT3›], and metal-industry electricity load
with labeled extreme-weather event windows [‹EWELD›], each over horizons {96,
192, 336, 720}. Every table reports a
naive-persistence baseline, making the difficulty of each forecasting task
explicit. Beyond clean-input accuracy, we assess reliability against four
classes of shop-floor sensor faults, additive Gaussian noise, missing values,
impulse outliers, and quantization distortion, across ‹TBD› corruption
conditions, an evaluation axis underrepresented in the forecasting literature
and directly relevant to field-deployed predictive maintenance (§Robustness).
The eight standard LTSF benchmarks are retained as a separate generalization
study (§Generalization).

> 이러한 설계는 FEATHer를 H=96 지평에서 다섯 개의 엣지-규모(D<=14) 제조
> 데이터셋에 대해 sub-1K 파라미터 영역(‹402›~‹866› 파라미터)에 위치시키며,
> 고로 공정(D=28)은 명시된 고차원 스트레스 케이스로 유지한다; 배포는 스트리밍
> 배치-1 추론에서 16-64 KB RAM 예산 아래 Cortex-M3 하드웨어로 검증된다(§Edge). 우리는
> FEATHer를 엣지 장기 예측 벤치마크로 캐스팅한 여섯 개의 이질적 산업 도메인에서
> 평가한다: 철강 플랜트 에너지, 풍력 터빈 SCADA, 고로(blast-furnace) 공정 신호,
> 산업용 N2O 배출, 지하철 압축기 센서, 극한기상 이벤트 창이 라벨된 금속산업 전력
> 부하, 각각 지평 {96, 192, 336, 720}에 대해. 모든 표는 나이브-지속(persistence) 베이스라인을 보고하여 각
> 예측 과제의 난이도를 명시한다. 클린-입력 정확도를 넘어, 우리는 네 종류의 현장
> 센서 결함, 즉 가산 가우시안 잡음·결측·임펄스 이상치·양자화 왜곡에 대한
> 신뢰성을 ‹TBD›개 손상 조건에 걸쳐 평가하는데, 이는 예측 문헌에서 과소대표된
> 평가 축이며 현장 배포 예지보전에 직접적으로 관련된다(§Robustness). 여덟 개의
> 표준 LTSF 벤치마크는 별도의 일반화 연구로 유지된다(§Generalization).

## Contributions

- **(C1)** An ultra-lightweight multiscale decomposition that separates input
  dynamics into four frequency-aligned pathways, enabling band-specialized
  temporal modeling under extreme parameter budgets while suppressing
  cross-frequency interference.
  > (C1) 입력 동역학을 네 개의 주파수-정렬 경로로 분리하여, 극단적 파라미터
  > 예산 아래에서 대역-특화 시간 모델링을 가능하게 하고 교차-주파수 간섭을
  > 억제하는 초경량 다중스케일 분해.

- **(C2)** A shared Dense Temporal Kernel (DTK) that captures temporal
  dependencies without recurrence or self-attention and prevents parameter
  growth in multi-branch designs.
  > (C2) 재귀나 셀프-어텐션 없이 시간 의존성을 포착하고 다중-분기 설계에서
  > 파라미터 증가를 막는 공유 Dense Temporal Kernel(DTK).

- **(C3)** A frequency-aware gating mechanism that adaptively fuses the
  multiscale pathways from the FFT magnitude of the input, providing structured
  adaptation to nonstationary dynamics with negligible parameter overhead.
  > (C3) 입력의 FFT 크기로부터 다중스케일 경로를 적응적으로 융합하여, 무시할
  > 만한 파라미터 오버헤드로 비정상 동역학에 대한 구조적 적응을 제공하는
  > 주파수-인식 게이팅 메커니즘.

- **(C4)** A Sparse-Period Kernel (SPK) for phase-aligned long-horizon
  reconstruction that captures periodic and seasonal structure with parameter
  count exactly nm = (L/P)(H/P) (Theorem ‹thm:spk-minimal›), without increasing
  depth.
  > (C4) 깊이를 늘리지 않고 파라미터 수 정확히 nm = (L/P)(H/P)로 주기·계절
  > 구조를 포착하는, 위상-정렬 장기 재구성을 위한 Sparse-Period Kernel(SPK).

- **(C5)** An edge long-horizon forecasting benchmark on six heterogeneous
  manufacturing domains (steel energy, wind SCADA, blast furnace, N2O
  emissions, metro compressor, extreme-weather load) against eleven recent
  baselines spanning five orders of magnitude in parameter count (SparseTSF and
  DiPE-Linear at <=10^3 to MDMLP-EIA at 1.6x10^6), with per-(method, dataset)
  validation-selected learning rates under fixed official architectures,
  multiple random seeds (‹N›), naive-persistence baselines, and pairwise
  Wilcoxon signed-rank tests on every comparison (§Results). To our knowledge
  this is the first study to cast these manufacturing sensor datasets as an edge
  long-horizon forecasting benchmark; the eight standard LTSF datasets are
  retained as a generalization study.
  > (C5) 파라미터 수에서 5차수(SparseTSF·DiPE-Linear의 <=10^3에서 MDMLP-EIA의
  > 1.6x10^6까지)에 걸친 11개의 최신 베이스라인과 비교하여, 여섯 개의 이질적
  > 제조 도메인(철강 에너지, 풍력 SCADA, 고로, N2O 배출, 지하철 압축기, 극한
  > 기상 부하)에서의 엣지 장기 예측 벤치마크. 고정된 공식 아키텍처 아래
  > (방법, 데이터셋)별 검증-선택 학습률, 다중 랜덤 시드(‹N›), 나이브-지속
  > 베이스라인, 모든 비교에 대한 쌍별 Wilcoxon 부호순위 검정을 사용한다.
  > 우리가 아는 한, 이는 이들 제조 센서 데이터셋을 엣지 장기 예측 벤치마크로
  > 캐스팅한 최초의 연구이며, 여덟 개의 표준 LTSF 데이터셋은 일반화 연구로
  > 유지된다.

- **(C6)** A shop-floor sensor-fault robustness evaluation across four fault
  axes on the manufacturing datasets, with trained checkpoints reused at
  inference so the reported behavior reflects the deployed model exactly
  (§Robustness).
  > (C6) 제조 데이터셋에서 네 개의 결함 축에 걸친 현장 센서-결함 견고성 평가.
  > 학습된 체크포인트를 추론에 재사용하여, 보고된 거동이 배포된 모델을 정확히
  > 반영하도록 한다(§Robustness).

- **(C7)** A Cortex-M3 on-device measurement protocol reporting peak RAM, flash
  usage, activation arena, and per-sample latency under 16-64 KB RAM budgets,
  making the deployability claim verifiable rather than abstract (§Edge).
  > (C7) 16-64 KB RAM 예산 아래에서 최대 RAM, 플래시 사용량, 활성화 아레나,
  > 샘플당 지연시간을 보고하는 Cortex-M3 온디바이스 측정 프로토콜로,
  > 배포가능성 주장을 추상적이지 않고 검증 가능하게 만든다(§Edge).

## Paper organization

‹standard "the remainder is organized as follows" paragraph — fill once the
final section list for the JMS version is fixed (theory section likely trimmed
for the applied venue).›

---
### Reframe notes (not for the manuscript)

- **Rewrite 2026-07-16**: reframed from the 2026-07-02 5-domain / two-setting
  (long-horizon + short-horizon PdM) version to the current **manufacturing-6**
  main table (Steel, WindSCADA, BlastFurnace, N2O, MetroPT3, EWELD; all
  [96..720]). GasTurbine removed. Short-horizon PdM (C-MAPSS/PMSM) dropped from
  the paper (lr-banked shelf). LTSF-8 demoted to a generalization study.
- **High-rate "bridge" (Para 1, motivation only)**: the high-rate / HR-SHM
  argument (fast sensing forces many-step forecasting, decimation aliases the
  event) is cited to motivate the many-step problem, NOT claimed as our
  evaluation regime. We explicitly state "this work targets the slow-process
  regime". Wording rule: never let a reviewer map us onto HR-SHM data and find
  a mismatch. Bib keys needed: HR-SHM overview (Dodson et al., high-rate SHM),
  multi-step forecasting for high-rate dynamic systems.
- **Robustness stated conservatively** (user decision 2026-07-16): one
  reliability axis in Para 1/Para 4/C6, not yet the central pillar. Central-
  pillar strength ("robustness, not accuracy, justifies the params") is
  deferred until the robustness sweep / spot-check lands. Do NOT over-claim
  graceful degradation in the intro before the evidence exists.
- **Seeds `‹N›`**: currently 3-seed for the running sweeps (compute triage;
  top-up to 5 in revision by re-running with `--num_seeds 5`). Freeze the
  number at table time.
- **Sub-1K scoping**: quoted counts are H=96 values. SPK backbone scales as
  (L/P)(H/P), so params grow with H; at H=720, D>=11 exceeds 1K. Keep the
  sub-1K headline scoped to H=96 (or give per-horizon counts). Do not repeat
  the IoT-J habit of attributing growth to channel count D only.
- **Two-tier / classic baselines (LSTM/GRU/TCN/XGBoost) intentionally NOT in
  the intro** — classic-baseline inclusion is uncommitted (code local, server
  not activated). Add only after that decision lands.
- Bib keys needed: Sathishkumar2021Steel, Plumley2022Kelmarsh, plus new
  dataset-origin keys for BlastFurnace (Mendeley 6d7jbc7tb5, Data in Brief
  2025), N2O, MetroPT3 (MetroPT-3), EWELD, and the two HR-SHM refs above.
