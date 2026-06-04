# Abstract

> **Source**: ported from `tex_workspace/feather_raw.tex` lines 7--40,
> rewritten for IoT-J targeting. Each English paragraph is followed by
> the Korean translation as a blockquote. Reviewer-fix anchors are
> noted inline as HTML comments (invisible in rendered Markdown).

---

<!-- R8 #1, R8 #2 tone-down applied. Word count target: 220-250. -->

Time-series forecasting underpins critical functions in modern industrial systems, including production scheduling, predictive maintenance, energy balancing, and safety monitoring across manufacturing, logistics, and transportation infrastructure.
> 시계열 예측은 제조, 물류, 운송 인프라 전반의 생산 일정 관리, 예지 보전, 에너지 균형, 안전 감시와 같은 현대 산업 시스템의 핵심 기능을 뒷받침한다.

As these domains shift toward cyber-physical automation, forecasting models are increasingly required to run directly on edge devices such as programmable logic controllers and embedded microcontrollers, where strict latency, memory, and energy budgets limit deployable models to at most a few thousand parameters.
> 이러한 영역이 사이버 물리 자동화로 전환됨에 따라, 예측 모델은 프로그래머블 로직 컨트롤러 및 임베디드 마이크로컨트롤러와 같은 에지 기기에서 직접 동작해야 하는 경우가 증가하고 있으며, 엄격한 지연 시간, 메모리, 에너지 제약은 배포 가능한 모델 크기를 최대 수천 개의 파라미터로 제한한다.

<!-- Repositioning: not "method paper" but "deployable system paper". -->

To meet these constraints, we present FEATHer, a deployable forecasting system built around four lightweight components: (i) a multiscale temporal decomposition that splits the input into point-, high-, mid-, and low-frequency pathways via depthwise 1D convolutions; (ii) a shared Dense Temporal Kernel that performs temporal mixing through projection, depthwise convolution, and projection, with no recurrence or attention; (iii) a frequency-aware branch gating mechanism that fuses multiscale representations using the spectral signature of the normalized input; and (iv) a Sparse Period Kernel that reconstructs long-horizon outputs through period-wise reshaping and a shared linear projection.
> 이러한 제약을 충족하기 위해, 본 연구는 네 가지 경량 구성 요소를 중심으로 한 배포 가능한 예측 시스템 FEATHer를 제안한다: (i) 깊이별 1D 합성곱을 통해 입력을 점, 고주파, 중주파, 저주파 경로로 분할하는 다중 스케일 시간 분해, (ii) 순환 신경망이나 어텐션 없이 사영, 깊이별 합성곱, 사영을 통해 시간적 혼합을 수행하는 공유 Dense Temporal Kernel(DTK), (iii) 정규화된 입력의 스펙트럼 특징을 활용해 다중 스케일 표현을 융합하는 주파수 인식 분기 게이팅, (iv) 주기 단위 재구성과 공유 선형 사영을 통해 장기 예측을 재구성하는 Sparse Period Kernel(SPK).

<!-- R8 #2 fix: sub-1K only holds for some configurations. -->

FEATHer operates in a configuration-dependent ultra-light regime, requiring 453 parameters on ETTh1 with seven channels and scaling to a few thousand parameters on higher-channel datasets such as Weather and Solar; the model is deployable on Cortex-M3 hardware under 16, 32, and 64 KB RAM budgets in streaming batch-one inference.
> FEATHer는 설정에 따라 달라지는 초경량 영역에서 동작하며, 7 채널의 ETTh1에서는 453개의 파라미터를 사용하고 Weather나 Solar와 같이 채널 수가 큰 데이터셋에서는 수천 개 수준으로 확장된다. 본 모델은 16, 32, 64 KB 램 예산 하에서 배치 크기 1의 스트리밍 추론으로 Cortex-M3 하드웨어에 배포 가능하다.

<!-- R8 #1 fix: tone down "broadly SOTA". -->

Across eight long-term forecasting benchmarks (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, Electricity, Traffic) and four prediction horizons with five seeds per cell, FEATHer is competitive in the ultra-light parameter regime against 11 baselines spanning five orders of magnitude in parameter count, including DiPE-Linear (DASFAA 2026) and MDMLP-EIA (AAAI 2026); larger-capacity baselines remain stronger on high-channel datasets such as Solar, Traffic, and Electricity, which we report transparently rather than averaging over.
> 8개의 장기 예측 벤치마크(ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange, Electricity, Traffic)와 4개 예측 구간에서 셀당 5개의 시드를 사용해 평가한 결과, FEATHer는 파라미터 수에서 5자리수 차이를 보이는 11개 기준 모델(DiPE-Linear (DASFAA 2026)과 MDMLP-EIA (AAAI 2026) 포함)에 대해 초경량 파라미터 영역에서 경쟁력을 보였으며, Solar, Traffic, Electricity와 같이 채널 수가 큰 데이터셋에서는 더 큰 용량의 기준 모델이 여전히 우세한 결과를 평균화하지 않고 투명하게 보고한다.

<!-- IoT-J selling point: R6 #7a robustness. -->

Beyond clean-input accuracy, we evaluate FEATHer's robustness against four classes of sensor faults that arise in field deployments — additive Gaussian noise, missing values, impulse outliers, and quantization-induced distortion — across 19,200 corruption conditions, and provide pairwise Wilcoxon signed-rank significance tests over five seeds to support all reported comparisons.
> 깨끗한 입력에서의 정확도를 넘어, 실제 배포 환경에서 발생하는 가산 가우시안 잡음, 결측값, 임펄스 이상치, 양자화로 인한 왜곡의 4개 센서 결함 범주에 대해 총 19,200개의 부패 조건에 걸친 FEATHer의 견고성을 평가하고, 보고된 모든 비교에 대해 5개 시드에서 쌍 별 Wilcoxon 부호 순위 검정의 유의성 결과를 함께 제공한다.

These results demonstrate that reliable long-horizon forecasting is achievable on resource-constrained industrial edge hardware, and they identify a practical operating regime for next-generation industrial IoT systems that require continuous, low-power inference under sensor noise and missing data.
> 이러한 결과는 자원이 제약된 산업용 에지 하드웨어에서 신뢰성 있는 장기 예측이 가능함을 입증하고, 센서 잡음과 결측 데이터 하에서 지속적이고 저전력의 추론을 요구하는 차세대 산업용 IoT 시스템을 위한 실용적인 동작 영역을 제시한다.

---

**Index Terms** — Time-series forecasting, edge AI, industrial IoT, ultra-lightweight models, on-device inference, sensor robustness.
> **색인어** — 시계열 예측, 에지 AI, 산업용 IoT, 초경량 모델, 온디바이스 추론, 센서 견고성.
