# Introduction

> **Source**: `tex_workspace/feather_raw.tex` lines 50--178.
> Body text retained as-is; citations kept as raw `[N]` markers for now
> (replaced with `\cite{}` keys during bib migration). Reviewer-fix
> anchors marked as HTML comments.

---

Time-series forecasting is a core capability in modern industrial intelligence [1], enabling production scheduling [2], anomaly detection [3], predictive maintenance [4], energy balancing [5], safety monitoring [6], and process control across manufacturing [7], logistics [8], transportation infrastructure [9]. Accurate long-horizon forecasting is increasingly critical as industrial environments transition toward cyber-physical and autonomous operation, where decisions must be made continuously using streaming sensor data [10].
> 시계열 예측은 현대 산업 지능의 핵심 역량 [1] 이며, 제조 [7], 물류 [8], 운송 인프라 [9] 전반의 생산 일정 [2], 이상 탐지 [3], 예지 보전 [4], 에너지 균형 [5], 안전 감시 [6], 공정 제어를 가능하게 한다. 산업 환경이 사이버 물리 및 자율 운영으로 전환됨에 따라, 스트리밍 센서 데이터를 사용해 지속적으로 의사결정을 수행해야 하는 장기 예측의 정확성은 점점 더 중요해지고 있다 [10].

<!-- R8 #7 IoT-J angle: tighten edge-deployment framing here. -->

At the same time, practical deployment increasingly requires forecasting models to run directly on edge platforms, such as programmable logic controllers (PLCs), embedded microcontrollers, industrial IoT sensors, and low-power gateways, under strict constraints on latency, memory, and energy [11]. These devices typically offer limited CPU throughput and small memory footprints while still demanding millisecond-level response times and reliable behavior under nonstationary operating conditions [12]. Under such constraints, large Transformer-based architectures, deep convolutional encoders, and even moderately sized state-space models are often impractical for real-world deployment [13]. This motivates the need for resource-constrained forecasting models that maintain accuracy over long horizons under extremely tight parameter budgets [14].
> 동시에 실용적 배포는 프로그래머블 로직 컨트롤러(PLC), 임베디드 마이크로컨트롤러, 산업용 IoT 센서, 저전력 게이트웨이와 같은 에지 플랫폼에서 예측 모델이 엄격한 지연 시간, 메모리, 에너지 제약 하에 직접 실행되어야 함을 요구하고 있다 [11]. 이러한 기기는 일반적으로 제한된 CPU 처리량과 작은 메모리 풋프린트를 제공하면서도 밀리초 단위의 응답 시간과 비정상 동작 조건 하에서의 안정적인 동작을 요구한다 [12]. 이러한 제약 하에서 대형 Transformer 기반 아키텍처, 심층 합성곱 인코더, 심지어 중간 크기의 상태 공간 모델조차 실제 배포에는 종종 비실용적이다 [13]. 이로 인해 매우 빠듯한 파라미터 예산 하에서 장기 예측 정확도를 유지하는 자원 제약형 예측 모델이 필요해진다 [14].

<!-- R4 #2-3 outdated baselines: DiPE-Linear (DASFAA 2026), MDMLP-EIA
     (AAAI 2026) should be added to the lightweight-method list below
     during IoT-J revision. -->

Recent research has explored lightweight forecasting architectures, including DLinear [15], TiDE [16], TSMixer [17], FITS [18], CycleNet [19], and SparseTSF [20]. While these methods demonstrate the potential of linear decomposition, shallow temporal mixing, and period-aware sparse projections, several challenges remain when applying them to industrial long-horizon forecasting. First, many designs rely on a single temporal scale or a fixed periodic structure, which is insufficient to represent the hierarchical patterns commonly observed in industrial signals, ranging from rapid fluctuations to medium-range transitions and long-term seasonal drifts [21]. Second, lightweight architecture often lacks explicit mechanism for structured frequency decomposition, forcing heterogeneous temporal components into a single representational pathway [22]. This can induce cross-frequency interference and degrade temporal resolution [23]. Third, despite being labeled lightweight, many models still require tens of thousands of parameters, which may exceed the extreme budgets imposed by tightly constrained edge controllers [24].
> 최근 연구는 DLinear [15], TiDE [16], TSMixer [17], FITS [18], CycleNet [19], SparseTSF [20]를 포함한 경량 예측 아키텍처를 탐색해 왔다. 이러한 방법들이 선형 분해, 얕은 시간적 혼합, 주기 인식 희소 사영의 가능성을 입증해 왔지만, 산업용 장기 예측에 적용할 때 몇 가지 과제가 남아 있다. 첫째, 많은 설계가 단일 시간 스케일 또는 고정된 주기 구조에 의존하고 있으며, 이는 빠른 변동에서 중기 전이, 장기 계절성 드리프트에 이르기까지 산업 신호에서 흔히 관찰되는 계층적 패턴을 표현하기에는 불충분하다 [21]. 둘째, 경량 아키텍처는 종종 구조화된 주파수 분해를 위한 명시적 메커니즘이 부족하여, 이질적인 시간 성분을 단일 표현 경로에 강제 결합한다 [22]. 이는 교차 주파수 간섭을 유발하고 시간 해상도를 저하시킬 수 있다 [23]. 셋째, 경량으로 분류되었음에도 불구하고 많은 모델이 여전히 수만 개의 파라미터를 필요로 하며, 이는 엄격하게 제약된 에지 컨트롤러가 부과하는 극단적 예산을 초과할 수 있다 [24].

<!-- R2 #4: multi-scale already in TimesNet/SCINet/Pyraformer. Need
     explicit differentiation paragraph in Sec 2 (Related Work). -->

To address these challenges, we propose Fourier-Efficient Adaptive Temporal Hierarchy Forecaster (FEATHer), an ultra-lightweight model designed for accurate long-horizon forecasting under severe resource constraints. FEATHer follows a structured design principle in which representations are explicitly organized across temporal scales and adaptively fused based on the spectral characteristics of the input. Specifically, FEATHer generates multiscale representations using lightweight temporal filtering via depthwise operations to obtain point-, high-, mid-, and low-frequency pathways. Each pathway is processed by a shared Dense Temporal Kernel (DTK), composed of a linear projection, a depthwise temporal convolution, and a reverse projection, enabling efficient temporal mixing without recurrence or self-attention. FEATHer further incorporates a frequency-aware gating module that dynamically reweights multiscale pathways by analyzing the spectrum of the normalized input, improving robustness under nonstationary dynamics. For long-horizon forecasting, FEATHer employs a Sparse Period Kernel (SPK) that reconstructs periodic and seasonal structure through phase-aligned and period-aligned reorganization with a shared linear transformation, enabling effective long-horizon modeling with minimal additional parameters.
> 이러한 과제를 해결하기 위해, 본 연구는 심각한 자원 제약 하에서 정확한 장기 예측을 위해 설계된 초경량 모델 Fourier-Efficient Adaptive Temporal Hierarchy Forecaster(FEATHer)를 제안한다. FEATHer는 표현을 시간 스케일에 따라 명시적으로 조직하고 입력의 스펙트럼 특성에 기반하여 적응적으로 융합하는 구조적 설계 원칙을 따른다. 구체적으로 FEATHer는 깊이별 연산을 통한 경량 시간 필터링으로 다중 스케일 표현을 생성하여 점, 고주파, 중주파, 저주파 경로를 얻는다. 각 경로는 선형 사영, 깊이별 시간 합성곱, 역 사영으로 구성된 공유 Dense Temporal Kernel(DTK)에 의해 처리되며, 순환이나 자기 어텐션 없이 효율적인 시간적 혼합을 가능하게 한다. FEATHer는 정규화된 입력의 스펙트럼을 분석하여 다중 스케일 경로를 동적으로 재가중하는 주파수 인식 게이팅 모듈을 추가로 통합하여, 비정상 동작 하에서의 견고성을 향상시킨다. 장기 예측을 위해 FEATHer는 위상 정렬 및 주기 정렬된 재구성과 공유 선형 변환을 통해 주기 및 계절 구조를 복원하는 Sparse Period Kernel(SPK)을 사용하여, 최소한의 추가 파라미터로 효과적인 장기 모델링을 가능하게 한다.

<!-- R8 #1 tone-down: "competitive accuracy" wording is OK; avoid
     "state-of-the-art" elsewhere. -->
<!-- R8 #2 tone-down: replace "fewer than 1,000 parameters" with
     "configuration-dependent ultra-light regime" during revision. -->

Owing to these architectural choices, FEATHer operates under an ultra-compact parameter budget, for example, fewer than 1,000 parameters in compact configurations, while delivering competitive accuracy across standard long-horizon forecasting benchmarks. These results suggest that FEATHer is practical for real-time industrial edge deployment under stringent latency, memory, and energy constraints. The key contributions of this work are as follows:
> 이러한 아키텍처 선택 덕분에 FEATHer는 초경량 파라미터 예산 하에서 동작하며, 예를 들어 작은 설정에서는 1,000개 미만의 파라미터를 사용하면서도 표준 장기 예측 벤치마크 전반에서 경쟁력 있는 정확도를 제공한다. 이러한 결과는 FEATHer가 엄격한 지연 시간, 메모리, 에너지 제약 하에서 실시간 산업 에지 배포에 실용적임을 시사한다. 본 연구의 주요 기여는 다음과 같다:

- **(C1)** We introduce an ultra-lightweight multiscale decomposition that separates input dynamics into frequency-aligned pathways, enabling hierarchical temporal modeling under extreme parameter budgets while reducing cross-frequency interference.
  > **(C1)** 입력 동역학을 주파수 정렬 경로로 분리하는 초경량 다중 스케일 분해를 도입하여, 극단적 파라미터 예산 하에서 계층적 시간 모델링을 가능하게 하고 교차 주파수 간섭을 줄인다.

- **(C2)** We propose the DTK, a shared lightweight mixing block that captures temporal dependencies without recurrence or self-attention, preventing parameter growth in multi-branch designs.
  > **(C2)** 다중 분기 설계에서 파라미터 증가를 방지하면서 순환이나 자기 어텐션 없이 시간적 의존성을 포착하는 공유 경량 혼합 블록 DTK를 제안한다.

- **(C3)** We develop a frequency-aware gating mechanism that adaptively fuses multiscale representations based on the input spectrum, enabling structured adaptation to nonstationary dynamics without additional learnable parameters.
  > **(C3)** 입력 스펙트럼에 기반하여 다중 스케일 표현을 적응적으로 융합하는 주파수 인식 게이팅 메커니즘을 개발하여, 추가 학습 파라미터 없이 비정상 동역학에 구조적으로 적응할 수 있게 한다.

- **(C4)** We design the SPK for phase-aligned and period-aligned long-horizon reconstruction, capturing periodic and seasonal structure without increasing model depth.
  > **(C4)** 위상 정렬 및 주기 정렬된 장기 재구성을 위한 SPK를 설계하여, 모델 깊이를 증가시키지 않고 주기 및 계절 구조를 포착한다.

- **(C5)** We show that FEATHer achieves competitive performance on long-horizon forecasting benchmarks while maintaining an ultra-compact footprint, supporting practical deployment in constrained industrial edge environments.
  > **(C5)** FEATHer가 초경량 풋프린트를 유지하면서도 장기 예측 벤치마크에서 경쟁력 있는 성능을 달성하여, 제약된 산업 에지 환경에서의 실용적 배포를 지원함을 보인다.

<!-- IoT-J revision: add (C6) sensor-fault robustness sweep, (C7)
     Cortex-M3 on-device measurement. Update paper-organization
     sentence below to mention Robustness (Sec 8) and Edge Deployment
     (Sec 9). -->

The remainder of this paper is organized as follows. Section 2 reviews related work on lightweight forecasting. Section 3 describes the FEATHer architecture and its components. Section 4 presents theoretical analyses of stability, expressiveness, and computational efficiency. Section 5 reports the experimental setup and results. Section 6 presents ablation studies and empirical validation of the theoretical analysis. Section 7 concludes with limitations and future directions.
> 본 논문의 나머지 부분은 다음과 같이 구성된다. 2절에서는 경량 예측에 대한 관련 연구를 검토한다. 3절에서는 FEATHer 아키텍처와 그 구성 요소를 설명한다. 4절에서는 안정성, 표현력, 계산 효율성에 대한 이론적 분석을 제시한다. 5절에서는 실험 설정과 결과를 보고한다. 6절에서는 절제 연구와 이론적 분석의 실증적 검증을 제시한다. 7절에서는 한계와 향후 방향으로 결론짓는다.
