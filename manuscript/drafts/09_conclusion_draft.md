# Conclusion

> **Source**: `tex_workspace/feather_raw.tex` lines 1985--2029.
> Original TPAMI conclusion is dense and reuses material from
> the introduction; R5, R6 #9, R8 all flagged it as too long.
> IoT-J revision should trim, add a "Limitations" paragraph
> (Solar / Traffic / Electricity losses, sub-1K only some
> configs, Cortex-M3-only deployment so far), and a "Future
> Work" line on broader hardware classes.

---

This paper proposes the Fourier-Efficient Adaptive Temporal Hierarchy Forecaster (FEATHer) to address the challenge of achieving both high accuracy and high efficiency in LTSF under the severe resource constraints typical of industrial edge devices, such as PLCs and embedded microcontrollers. FEATHer is deliberately designed to operate with an ultra-lightweight parameter budget while avoiding computationally intensive mechanisms such as recurrence and self-attention. The core of FEATHer is a structured multiscale temporal decomposition module that separates the input sequence into four frequency-structured pathways, namely point-scale, high-frequency, mid-frequency and low-frequency components, using simple temporal filtering operations. This design is theoretically shown to behave as a near-orthogonal filter bank that promotes signal disentanglement across frequency bands. Each resulting representation is then processed by a shared DTK, which efficiently mixes temporal information through linear projection, depthwise temporal convolution and inverse projection. The stability of this temporal mixing process is guaranteed by the Lipschitz continuity of the kernel.
> 본 논문은 PLC와 임베디드 마이크로컨트롤러와 같은 산업 에지 기기의 특징적인 심각한 자원 제약 하에서 LTSF에서 높은 정확도와 높은 효율성을 모두 달성하는 과제를 해결하기 위해 Fourier-Efficient Adaptive Temporal Hierarchy Forecaster(FEATHer)를 제안한다. FEATHer는 순환이나 자기 어텐션과 같은 계산 집약적 메커니즘을 회피하면서 초경량 파라미터 예산으로 동작하도록 의도적으로 설계되었다. FEATHer의 핵심은 간단한 시간 필터링 연산을 사용하여 입력 시퀀스를 점 스케일, 고주파, 중주파, 저주파 성분의 네 개의 주파수 구조화된 경로로 분리하는 구조화된 다중 스케일 시간 분해 모듈이다. 이 설계는 이론적으로 주파수 대역에 걸쳐 신호 분리를 촉진하는 근직교 필터 뱅크로 동작함이 입증된다. 그렇게 얻어진 각 표현은 그런 다음 선형 사영, 깊이별 시간 합성곱, 역 사영을 통해 효율적으로 시간 정보를 혼합하는 공유 DTK에 의해 처리된다. 이러한 시간 혼합 과정의 안정성은 커널의 Lipschitz 연속성에 의해 보장된다.

In addition, the frequency-aware branch gating module dynamically fuses multiscale representations by deriving gating signals from the spectral profile of the input sequence. This mechanism enables FEATHer to adaptively emphasize the most informative frequency bands and respond effectively to nonstationary temporal dynamics, resulting in band-wise adaptive projection. For long-horizon forecasting, the SPK efficiently reconstructs periodic and seasonal structure by transforming period-aligned blocks through a shared linear mapping. This design achieves the theoretical lower bound on parameter complexity, $nm$, required for period-aligned reconstruction.
> 또한 주파수 인식 분기 게이팅 모듈은 입력 시퀀스의 스펙트럼 프로파일로부터 게이팅 신호를 유도하여 다중 스케일 표현을 동적으로 융합한다. 이 메커니즘은 FEATHer가 가장 유익한 주파수 대역을 적응적으로 강조하고 비정상 시간 동역학에 효과적으로 응답할 수 있게 하여, 대역별 적응 사영을 결과로 산출한다. 장기 예측을 위해, SPK는 공유 선형 매핑을 통해 주기 정렬 블록을 변환함으로써 주기 및 계절 구조를 효율적으로 재구성한다. 이 설계는 주기 정렬 재구성에 필요한 파라미터 복잡도의 이론적 하한 $nm$을 달성한다.

<!-- R8 #1 tone-down anchor: replace "state-of-the-art ... five
     of eight" with the honest IoT-J result. Acknowledge the
     Solar / Traffic / Electricity losses up front. -->
<!-- R8 #2 tone-down anchor: "as few as 400 trainable parameters"
     -> "as few as 453 parameters on D=7 datasets, configuration-
     dependent ultra-light regime". -->

Empirically, FEATHer achieves state-of-the-art performance on benchmarks such as ETTh1 and ETTh2 with as few as 400 trainable parameters, records the best overall performance on five of eight long-term forecasting benchmarks and demonstrates markedly superior inference efficiency. For example, on the large Solar-Energy dataset with a forecasting horizon of 720, FEATHer requires only 1.40 ms for inference, substantially outperforming much larger models as well as other lightweight alternatives. Collectively, these results demonstrate that high forecasting accuracy can be attained under extremely tight parameter budgets, confirming that reliable long-range forecasting is feasible on highly constrained edge hardware. This work therefore suggests a practical direction for next-generation industrial systems that require real-time inference with minimal computational cost.
> 경험적으로, FEATHer는 400개 정도의 학습 가능 파라미터로 ETTh1과 ETTh2와 같은 벤치마크에서 최신 성능을 달성하고, 8개의 장기 예측 벤치마크 중 5개에서 전반적으로 가장 좋은 성능을 기록하며, 현저히 우수한 추론 효율성을 입증한다. 예를 들어, 예측 구간 720의 큰 Solar-Energy 데이터셋에서, FEATHer는 추론에 1.40 ms만 요구하여, 훨씬 큰 모델뿐만 아니라 다른 경량 대안들을 상당히 능가한다. 종합적으로 이러한 결과는 극도로 빠듯한 파라미터 예산 하에서도 높은 예측 정확도가 달성될 수 있음을 입증하며, 신뢰성 있는 장기 예측이 매우 제약된 에지 하드웨어에서 실현 가능함을 확인한다. 따라서 본 연구는 최소한의 계산 비용으로 실시간 추론을 요구하는 차세대 산업 시스템을 위한 실용적인 방향을 제시한다.

---

## Limitations (R5 / R6 #9 / R8 — new paragraph for IoT-J)

<!-- IoT-J revision adds an explicit Limitations paragraph the
     TPAMI version was missing. -->

> **To draft after Phase 4 numbers land**. Cover at least:
> 1. **Configuration-dependent parameter regime** — sub-1K only on
>    low-channel datasets (ETT-family, Exchange); Weather sits at
>    1.29-1.71 K, Solar at 23.79-24.21 K. The ultra-light claim
>    therefore applies to a regime, not the entire benchmark.
> 2. **High-channel accuracy gap** — larger-capacity baselines
>    (PatchTST, iTransformer, TQNet) outperform FEATHer on Solar /
>    Traffic / Electricity. We argue this trade is acceptable
>    *for the deployment scope* but flag it explicitly.
> 3. **Single-hardware deployment evidence** — Cortex-M3 only so
>    far; broader hardware classes (Cortex-M4F, RISC-V MCUs, FPGAs)
>    are future work.
> 4. **Theoretical scope** — Theorems 1-5 are tied to implemented
>    operators and do not establish optimality beyond the
>    phase-aligned linear class.

## Future Work (R5 / R6 #9)

> 1. Quantization-aware training to push the parameter / RAM
>    footprint further (R8 #7 explicitly asks about bit-width).
> 2. Energy-per-inference measurements on lower-tier MCUs (Cortex-M0+, RISC-V).
> 3. Online / streaming-mode evaluation under data drift, beyond
>    the offline LTSF protocol.
> 4. Integration with anomaly detection for closed-loop industrial monitoring.
