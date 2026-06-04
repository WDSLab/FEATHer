# Experiments

> **Source**: `tex_workspace/feather_raw.tex` lines 774--1528 (Sec 5).
> Body text retained; numeric tables are deferred to
> `tools/paper/main_table.py` output (Phase 5) — Tables 2/3/4/5 will
> be regenerated from `results/fcst_results.csv` after the 5-seed
> sweep completes. Reviewer-fix anchors as HTML comments.

---

## 5.1 Experimental Setting

We conducted extensive experiments on eight multivariate time-series datasets to systematically evaluate the performance of long-term forecasting models.
> 본 연구는 장기 예측 모델의 성능을 체계적으로 평가하기 위해 8개의 다변량 시계열 데이터셋에 대해 광범위한 실험을 수행했다.

**TABLE 1 — Summary of datasets** *(content as in the TPAMI submission; same row order: ETTh1, ETTh2, Airquality, SML, Weather, Solar-Energy, Traffic, Electricity)*. Channels range from 7 (ETT) to 321 (Electricity); sampling frequencies span 10-minute to hourly resolution; timesteps from 4,137 (SML) to 52,695 (Weather).

As summarized in Table 1, the datasets span a wide range of domains, including energy (ETTh1, ETTh2), meteorology (Weather), solar power generation (Solar-Energy), air quality (AirQuality), indoor environmental sensing (SML), traffic flow (Traffic), and household electricity consumption (Electricity). These datasets vary substantially in channel count, sampling frequency, and overall sequence length. All datasets were processed in strict chronological order, and the training, validation, and test splits were set to a ratio of 6:2:2. Each input variable was standardized using z-score normalization computed from the training set, with the same normalization parameters applied to the validation and test sets.
> Table 1에 요약된 바와 같이, 데이터셋은 에너지(ETTh1, ETTh2), 기상(Weather), 태양광 발전(Solar-Energy), 대기질(AirQuality), 실내 환경 감지(SML), 교통 흐름(Traffic), 가정용 전력 소비(Electricity)를 포함한 광범위한 영역을 다룬다. 이러한 데이터셋은 채널 수, 샘플링 주파수, 전체 시퀀스 길이에서 상당히 다양하다. 모든 데이터셋은 엄격한 시간 순서로 처리되었고, 학습/검증/테스트 분할은 6:2:2 비율로 설정되었다. 각 입력 변수는 학습 세트에서 계산된 z-score 정규화로 표준화되었으며, 동일한 정규화 파라미터가 검증 및 테스트 세트에 적용되었다.

<!-- R8 #3 reproducibility: list every per-(method, dataset) lr /
     d_model / e_layers etc. as a supplementary table; cite the
     public code repo. The values are in baselines/__init__.py:
     _METHOD_DEFAULTS + _DATASET_OVERRIDES. -->

For all experiments, the input sequence length was fixed at 96 time steps. Four forecasting horizons, namely 96, 192, 336, and 720 steps, were used to evaluate short-, medium-, and long-term predictive performance. All models were configured to generate the full forecasting horizon in a single forward pass without autoregressive decoding. The baselines include a broad set of state-of-the-art forecasting models spanning diverse architectural families, including Transformer-based models (Autoformer, PatchTST, iTransformer), attention-based models (TQNet), linear models (DLinear), LLM-empowered models (TimeCMA), frequency-inspired architectures (FITS), and sparsity-driven models (SparseTSF).
> 모든 실험에서 입력 시퀀스 길이는 96 시간 단계로 고정되었다. 네 가지 예측 구간(96, 192, 336, 720 단계)이 단기, 중기, 장기 예측 성능을 평가하는 데 사용되었다. 모든 모델은 자기회귀 디코딩 없이 단일 순방향 패스로 전체 예측 구간을 생성하도록 구성되었다. 기준 모델은 Transformer 기반 모델(Autoformer, PatchTST, iTransformer), 어텐션 기반 모델(TQNet), 선형 모델(DLinear), LLM 활용 모델(TimeCMA), 주파수 영감 아키텍처(FITS), 희소성 기반 모델(SparseTSF)을 포함한 다양한 아키텍처 군에 걸친 광범위한 최신 예측 모델을 포함한다.

<!-- R4 #2-3: IoT-J revision adds DiPE-Linear (DASFAA 2026),
     MDMLP-EIA (AAAI 2026), TimeMixer, LMS-AutoTSF to this list. -->

To ensure a fair and rigorous comparison, training configurations were standardized across all models. Key hyperparameters were selected through a comprehensive grid search on the validation set to promote stable convergence. We used the AdamW optimizer with an initial learning rate of $1 \times 10^{-2}$, cosine annealing scheduling, and a weight decay of $1 \times 10^{-4}$. The batch size was set to 32, and training was conducted for 30--50 epochs depending on dataset size. MSE was used as the training objective, while MAE and the correlation coefficient (COR) were used for evaluation. To ensure statistical reliability, each model was trained and evaluated 30 times under identical experimental settings, varying only the random seed for model initialization and data shuffling. The final reported results correspond to the average performance across the 30 independent runs, reducing variance arising from stochastic training dynamics. All experiments were conducted in a unified computational environment with consistent settings applied to all models to ensure fairness and reproducibility.
> 공정하고 엄밀한 비교를 보장하기 위해, 학습 설정은 모든 모델에 걸쳐 표준화되었다. 핵심 하이퍼파라미터는 안정적 수렴을 촉진하기 위해 검증 세트에서의 포괄적인 그리드 탐색을 통해 선택되었다. 우리는 초기 학습률 $1 \times 10^{-2}$, 코사인 어닐링 스케줄링, 가중치 감쇠 $1 \times 10^{-4}$의 AdamW 최적화기를 사용했다. 배치 크기는 32로 설정되었고, 학습은 데이터셋 크기에 따라 30-50 에포크 동안 수행되었다. MSE는 학습 목표로 사용되었으며, MAE와 상관 계수(COR)는 평가에 사용되었다. 통계적 신뢰성을 보장하기 위해, 각 모델은 동일한 실험 설정 하에서 30회 학습 및 평가되었으며, 모델 초기화와 데이터 셔플링에 대한 무작위 시드만 변경되었다. 최종 보고된 결과는 30회의 독립 실행에 걸친 평균 성능에 해당하며, 확률적 학습 동역학에서 발생하는 분산을 줄인다. 모든 실험은 모든 모델에 일관된 설정이 적용된 통일된 계산 환경에서 수행되어 공정성과 재현성을 보장한다.

<!-- R8 #3 audit: TPAMI says "30 repetitions" — IoT-J version
     should align with the code-side 5-seed protocol (2025-2029).
     Either keep 30 (and document seeds) or switch to 5 with
     Wilcoxon. Phase 4 sweep produces 5-seed numbers; default
     plan is 5 seeds + std + Wilcoxon. -->

## 5.2 Experimental Results

**Main Forecasting Performance.** The quantitative forecasting results on the eight multivariate datasets are summarized in Table 2 and Table 3.
> **주요 예측 성능.** 8개의 다변량 데이터셋에 대한 정량적 예측 결과는 Table 2와 Table 3에 요약되어 있다.

**TABLE 2 — Forecasting Performance Results on ETTh1, ETTh2, Air quality, and SML.** *(Placeholder; populate from `results/fcst_results.csv` via `tools/paper/main_table.py --metric MSE --format latex`)*

**TABLE 3 — Forecasting Performance Results on Weather, Solar-Energy, Traffic, and Electricity.** *(Same — generated by Phase 5 tool.)*

<!-- R8 #1 tone-down: re-read these paragraphs after Phase 4
     numbers land. If FEATHer loses on Solar / Traffic /
     Electricity, say so up front rather than burying it. -->

The proposed model demonstrates strong overall performance, achieving the lowest MSE and MAE on most datasets and forecasting horizons. In particular, it consistently outperforms strong Transformer-based baselines such as PatchTST and iTransformer, as well as the recently proposed LLM-empowered model TimeCMA. For example, on the Weather dataset with a horizon of 96, the proposed model records an MSE of 0.216, representing a clear error reduction compared with PatchTST (0.239) and iTransformer (0.240). Notably, although TimeCMA leverages semantic knowledge from pre-trained Large Language Models (LLMs) through cross-modality alignment to enhance robustness, the proposed model achieves superior forecasting accuracy. This result indicates that, for time-series forecasting, the proposed domain-specific design — particularly the FFT-based Frequency-Adaptive Gating mechanism — is more effective at capturing intrinsic temporal dynamics than aligning time-series representations with textual modalities, which may introduce domain mismatch or representation entanglement.
> 제안된 모델은 강력한 전반적 성능을 보이며, 대부분의 데이터셋과 예측 구간에서 가장 낮은 MSE와 MAE를 달성한다. 특히 PatchTST와 iTransformer와 같은 강력한 Transformer 기반 기준 모델은 물론, 최근 제안된 LLM 활용 모델 TimeCMA를 일관되게 능가한다. 예를 들어, 구간 96의 Weather 데이터셋에서 제안 모델은 0.216의 MSE를 기록하며, 이는 PatchTST (0.239)와 iTransformer (0.240)와 비교하여 명확한 오차 감소를 나타낸다. 특히 TimeCMA가 견고성을 향상시키기 위해 교차 모달리티 정렬을 통해 사전 학습된 대규모 언어 모델(LLM)의 의미론적 지식을 활용하지만, 제안된 모델은 우수한 예측 정확도를 달성한다. 이 결과는 시계열 예측에 있어 제안된 도메인 특화 설계(특히 FFT 기반 주파수 적응형 게이팅 메커니즘)가 도메인 불일치 또는 표현 얽힘을 도입할 수 있는 시계열 표현과 텍스트 모달리티의 정렬보다 본질적인 시간 동역학을 포착하는 데 더 효과적임을 나타낸다.

The proposed model also exhibits strong robustness in long-horizon forecasting scenarios. As the forecasting horizon extends to 720 steps, many Transformer-based models experience notable performance degradation, often attributed to the dispersion of attention weights. In contrast, the proposed model maintains stable performance with limited error accumulation, highlighting its suitability for long-range forecasting under constrained settings.
> 제안된 모델은 또한 장기 예측 시나리오에서 강한 견고성을 보인다. 예측 구간이 720 단계까지 확장됨에 따라, 많은 Transformer 기반 모델은 종종 어텐션 가중치의 분산에 기인하는 두드러진 성능 저하를 겪는다. 대조적으로, 제안된 모델은 제한된 오차 누적과 함께 안정적인 성능을 유지하여, 제약된 환경에서의 장기 예측에 대한 적합성을 강조한다.

**TABLE 4 — Overall Performance Rankings Across All Datasets and Horizons.** *(Phase 5 tool: aggregate first-place / second-place counts and average rank per model.)*

This robustness is attributed to the SPK, which aggregates information according to inherent periodicity rather than relying on simple point-wise mappings, thereby preserving long-range dependencies more effectively. In the comprehensive ranking analysis reported in Table 4, the proposed model ranks first in 60 experimental settings and achieves an average rank of 2.05. This result substantially outperforms the second-tier group, including PatchTST with an average rank of 3.71 and iTransformer with 3.93, demonstrating that the proposed method provides strong generalization across diverse domains such as energy, traffic, weather, and electricity.
> 이러한 견고성은 단순한 점별 매핑에 의존하기보다는 본질적 주기성에 따라 정보를 집계하는 SPK에 기인하며, 따라서 장기 의존성을 더 효과적으로 보존한다. Table 4에 보고된 포괄적 순위 분석에서, 제안된 모델은 60개의 실험 설정에서 1위를 차지하고 평균 순위 2.05를 달성한다. 이 결과는 평균 순위 3.71의 PatchTST와 3.93의 iTransformer를 포함한 두 번째 그룹을 상당히 능가하며, 제안된 방법이 에너지, 교통, 기상, 전기와 같은 다양한 영역에 걸쳐 강력한 일반화를 제공함을 입증한다.

<!-- R8 #1 audit: 60 wins / avg rank 2.05 was the TPAMI claim
     against 8 baselines. After IoT-J expansion to 11 baselines
     and 5-seed averaging, the win count will shift. Update with
     Phase 4 numbers before submitting. -->

**Model Efficiency and Computational Cost.** Table 5 presents a detailed comparison of model complexity and inference efficiency, highlighting the structural advantages of the proposed method. Existing state-of-the-art models face a pronounced trade-off between accuracy and efficiency. Transformer-based architectures such as PatchTST and iTransformer incur quadratic computational complexity $\mathcal{O}(L^2)$, leading to high memory consumption and latency, particularly on large-scale datasets such as Solar-Energy. Similarly, although TimeCMA seeks to reduce the computational burden of large language models by freezing weights and storing embeddings, the underlying LLM backbone still introduces substantial parameter overhead. Even comparatively efficient attention-based models such as TQNet, which optimize the query mechanism, require matrix operations that scale with sequence length.
> **모델 효율성과 계산 비용.** Table 5는 모델 복잡도와 추론 효율성의 상세한 비교를 제시하며, 제안된 방법의 구조적 이점을 강조한다. 기존 최신 모델은 정확도와 효율성 사이의 두드러진 절충에 직면한다. PatchTST와 iTransformer와 같은 Transformer 기반 아키텍처는 이차 계산 복잡도 $\mathcal{O}(L^2)$를 초래하여, 특히 Solar-Energy와 같은 대규모 데이터셋에서 높은 메모리 소비와 지연 시간을 유발한다. 마찬가지로, TimeCMA가 가중치를 고정하고 임베딩을 저장하여 대규모 언어 모델의 계산 부담을 줄이고자 하지만, 기저 LLM 백본은 여전히 상당한 파라미터 오버헤드를 도입한다. 쿼리 메커니즘을 최적화하는 TQNet과 같이 비교적 효율적인 어텐션 기반 모델조차 시퀀스 길이에 따라 스케일되는 행렬 연산을 요구한다.

In contrast, the proposed model adopts a DTK based on depthwise convolution, which enables temporal correlation modeling with linear complexity $\mathcal{O}(L)$. This structural design allows the model to achieve state-of-the-art accuracy while using substantially fewer parameters and MACs (Multiply-Accumulate Operations). As shown in Table 5, on the Solar-Energy dataset with a forecasting horizon of 720, the proposed model requires only 1.40 ms for inference. This is markedly faster than PatchTST at 44.7 ms and also outperforms lightweight frequency-domain models such as FITS at 2.79 ms. These results indicate that the proposed model effectively resolves the accuracy-efficiency trade-off, making it well-suited for real-time forecasting applications under limited computational resources.
> 대조적으로, 제안된 모델은 깊이별 합성곱에 기반한 DTK를 채택하여, 선형 복잡도 $\mathcal{O}(L)$로 시간 상관 모델링을 가능하게 한다. 이 구조적 설계는 모델이 상당히 적은 파라미터와 MAC(곱셈-누산 연산)을 사용하면서도 최신 정확도를 달성할 수 있게 한다. Table 5에 나타난 바와 같이, 예측 구간 720의 Solar-Energy 데이터셋에서 제안된 모델은 추론에 1.40 ms만 필요로 한다. 이는 PatchTST의 44.7 ms보다 현저히 빠르며 FITS의 2.79 ms와 같은 경량 주파수 영역 모델도 능가한다. 이러한 결과는 제안된 모델이 정확도-효율성 절충을 효과적으로 해결하여, 제한된 계산 자원 하에서의 실시간 예측 응용에 적합함을 나타낸다.

## 5.3 On-device Deployment Results

> **R8 #7 anchor**: this section was the weakest part of the TPAMI
> submission ("O/X masks bit-width, peak RAM, energy, etc."). IoT-J
> reframing must add: per-model bit-width / peak RAM table, flash
> usage, activation arena size, energy in mJ per inference,
> arm-none-eabi-gcc flags, library version, and number of repetitions.

**TABLE 5 — Model Complexity and Inference Efficiency** *(parameters / MACs / inference time per dataset and horizon)*. **TABLE 6 — On-device deployability (O/X) under 16 KB / 32 KB / 64 KB RAM budgets on ETTh1 and Weather.** *(Placeholders for the existing TPAMI tables; IoT-J version adds bit-width and energy columns.)*

We evaluate the on-device deployability of FEATHer on a physical Cortex-M3-class embedded platform under strict memory constraints representative of extreme sensor-class industrial hardware. Specifically, we execute the inference firmware on the LM3S6965EVB (Stellaris) target (ARM Cortex-M3) compiled with arm-none-eabi-gcc. All deployability outcomes are obtained directly on the real board in order to reflect practical embedded execution constraints.
> 본 연구는 극단적인 센서급 산업 하드웨어를 대표하는 엄격한 메모리 제약 하에서 물리적 Cortex-M3급 임베디드 플랫폼에서 FEATHer의 온디바이스 배포 가능성을 평가한다. 구체적으로, 우리는 arm-none-eabi-gcc로 컴파일된 LM3S6965EVB(Stellaris) 타겟(ARM Cortex-M3)에서 추론 펌웨어를 실행한다. 모든 배포 가능성 결과는 실용적 임베디드 실행 제약을 반영하기 위해 실제 보드에서 직접 얻어진다.

Although the target board provides 64 KB RAM, real deployments must reserve memory for the firmware stack, interrupt handling, and I/O buffers. To capture this reality and to test robustness across progressively constrained environments, we consider three effective RAM budgets: 16 KB, 32 KB, and 64 KB. Each budget is enforced by restricting the memory region available to the inference runtime (e.g., limiting the heap/activation arena via the linker script and compile-time configuration), ensuring that the model and its intermediate buffers must fit within the specified budget during execution. We perform inference with batch size = 1, which matches typical streaming edge usage. A model is marked as deployable (O) if it completes inference successfully within the given RAM budget; otherwise, it is marked as non-deployable (X) due to out-of-memory or runtime failure. Importantly, this criterion goes beyond parameter counts: in embedded execution, feasibility is often dominated by the peak runtime memory footprint, including intermediate activations and temporary buffers.
> 타겟 보드가 64 KB RAM을 제공하지만, 실제 배포는 펌웨어 스택, 인터럽트 처리, I/O 버퍼를 위한 메모리를 예약해야 한다. 이러한 현실을 포착하고 점진적으로 제약된 환경에서의 견고성을 테스트하기 위해, 우리는 세 가지 유효 RAM 예산을 고려한다: 16 KB, 32 KB, 64 KB. 각 예산은 추론 런타임에서 사용 가능한 메모리 영역을 제한함으로써(예: 링커 스크립트와 컴파일 타임 구성을 통해 힙/활성화 아레나 제한) 강제되며, 모델과 그 중간 버퍼가 실행 중 지정된 예산 내에 맞아야 함을 보장한다. 우리는 전형적인 스트리밍 에지 사용과 일치하는 배치 크기 1로 추론을 수행한다. 모델이 주어진 RAM 예산 내에서 추론을 성공적으로 완료하면 배포 가능(O)으로 표시되고, 그렇지 않으면 메모리 부족 또는 런타임 실패로 인해 배포 불가능(X)으로 표시된다. 중요하게도, 이 기준은 파라미터 수를 넘어선다: 임베디드 실행에서 실현 가능성은 종종 중간 활성화와 임시 버퍼를 포함한 최대 런타임 메모리 풋프린트에 의해 지배된다.

We report deployability on two long-term forecasting datasets with different width settings: ETTh1 (Small) and Weather (Middle). Table 6 summarizes the on-device feasibility (O/X) under 16 KB / 32 KB / 64 KB for each baseline and FEATHer. The results reveal a clear separation between truly deployable ultra-compact models and conventional deep forecasting architectures. On ETTh1, FEATHer remains deployable even under the most stringent 16 KB budget, while most baselines fail. As the budget increases to 32 KB and 64 KB, a limited subset of lightweight baselines becomes feasible; however, larger transformer-family models and high-capacity methods remain non-deployable due to substantial activation and buffering demands. These observations indicate that even when parameter counts are moderate, runtime memory can still prevent embedded execution.
> 우리는 서로 다른 폭 설정을 갖는 두 개의 장기 예측 데이터셋, ETTh1(Small)과 Weather(Middle)에 대한 배포 가능성을 보고한다. Table 6은 각 기준 모델과 FEATHer에 대한 16 KB / 32 KB / 64 KB 하에서의 온디바이스 실현 가능성(O/X)을 요약한다. 결과는 진정으로 배포 가능한 초경량 모델과 기존의 심층 예측 아키텍처 간의 명확한 분리를 드러낸다. ETTh1에서 FEATHer는 가장 엄격한 16 KB 예산 하에서도 배포 가능한 상태를 유지하는 반면, 대부분의 기준 모델은 실패한다. 예산이 32 KB와 64 KB로 증가함에 따라 경량 기준 모델의 제한된 부분 집합이 실현 가능해지지만, 더 큰 transformer 계열 모델과 고용량 방법은 상당한 활성화와 버퍼링 요구로 인해 배포 불가능한 상태로 남는다. 이러한 관찰은 파라미터 수가 적당하더라도 런타임 메모리가 여전히 임베디드 실행을 방해할 수 있음을 나타낸다.

The Weather setting is substantially more challenging. While some lightweight methods may fit ETTh1 at 32 KB or 64 KB, most fail on Weather under 16 KB and 32 KB, and feasibility remains limited even at 64 KB. In particular, the results show that deployability does not necessarily transfer across datasets: increasing the model width and intermediate buffering requirements can push otherwise compact models beyond the memory budget. In contrast, FEATHer is the only method that remains deployable on Weather within the 64 KB budget, demonstrating robust feasibility under a more demanding dataset configuration.
> Weather 설정은 상당히 더 어렵다. 일부 경량 방법이 32 KB나 64 KB에서 ETTh1에 맞을 수 있지만, 대부분이 16 KB와 32 KB 하에서 Weather에 실패하고, 실현 가능성은 64 KB에서도 제한된 상태로 남는다. 특히 결과는 배포 가능성이 데이터셋에 걸쳐 반드시 전이되지 않음을 보여준다: 모델 폭과 중간 버퍼링 요구 사항의 증가는 그렇지 않으면 컴팩트한 모델을 메모리 예산을 초과하게 할 수 있다. 대조적으로, FEATHer는 64 KB 예산 내에서 Weather에 배포 가능한 상태를 유지하는 유일한 방법이며, 더 까다로운 데이터셋 구성 하에서 견고한 실현 가능성을 입증한다.

Overall, these findings validate that FEATHer maintains an exceptionally small footprint not only in parameter count but also in end-to-end runtime memory, enabling reliable on-device execution across progressively constrained RAM budgets. This supports FEATHer as a practical forecaster for next-generation industrial edge systems where real-time inference must be performed under extreme memory limitations.
> 전반적으로 이러한 결과는 FEATHer가 파라미터 수뿐만 아니라 종단 간 런타임 메모리에서도 예외적으로 작은 풋프린트를 유지하여, 점진적으로 제약된 RAM 예산에 걸쳐 신뢰성 있는 온디바이스 실행을 가능하게 함을 검증한다. 이는 FEATHer가 실시간 추론이 극단적인 메모리 제한 하에서 수행되어야 하는 차세대 산업 에지 시스템을 위한 실용적인 예측기임을 뒷받침한다.

<!-- IoT-J Sec 8 (new): Robustness on Sensor Faults — 19,200 row
     sweep generated by tools/paper/robust_summary.py. R6 #7a
     direct response. To be drafted as a new section after
     Phase 4b results land. -->
