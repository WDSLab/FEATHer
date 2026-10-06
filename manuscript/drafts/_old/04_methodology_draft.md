# Methodology — Fourier-Efficient Adaptive Temporal Hierarchy Forecaster

> **Source**: `tex_workspace/feather_raw.tex` lines 339--531
> (Sec 3 Method). Equations kept as inline LaTeX ($...$). Algorithm 1
> moved to `05_algorithms_word.md`. Reviewer-fix anchors as HTML comments.

---

This section presents the overall architecture of the Fourier-Efficient Adaptive Temporal Hierarchy Forecaster (FEATHer). FEATHer is designed to support stable long-horizon forecasting under stringent latency, memory, and energy constraints typical of industrial edge devices. The model integrates four core components: **(i)** a *structured multiscale temporal decomposition* that produces complementary scale representations, **(ii)** a shared *DTK* that performs efficient temporal mixing through linear projections and depthwise temporal filtering, **(iii)** a *frequency-aware branch gating* module that adaptively fuses scale representations based on the input spectrum and **(iv)** a *SPK* that reconstructs long-horizon periodic and seasonal structure through compact period-aligned transformations. FEATHer avoids computationally intensive recurrence and self-attention, relying instead on lightweight linear operators, depthwise filtering, and sparse period-wise mappings, making it suitable for deployment on resource-constrained hardware. Fig. 1 illustrates the overall architecture of FEATHer.
> 본 절에서는 Fourier-Efficient Adaptive Temporal Hierarchy Forecaster(FEATHer)의 전체 아키텍처를 제시한다. FEATHer는 산업용 에지 기기의 특징적인 엄격한 지연 시간, 메모리, 에너지 제약 하에서 안정적인 장기 예측을 지원하도록 설계되었다. 본 모델은 네 가지 핵심 구성 요소를 통합한다: **(i)** 보완적 스케일 표현을 생성하는 *구조화된 다중 스케일 시간 분해*, **(ii)** 선형 사영과 깊이별 시간 필터링을 통해 효율적인 시간 혼합을 수행하는 공유 *DTK*, **(iii)** 입력 스펙트럼에 기반하여 스케일 표현을 적응적으로 융합하는 *주파수 인식 분기 게이팅* 모듈, **(iv)** 컴팩트한 주기 정렬 변환을 통해 장기 주기 및 계절 구조를 재구성하는 *SPK*. FEATHer는 계산 집약적인 순환과 자기 어텐션을 피하고, 대신 경량 선형 연산자, 깊이별 필터링, 희소 주기 단위 매핑에 의존하여 자원이 제약된 하드웨어에 배포하기에 적합하다. Fig. 1은 FEATHer의 전체 아키텍처를 보여준다.

---

## 3.1 Problem Setup and Notation

We consider multivariate forecasting with an input window of length $L$, a forecasting horizon $H$, and $D$ variables. The input sequence is denoted by $X = [x_1, \ldots, x_L]^T \in \mathbb{R}^{L \times D}$, and the target horizon is $Y = [y_1, \ldots, y_H]^T \in \mathbb{R}^{H \times D}$. FEATHer produces $\widehat{Y} \in \mathbb{R}^{H \times D}$ from $X$. The model constructs $B \in \{2, 3, 4\}$ scale pathways; the active branch set is denoted by $\mathcal{B} \subseteq \{p, h, m, l\}$, corresponding to point, high, mid, and low frequency pathways, where $|\mathcal{B}| = B$. The DTK uses a latent width $S$, and the SPK uses a period $P$. All branch representations are aligned to the same temporal length $L$ so that they can be fused without cross-scale alignment overhead. In this work, scale refers to frequency-aligned temporal representations obtained via lightweight temporal filtering.
> 우리는 입력 윈도우 길이 $L$, 예측 구간 $H$, 변수 수 $D$를 갖는 다변량 예측을 고려한다. 입력 시퀀스는 $X = [x_1, \ldots, x_L]^T \in \mathbb{R}^{L \times D}$로 표기하고, 목표 구간은 $Y = [y_1, \ldots, y_H]^T \in \mathbb{R}^{H \times D}$로 표기한다. FEATHer는 $X$로부터 $\widehat{Y} \in \mathbb{R}^{H \times D}$를 생성한다. 본 모델은 $B \in \{2, 3, 4\}$개의 스케일 경로를 구성하며, 활성 분기 집합은 점, 고주파, 중주파, 저주파 경로에 해당하는 $\mathcal{B} \subseteq \{p, h, m, l\}$로 표기하고 $|\mathcal{B}| = B$이다. DTK는 잠재 폭 $S$를 사용하고, SPK는 주기 $P$를 사용한다. 모든 분기 표현은 동일한 시간 길이 $L$로 정렬되어 교차 스케일 정렬 오버헤드 없이 융합될 수 있다. 본 연구에서 스케일은 경량 시간 필터링을 통해 얻어진 주파수 정렬 시간 표현을 의미한다.

## 3.2 Structured Multiscale Temporal Decomposition

FEATHer begins by transforming $X$ into multiple scale representations that emphasize different temporal behaviors while remaining computationally inexpensive (Fig. 2).
> FEATHer는 $X$를 계산적으로 저렴하게 유지하면서도 서로 다른 시간적 거동을 강조하는 여러 스케일 표현으로 변환하는 것으로 시작한다 (Fig. 2).

<!-- R6 #1a: clarify the rationale for B ∈ {2,3,4} during IoT-J revision. -->

For each active branch $b \in \mathcal{B}$, we generate $X^{(b)} = \phi^{(b)}(X) \in \mathbb{R}^{L \times D}$, where $\phi^{(b)}(\cdot)$ is implemented using lightweight depthwise temporal operations. The point branch preserves instantaneous information through a kernel-size-1 depthwise operator, while the high and mid branches use short-support depthwise convolutions, for example, kernel sizes 3 and 5, to emphasize local and medium-range variations, respectively. The low branch isolates slow components by temporally downsampling the input using average pooling with stride $r$ and then interpolating back to length $L$, yielding an explicit low-pass bias without introducing heavy parameters. This decomposition provides complementary pathways that reduce cross-frequency interference under compact model budgets while keeping all representations time-aligned for subsequent fusion.

**Fig. 2.** Structure of the multiscale temporal decomposition. The instance-normalized input is fed into four parallel depthwise operators: a kernel-1 depthwise convolution (point), kernel-3 (high), kernel-5 (mid), and an average-pool / upsample pair (low). All four pathways are returned at the original length $L$ so they can be fused in the time-aligned representation $H \in \mathbb{R}^{L \times D}$.
> **Fig. 2.** 다중 스케일 시간 분해 구조. 인스턴스 정규화된 입력은 네 개의 병렬 깊이별 연산자에 공급된다: 커널 1의 깊이별 합성곱(점), 커널 3(고주파), 커널 5(중주파), 그리고 평균 풀링·업샘플 쌍(저주파). 네 경로 모두 원래 길이 $L$로 반환되어, 시간 정렬된 표현 $H \in \mathbb{R}^{L \times D}$에서 융합된다.
> 각 활성 분기 $b \in \mathcal{B}$에 대해, 우리는 $X^{(b)} = \phi^{(b)}(X) \in \mathbb{R}^{L \times D}$를 생성하며, 여기서 $\phi^{(b)}(\cdot)$는 경량 깊이별 시간 연산을 사용하여 구현된다. 점 분기는 커널 크기 1의 깊이별 연산자를 통해 순간 정보를 보존하고, 고주파 및 중주파 분기는 각각 국소 및 중간 범위 변동을 강조하기 위해 짧은 지원의 깊이별 합성곱(예: 커널 크기 3과 5)을 사용한다. 저주파 분기는 보폭 $r$의 평균 풀링으로 입력을 시간적으로 다운샘플링한 후 길이 $L$로 보간하여 느린 성분을 분리함으로써, 무거운 파라미터를 도입하지 않고도 명시적인 저역 통과 편향을 제공한다. 이러한 분해는 컴팩트 모델 예산 하에서 교차 주파수 간섭을 줄이는 보완적 경로를 제공하며, 모든 표현을 시간 정렬된 상태로 유지하여 후속 융합에 사용된다.

<!-- R5 #5: explain why low-freq uses pooling (linear) while high-freq
     uses conv (learned). Add justification paragraph here. -->

## 3.3 Dense Temporal Kernel (DTK)

Each decomposed pathway is processed by a shared DTK that performs efficient temporal mixing without recurrence or self-attention (Fig. 3). DTK adopts a projection--depthwise filtering--inverse projection structure. For a branch input $X^{(b)} \in \mathbb{R}^{L \times D}$, DTK first projects the input to a compact latent width $S$:
> 각 분해된 경로는 순환이나 자기 어텐션 없이 효율적인 시간 혼합을 수행하는 공유 DTK에 의해 처리된다 (Fig. 3). DTK는 사영-깊이별 필터링-역 사영 구조를 채택한다. 분기 입력 $X^{(b)} \in \mathbb{R}^{L \times D}$에 대해, DTK는 먼저 입력을 컴팩트한 잠재 폭 $S$로 사영한다:

$$Z^{(b)} = X^{(b)} W_{\text{in}} \in \mathbb{R}^{L \times S} \quad (1)$$

where $W_{\text{in}} \in \mathbb{R}^{D \times S}$. A depthwise temporal convolution then mixes information along the temporal dimension independently for each latent channel, where $k_t$ denotes the depthwise temporal convolution kernel:
> 여기서 $W_{\text{in}} \in \mathbb{R}^{D \times S}$이다. 깊이별 시간 합성곱은 각 잠재 채널에 대해 독립적으로 시간 차원을 따라 정보를 혼합하며, $k_t$는 깊이별 시간 합성곱 커널을 나타낸다:

$$U^{(b)} = \text{DWConv}_t(Z^{(b)}; k_t) \in \mathbb{R}^{L \times S} \quad (2)$$

$$H^{(b)} = U^{(b)} W_{\text{out}} \in \mathbb{R}^{L \times D} \quad (3)$$

The same DTK parameters $\{W_{\text{in}}, W_{\text{out}}, \text{DWConv weights}\}$ are shared across all branches $b \in \mathcal{B}$. This parameter sharing prevents growth in model size as the number of branches increases and keeps the architecture ultra-lightweight, while still allowing each branch to express scale-specific temporal dynamics through its distinct input signal $X^{(b)}$.
> 동일한 DTK 파라미터 $\{W_{\text{in}}, W_{\text{out}}, \text{DWConv 가중치}\}$가 모든 분기 $b \in \mathcal{B}$에 걸쳐 공유된다. 이러한 파라미터 공유는 분기 수가 증가해도 모델 크기 증가를 방지하고 아키텍처를 초경량으로 유지하면서도, 각 분기가 서로 다른 입력 신호 $X^{(b)}$를 통해 스케일별 시간 동역학을 표현할 수 있게 한다.

**Fig. 3.** Structure of the Dense Temporal Kernel (DTK). The branch input $X^{(b)} \in \mathbb{R}^{L \times D}$ is first projected channel-wise by $W_{\text{in}} \in \mathbb{R}^{D \times S}$ into a compact latent state $Z^{(b)} \in \mathbb{R}^{L \times S}$. A depthwise temporal convolution then mixes information along time independently per latent channel using the shared kernel $k_{\text{temp}}$. Finally, the latent representation is projected back to the model dimension by $W_{\text{out}} \in \mathbb{R}^{S \times D}$. All three weights are shared across the active branches $b \in \mathcal{B}$, so the parameter count does not grow with the number of pathways.
> **Fig. 3.** Dense Temporal Kernel(DTK) 구조. 분기 입력 $X^{(b)} \in \mathbb{R}^{L \times D}$는 먼저 $W_{\text{in}} \in \mathbb{R}^{D \times S}$에 의해 채널별로 컴팩트한 잠재 상태 $Z^{(b)} \in \mathbb{R}^{L \times S}$로 사영된다. 그런 다음 깊이별 시간 합성곱이 공유 커널 $k_{\text{temp}}$를 사용하여 잠재 채널별로 독립적으로 시간 방향 정보를 혼합한다. 마지막으로 잠재 표현은 $W_{\text{out}} \in \mathbb{R}^{S \times D}$에 의해 모델 차원으로 다시 사영된다. 세 가중치 모두 활성 분기 $b \in \mathcal{B}$에 걸쳐 공유되므로, 파라미터 수는 경로 수에 따라 증가하지 않는다.

## 3.4 Frequency-aware Branch Gating

The importance of each temporal scale varies across instances and operating regimes. FEATHer therefore employs a frequency-aware gating mechanism that assigns branch weights based on the input spectral profile, enabling instance-adaptive fusion while remaining lightweight (Fig. 4). We first normalize the input sequence along the temporal dimension for each variable. We then compute a real FFT along the temporal axis,
> 각 시간 스케일의 중요도는 인스턴스와 운영 영역에 걸쳐 달라진다. 따라서 FEATHer는 입력 스펙트럼 프로파일에 기반하여 분기 가중치를 할당하는 주파수 인식 게이팅 메커니즘을 사용하여, 경량을 유지하면서도 인스턴스 적응형 융합을 가능하게 한다 (Fig. 4). 우리는 먼저 각 변수에 대해 시간 차원을 따라 입력 시퀀스를 정규화한다. 그런 다음 시간 축을 따라 실수 FFT를 계산한다:

$$F = \mathcal{F}(\overline{X}), \quad (4)$$

and take the magnitude spectrum
> 그리고 진폭 스펙트럼을 취한다:

$$A = |F| \quad (5)$$

A compact spectral descriptor is obtained by averaging across variables,
> 변수에 걸쳐 평균을 취하여 컴팩트한 스펙트럼 기술자를 얻는다:

$$a = \frac{1}{D} \sum_{d=1}^{D} A_{:,d} \in \mathbb{R}^{L_f} \quad (6)$$

where $L_f = \lfloor L/2 \rfloor + 1$. A lightweight gating network $\psi(\cdot)$ maps $a$ to branch logits,
> 여기서 $L_f = \lfloor L/2 \rfloor + 1$이다. 경량 게이팅 네트워크 $\psi(\cdot)$는 $a$를 분기 로짓으로 매핑한다:

<!-- R6 #1d: Eq (6) is in R^{L_f}, Eq (7) is in R^B. The dimension
     transition from L_f to B happens inside ψ(·). Clarify that ψ is
     "Conv1D → global pool → linear to B" so the reader can trace
     the dimensions, not just see logit output. -->

$$u = \psi(a) \in \mathbb{R}^B \quad (7)$$

and a softmax operation yields branch weights,
> 소프트맥스 연산으로 분기 가중치를 얻는다:

$$g = \text{softmax}(u) \in \mathbb{R}^B \quad (8)$$

The fused representation is formed by weighting the DTK outputs:
> 융합된 표현은 DTK 출력에 가중치를 곱하여 형성된다:

$$H = \sum_{b \in \mathcal{B}} g_b H^{(b)} \in \mathbb{R}^{L \times D} \quad (9)$$

In practice, $\psi(\cdot)$ can be implemented as a small Conv1D stack operating on $a$. Because the descriptor length is only $L_f$, this module introduces negligible overhead while allowing the model to emphasize the most informative scales for each input.
> 실제로 $\psi(\cdot)$는 $a$에 대해 동작하는 작은 Conv1D 스택으로 구현될 수 있다. 기술자 길이가 $L_f$에 불과하기 때문에, 이 모듈은 무시할 수 있는 오버헤드만 도입하면서도 모델이 각 입력에 대해 가장 유익한 스케일을 강조할 수 있게 한다.

**Fig. 4.** Structure of the frequency-aware branch gating module. The instance-normalized input $\widetilde{X}$ is transformed by a real FFT along the temporal axis; the magnitude spectrum is averaged across channels to obtain the compact spectral descriptor $a \in \mathbb{R}^{L_f}$ with $L_f = \lfloor L/2 \rfloor + 1$. A lightweight Conv1D stack $\psi$ then maps $a$ to branch logits $u \in \mathbb{R}^B$, which a softmax converts to non-negative weights $g \in \Delta^B$. The DTK outputs of the active branches are fused as $H = \sum_{b \in \mathcal{B}} g_b H^{(b)}$.
> **Fig. 4.** 주파수 인식 분기 게이팅 모듈 구조. 인스턴스 정규화된 입력 $\widetilde{X}$는 시간 축을 따라 실수 FFT로 변환되고, 진폭 스펙트럼은 채널에 걸쳐 평균되어 $L_f = \lfloor L/2 \rfloor + 1$인 컴팩트한 스펙트럼 기술자 $a \in \mathbb{R}^{L_f}$를 얻는다. 경량 Conv1D 스택 $\psi$가 $a$를 분기 로짓 $u \in \mathbb{R}^B$로 매핑하고, 소프트맥스가 이를 비음수 가중치 $g \in \Delta^B$로 변환한다. 활성 분기들의 DTK 출력은 $H = \sum_{b \in \mathcal{B}} g_b H^{(b)}$로 융합된다.

## 3.5 Sparse Period Kernel (SPK) for Long-Horizon Reconstruction

To produce $\widehat{Y}$ efficiently, FEATHer employs a SPK that reconstructs periodic and seasonal structure through compact period-aligned transformations (Fig. 5). Starting from $H \in \mathbb{R}^{L \times D}$, SPK first applies a depthwise temporal aggregation:
> $\widehat{Y}$를 효율적으로 생성하기 위해, FEATHer는 컴팩트한 주기 정렬 변환을 통해 주기 및 계절 구조를 재구성하는 SPK를 사용한다 (Fig. 5). $H \in \mathbb{R}^{L \times D}$로부터 시작하여, SPK는 먼저 깊이별 시간 집계를 적용한다:

$$H'(t) = (H * w)(t) \quad (10)$$

where $w$ is a learnable 1D filter with padding that preserves the temporal length $L$. SPK then reorganizes $H'$ into phase-aligned groups using a period $P$. When $L$ and $H$ are divisible by $P$, we define $n = L/P$ and $m = H/P$. For each variable $d$, $H'_{:,d}$ is reshaped into $\widetilde{H}_d \in \mathbb{R}^{P \times n}$, where each row corresponds to a fixed phase within the period. A shared linear mapping $W \in \mathbb{R}^{n \times m}$ is then applied along the period axis:
> 여기서 $w$는 시간 길이 $L$을 보존하는 패딩을 갖는 학습 가능한 1D 필터이다. SPK는 그런 다음 주기 $P$를 사용하여 $H'$를 위상 정렬된 그룹으로 재조직한다. $L$과 $H$가 $P$로 나누어 떨어질 때, 우리는 $n = L/P$와 $m = H/P$로 정의한다. 각 변수 $d$에 대해, $H'_{:,d}$는 $\widetilde{H}_d \in \mathbb{R}^{P \times n}$로 재구성되며, 각 행은 주기 내의 고정된 위상에 해당한다. 공유 선형 매핑 $W \in \mathbb{R}^{n \times m}$이 주기 축을 따라 적용된다:

$$Y_d(p, :) = \widetilde{H}_d(p, :) W, \quad p = 1, \ldots, P \quad (11)$$

The phase-wise predictions are reassembled by interleaving phases to form $\widehat{Y} \in \mathbb{R}^{H \times D}$. When $L$ or $H$ is not divisible by $P$, we pad to the next multiple of $P$ and crop the final output back to length $H$, which keeps the mapping well-defined without restricting the experimental setup. Sharing $W$ across phases and variables yields a compact horizon mapping that avoids timestep-specific parameters and remains compatible with stringent edge constraints.
> 위상별 예측은 위상을 인터리빙하여 $\widehat{Y} \in \mathbb{R}^{H \times D}$를 형성하도록 재조립된다. $L$ 또는 $H$가 $P$로 나누어 떨어지지 않을 때, 우리는 $P$의 다음 배수로 패딩하고 최종 출력을 길이 $H$로 잘라내어 실험 설정을 제한하지 않으면서도 매핑이 잘 정의된 상태로 유지한다. 위상과 변수에 걸쳐 $W$를 공유하면 시간 단계별 파라미터를 피하는 컴팩트한 구간 매핑이 얻어지며, 엄격한 에지 제약과 양립한다.

**Fig. 5.** Structure of the Sparse Period Kernel (SPK). The fused representation $H \in \mathbb{R}^{L \times D}$ first passes through a depthwise residual aggregation $H' = H + \text{DWConv}_t(H; k_{\text{slide}})$ that smooths short-term fluctuations while preserving the input length $L$. Each channel of $H'$ is then reshaped by period $P$ into $\widetilde{H}_d \in \mathbb{R}^{P \times n}$ with $n = L/P$ rows of phase-aligned cycles. A single shared linear map $W \in \mathbb{R}^{n \times m}$ ($m = H/P$) projects each phase from $n$ past cycles to $m$ future cycles, and the per-phase outputs are interleaved to produce $\widehat{Y} \in \mathbb{R}^{H \times D}$. The parameter count $nm$ matches the lower bound for any phase-aligned linear cycle map (cf. Theorem 2 in Sec 4).
> **Fig. 5.** Sparse Period Kernel(SPK) 구조. 융합된 표현 $H \in \mathbb{R}^{L \times D}$는 먼저 입력 길이 $L$을 보존하면서 단기 변동을 평활화하는 깊이별 잔차 집계 $H' = H + \text{DWConv}_t(H; k_{\text{slide}})$를 통과한다. 그런 다음 $H'$의 각 채널은 주기 $P$로 $n = L/P$개의 위상 정렬 사이클 행을 갖는 $\widetilde{H}_d \in \mathbb{R}^{P \times n}$로 재구성된다. 단일 공유 선형 매핑 $W \in \mathbb{R}^{n \times m}$($m = H/P$)이 각 위상을 $n$개의 과거 사이클에서 $m$개의 미래 사이클로 사영하고, 위상별 출력이 인터리빙되어 $\widehat{Y} \in \mathbb{R}^{H \times D}$를 생성한다. 파라미터 수 $nm$은 임의의 위상 정렬 선형 사이클 매핑에 대한 하한과 일치한다 (Sec 4의 Theorem 2 참조).

## 3.6 End-to-end FEATHer Pipeline

Given $X$, FEATHer first generates multiscale signals $\{X^{(b)}\}_{b \in \mathcal{B}}$, processes each pathway using the shared DTK to obtain $\{H^{(b)}\}_{b \in \mathcal{B}}$, computes spectrum-conditioned weights $g$, and fuses the branch representations into $H$. The SPK then produces the final forecast $\widehat{Y}$. The model is trained end-to-end by minimizing a forecasting loss $\mathcal{L}(\widehat{Y}, Y)$, such as mean absolute error (MAE) or mean squared error (MSE). The forward computation is identical during training and inference. During training, parameters are additionally updated through gradient-based optimization, whereas inference executes the forward pass only. For clarity, the algorithmic description in this section focuses on the forward pipeline, while optimization details are described in the training procedure in Section 5.
> $X$가 주어지면, FEATHer는 먼저 다중 스케일 신호 $\{X^{(b)}\}_{b \in \mathcal{B}}$를 생성하고, 공유 DTK를 사용하여 각 경로를 처리해 $\{H^{(b)}\}_{b \in \mathcal{B}}$를 얻은 다음, 스펙트럼 조건부 가중치 $g$를 계산하고, 분기 표현을 $H$로 융합한다. SPK는 그런 다음 최종 예측 $\widehat{Y}$를 생성한다. 본 모델은 평균 절대 오차(MAE)나 평균 제곱 오차(MSE)와 같은 예측 손실 $\mathcal{L}(\widehat{Y}, Y)$를 최소화하여 종단 간 학습된다. 순방향 계산은 학습과 추론에서 동일하다. 학습 중에는 파라미터가 경사도 기반 최적화를 통해 추가로 갱신되는 반면, 추론은 순방향 패스만 실행한다. 명확성을 위해, 본 절의 알고리즘 설명은 순방향 파이프라인에 초점을 맞추고, 최적화 세부 사항은 Section 5의 학습 절차에서 설명된다.

---

**Figure source files** (in `manuscript/figures/`, to be redrawn for IoT-J — R3 #2 / R6 #4):
- Fig. 1 → `fig1_overall.jpeg` *(placed at end of Sec 2.4 in 03_related_work_draft.md)*
- Fig. 2 → `fig2_multiscale.svg` *(placed in Sec 3.2 above)*
- Fig. 3 → `fig3_dtk.svg` *(placed in Sec 3.3 above)*
- Fig. 4 → `fig4_gate.svg` *(placed in Sec 3.4 above)*
- Fig. 5 → `fig5_spk.svg` *(placed in Sec 3.5 above)*
