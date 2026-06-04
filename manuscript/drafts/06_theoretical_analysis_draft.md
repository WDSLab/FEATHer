# Theoretical Analysis

> **Source**: `tex_workspace/feather_raw.tex` lines 613--772.
> **R8 #4 critical**: TPAMI body defined Theorems 1 and 2 only, yet
> Sec 6.1 / 6.4 / 6.5 referenced Theorems 3, 4, and 5. The
> corresponding formal statements must either be added here or the
> Section 6 references rewritten. Anchors are flagged inline.

---

This section provides theoretical support for FEATHer by formalizing **(i)** the stability and controlled expressiveness of the shared DTK, **(ii)** the correctness and parameter minimality of the period-aligned reconstruction in the SPK, and **(iii)** the computational scaling properties that enable Sub-1K parameter deployment. Rather than introducing overly strong claims about spectral separation, we present guarantees that directly correspond to the implemented operators and remain verifiable under compact parameter budgets.
> 본 절에서는 FEATHer에 대한 이론적 지지를 제공하기 위해 **(i)** 공유 DTK의 안정성과 제어된 표현력, **(ii)** SPK의 주기 정렬 재구성의 정확성과 파라미터 최소성, **(iii)** Sub-1K 파라미터 배포를 가능하게 하는 계산 스케일링 특성을 형식화한다. 스펙트럼 분리에 대한 지나치게 강한 주장을 도입하기보다는, 구현된 연산자에 직접 대응되고 컴팩트 파라미터 예산 하에서도 검증 가능한 보장을 제시한다.

## 4.1 Preliminaries and Operator Notation

We view each temporal module in FEATHer as a linear operator acting along the temporal dimension and applied channel-wise unless specified otherwise. For a matrix sequence $X \in \mathbb{R}^{L \times D}$, we denote the induced operator norm by $\|\cdot\|_2$. Let $\text{DWConv}_t(\cdot; k)$ be a depthwise 1D convolution along time. In implementation, the DTK temporal convolution uses left padding followed by output trimming to preserve sequence length $L$, yielding a causal-style local mixing operator. The SPK aggregation uses a residual formulation $H + \text{DWConv}_t(H; k)$.
> 본 연구에서는 FEATHer의 각 시간 모듈을 시간 차원을 따라 작용하고 별도로 명시되지 않은 한 채널 단위로 적용되는 선형 연산자로 본다. 행렬 시퀀스 $X \in \mathbb{R}^{L \times D}$에 대해, 유도된 연산자 노름을 $\|\cdot\|_2$로 표기한다. $\text{DWConv}_t(\cdot; k)$를 시간에 대한 깊이별 1D 합성곱이라고 하자. 구현 측면에서 DTK 시간 합성곱은 시퀀스 길이 $L$을 보존하기 위해 좌측 패딩과 출력 트리밍을 사용하여 인과 스타일의 국소 혼합 연산자를 생성한다. SPK 집계는 잔차 형식 $H + \text{DWConv}_t(H; k)$를 사용한다.

> **Remark 1 (Implementation faithfulness).** All statements below are written for the operators as implemented (left padding + trimming in DTK, residual aggregation in SPK). This design choice prevents theory-implementation mismatch and ensures that the analysis is directly reproducible.
> **Remark 1 (구현 충실성).** 이하의 모든 진술은 구현된 연산자(DTK의 좌측 패딩 + 트리밍, SPK의 잔차 집계)에 대해 작성되었다. 이 설계 선택은 이론-구현 불일치를 방지하고 분석이 직접 재현 가능함을 보장한다.

We use $\|\cdot\|_2$ to denote the Euclidean norm for vectors and the induced spectral norm for matrices and linear operators. For a sequence $x \in \mathbb{R}^L$, we denote its discrete-time Fourier transform by $\mathcal{F}(x)(\omega)$ and its magnitude spectrum by $|\mathcal{F}(x)(\omega)|$. For multivariate inputs $X \in \mathbb{R}^{L \times D}$, transforms are applied along the temporal axis, and each channel is treated independently.
> 본 연구에서는 $\|\cdot\|_2$를 벡터에 대해 유클리드 노름으로, 행렬과 선형 연산자에 대해 유도된 스펙트럼 노름으로 사용한다. 시퀀스 $x \in \mathbb{R}^L$에 대해, 그 이산 시간 푸리에 변환을 $\mathcal{F}(x)(\omega)$로 표기하고 그 진폭 스펙트럼을 $|\mathcal{F}(x)(\omega)|$로 표기한다. 다변량 입력 $X \in \mathbb{R}^{L \times D}$에 대해, 변환은 시간 축을 따라 적용되고 각 채널은 독립적으로 처리된다.

## 4.2 Stability of the DTK

DTK applies a sequence of linear projection, depthwise temporal filtering, and inverse projection:
> DTK는 선형 사영, 깊이별 시간 필터링, 역 사영의 연속을 적용한다:

$$\text{DTK}(H) = \text{DWConv}_t(H W_{\text{in}}; k_{\text{temp}}) \, W_{\text{out}} \quad (12)$$

where $W_{\text{in}} \in \mathbb{R}^{D \times S}$ and $W_{\text{out}} \in \mathbb{R}^{S \times D}$.
> 여기서 $W_{\text{in}} \in \mathbb{R}^{D \times S}$이고 $W_{\text{out}} \in \mathbb{R}^{S \times D}$이다.

> **Theorem 1 (Global Lipschitz stability of DTK).** Let $\mathcal{K}$ denote the linear operator induced by the implemented depthwise temporal convolution in DTK (left padding + trimming). Then, for any $H, H' \in \mathbb{R}^{L \times D}$,
> $$\|\text{DTK}(H) - \text{DTK}(H')\|_2 \leq \|W_{\text{in}}\|_2 \, \|\mathcal{K}\|_2 \, \|W_{\text{out}}\|_2 \, \|H - H'\|_2.$$
> Moreover, if each depthwise kernel satisfies $\|k_{\text{temp}}\|_1 \leq \kappa$, then $\|\mathcal{K}\|_2 \leq \kappa$, hence DTK is globally Lipschitz with constant $\|W_{\text{in}}\|_2 \, \kappa \, \|W_{\text{out}}\|_2$.

> **Theorem 1 (DTK의 전역 Lipschitz 안정성).** $\mathcal{K}$를 DTK에서 구현된 깊이별 시간 합성곱(좌측 패딩 + 트리밍)이 유도하는 선형 연산자라고 하자. 그러면 임의의 $H, H' \in \mathbb{R}^{L \times D}$에 대해,
> $$\|\text{DTK}(H) - \text{DTK}(H')\|_2 \leq \|W_{\text{in}}\|_2 \, \|\mathcal{K}\|_2 \, \|W_{\text{out}}\|_2 \, \|H - H'\|_2$$
> 가 성립한다. 또한 각 깊이별 커널이 $\|k_{\text{temp}}\|_1 \leq \kappa$를 만족하면 $\|\mathcal{K}\|_2 \leq \kappa$이며, 따라서 DTK는 상수 $\|W_{\text{in}}\|_2 \, \kappa \, \|W_{\text{out}}\|_2$를 갖는 전역 Lipschitz이다.

**Proof.** DTK is a composition of linear operators, and the induced norm bound follows from submultiplicativity. For depthwise convolution, the operator norm is bounded by the $\ell_1$ norm of the kernel (standard discrete time filtering bound). The use of left padding and trimming preserves sequence length and does not increase the induced operator norm beyond that of the corresponding convolution. $\square$
> **증명.** DTK는 선형 연산자들의 합성이며, 유도된 노름 경계는 부등식 곱에서 따라온다. 깊이별 합성곱의 경우 연산자 노름은 커널의 $\ell_1$ 노름에 의해 경계 지어진다 (표준 이산 시간 필터링 경계). 좌측 패딩과 트리밍의 사용은 시퀀스 길이를 보존하며, 해당 합성곱의 유도된 연산자 노름을 초과하여 증가시키지 않는다. $\square$

> **Remark 2 (Implications for long-horizon forecasting).** Theorem 1 guarantees that the shared temporal mixer cannot arbitrarily amplify perturbations in the branch representations. This property is particularly important for long-horizon forecasting in industrial signals, where sensor noise and operating regime shifts may otherwise destabilize extrapolation. The bound also clarifies that stability is governed by a small set of parameters ($W_{\text{in}}, W_{\text{out}}, k_{\text{temp}}$), which is desirable in ultra-compact regimes.
> **Remark 2 (장기 예측에 대한 함의).** Theorem 1은 공유 시간 혼합기가 분기 표현의 섭동을 임의로 증폭시킬 수 없음을 보장한다. 이 특성은 센서 잡음과 운영 영역 전환이 외삽을 불안정하게 할 수 있는 산업 신호의 장기 예측에 특히 중요하다. 또한 이 경계는 안정성이 작은 파라미터 집합($W_{\text{in}}, W_{\text{out}}, k_{\text{temp}}$)에 의해 지배됨을 명확히 하며, 이는 초경량 영역에서 바람직하다.

## 4.3 What the Frequency Gate Is Doing (Energy-consistent Interpretation)

FEATHer computes branch weights from the magnitude spectrum of the instance-normalized input. Let $\widetilde{X}$ be the instance-normalized input and let $a$ be a reduced spectral descriptor (e.g., channel-averaged magnitude). A lightweight gating network produces logits and corresponding weights $g \in \Delta^B$ (simplex), and FEATHer fuses branch outputs $H = \sum_{s \in \mathcal{B}} g_s H^{(s)}$.
> FEATHer는 인스턴스 정규화된 입력의 진폭 스펙트럼으로부터 분기 가중치를 계산한다. $\widetilde{X}$를 인스턴스 정규화된 입력으로, $a$를 축소된 스펙트럼 기술자(예: 채널 평균 진폭)로 두자. 경량 게이팅 네트워크가 로짓과 해당 가중치 $g \in \Delta^B$(심플렉스)를 생성하고, FEATHer는 분기 출력을 $H = \sum_{s \in \mathcal{B}} g_s H^{(s)}$로 융합한다.

> **Proposition 1 (Energy-consistent soft selection under a surrogate objective).** Let $E_s \geq 0$ denote a branch relevance score derived from spectral energy captured by branch $s$. Consider the entropy-regularized surrogate objective $\max_{g \in \Delta^B} \sum_{s \in \mathcal{B}} g_s E_s + \tau \sum_{s \in \mathcal{B}} g_s \log g_s$. The optimizer satisfies $g_s \propto \exp(E_s / \tau)$, yielding a soft selection rule that is monotone in $E_s$ and approaches hard selection as $\tau \to 0$.
> **Proposition 1 (대리 목적 함수 하에서의 에너지 일관성 있는 소프트 선택).** $E_s \geq 0$를 분기 $s$가 포착한 스펙트럼 에너지로부터 유도된 분기 관련성 점수라고 하자. 엔트로피 정규화된 대리 목적 함수 $\max_{g \in \Delta^B} \sum_{s \in \mathcal{B}} g_s E_s + \tau \sum_{s \in \mathcal{B}} g_s \log g_s$를 고려하자. 최적화기는 $g_s \propto \exp(E_s / \tau)$를 만족하며, 이는 $E_s$에 대해 단조적이고 $\tau \to 0$일 때 강한 선택에 접근하는 소프트 선택 규칙을 산출한다.

> **Remark 3 (Why we use spectrum).** This proposition provides a principled justification for the design choice without overclaiming optimality. Using spectral features to compute $g$ is consistent with selecting the most relevant temporal scales for each instance. The learned gating network implements a compact approximation of this energy-based policy while keeping parameter overhead negligible.
> **Remark 3 (스펙트럼을 사용하는 이유).** 이 명제는 최적성을 과도하게 주장하지 않으면서 설계 선택에 대한 원리적 정당화를 제공한다. $g$를 계산하기 위해 스펙트럼 특징을 사용하는 것은 각 인스턴스에 대해 가장 관련 있는 시간 스케일을 선택하는 것과 일관된다. 학습된 게이팅 네트워크는 파라미터 오버헤드를 무시할 수 있는 수준으로 유지하면서 이 에너지 기반 정책의 컴팩트한 근사를 구현한다.

## 4.4 Parameter Minimality of SPK

SPK first applies a residual aggregation $H_{\text{agg}} = H + \text{DWConv}_t(H; k_{\text{slide}})$, then reshapes each channel into a phase-aligned representation with period $P$, and applies a shared linear map $W \in \mathbb{R}^{n \times m}$, where $n = L/P$ and $m = H/P$.
> SPK는 먼저 잔차 집계 $H_{\text{agg}} = H + \text{DWConv}_t(H; k_{\text{slide}})$를 적용하고, 각 채널을 주기 $P$를 갖는 위상 정렬 표현으로 재구성한 다음, 공유 선형 매핑 $W \in \mathbb{R}^{n \times m}$을 적용한다. 여기서 $n = L/P$, $m = H/P$이다.

> **Theorem 2 (Parameter minimality of SPK for phase-aligned linear cycle mapping).** Consider the class of functions in which, for each phase, the $n$ observed cycle values are mapped linearly to $m$ future cycle values using a phase-shared linear map. Any linear operator realizing this mapping requires at least $nm$ degrees of freedom. SPK uses exactly $nm$ parameters in $W$ and is therefore parameter minimal for this function class.
> **Theorem 2 (위상 정렬 선형 사이클 매핑에 대한 SPK의 파라미터 최소성).** 각 위상에 대해 $n$개의 관측된 사이클 값이 위상 공유 선형 매핑을 사용하여 $m$개의 미래 사이클 값으로 선형 매핑되는 함수 클래스를 고려하자. 이 매핑을 실현하는 임의의 선형 연산자는 최소 $nm$개의 자유도를 요구한다. SPK는 정확히 $nm$개의 파라미터를 $W$에 사용하므로 이 함수 클래스에 대해 파라미터 최소이다.

**Proof.** A linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ has $nm$ degrees of freedom. Since SPK applies such a map in the phase-aligned space using a shared $W$, it exactly matches this lower bound. $\square$
> **증명.** $\mathbb{R}^n$에서 $\mathbb{R}^m$로의 선형 매핑은 $nm$개의 자유도를 갖는다. SPK는 공유 $W$를 사용하여 위상 정렬 공간에서 이러한 매핑을 적용하므로 이 하한과 정확히 일치한다. $\square$

> **Remark 4 (Scope and limitations of the guarantee).** Theorem 2 does not claim that all time series are periodic. Instead, it states that when long-horizon structure is well explained by phase-aligned cycle dynamics, SPK allocates parameters exactly to the intrinsic degrees of freedom of that structure, thereby avoiding horizon-specific decoders. This inductive bias is particularly effective under Sub-1K parameter budgets.
> **Remark 4 (보장의 범위와 한계).** Theorem 2는 모든 시계열이 주기적이라고 주장하지 않는다. 대신 장기 구조가 위상 정렬 사이클 동역학에 의해 잘 설명될 때, SPK가 그 구조의 본질적 자유도에 정확히 파라미터를 할당함으로써 구간별 디코더를 회피한다고 말한다. 이러한 귀납적 편향은 Sub-1K 파라미터 예산 하에서 특히 효과적이다.

<!-- =====================================================
     R8 #4 CRITICAL — TPAMI Sec 6.1/6.4/6.5 referenced
     Theorems 3, 4, 5 without ever stating them in Sec 4.
     Two options for IoT-J revision:
       (A) Add the three formal statements here (preferred).
       (B) Rewrite Sec 6.1/6.4/6.5 to cite Theorem 1/2 plus
           Remarks/Propositions instead.
     The placeholders below mark the statements that need
     formalization — extract intended content from
     08_ablation_draft.md Sec 6.1/6.4/6.5 and write proper
     formal claims.
     ===================================================== -->

## 4.5 [R8 #4 — to add] Implicit Band-wise Spectral Separability

> **Theorem 3 (Implicit Band-wise Spectral Separability) — placeholder.** *Section 6.1 cites this theorem as the justification for the multiscale decomposition producing near-orthogonal filter banks. Formalize the claim: under the FEATHer decomposition with point/high/mid/low branches, the corresponding frequency responses $\{H_s(f)\}_{s \in \mathcal{B}}$ concentrate energy in approximately disjoint spectral bands when measured at the kernel-size choice $(k_p=1, k_h=3, k_m=5, k_l=\text{avg-pool stride } r)$. Alternative: remove the "Theorem 3" reference in Sec 6.1 and reframe as an empirical observation.*
> **Theorem 3 (암묵적 대역별 스펙트럼 분리 가능성) — placeholder.** *Section 6.1에서 다중 스케일 분해가 근직교 필터 뱅크를 생성한다는 정당화로 이 정리를 인용한다. 정형화 필요: 점/고주파/중주파/저주파 분기를 갖는 FEATHer 분해 하에서 해당 주파수 응답 $\{H_s(f)\}_{s \in \mathcal{B}}$가 커널 크기 선택 $(k_p=1, k_h=3, k_m=5, k_l=\text{평균 풀링 보폭 } r)$에서 측정될 때 대략 서로소인 스펙트럼 대역에 에너지를 집중시킨다. 대안: Section 6.1의 "Theorem 3" 참조를 제거하고 경험적 관찰로 재구성.*

## 4.6 [R8 #4 — to add] Phase-preserving Reconstruction

> **Theorem 4 (Phase-preserving reconstruction of SPK) — placeholder.** *Section 6.4 cites this as the justification for SPK preserving autocorrelation structure. Formalize: when the dominant period of the input is $P$ (or a divisor of $P$), SPK reconstruction commutes with phase shifts and preserves the autocorrelation at lag $kP$ up to $\mathcal{O}(\|w\|_2)$ filter error. Alternative: state empirically.*
> **Theorem 4 (SPK의 위상 보존 재구성) — placeholder.** *Section 6.4에서 SPK가 자기상관 구조를 보존한다는 정당화로 인용된다. 정형화: 입력의 지배 주기가 $P$(또는 $P$의 약수)일 때, SPK 재구성은 위상 이동과 가환이며 시차 $kP$에서의 자기상관을 필터 오차 $\mathcal{O}(\|w\|_2)$까지 보존한다. 대안: 경험적으로 진술.*

## 4.7 [R8 #4 — to add] Parameter Lower Bound for Phase-aligned Linear Heads

> **Theorem 5 (Parameter lower bound for any phase-aligned linear head) — placeholder.** *Section 6.5 cites this as the bound any phase-aligned linear forecaster must satisfy. Formalize: any linear forecasting head that maps $n = L/P$ observed cycles to $m = H/P$ future cycles in a phase-shared way must use at least $\max(nm, P)$ parameters. SPK achieves $nm$ when $P \leq nm$. Note this overlaps with Theorem 2 — distinguish or merge.*
> **Theorem 5 (위상 정렬 선형 헤드에 대한 파라미터 하한) — placeholder.** *Section 6.5에서 위상 정렬 선형 예측기가 만족해야 하는 경계로 인용된다. 정형화: $n = L/P$개의 관측 사이클을 위상 공유 방식으로 $m = H/P$개의 미래 사이클로 매핑하는 임의의 선형 예측 헤드는 최소 $\max(nm, P)$개의 파라미터를 사용해야 한다. SPK는 $P \leq nm$일 때 $nm$을 달성한다. 이는 Theorem 2와 중첩됨에 유의 — 구분하거나 병합 필요.*

## 4.8 Complexity and Sub-1K Regime

We summarize parameter scaling to clarify why FEATHer can operate in sub-1K regimes. DTK parameters are $2DS + S k_{\text{temp}}$. SPK parameters are $k_{\text{temp}} + nm$. The decomposition filters and spectral gating module introduce only minor overhead. Because DTK is shared across branches, the parameter count does not grow with $B$, except for negligible branch-specific filters and gating outputs. Runtime complexity is linear in $L$ and $D$, up to small constants determined by kernel sizes and latent width $S$, which is compatible with the constraints of industrial edge execution.
> 본 연구에서는 FEATHer가 sub-1K 영역에서 동작할 수 있는 이유를 명확히 하기 위해 파라미터 스케일링을 요약한다. DTK 파라미터는 $2DS + S k_{\text{temp}}$이다. SPK 파라미터는 $k_{\text{temp}} + nm$이다. 분해 필터와 스펙트럼 게이팅 모듈은 부수적인 오버헤드만 도입한다. DTK가 분기에 걸쳐 공유되기 때문에 무시할 수 있는 분기별 필터와 게이팅 출력을 제외하면 파라미터 수는 $B$에 따라 증가하지 않는다. 런타임 복잡도는 커널 크기와 잠재 폭 $S$에 의해 결정되는 작은 상수까지 $L$과 $D$에 대해 선형이며, 이는 산업 에지 실행의 제약과 양립한다.

> **Remark 5 (Why stronger theoretical claims are avoided).** Under compact designs, the most meaningful guarantees are those directly tied to implemented operators and measurable quantities, such as stability bounds, parameter minimality, and scaling behavior. Stronger claims, including strict spectral disjointness or near orthogonality, would require assumptions that are difficult to verify in practice and are not necessary to support FEATHer's core contributions.
> **Remark 5 (더 강한 이론적 주장을 회피하는 이유).** 컴팩트 설계 하에서 가장 의미 있는 보장은 안정성 경계, 파라미터 최소성, 스케일링 거동과 같이 구현된 연산자와 측정 가능한 양에 직접 연결된 것이다. 엄격한 스펙트럼 서로소성 또는 근직교성과 같은 더 강한 주장은 실제로 검증하기 어려운 가정을 요구하며 FEATHer의 핵심 기여를 뒷받침하는 데 필요하지 않다.

All subsequent theoretical statements are formulated with respect to the specific implementation of temporal operators, including left-padding and output trimming. This ensures that the derived stability bounds and complexity limits directly govern the empirical behavior of the FEATHer architecture.
> 모든 후속 이론적 진술은 좌측 패딩과 출력 트리밍을 포함한 시간 연산자의 특정 구현에 대해 정형화된다. 이는 유도된 안정성 경계와 복잡도 한계가 FEATHer 아키텍처의 경험적 거동을 직접 지배하도록 보장한다.
