# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

## Current status (2026-08-26)

**⚡⚡⚡⚡⚡⚡⚡⚡⚡ 2026-08-25/26 (부품 해체 + 주기 설정 오류 + 호라이즌 발견;
상세 = `memory/project_component_teardown_0825.md`). ⚠ 사용자 판단 = "이 방향으론
아무것도 안 될 것 같다" → 새 세션에서 원점 재검토. 아래는 방향과 무관한 사실:**
- **① ★★★ 시그니처 부품 전부 무용 확정**: `--num_bands 1 --point_kernel 7
  --dtk_mode none`(밴드4+게이트+DTK → **conv 하나**)이 mfg **+1.38%**, LTSF **+1.92%**
  = 둘 다 노이즈 안 동률, **파라미터 −30%**. **채널모드가 다른 두 도메인에서 같은 결론**.
  Steel은 −2.67%로 개선. 1시드라 3시드 재확인 필요.
- **② ★★★ 단일 부품 ablation이 이 구조에선 무의미**: 축마다 부품이 **중복**
  (시간축 밴드conv↔DTKconv / 채널축 in_proj↔잔차MLP). `dtknone` −0.98%·`band1` −0.75%로
  각각은 괜찮은데 **둘 다 빼면 +2.2%**(conv_point가 k=1이라 시간축 필터가 0이 됨).
  **옛 ablation 표와 현 ablation이 정반대인 이유도 이것** — 둘 다 단일부품 방식.
- **③ ★★★ 주기 설정 오류(신규)**: detrend 후 acf 봉우리 = Steel/EWELD **96**(0.71/0.72)인데
  설정은 **24**(acf **음수**). WindSCADA 설정48 acf **−0.271**. BlastFurnace는 **봉우리 없음**.
  **구조적 강제**: period가 seq_len·pred_len을 나눠야 해 gcd(96,336)=48 → **P≤48**,
  P=96이면 n=1이라 SPK 무의미. **→ SPK는 주기 인수분해가 아니라 근거 없는 스트라이드였음.**
  일반화 = **주기모델(룩백≫주기) × MCU예산(짧은 룩백) × 산업샘플링(주기가 룩백만큼)
  = 3제약 양립불가.** SparseTSF/CycleNet도 동일 규칙이라 같은 핸디캡(공정성 유지).
- **④ ★★★★ 호라이즌별 배포가능 1위 — 가장 강한 결과**: flash 128KB 기준 H=96은 DLinear
  0.5662 > FEATHer 0.5849(2위)인데, **H=192 FEATHer 1위(+3.5%)·H=336 1위(+6.0%)·
  H=720 1위(+5.8%)** — DLinear(H192서 167KB)·CycleNet(H336서 165KB) 탈락. 이유는 정확도가
  아니라 **파라미터 호라이즌 불변**(4496→4522=1.006배 vs DLinear 7.5배). **정확도 데이터는
  이미 전부 있음. 막힌 건 RAM뿐.**
- **⑤ ★★ RAM이 병목·파라미터는 예측 못 함**: params↔flash rho **+0.94(유의)** vs
  RAM **+0.54(무의)**·latency +0.60(무의)·정확도 **−0.83(유의)**. **잘 예측하는 축(flash)은
  여유(25.7/128KB), 못 예측하는 축(RAM)이 벽(18.6/20KB)**. FEATHer가 RAM 7모델 중 **꼴찌**.
  **밀집 선형헤드가 H=192까지 flash 통과**(62/99KB)하고 **SPK보다 3.6% 정확** →
  **H=96에선 SPK 쓸 이유 없음**, 압축은 H≥336부터 의미.
- **⑥ ★★ 출력 스트리밍 수치검증**: SPK backbone이 위상 독립 → **출력주기 단위로 루프를 뒤집으면**
  24스텝씩 연속 방출, **max|diff| 2.68e-07 = 동일**(순수 루프 재정렬, **PyTorch 수정 불요**).
  출력버퍼 19.7KB→0.66KB, **총 6.8KB로 H 무관**. ⚠ **SparseTSF도 같은 구조 = 차별점 아님**.
- **⑦ ★★ 다운스트림 KPI 실패(Steel)**: 전날 피크 회피 → **climatology가 학습모델 전부를 이김**,
  최고가 달성가능분의 11%. 원인 = **원리적 불가**(실제 피크가 하루 3.9h 이동, 모델 시각오차
  3.5~4.2h = 찍기 수준). Spearman(MSE, relief) −0.09~−0.49. ⚠ **KPI 갈아끼우기 = p-해킹**.
  웹검색: decision-focused learning으로 **이미 확립된 방법론** → **노벨티 아님, JMS 입장권**.
- **⑧ 계열 전체 선형(소스 검증)**: SparseTSF-Linear/MixLinear/FITS/DLinear 비선형 0,
  **DiPE는 softmax가 `nn.Parameter`에 걸려 추론 시 상수**(`StaticTimeWeight`, temperature=NaN).
  유일한 예외 = SparseTSF-MLP의 ReLU 헤드. **비선형은 넣을 때마다 이득**(잔차 +2~9%,
  headmlp −4.64%, STSF-MLP가 Linear보다 3%↑), **선형 구조는 무엇을 해도 0**.
- **⑨ LTSF 고채널(24런 완주)**: **Traffic 862ch에서 FEATHer 0.6190 = 경량 1위**(226 파라미터).
  Electricity 0.2237 = SparseTSF와 동률·중하위.
- **⑩ ⚠ 옛 ablation 표(TABLE 7~11) 사용 금지**: 현 측정과 **8개 항목 전부 반대**이고
  (MLP헤드: 옛표 최악 vs 현재 최고), 그 표의 ETTh1 0.373 = **test-peeking 수치**(08-09 확정).
- **▶ 커밋(전부 기본값 byte-identical 검증)**: `--dump_preds`/`--dump_stride`(+`self.border1`,
  역변환 2.9e-06 재현·타임스탬프 정렬 확인) · `--point_kernel`/`--no_in_proj` ·
  `tools/paper/kpi_steel_peak.py`.
- **▶ 미완**: mfg 실험 B/C(채널축 중복, 중지됨) · A 3시드 · **정합 주파수밴드 대조군(미구현,
  없으면 "구현을 잘못한 것" 반박 못 막음)** · SPK 내부 비선형(`spk_hidden` 50~150p) ·
  스트리밍 C구현→골든→H192/336/720 빌드·실측 · 배포가능성 행렬 · DLinear 포팅 ·
  MetroPT3 타당성(라벨 없음) · ablation specHi 3칸(서버 221/240 정체).
  [[project-component-teardown-0825]]

## Current status (2026-08-24)

**⚡⚡⚡⚡⚡⚡⚡⚡ 2026-08-24 (보드 7모델 전수실측 + ablation 부품기여 0 + JMS 리뷰어
심사; 상세 = `memory/project_board_session_ablation_0824.md`):**
- **① ★ 보드 실측 완결**(`deployment/BOARD_RESULTS_20260824.md`, 한 세션·한 빌드,
  golden 전부 PASS): persistence 0.07ms/10.4KB · SparseTSF-Linear 41p/47.4ms/13.5KB ·
  MixLinear 49p/79.7ms/15.8KB · DiPE 339p/1312ms/21.7KB · **FEATHer-N 938p/655ms/25.7KB/
  RAM 18.6KB** · SparseTSF-MLP 1181p/431ms/22.3KB · FITS 1200p/796ms/28.9KB. **4모델이
  07-27과 마이크로초 일치 = 측정 재현성.** 전류 7모델 전부 29.6~32mA(**모델 무관**) →
  에너지 = latency 함수.
- **② ★★ 발견 3개(선례 없음)**: (a) **파라미터로 latency 예측 불가**(DiPE 339p가
  FEATHer 938p의 2배 느림) (b) **flash도 예측 불가**(params×4B가 최대 **84배** 오차;
  펌웨어 바닥 10.4KB 지배, flash 범위 2배 vs params 29배) (c) **matmul-only MAC은
  r²=0.01**, 정규화/활성/복소연산 + **보드의 실제 O(N²) DFT**까지 계상하면 **r²=0.87**
  (`tools/paper/measure_complexity.py --validate`). FEATHer가 FITS를 4축 전부 지배;
  SparseTSF-MLP엔 **속도 짐(655 vs 431ms)·정확도 승(0.6940 vs 0.7271, 4.8%)**.
- **③ ★★ ablation(mfg 4데이터×4H, 64%)**: **시그니처 부품 전부 flat**(노이즈 1.2%,
  임계 2%) — `dtknone` **−0.98%**·`band1` **−0.75%**(=**빼는 게 나음**)·`gatesm` +0.21%
  (FFT 불필요 직접증거)·gateuni +1.32%. **실제 기여 = 채널믹싱 5.5% > 잔차MLP 2.16% >
  나머지 ~0**(둘 다 표준 차용 부품). `headmlp` −8.37%지만 37,899p(40배)=MCU 초과.
  08-16 선형붕괴 예측 적중.
- **④ 현 config가 파레토 밖**: ds16_k7_B4(938p, 0.6926)가 **ds8_k7_B4(770p, 동일 MSE)
  한테 지배됨**; 27config 중 18개가 노이즈 안 동률. **⚠ 그리드 27개 전부 잔차 ON →
  `s_lat = L×max(d_state, mlp_rank)`라 d_state 낮춰도 RAM 안 줄어듦**(전부 18.7KB);
  **잔차를 빼야 d_state가 메모리 레버**(ds8_k3_B4_nr 491p/15.7KB). `run_config_sweep.py`
  (6 arm) 커밋됨·**미발사**.
- **⑤ ⚠ 호라이즌 막힘**: FEATHer H=192 = 24.5KB > 20KB SRAM. 우리가 이기는 구간
  (카운터 예측 H720서 866 vs 2275ms = 2.6배 승)을 보드가 못 담음. 밴드 줄여도 무효.
- **⑥ ★★★ JMS 리뷰어 심사(에이전트)**: **현 상태 reject 확률 85~90%**(그중 30~40%는
  **데스크리젝**). 치명 공격 = (1) **H=96에선 DLinear가 128KB에 들어갈 수 있음**
  (18.6Kp≈74KB+10.4KB<128KB)이고 DLinear가 3.7% 더 정확 ↔ H≥192는 우리가 배포 불가 →
  "1위 주장이 성립하는 칸이 어디냐" (2) **"신규 부품은 기여 0, 성능 내는 부품은 신규
  아님"** (3) **SPK가 SparseTSF 헤드와 사실상 동일한데 Methodology 미귀속 = 윤리 등급
  지적** (4) 제조 시스템 기여 부재(다운스트림 KPI 0) (5) 배포집합 순환성 + 자기 파레토
  위반. **권고 = 헤드라인을 "모델 제안"에서 "배포비용 실측 특성화 + 설계모티프 반증"
  으로 전환**(FEATHer는 벤치마크의 한 점으로 강등); 선형붕괴 분석을 **FITS/MixLinear/
  DiPE로 확장**해야 "우리 모델 실패"가 "이 설계 계열 전체 반증"이 됨. 다운스트림 KPI
  (MetroPT3 알람 리드타임 or EWELD 피크부하 페널티, **재학습 불요·저장 예측만으로**)
  없으면 **JMS 대신 IEEE TII 권장**.
- **▶ 남은 것**: (1) ablation 잔여 86런(headmlp/headconv/headlin/spec0/specHi) 완주 →
  A유지/B축소/C부품제거 결정 (2) **LTSF Electricity/Traffic 24런 미발사**(서버):
  `FORCE_OMP=4 python run_forecast.py --model FEATHer --data Electricity,Traffic
  --exp_tag main --num_seeds 3 --results_csv results/ltsf_highD.csv --ngpu 1 --gpu 0
  --jobs_per_gpu 2` (3) Tier-0 무료 수정 3건(SPK 귀속·작은 config 교체·robustness를
  Limitations에 공개) (4) deployability matrix(모델×데이터×호라이즌 flash/SRAM).
- **▶ 커밋(personal/ijaehun, 전부 push)**: `ac2a138`(metrics CORR/R² NaN 버그+compute_r2)
  ·`285ad2e`~`a644678`(ablation, override precedence 버그 포함)·`0847ffb`(표 생성기)·
  `f3bb4d9`(FEATHer 938p 보드포팅)·`d267481`(SparseTSF-MLP 포팅)·`b322342`(SRAM)·
  `137b966`(보드 결과)·`3b03dff`(run_config_sweep). [[project-board-session-ablation-0824]]

**?�⚡?�⚡?�⚡??2026-08-14 (최종 HP ?�정 + 배포 ?�레?�밍 ?�립 + mfg final ?�료;
?�세 = `memory/project_final_hp_framing_0814.md`):**
- **??모델 ?��?�??�정 = 2변�?채널모드 ?�위�?**: **mfg = FEATHer-N**(channel_mode `mix`
  + ?�차MLP rank16, canonical **`ds16_k7_B4`** = d_state16/kernel7/num_bands4, **??mfg
  ?�일 1�?*, period�??�이?�별 FFT) / **LTSF = FEATHer-CI**(`ci_full` + ?�차, **per-dataset**
  6�?config). **????MSE ?�습**, 차이 = 채널모드 ?�나. "채널?��? 강한 ?�업?�서=mix, ?�한
  ?�채널=?�립" = inductive bias(DLinear/NLinear ?��?). ?�일 config�?????못잡???�연).
- **??그리?�서�?�?27config, val-rank)**: **d_state�??�의�?*(d16>d8>d4), **kernel/num_bands
  flat**(?�이�?. mfg mean-rank 1??ds16_k3_B3?�나 **num_bands=4 채택**(4밴드 ?�그?�처 +
  d16??test-best) ??**ds16_k7_B4**. LTSF: **CIN(0.3588) < L(0.3647)** ??CI ?�정.
- **????mfg final 3seed ?�료 = 배포가??1??*: `feather_final.csv` 72?? **FEATHer-N OVERALL
  0.6940** > CycleNet 0.720 > FITS 0.726 > DiPE 0.729 > Mix/Sparse 0.749. ?�??=
  `analysis/01_accuracy_mfg.xlsx`(FEATHer 822p). **LTSF final(--group ltsf) ?�직** ??03
  FEATHer blank(?�리�?채�?). LTSF CIN val=0.3588 ??FITS 0.3582(**4/6 ?�이????*, ?�률).
- **???�★ ?�레?�밍 = "128KB MCU 배포가??�?최정??(sub-1K ?�기)**: sub-1K 강제?�면 **DiPE
  (339p,0.729)?�테 �?*(sub-1K 모델 �?DiPE 1?�·CIN 2??=?�멸 ?�레?? MCU-deployable�?가�?  N 1??+ 물리??�?N 최�? 4522p=18KB<<128KB). **배포집합 6�?*(SparseTSF/MixLinear/DiPE/
  FITS/FEATHer-N/CIN); **CycleNet(H720 288KB)·DLinear(545KB) flash ??���?배제**(FITS 20KB??  CycleNet 288KB=14배갭). **?�력버퍼 방어**: L=720 ?�도??720×D×4B)>20KB SRAM(Steel??19.7KB)
  ??L=96????보드 최�? 룩백, **모델무�?**(채널?�립 ?�외 ?�음).
- **??config�?`_DATASET_OVERRIDES`???�결**(`_FEATHER_MFG_N`/`_FEATHER_LTSF_CI` 공유dict +
  per-dataset period/arch) ??**`run_forecast.py --model FEATHer --group {mfg,ltsf}`�?직접**
  (???�구 ??만듦; arch???�동 `--model_overrides` forward; 그리?�는 ?�커 직접?�출?�라 무영??.
- **????FASTer 미스?�리 종결**: ??코드 **test-peeking**(val?�이 min-over-epochs test) ??  0.373 ?�현·?�용 불�?(?�중?�라?�로??ETT 못따?�감=코드 ?�티?�트). ?�이브리??????0.347)=
  cherry-picking 금�?. `--oracle_test` ?�래그로 ?�리??천장 �????�게 ??진단?�용).
  [[project-old-faster-test-peeking-0809]]
- **??커밋(personal/ijaehun; ?�버 origin=ijaehun)**: b925a12(--oracle_test)·529f56e(NaN-tolerant
  resume torn-row)·ef1cdc8(mfg N override)·7689840(LTSF CI override)·76b80d1(--best_only)·
  cf704f9(build_paper_tables?�final). ???�버 VSCode ?�뎃 ??askpass 깨짐 ?????��????�는
  `GIT_ASKPASS= git pull origin main`.
- **???�음**: (1) **LTSF final** 발사(`--group ltsf --exp_tag final`, config 코드???�음)??  03 채�? + best_only 천장 ?�리 (2) **보드 ?�측??*(최종 ds16_k7_B4 golden ?�생?? 07-27?�
  ??453p) = 배포?�성??3�?6모델 latency/energy/memory ?�측 + 배터�?projection + ?�라?�즌
  flash�? (3) **Ablation**(?�계??: A 부???�차/게이??밴드/DTK/SPK/RevIN)·B 채널모드·C HP민감??  (??게이??밴드 ?�할�??�직보고) (4) GRU/XGBoost=?�음(리비??카드). [[project-final-hp-framing-0814]]

**?�⚡?�⚡?�⚡ 2026-08-05 (배포집합 ?�레?�밍 + L/N/CI ?��?�?+ 천장?�캔 발사 +
MixLinear 보드?�팅 ?�료; ?�세 = `memory/project_deployment_family_pivot_0805.md`):**
- **???�레?�밍 ?�정 = "MCU 배포가??모델 �?최정??** (경량tier ?�님). DLinear??배포?�서
  **size�??�연?�락**(ETTh1 H720 = 139,680 params ×4B = **545KB > 128KB flash**, ?��? 로그�?.
  MCU params ?�한 ??**2.7�?float32)** = (128KB??15KB?�웨?�−~1.5KB?�이�?÷4B. robustness ?�거
  ?�확??
- **??FEATHer ?��?�?= 2�?4�?* (DLinear Linear/NLinear/DLinear 미러): 채널처리(mix/CI) ×
  비선?�잔�?off/on) ??**L**(base canonical)·**N**(mix+?�차, ?�채널??·**CI**(ci_full=채널?�립,
  **D무�? ~140~370 params sub-1K**)·**CIN**. **D²범인 = `in_proj=nn.Linear(eff_d,eff_d)`
  (FEATHer.py line338)**; CI가 ?�걸 ?�거(Electricity 125K??55). **CI??params???�??MACs??*
  (채널반복). ?�차 ?�설�? ?�??��(rank16)?�단??비선??**D??D?�D**(천장?�캔?� ??rank16). **??  ?��?�?2�?L/N) vs 3�?L/N/CI) = PENDING**, 게이??= **mfg BlastFurnace(D=28)??CI가
  FITS(1.005) ?�기??* (지�?mix가 지???�일 mfg). `--channel_mode {mix,ci_full}`(2330b9c)·
  `--use_residual_mlp` ?��? ?�고 orchestrator ?�다 forward(검�?.
- **??천장?�캔 발사 (?�버, ?�기중)**: 4�?× 2loss(mse/l1) × 12?�이??mfg6+LTSF?�채널6) × 4H ×
  **1seed = 384??*, 격리 CSV `results/fcst_FEATHer_results.csv`(WDSLab?�크), exp_tag
  `feather_{L,N,CI,CIN}`+`_l1`. 1seed�?천장(test-MSE) 본뒤 best�?3seed. **?�션??64/384**
  (mfg L ?�료, N 진행). **??mfg=MSE ?�정**(L: MSE?�습 0.713 < MAE 0.744 ?�확?? MAE??LTSF�?.
  ?�림 = N2O/MetroPT3 ?�??+ oversubscription(MSE+MAE+lookback 2GPU) ??**FORCE_OMP/?�시sweep??*.
- **??R² ?�롭**: mfg/LTSF ?��? -?�조(r2_mean???�도?�내부 T축분?�으�??�눔 ??near-constant
  채널 ??��). **MSE/MAE�?*(문헌?��?). 교수?�하�?R²=1?�MSE/Var(?�역) ?�처�??�실?�불??
  ???�MSE=?�위?�일).
- **??복잡?�표 T5 = `analysis/04_complexity.xlsx` ?�성**(베이?�라?? FEATHer placeholder):
  **FFT-aware MACs 카운??* `tools/paper/measure_complexity.py`(matmul=FlopCounterMode÷2 +
  FFT=N·log2N; **DiPE 0??1K**; params CSV?�치). `results/complexity.csv`(288). **??FEATHer??  MACs 최소 ?�님**(FEATHer-L 79K vs FITS 17K); ??참고??"min 1.67M"=?�른규약. **차별??=
  보드 latency**(FITS 796ms > FEATHer 349ms = MACs?�어???�칩?�림).
- **????MixLinear 보드?�팅 ?�료 + x86 golden PASS** (배포집합 ?�일 구멍?�었?? 49p=FEATHer
  직접 sub-1K ?�이�?: `deployment/board/export_golden_mixlinear.py`(period24 SparseTSF?��?
  staged **0 diff**) + `src/mixlinear_infer.{c,h}`(공용커널+복소linear; forward=mean-sub?�shared
  conv1d(k25)?�차?�segment?�time(factored Lin2×2)?�freq(len4 FFT+복소Lin4????+iFFT-real)?�mix
  α0.5) + golden 3?�더 + `test/test_mixlinear.c` + `_mixlinear_scalar.py`. **x86 golden PASS**
  (**WSL gcc 11.4 로컬**, 4?�계 ~1e-7). **보드코드 6�??��? ?�료**; ?��?�?MixLinear 물리측정
  (main.c 4?�집=`#define RUN_MIXLINEAR`+분기, BOARD_RUNBOOK 51-60 ?�턴) + FEATHer최종변???�측??
- **??GRU/XGBoost = mfg�?* (LTSF ?�킵). **?��?�?*: ?�조PdM ?�문 RF/XGBoost/LSTM/GRU 관?�적
  ?�함(XGBoost ?�규모mfg 강자+?�석?? = practitioner-tier credibility + "generic RNN sub-1K
  ?�됨" ?�거. **보드 ?�올�?*(DLinear처럼 size배제). **공통�?= flash바이??*(XGBoost=?�리??  params N/A ??KB�?비교). ?�닝 = GRU hidden{32,64,128}·XGBoost depth{4,6,8} mean-rank 1�?+
  val?�택. 코드 미커�?list_models15=?�윕?�장 ?�리�? 별도커밋).
- **???�버 CSV ?�리**: `fcst_results.csv` FEATHer 192???�거?�아카이�?+ 5seed?�분(2028/2029)
  352??분리??_cleanup_20260805/`(백업+README 머�?백법). 메인 = 베이?�라??3 균일3?�드 3429??
  `baseline_organized.csv`+`organize_baseline_results.py`(?�실?�용). lookback `--exclude
  FEATHer,천장7` seq336/720 ?�어�?
- **??미커�?로컬**: MixLinear ?�팅 ?��?(export/C/golden/test) + 복잡?�도�?measure_complexity/
  build_complexity_xlsx) + organize_baseline_results + run_grid_ltsf_v2. **???�음 ?�션**:
  천장?�캔 ?�주 ?��?CI/CIN ?�오�?**?��?�?L/N vs L/N/CI 결정**) ???�차 최종?????�튜??3seed ??  ??T1 ?�인??flash축·T4 배포·01/03 갱신·Pareto) ??MixLinear 보드측정 + GRU/XGBoost.
  [[project-deployment-family-pivot-0805]]

**??2026-08-04b (lit_review vault ?�면?�리 + baseline provenance/?�현불�? ?�량증거;
?�세 = `memory/project_litreview_vault_provenance_0804b.md`):** ?�험 ?��?�?**문서·분석
?�션**(코드/?�버 변�?0, gitignored vault).
- **?�출**: `analysis/lit_review/experiments/`(?�규) foundation 11�?protocol·lineup·
  datasets·**master_table**·**number_crosscheck**·**datasets_credibility** ?? +
  `papers/`??baseline method ?�트 **9�??�규**(iTransformer·PatchTST·TQNet·TimesNet·
  TimeMixer·MDMLP_EIA·LMS·CycleNet·DiPE) + ??21??`category::` ?�그 + baseline PDF
  13�?TimeXer ?�운(pdfs/ 22�? arXiv ID�??�도 ?�정).
- **??lookback 2?�태�?*: L=96 native(TSLib 계보) vs **L=720 native(?��?-linear=
  SparseTSF/FITS/MixLinear/DiPE)** ??uniform L=96가 **?�리가 ?�겨????sub-1K ?�이벌을
  ?�디�?* ??방어 = 배포 바이?�예??+ **SparseTSF@720 부�?rank-stability**.
- **??복사-??*(PDF 직접?�인): MixLinear?�FITS(byte-identical) · DLinear??Transformer??  FEDformer(캡션 ?�백) · DiPE비선?�·CycleNet?�iTf+TimeMixer · iTf/PatchTST?�Zeng2022 ·
  TimesNet/LMS/TimeMixer=TSLib ?�run(뿌리) · TQNet=?�용 · MDMLP=불명.
- **?�★ ?�현불�? ?�량증거(방법�?최강 방어)**: **DLinear ETTh1@720, ?��? nominally L=96·
  ?�당??같�? TSLib repo?�데 0.513(TimeXer/?�규?�러?�터)~0.695(TimesNet)=35% ?�프?�드.**
  최신 3??LMS/TQNet/MDMLP)=?�로 byte-identical(?��?공유), TimeXer=endpoints�?부분중�?
  ?�스 ?�님); TimeXer·LMS ?�다 "TSLib ?�run" 명시?�데 �??�름 ??**같�? repo ?�run??  ?�현 ????** ???�문 방법론에 "공표?�자 직접비교 불�? ???�일 L=96 from-scratch ?�면
  ?�현만이 공정" 문장?�로 직행.
- **??mfg-6 공신???��?�?4�??�규 vet)**: ?��? credible ??Steel(UCI#851/BRI Q1)·
  WindSCADA(Zenodo/Wind Energy Q1)·BlastFurnace(**Data in Brief'25** DOI)·N2O(descriptor=
  arXiv, **?�커=Water Research'25**)·MetroPT3(**Nature Sci Data'22**/UCI#791)·EWELD(**Nature
  Sci Data'23**/Figshare). ?�무??96??20 LTSF ?��????�조 ?��? LTSF 최초" ?�효. **?�리
  ?�용=권고 ?�합**(BlastFurnace 29,602??raw[200??분류?�일 ?�님]·MetroPT3 ?�날로그7�?.
  ?�문 to-do: N2O?�Water Research ?�용·EWELD seasonal-naive ?�수·MetroPT3 ?�날로그7+결측
  17% 마스??명시·Steel ?�채??공동?�측 명시.
- **??결정(코드 0)**: (1) **배포 ?�험 ?�프?�이�?* ??FEATHer??최소·최속·최�??�너지 ??  ?�님(SparseTSF가 ?��?) ??**?�율경쟁 ?�레?�밍 금�?, accuracy-per-resource Pareto�?*:
  FEATHer가 **FITS/DiPE???�너지×?�확??지�?*(??빠르�????�확), SparseTSF/MixLinear??  ?�율�?�?�?명확?????�확). **?�드?�인=?�확??sub-1K 최정??MCU ?�재), 배포=feasibility
  +FITS/DiPE지�?보조.** (2) **???��? 보드 구매 NO**(?�력버퍼 모델무�?·FEATHer ?�전??  ?�고 ?�림·SparseTSF ?�너지 못이김=보드무�? ???�매????고침; 배터�?각도�??�하�?그때
  L053R8 ULP 1�?. (3) **?�인??추�? NO**(S4D/SegRNN/MambaSimple/Koopa: Mamba=?�칙?�외,
  ?�머�?Related Work ?�용+escalation; 14 ?��?).
- **???�음**: ?�버 ?��?(CycleNet/MixLinear LTSF + baseline 룩백 ?�윕 `--exp_tag lookback`
  seq48/192/336/720 edge-6) + **FEATHer v2(MLP 640p) 보드 ?�측??*(?�확??v2 / 배포?�측=
  canonical 453p = 불일�?구멍). [[project-litreview-vault-provenance-0804b]]

**?�⚡?�⚡?�⚡ 2026-08-04 (JMS 배포-중심 ?�벗 + robustness ?�망?�정 + FEATHer-MLP v2;
?�세 = `memory/project_jms_deployment_pivot_0804.md`):**
- **??robustness thesis ?�이?�로 ?�망**: `robust_feather_v2_mlp.csv` 6 mfg ?�주 ??  FEATHer-MLP(canonical+MLP+RevIN)???��?-7 �?**꼴찌** ??Dw(?��?) 1.211 vs peer
  1.006~1.034 = **7/7**, corruptMSE(?��?) 0.824 vs 0.688~0.749 = **7/7**(clean 좋아??  �?구제). no_revin(EWELD ??�� 비보??·robust-RevIN(median/MAD ?�패)·center(N2O ?�화)
  = **?�동?�는 보편 fix ?�음.** ??**JMS??robustness ??주장 ?�째 ?�거**(?�으�?"?�일
  ??강건"?�라 ??���? 주장 ???�면 ???�스?�해????. robustness = ?�후 ???�규???�기로만.
- **??JMS = 배포-중심 ?�벗(?�정)**: 기둥 = accuracy-per-param + Cortex-M3 ?�측. ?�라미터
  ?�당??"453p>41p??robustness"(죽음) ??**"?�확??**(FEATHer가 SparseTSF/MixLinear보다
  ?�확)�??�환.
- **???�확??= FEATHer-MLP(v2) #2**: `feather_v2_mlp.csv` ?�주(14?�이??6mfg+8LTSF,
  3seed, **MLP rank16 + loss mse**). mfg ?��?-7 OVERALL MSE: DLinear 0.668 > **FEATHer
  0.694(640p)** > CycleNet 0.720 > FITS 0.726 > DiPE 0.729 > MixLinear/SparseTSF 0.749.
  = **#2, 29�???DLinear 빼곤 ???��?, Steel 1??DLinear ?�침).** DLinear �?�� **79%가
  BlastFurnace ?�나**(D=28 고채?? MLP 채널?�합 무익; CI가 ?�버 ?�보??미�?�?.
- **??2-??구조(DLinear 처리)**: ?�확?�표=**?��?-7 ?��? ?��?**(DLinear ?�함 ??빼면 "?�기??  ???��?"=리뷰??즉살) / 배포??**MCU ?�어가??것만**(F103RB 128KB flash): ?�측 5�?  (persistence/SparseTSF/DiPE/FITS/FEATHer)+MixLinear(?�팅 ??. **DLinear ?�락**(H720??  138K??52KB>128KB = 물리??미탑??= dodging ?�니??발견).
- **???�래??베이?�라??*(GRU/XGBoost) 배포??추�? = ?�조 리뷰??credibility. 규칙 =
  ?�리?�링 금�?(**budget ??best config**) + **?�연?�기 병기**(조작 ?�님 증명 + "generic
  RNN?� sub-1K�???줄어??=?�리 ?�거) + **XGBoost??flash 바이?�축**(?�리??params 무의�?.
  코드 ?��? ?�음(미커�? [[project-jms-baseline-repositioning]]).
- **??LTSF = FEATHer #5(?�함)**: 0.406 vs FITS 0.376/SparseTSF 0.380/DiPE 0.380/DLinear
  0.399. MLP ?�어???�함 = ?�조 ?��??�계, LTSF ???�님. ??LTSF??accuracy-per-param??  불리(SparseTSF 41p<640p?�데 ???�확). **공정??OK**(FEATHer LTSF per-dataset ?�닝??
  베이?�라?�보????. 근데 **?�튜???�기 ?�당 = ?�중 미스매치**(arch가 L1+MLP?�이 ?�닝?�는??  지�?MSE+MLP). **목표="?�급 ?�반?�증�? not ?��?**(07-22: arch?�치 최적=degenerate·FITS
  �??��?); 리뷰??"mfg ?�버??" 방어??
- **??코드(미커�?로컬, push ?�요)**: `run_grid_ltsf_v2.py`(?�규 ??MLP+MSE-aware,
  **kernel?? 강제**(kernel=1=DTK 죽여 ?�형??degenerate 배제, FEATHer ?�체???��?),
  144config, 격리 CSV `results/grid_ltsf_v2.csv`, 기본 6 ?�채널 LTSF) + `build_paper_tables.py`
  (FEATHer ?�스 canonical?�feather_v2_mlp, params 453??40, **03_accuracy_ltsf.xlsx** ?�트
  추�?). **????baselines/__init__.py ??건드�?= ?�윕 ?�동?�장 ?�리�??�음 = ?�전 커밋**
  (?�버 grid pull ?�해 push ?�요).
- **???�버 ?�태**: **CycleNet·MixLinear LTSF 미실??*(baseline ?�중 추�???mfg�???. ??  orchestrator `--model`?� **?�일�?*(콤마 X, `--exclude`�?콤마) ???�로 발사:
  `python run_forecast.py --group ltsf --model CycleNet --exp_tag main --num_seeds 3
  --ngpu 2 --jobs_per_gpu 3`(그다??MixLinear). 기본 `results/fcst_results.csv` resume.
  ??3�?갱신(`analysis/0{1,2,3}_*.xlsx`).
- **???�음 ?�션**: (1) CycleNet/MixLinear LTSF 발사?�표 ?�생??(2) grid ?�구 push?�서�?pull??  ETTh1 ?�로�?576??1seed)??--summary` ?�정 (3) 게이???�과 ??6�??�장?�best per-dataset??  `_DATASET_OVERRIDES` 갱신??seed 최종 (4) 배포??GRU/XGBoost budget-best+?�연?�기+바이?�축)
  ?�업. [[project-jms-deployment-pivot-0804]]

**?�⚡?�⚡??2026-08-03 (체계??문헌리뷰 ???�벨??진단 + ?�후 방향; ?�세 =
`memory/project_lit_review_novelty_0803.md`, vault = `analysis/lit_review/`):**
- **?�기**: ?�용??"FEATHer=짬뽕(기존 ?�릭 조합), 방법 ?�벨???�다" ?�려 ??**Obsidian
  vault 체계??리뷰**(md+wikilink+graph, gitignored). PDF 1차출�??�독?�구조화 ?�트 1?�씩,
  **10???�료**: SparseTSF·FITS·MixLinear·DLinear/NLinear·RevIN·OnTheRoleOfRevIN·FAN·
  FredNormer·SAN·TS-Fault·SensorFault-Bench.
- **??결론(?�정)**: **??FEATHer 방법 ?�벨??= 0.** 부???��? ?�점(center=NLinear·CI=
  DLinear·?�터robust=FITS/MixLinear·FFT?�규??FAN). **FFT 게이??부�?*(07-31b 기여~0 +
  FAN???�리?�으�??????�망?�스???�??. 조립?� ?�벨???�님 = ?�용??직감 ?�확.
- **강건 벤치마크 2?�이 ?�점(방법 ?�니??진단?�구=?�리 방법?�벨???�협 X, ?�용?�군)**:
  TS-Fault(clean?�robust 반비례 ?=??.544) · **SensorFault-Bench(RevIN??clean ?�고
  degradation ?��? ???�리 07-31b 진단??benchmark로·우�??�메??wind SCADA???�점).**
- **?�규???�리 공간 ?�화**: FAN(NeurIPS'24)·FredNormer(ICLR'25?�사)·SAN(NeurIPS'23) ??  **clean 비정?�성** ?�?? **OnTheRoleOfRevIN**: "scale ?�거=predictive context 버림" ??  **center??N2O/EWELD clean ?�화??버그 ?�니???�려�??�레?�드?�프**(center 만능?�리??
  robust-RevIN???�반??. **RobustTSF(ICLR'24)**: MAE robust·MSE non-robust 증명(07-30
  L1?�MSE ?�고 ?�?? + regime=**training-?�염 vs ?�리 inference-?�상**=?�일 깨끗??차별??
- **???�일?�게 �?방법 ?�리**: **?�상 가?�성 ?�는 ?�력 ?�도?�서 reversible mean/std�?  within-window robust 추정, scale(context) ?��?, sub-1K.** 경계: 주파?�❌(FAN/FredNormer/
  SAN)·미래?�측??SAN)·across-dataset??FredNormer). ?�패근거=median(50%miss breakdown)·
  center(scale버림)=???�리가 ?�렵???�직 �?
- **??결정 (?�용??**: **FEATHer = ?��?�??�험 ????JMS ?�출**(?�설�????? ?�정).
  **?�후(?�음 ?�문) 2갈래**: ?�모?�링 ?�설�?"진단?�아?�텍�?, 부???�망?�스?? + **?�손??
  robust ?�규??*(model-agnostic=?�립 방법?�문 ?��?venue + ?�의 ?�벨??척추). 그림: FEATHer
  (JMS)?�②(?��?venue)?�①(?�품?� ?�설�?. ?�출=`analysis/lit_review/{FUTURE_WORK.md,
  gaps/SYNTHESIS_solution_approach.md, _MOC.md, papers/*.md}`.
- **??모드 ?�환**: 리뷰 마무�? ?�험(�??�버 FEATHer+MLP full rerun ?? ?�기하�?pending
  처리 모드�? [[project-lit-review-novelty-0803]]

**?�⚡?�⚡ 2026-07-31b (?�확??착�? + MLP ?�험 + RevIN robustness 진단; ?�세 =
`memory/project_mlp_revin_diagnosis_0731.md`):**
- **???�확??착�? (?��?-7, mfg ?�주 4 = Steel/WindSCADA/BlastFurnace/EWELD, 3seed×4H,
  ?��? mse?�습·mse채점):** FEATHer = **sub-1K ?�확??리더.** MSE ?�균 **3??7** (DLinear
  0.645[18.6K]·CycleNet 0.708[9.5K]�??? FEATHer 0.7095[453]) ??**진짜 sub-1K(MixLinear
  49·SparseTSF 41·DiPE 267·FITS 1.2K) ?��? ?��?, MixLinear??16?� ?��? ??** MAE 4??  (MixLinear/SparseTSF???��?). **07-30 L1/MSE confound ?�소 ?�정** (val_loss ?��??�로??  mse ?�습 검�?. ??MetroPT3/N2O 꼬리(FEATHer/MixLinear/CycleNet) 미완. DLinear �?**?��?
  BlastFurnace(D=28)+WindSCADA(D=15) 고채??* ??Steel/EWELD??FEATHer 2??
- **???�차 MLP ?�험 (?�규, `--use_residual_mlp/--mlp_rank` 기본 OFF, canonical
  byte-identical):** 게이???�현 h??**채널�??�??�� 비선???�차** `h+=W_up(GELU(W_down(h)))`
  (D?�rank?�D, +2·D·r; Steel 453??40@r16, **sub-1K ?��?**). **clean MSE: Steel ??%·
  WindSCADA ??% (3seed), BlastFurnace ~0**(고채?��? 채널?�합???�질채널 ?�어 무익). 추정
  FEATHer+MLP ~0.68 ??**CycleNet ?�치�?#2**. **robustness 중립**(Dw 본전, ?��??�차??개선).
- **????RevIN robustness 진단 (결정?? Steel 3-arm, ??MLP-on):** canonical FEATHer가
  ?��? �?**robustness 최악**(Steel Dw~1.24 vs ?�이�?1.03~1.06; 최�? severity??즉시 계단,
  impulse@0.05 Dw 2.17). **범인 ?�정 = RevIN, FFT 게이???�님.** gate-off(uniform) Dw
  1.202??.200(무�??? / **no_revin Dw 1.202??.076(?�이�?급락)**. **게다가 no_revin??clean
  MSE??최고(Steel 0.477, ??3%) = robustness+clean ?�중 ?�득.** 메커?�즘=?�상 ?�도?�로
  mean/std 계산?�impulse ?�파?�크가 ?�계 ?�염?�정규화·??��규화 증폭. (?�용??RevIN 가???�중,
  Claude 게이??가???�답.) **FFT 게이?�는 clean/robust 기여 ~0 = 값어�?미�?�?*(gate ablation
  ?�상 최초�???.
- **?�략 ?�환**: (a) **accuracy-first** ?�레?�밍(robustness-not-accuracy???�이?�로 ??��?�으??  RevIN-fix�??�당 부??가??. (b) **?�벨??척추 = "?�서고장 ??instance-norm(RevIN) 취약??  진단 + robust-RevIN(median/MAD) fix"** ??JMS(?�조?�?? ML?�벨??�???��)???? ?�조 문제?�
  ?�치. (c) **DLinear 추월?� BlastFurnace 고채???�나???�림** ??근데 mfg??**per-dataset
  ?�닝 금�?(??모델 default config ?��?** ???�일 canonical ?�키?�처 변경으�?고채???�리??  ?�채널 ??망쳐???�려?�). **?��?부??FFN)?�로 DLinear 쫓기=?�벨??0, 지??** (d) FITS가
  robust???�유=FFT�?**?�터(low-pass cut_freq=seq_len//4)�??�**(반응 ?�님)?�고주파 ?�상 ?�기.
- **???�음 (RevIN???�심)**: WindSCADA/EWELD�?no_revin ?�확??Steel?� ?�상??커서 ?�거 ?�득??  ???�음, 비정?�성 ?�이?�선 clean ?�칠 ?�도??*robust-RevIN???�반??*) ??**robust-RevIN(median/
  MAD) 변�?구현=?�문 기여 ?�심.** + MLP A/B EWELD/BlastFurnace ?�주 + ?�확??MetroPT3/N2O 꼬리.
- **??코드 (personal push): `ffbffde` MLP flag** + robustness MLP 배선 + **NaN-tolerant
  resume**(?�시 append torn-row ?�스) + **`--no_revin`/`--gate_uniform` 진단 ?��?**(기본 OFF).
  격리 CSV: `feather_mlp_ab.csv`(?�확??·`robust_mlp_ab.csv`(robustness). robustness??  **jobs_per_gpu 2 권장**(torn-row). robustness arm?� ?�습�?**flag ?�일**?�야(forward ?�치).
  [[project-mlp-revin-diagnosis-0731]]

**?�⚡??2026-07-31 ?�개편 (?��?-tier ?�구조화; ?�세 =
`memory/project_edge_tier_restructure_0731.md`):**
- **?�레?�밍**: JMS ?��?(IoT-J ?�동 X) + **IIoT ?�레?�밍�?*(?�이??교체 X, "?�조"??  "?�업 IoT ?��?" ?�술�?. **메인 비교 = ?��?-tier�? 천장(?�??=Pareto plot ?�으로만**
  (??�� X, 비교??X ??"??SOTA ?�냐" 방어 + accuracy-per-param ?��?). ?��? ?�의 =
  "published default?�서 ?�연???��?" 모델(?��? 축소=strawman 금�?).
- **?�인??14**: ?��?-7 (SparseTSF 41·**MixLinear 49**·DiPE 267·**FEATHer 453**·FITS
  1.2K·CycleNet-linear 9.5K·DLinear 18.6K) + 천장-7 (PatchTST~MDMLP 35K~1.6M, **기존
  ?�이???�사?�·재?�행 X**). **?�래??LSTM/GRU/TCN/XGBoost)=`_archive/` ?�카?�브**(리비??.
- **??MixLinear(ICLR'26, 49p) = FEATHer 최�? ?�협** ??dual time+freq, sub-1K SOTA 주장,
  FEATHer niche 직접 경쟁?? ?�확?�로 ?�기�?sub-1K ?�확??리더 주장 ?�고 robustness/배포�?
  **벤치마크 급함 = Exp1 �?관�?** MixLinear/CycleNet 배선(주기=per-dataset FFT 미러
  SparseTSF/TQNet, lr=?�문 ?�크립트 default 0.03/0.005; wrapper=debug print ?�제+lpf clamp).
- **FEATHer = canonical(d8/k7/B3+FFT period) + loss l1?�mse** (07-30 confound: mse?�습??  MSE???��?). **??14모델 train=mse/eval=mse = 공정.** arch/lr ?�치(grid/OFAT/mfg-search/
  lr-search) ?��? `_archive/`. **공정??감사 ?�료**(??모델 ?�문 default).
- **robustness = ?��?-7 × mfg-6** (LTSF ?�커 ETTh1/Weather ?�롭 ???�확?�표 커버리�? ?�치),
  4 fault�?report??quant ?�롭). **체크?�인??= 모델�??�더**(`checkpoints/<tag>/<model>/`,
  `migrate_checkpoints.py`�?2151�??�동 ?�료). **dispatch = 경량 먼�?**(`_MODEL_DISPATCH_ORDER`).
- **최종 ?�험 9**: 메인(1 ?�확?�L96·2 robustness·3 lookback ?��?7×mfg6×L{48,192,336}·
  4 ablation-clean 5�? / 가�?5 ?�이?�효?�cold-start·6 ablation-under-corruption·
  7 ?�너지envelope 계산) / ??8 Pareto·9 ??. **FFT 게이???�석??버림.**
- **???�버 ?�태**: **Exp1(mfg ?�확?? ?�GPU ngpu2)** + **Robustness(?��?7×mfg6, GPU0,
  FORCE_OMP=4)** 병행 RUNNING. Exp1 경량먼�???MixLinear?�FEATHer?�CycleNet ??체크?�인??  ?�성. robustness=체크?�인???�는 것만(멱등, ?�실?�으�?채�?). ??**박스 공유 ??FORCE_OMP
  ?�수**(�?orchestrator ?�기?�커?�로 ?�동?�?�동??oversubscription?�CPU?�레?�→GPU 80%??0%).
- **?�험 코드 = ?��? 준�??�료** (board ?�팅�??�음): `--train_frac`(?�이?�효??`ff139fd`)
  + `pareto_plot.py`(`19c83d4`) + `energy_envelope.py`(`93f0009`) + `run_robustness
  --ablation_axis`(ablation-under-corruption `9df1b40`). lookback=`--seq_len` 반복,
  ablation-clean=기존 ?�프?? ??**?�너지 발견**: "?�산=?��?1.5%=공짜"??NUCLEO sleep
  (비�??? 기�?; production STOP???�산 90% 지�???**진짜 ?�거="FEATHer 짧�? latency??  배터�??�래�?latency?�params)"**. **?��? 코드 = board C ?�팅**(MixLinear ?�선, config
  ?�정 ??WSL/보드 배치; 로컬 gcc ?�음). **병행 to-do = ?�고 리프?�임**(*_jms.md ??  ?�레?�밍?�IIoT/?��?tier/14모델/MixLinear/mse). push `e22236f`~`9df1b40`(personal).
  체크 명령: `results_table.py --data Steel,WindSCADA,EWELD` / `robust_summary.py --csv
  results/robust_results.csv`. [[project-edge-tier-restructure-0731]]

**?�⚡ 2026-07-30 발견 (L1/MSE ?�실 confound ???�세 =
`memory/project_l1_mse_loss_confound_0730.md`):** mfg ?�치 중간?��? �??�용?��?
"FEATHer가 clean ?�확?�로 ?�도?�으�??�쁘?? ???�인 진단 결과 **?�실/지??불일�?*.
- **FEATHer�?`loss=l1`, 베이?�라??11개는 ?��? `loss=mse`**(코드 `get_method_defaults`
  ?�인)?�데 ?�린 ?�확???�는 **MSE�?채점**. L1?� ?�상치에 robust ?????�차�?방치 ??  ?�파?�크 많�? 공정?�호(Steel ???�서 �??�차가 **MSE?�서 ?�곱????��**.
- **증거(?�주�?Steel/WindSCADA/EWELD, 같�? ?�측·지?�만 교체)**: FEATHer\*(?��?  min-config ?�라?? **MSE 채점 10.3/12(최하??** ??**MAE 채점 4.3/12(sub-1K 1??* ??  FITS·DiPE·SparseTSF·DLinear·PatchTST·TQNet·MDMLP ?��? ?��?). ??모델(TimesNet/LMS)?�
  ?�느 지?�든 ?�전???? ??"sub-1K가 500K-param???�린?�로 �??�긴?????�효, 바�???  ?�는 �?"**sub-1K tier ?�에??FEATHer 최강**"?�라????강한 ?�확??주장.
- **?�급**: ETTh1 그리??"clean 8/9?? 결론??MSE 기�??�라 **같�? confound** ??07-28
  "FEATHer clean �??��?" ?��????�??MSE 지??착시 ?�분).
- **?�용??결정 = L1·MSE ?????�학??비교.** `search_feather_mfg.py`??**`--loss
  {l1,mse}` 추�?**(커밋 `eb91d39`, personal push): `mse`??results/feather_mfg_search_mse.csv`
  ?�동 ?�우??+ exp_tag `search_mse_*` ?�두?????�던 L1 ?�치(`feather_mfg_search.csv`,
  `search_ds*`)?� **CSV·resume???�전 격리, ?�시 ?�행 ?�전**(l1 경로/?�그 무�?�?.
- **??07-30 발사**: �??�버 GPU0 `FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --loss mse --ngpu 1 --jobs_per_gpu 2`(324??. ?�정 =
  `--loss mse --summary`. ???�치 ??MSE·MAE 기록 ??**4분면(L1?�습/MSE?�습 ×
  MSE??MAE??**?�로 ?�드?�인 지?�·손??최종 결정, canonical?� 3seed 최종??
  ??**MSE-?�습 ?�치 결과 ?�직 ?�음**(방금 발사) ????10?? ??��?� L1-?�습�?MAE
  ?�채?�으로만 ?�인. [[project-l1-mse-loss-confound-0730]]
- **???�버 ?�태 (07-30):** �??�버 GPU1 = mfg 베이?�라???�윕 RUNNING(505/792 ??4%;
  Steel/BF/WindSCADA ?�주, EWELD 진행 �?heavy 3모델 꼬리[iTransformer 0·TimesNet 4·
  TimeMixer 9], N2O/MetroPT3 미착??. �??�버 GPU0 = FEATHer mfg ?�치 L1(135/324) +
  **MSE ?�치 ?�규(0/324)**. LTSF `fcst_results.csv` 거의 ?�주(?�드 ?�재). �?건강
  (NaN·zero·dup 0).

**??2026-07-28 결정 (?�조-가�??�험 ?�코???�정; ?�세 =
`memory/project_mfg_value_experiments_0728.md`):** ETTh1 그리???�주(360/360) ???�라?????�인 ??**FEATHer clean ?�확?�로 �??��? ?�확??*(test-peek ?�라??줘도
ETTh1 16�?�?1칸만 1?? 공정 val-?�택 9/13??. ?��?= robustness + 배포. �?배포 기둥??"3D"�??�다?�오???�다?�재?�닝 ?�이 ?�다) 강화?�려 ?�③ 검????결론:
- **??one-config-fits-all-sensors = ?�기.** ?�조??`_DATASET_OVERRIDES`??**?�원
  ?�문 default config + period 모델�?FFT ?�동주기** = 모두 "??config" ??FEATHer�?  강점 ?�님, 비교 ?�위 ?�음. (??full=canonical vs per-dataset-최적 FEATHer �??
  ?�조 그리??8.6k?�이??리뷰???�구 ?�에�? 지금�? 문장 ?�나�?강등.)
- **???�너지/배터�?envelope = 채택** (?�조-가�?1�?. ?�측(35mJ/추론·latency·
  ?�성?�류 30mA) 계산�? **???�습/보드 ?�험 불요.** ??NUCLEO sleep ?�측??금�?
  (11.4mA=ST-LINK 지�?비�??? ??production bare-MCU **?�이?�시??STOP(~14-24µA)**
  기�?. ?�문??**?�자(배터�??�명 projection "코인?� N??, sleep=?�이?�시??
  projection 명시) + ?�자(?�수 ?�측 "35mJ/추론, duty???�산 1.5%") ????** ?�심=
  latency?�param?�→FEATHer가 FITS/DiPE보다 배터�??�래감→?�너지×robustness Pareto.
- **??PdM 조기경보 리드?�??= NO-GO.** 구조??anomaly detection(residual?�경�??�라
  forecasting ?�체???�리�?AD-baseline flank ?�림 + small N(MetroPT 고장 3~4�? +
  ?�벨 ?�이?�라?�에 ?�음(???�문?�만). **?�기/future-work ??문장?�로�?** ?�조-가�?
  ???�너지 + 기존 ?�확??robustness/배포�?충분.
- **부???�인**: seq_len=96·H={96,192,336,720}; ?�조 ?��??�로?�콜(?�원 default+FFT
  period) 공정???�슈 ?�음 = settled. **Results ?�름 ?�정**: §6.1 ?�확??mfg+Pareto)
  ?��?.2 robustness?��?.3 배포?��?.4 ?�너지(???��?.5 LTSF(강등)?��?.6 ablation.
- **?�출�?*: `ltsf_baseline_results.xlsx` FEATHer* grid-ceiling ?�을 ?�주 ETTh1(360)
  +prelim h2/m1/m2 ?�라?�로 갱신(min-MSE config MSE/MAE/CORR, ?�이?�이???�계??.
- **???�버 ?�태 (07-28, 3 ?�윕 ?�시):** **�??�버 GPU1 = mfg 베이?�라???�윕
  RUNNING** (792 / **245 done** ??1%, FEATHer ?�외 11모델; Steel 먼�?·N2O/MetroPT3
  꼬리; Dropbox 0?��? ?�크지?? LTSF ??32/32 ?�주). **�??�버 GPU0 = FEATHer mfg
  arch ?�치 RUNNING** (`FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --ngpu 1 --jobs_per_gpu 2`, 324??27config×3ds×4H×1seed,
  격리 CSV `feather_mfg_search.csv`). **???�버 = FEATHer LTSF 그리??RUNNING** on
  ETTh1/h2/m1/m2+Exch/Weather. **CSV 3�??�라 ?�시 ?�전.**
- **??FEATHer mfg HP = per-dataset ?�닝 금�?(?��?·?�치?�단??canonical ?�택.** ?�치
  1?�로 (a) per-dataset best[진단+one-config-gap 부�? + (b) mean-rank canonical vs
  base d8/k7/B3 ??�? arch 27�?d_state{4,8,16}×kernel{3,5,7}×num_bands{2,3,4},
  k=1 ?�외), period=FFT·lr/λ 고정. ?�구 `search_feather_mfg.py`(personal `b9c9ed7`).
  ?�나�?`--summary`�??�정?�확?�히 ?�으�?canonical 교체, ?�니�??��?. robustness
  Plan B/sub-1K tier?�는 ?�이???�온 ?? [[project-mfg-value-experiments-0728]]

**?�⚡ 2026-07-27b 결정 (mfg lr ?�치 ?�기 ??default lr; ?�세 =
`memory/project_mfg_lr_default_decision.md`):**
- **?�조 lr ?�치(?�a baseline + ?�b FEATHer)�??�면 ?�기?�고 �?모델??  `_METHOD_DEFAULTS` lr???�.** 결정??증거 = `lr_search.csv` 44�?mfg ?�?�서
  default lr vs 5-lr ?�치 best-lr??val_loss 개선??**중앙�?0.1%**(?�균 0.4%,
  최�? 6.4%=MDMLP·Steel ?�나; >3% 1/44·>10% 0/44). best-lr??default?� 10�?  ?�라??val_loss??0.5%�??�직임 ??mfg??**loss ?�면??lr???�???�평**, ?�치
  best???�이�?
- **??BASELINE?�만 ?�용, FEATHer???�외.** 베이?�라??default??�??�문 ?�??  공표값이???�당?��?�? FEATHer???�리 모델?�라 **공표 default ?�음**(1e-3?� ?�리가
  ?�의�?찍�? �? ??"default ?�킵" ?�리 ?�립 ???? FEATHer??진행 중인 **그리?�서�?  (ETTh1/h2/m1/m2+?�장) 결과�?config ?�정 ???�용?��? ?�께 보고 별도 발사**(arch=
  canonical mean-rank, period=FFT, lr=그리?�기�?. FEATHer??sub-1K??병목 ?�님.
- **?�과: 베이?�라??mfg �??�윕(11모델, `--exclude FEATHer`) 즉시 발사 가??**
  `--num_seeds 3`(?�출?? ?�중??`--num_seeds 5` resume=�?5?�드 ?�문�? LTSF?� ?�일
  ?�책) + `--save_model`. 베이?�라??mfg lr?� `_DATASET_OVERRIDES`??lr ?�음?�default
  ?�백(period_len/cycle�?override), **코드 ?�정 불요**. heavy(iTransformer 0?�·OOM
  ?�꾹�? lr ?��??�멸.
- **LTSF??무영??* ??baseline?� �??�문 per-dataset 공표 lr ?��?, FEATHer LTSF
  ?�결 OFAT ???��?. ??결정?� **mfg 축에�?* ?�용, [[project-open-lr-policy]]�?  mfg?�서 ?��?
- `lr_search.csv`???�기 ?�니??**??결정??증거�?*�?보존(?�용??결정: ??�� X).
  LR_SEEDS 2?? TEMP ??`5398d89`)?� ?�제 무의�?mfg lr ???? correctness 무해,
  �?�� ??[2025,2026] ?�복). ??SparseTSF default(0.02)???�치 격자 밖이??공표값이???��?.
- **???�버 ?�태 (07-27b 발사 ?????�버 가??:**
  - **�??�버 = mfg 베이?�라???�윕 RUNNING** ??`--exp_tag main --save_model
    --exclude FEATHer --num_seeds 3 --ngpu 1 --jobs_per_gpu 3` (11모델×6mfg×4H×3seed
    =792????`fcst_results.csv` mfg ??. seed 3=?�출?? ?�중 `--num_seeds 5` resume=
    �?5?�드. `--save_model`=robustness 체크?�인??
  - **???�버 = 그리?�서�?RUNNING** ??ETT(ETTh1 ~?�주/h2·m1·m2 진행) + **Exchange??    Weather ?�장(병렬, ETT?� byte-identical 360격자; Electricity/Traffic?� ?�림+
    non-sub-1K???�킵)**. **lr ?�치 중�?**(mfg lr ?�기�?무의�?.
  - **?�출�?*: `results/ltsf_baseline_results.xlsx` 갱신 = 11 베이?�라?????�드?�재
    seed?�균) + **FEATHer\* grid-ceiling ??*(sub-1K 칸별 min-MSE=test-peek, ETTh1�?    ?�뢰/h2m1m2 preliminary/�?ETT 공백; 1?�초록�??�노?? FEATHer??ETTh1@336�?1??    =clean ?��??�확???��? ?�님 ?�확??. ??LTSF 베이?�라??11모델 32/32 ?�주(?�드?�재:
    4모델 5?�드·2모델 24×5+8×3·5모델 3?�드).
  - **?�음 게이??*: mfg 792???�주?�robustness ?�윕 / 그리???�주?�FEATHer config ?�정
    (kernel=1 arch-degenerate 지배라 canonical 교체 X ?�상)?�FEATHer mfg+LTSF 발사.
- **??dispatch ?�서 ?�일 (커밋 `67d69ea`??3c4d27f`, personal push):** `run_forecast.py`
  `_DATA_DISPATCH_ORDER` ??mfg+ltsf **?�일**, ?��? ?�이?�셋 먼�? ??�??�중(tier?�행×채널)
  ??빠른 결과 먼�?·?�린 꼬리가 ??막음. mfg: Steel?�BlastFurnace?�WindSCADA?�EWELD??  **MetroPT3?�N2O**(N2O 210k·MetroPT3 306k?�이 최장); ltsf: Exchange?�ETTh1/h2/m1/m2??  Weather??*Electricity?�Traffic**(채널 321·862가 최장). ?�제 **dataset-major**(???�이?�셋
  11모델 ???�고 ?�음). **�??�버 ?�용=?�던 mfg Ctrl+C(`37e4758`)??git pull`?�같?� 명령
  ?�실??resume+???�서)**. `run_forecast.py`�?push(?�래???��? ?�외).

**??2026-07-27 ?�데?�트 (MCU 배포 ?�측 ?�료 ???�랙 ?�일 ?�스 =
`deployment/EDGE_EXPERIMENT.md`, ?�차??= `deployment/board/BOARD_RUNBOOK.md`):**
- **보드(NUCLEO-F103RB) ?�착 ??5모델 ?�보???�측 ?��? ?�료** (기여 4�?"Cortex-M3
  배포 증거"가 추정?�실�?. CubeIDE ?�로?�트 = `C:\Users\lee\STM32Cube_projects\
  blink_test`(repo �?. golden 5모델 **bit-exact PASS**. **FEATHer @F103RB
  (p12/453, -O2 soft-float): 349ms / �?flash 17.6KB(learnable 1,812B+twiddle
  768B=§8 ?�증) / ~35mJ/추론 / ?�성 30mA·?��?11.4mA**. ?�이?�시: pers 0.08 /
  STSF 47 / FEATHer 349 / FITS 796 / DiPE 1312 ms (**?�라미터?�≠?�이?�시??*=?�거;
  ?�성?�류??5모델 ?�수 29-31mA=모델무�? ?�증). ??배포 ?�세??**CLAUDE.md??  ???�**(규칙) ???��? `EDGE_EXPERIMENT.md` 07-27 로그. **?�음 = FEATHer p24(489p)
  골든 ?�생?????�측???�버 config ?�정 ?? + ?�문 §7/§8 ?�·문??**
  [[project-deployment-layer3-plan]]

**??2026-07-26/27 ?�데?�트 (중간?��? + lr ?�도?? ?�세 = `memory/project_grid_search_pivot_0722.md`):**
- **?�버 ?�태(?�재)**: �??�버 = ??LTSF(`fcst_results.csv`) 계속. ???�버 =
  ETTh1 그리??`grid_search.csv`) + ?�a ?�조 lr(`lr_search.csv`) ?�시. CSV??  `Dropbox\WDSLab\05. ?�험\FEATHer\results\`�??�크??로컬???�람 가??
- **??LTSF 커버리�?**: 11모델 거의 ?�주(iTransformer�?Traffic/Weather 미완,
  TimesNet 2?�). ???�드 ?�재 ??6모델 5?�드(??·5모델 3?�드(07-16 ?�후). 최종??  ???�드 ?�책 ?�정 ?�요.
- **그리??커버리�?**: **ETTh1 95.1%(1,370/1,440)**, ?��? 70 = `grid_ds8_k9_*`
  꼬리(?�렬??마�?막이???�상). ETTh2/m1/m2 = 방금 ?�작(~33 config). ?�용??  **?�차 ?�실??*(ETTh1?�h2?�m1?�m2, `foreach($d){run_grid_search.py --data $d ??`).
- **ETTh1 부분판??95%·1?�드)**: ?�식??"grid BEATS base(val)"지�?**?�자가
  arch-degenerate** ??base d8/k7/p12/B3 = sub-1K 272�?val **160??중위)**, best
  sub-1K `grid_ds16_k1_p4_B3`(981p) ?val+0.0165·MSE??.6%, **top8??kernel=1
  지�?*(DTK=temporal kernel 죽임). clean?�확?�는 ?�그?�처 arch 뭉갤 ?�만????  **robustness-not-accuracy 증거지 canonical 교체 근거 ?�님, ?�동�?금�?**. 공정
  ??val?�택) FEATHer = ETTh1 **9/13??*·ETTh2 **10/12??*(?�SparseTSF), ?�라??  (test-peek)=ETTh1 1?��?�?반칙. ?��?= accuracy-per-param + robustness.
- **?�a ?�조 lr = 72%(237/330 mfg?�)**: 가벼운 7모델 ?�료, **무거??4�?  (iTransformer 0·TimeMixer/TimesNet Steel만·TQNet 부�?** ?�음(=OOM ?�꾹�?  모델??. ?�림?�인 = heavy × big-ROW(N2O 210k·MetroPT3 306k ?? 채널?� ?�음
  6??8) × 50ep × 5lr × 2seed.
- **??lr ?�도??(커밋 `5398d89`, personal push)**: `LR_SEEDS 2??` (**TEMP ??  lr ?�나�?`[2025, 2026]` ?�복 ?�수**) + `run_lr_search.py --model
  iTransformer,TimeMixer,TimesNet,TQNet --num_epochs 20 --jobs_per_gpu 1` =
  **4~6�?*. **?�전근거 = lr?� ?�택?�구**(출력=best-lr만→`_DATASET_OVERRIDES`,
  최종?�자???�서 50ep/3seed ?�산�???ep/seed 컷이 보고�?무영?? per-cell
  ?�립?�라 불공??X). 1seed??pull�?받게 커밋 ?�요(미커밋�? pull ????.
  **??lr `--summary` ??LR_SEEDS ?�복 + heavy best-lr??격자??0.0001/0.01)
  ?�닌지 ?�인.** ??미커�?로컬 ?�전 = ?�래??베이?�라??+ ?�커 ?�로�??�래�?

**?�⚡ 2026-07-22 ?�데?�트 (?�번 ?�션 ??HP 방향?�환; ?�세 = `memory/project_grid_search_pivot_0722.md`):**
- **?�람 루프 1 ?�행??*: ??OFAT `--validate`??--summary` ??**LTSF 8???�동�?*
  (커밋 `0768e9b` + push ijaehun; validate가 raw combo 4/8 기각 ??base+1�??�백).
  **λ flat ?�확????confound CLOSED**; **canonical mfg d8/k7/p12/B3 ?��?**.
  (CLAUDE.md 07-21 ?�스?�도 ??커밋???�봉??)
- **TPAMI �?발견 (?�직)**: ?�동�?FEATHer가 ETTh1/h2 **clean 8/9??*
  (SparseTSF 69p·DiPE 511p?�테 �?, ??TPAMI FEATHer보다 **~5-7% ?�쁨**,
  **SPK@720 ?�기?�측 주장 ?�현 ????*(같�? L=96). thesis(robustness-not-
  accuracy)?� 모순?� ?�니??"ETT clean ?�리/??720 강세" 주장?� **?�??*.
- **?�로브로 ?�인 규명 (로컬 ETTh1, 3?�드)**: lr(0.01 불안??·loss(MSE가 L1보다
  ?�쁨)·**AdamW+wd+lr0.01(TPAMI ?�확?�현)=0.410 ???�쁨** ??**?�떤 training
  recipe??TPAMI 0.373 ?�복 �???* ????TPAMI ETT = **legacy ?�티?�트**(??  ?�결????축들??near-best). ?�커??`--optimizer/--weight_decay` ?�로�?  ?�래�?추�?(backward-compat, **로컬 미커�?*).
- **결정: OFAT ??조인??그리?�서�?* (?�일 ?��? ?�버 = arch ?�호?�용).
  **grid > Optuna**(?��? ?�산공간+초소??모델+방어??병렬). **불�?�?= val ?�택**
  (test-peek 금�?). **공정??= 모든 모델 per-dataset ?�닝?�라 ?��?*(OFAT????  버림 ??리뷰??R2#23 민감?�분??+ num_bands/λ flat 증거�??�전???�임).
  **2�?*: LTSF=per-dataset 그리??/ MFG=**mean-rank canonical 1�?*(?��?
  "one sub-1K config가 모든 ?�서?? ?�토�?강화; period=per-dataset FFT, lr=lr?�치).
- **`run_grid_search.py` ?�작·push** (`0f6dd63`+`0c2576a`): arch 4�??�수,
  λ/lr/loss 고정, `--seeds`(1?�드 ?�택), resumable/`--check`/`--summary`.
  `results/grid_search.csv`. **0722b 그리???�장(`fbe077d`, personal)**: 극단-?�형
  코너�??�힘 ??d_state{**2**,4,8,16}·kernel{**1**,3,5,7,9}·period{**2,4**,6,8,12,24}
  ·num_bands{2,3,4}=**360조합**, GRID_PREDS=**[96,192,336,720]**(period ?��?
  gcd=48 ?�눠 ?�롭 0). 극단코너(d_state2/k1/p2/H720) ?�모???�한(kernel=1?� DTK�?
  FFT게이??kernel=5 ?�드코딩; period=1?� SPK degenerate???�외). 커밋=
  run_grid_search.py **?�독**(?�래???�로�?미커�??��?).
- **RUNNING = go/no-go (???�버 ETTh1 그리??**: **??FEATHer 중�???* ??ETTh1
  그리?��? **?�a 베이?�라??mfg lr(`lr_search.csv`)�??�시** ?�행 �?(CSV ?�라 ?�전,
  ?�일 GPU???�림·RAM 주시). **0722b�??�일??1,440??*(360×4H×1?�드). ??�� ETTh1/h2
  ?��? `fcst_feather_ltsf.csv`??**?�동�??�퍼?�스�?보존**(비교?? ??�� 금�?).
  �??�버 = ??LTSF(`fcst_results.csv`) 계속. 로컬 = ?�리 ?�험 ?�음. **?�버 ?�발??*=
  ??그리??Ctrl+C??git pull personal main`??run_grid_search.py --data ETTh1
  --pred_len 96,192,336,720 --seeds 2025 --jobs_per_gpu 1`; **resume가 ??H96 144??  ?��?**(??44?�새360)·?�머지�?채�?(purge 불요). `--summary`??같�? `--pred_len/
  --seeds`+`PYTHONUTF8=1`. **?�정**: **grid?�base(val)** ??LTSF-8+mfg-6 ?�장·?�동결�?  ?�⑤ ?�실??loop-1 ?�오??. **base ties/wins** ??OFAT 충분, ???�결 ?��?·robustness.
- ??**???�버 ?�a lr ?�꾹�??�용??보류)**: MDMLP_EIA/MetroPT3/H96가 CUDA
  `illegal memory access`�?죽음(async??`.item()`/`empty_cache`????것뿐; �?  ?�로?�스 컨텍?�트�??�염, 그리?�는 �??�로?�스??무�?). ?�력=?�일32GB GPU�?  그리???�a 최중??MDMLP 1.6M)???�눠 ?�서 메모�??�박. 가짜행 ?�음(exit1+
  MASK-NAN-GRAD 가?�→missing ?��?, resume ?�시??. **?�중??�??��?격리+
  `CUDA_LAUNCH_BLOCKING=1`�??�현**(?�독?�과=메모리→?�a jobs 줄이�?/ ?�독?�패=
  모델·?�이??버그).
- ??**미커�?로컬**: ?�래??베이?�라??list_models 15) + ?�커 ?�로�??�래�?  (--optimizer/--weight_decay). ?�래?��? ?�전??push 금�?(?�윕 ?�동?�장 ?�리�?.
  (07-23: throwaway `results/probe_*.csv` + repro??CSV ??��, LTSF ?�현결과??  `results/ltsf_baseline_results.xlsx`�??�리 ??MSE/MAE/COR ?�점�?+ Coverage ?�트.)

**??2026-07-16 ?�데?�트 (?�약 ???�세???�래 블록/Decision history):**
- **베이?�라??5?? ?�드 (교수 결정)**: `--exclude FEATHer` ?�윕??`--num_seeds 3`
  ?�로 ??�� ?�출 ?�선, **리비?�에??`--num_seeds 5`�?보강**(3?�드=[2025,2026,
  2027]???�드??resume???��? 2개만 채�?; base_seed/exp_tag 불�? ?�수). **??LTSF??  3?�드�??�발?�됨.** ???�조 �??�윕?� FEATHer ?�함 ?�째�?`--num_seeds 3`(메인
  ?�이�?12모델 ?�드 ?��???. ?�a lr?� ?�래 2?�드 고정?�라 무�?. **코드 ?�정
  ?�음**(CLI ?�래�? ?�운?�트�??�구 ?��? `nunique` ?�적 집계, wilcoxon?� ?� ?�어
  ?�링이??3?�드???�의??충분). ??OFAT ?�택?� 3?�드�??��? �?�??�정??.
- **?�고 ?�스???�트 ?�작??(?�버 무�?)**: `manuscript/drafts/*_jms.md` 5�?  (abstract/intro/related/datasets/methodology+theory)�?mfg-6 + A+C 고속-"?�리"
  ?�레?�밍(?�기�? ?�리??slow-process ?��?) + **robustness 보수 ?�술** + SWIFT
  차별??+ half-RevIN + ?�론 ?�림(R8 placeholder Theorem 3/4/5 ?�거)?�로 갱신.
  ??`0X_*_draft.md`??IoT-J ?�카?�브. ?�세: `memory/project_manuscript_jms_drafts.md`.
  ?��? ?�스?? Conclusion 리프?�임 + Appendix 증명 2�?+ Algorithm 1 검??
- **고속 ?�레?�밍 ?�정**: HR-SHM(고속 ?�싱?�다?�텝 ?�측, decimation???�벤?��?
  aliasing) = ?�재 문헌?�나 ?�리 ?�이?�는 slow-process ??**?�기 ?�용�?*, ?�리
  ?��? regime?�라 주장 금�?("horizon=step count" ?�리�?봉합). EWELD = 금속?�업
  ?�력**부??*(??극한기상 부??) + 극한기상 ?�벤??�?

**??지�??�버 ?�태 (07-18b 2-?�버 ?�윕-분할 ??체인???�일 ?�스??`ROADMAP.md`):**
**?�용 GPU 1개짜�????�버 ?�보 ???�윕�?CSV 분리 ?�용??병렬 분배**(?�용??결정). ?�②?�a??CSV가 ?�라(`fcst_results`/`hp_search`/`lr_search`) ?�시 ?�전.
- **�??�버 (Linux, 공유 128GB, GPU1)** = **??LTSF** (`FORCE_OMP=5
  CUDA_VISIBLE_DEVICES=1 python run_forecast.py --group ltsf ??--ngpu 1
  --jobs_per_gpu 2 --exclude FEATHer --num_seeds 3`). `fcst_results.csv`.
- **???�버 (Windows, ?�용 32GB, GPU 1�? env `lee`, `C:\Users\user\Desktop\
  FEATHer`)** = **??FEATHer OFAT + ?�a ?�조 lr ????* (CSV ?�름 ???�시 ?�전).
  �??�버??멈춰?�던 hp�?`results/hp_search.csv` 복사???�사(resume). ?�일 GPU
  ?????�션 최소: **??OFAT = `python run_hp_search.py`(jobs 1 ?��? ??elec/traffic
  ?�커가 RAM 무거?�)**, **?�a lr = `python run_lr_search.py --exclude FEATHer
  --jobs_per_gpu 3`(mfg??RAM 가벼워 병렬 3�? GPU 10%·RAM ~8GB ?�유???�게 ?�도
  ?�버)**. RAM?� ??OFAT가 Traffic(862ch) ?�을 ?�만 조심(`free`). ?�a??mfg
  data/ + `lr_search.csv` ?�송 ?�요(뱅크 Steel/EWELD/BF resume).
- **?????�버 `bash`??bash 4+ ?�님** ??`setup_baselines.sh`??`declare -A`가
  조용???�패("division by 0" ??clone 0개인??"All present" 가�??�공). **11�?  upstream repo??cmd?�서 ?�동 `git clone --depth 1 <url> baselines/<N>/<N>-main`**
  (URL = `setup_baselines.sh`??REPOS). reformer_pytorch???��? ?�치?? ???�면
  베이?�라?�이 upstream import?�서 죽음(?? DiPE_Linear `No module 'timeprophet'`).
- **??규칙: 같�? CSV????박스?�서�?** `lr_search.csv`=???�버�?�??�버 lr 금�?),
  `hp_search.csv`=???�버�? `fcst_results.csv`=�??�버�? 최종??�?CSV�???  results/�?모으�?`check_duplicates.py` 검�? `FORCE_OMP` env = 커밋 `880c83e`
  (worker??OMP/MKL ?�레??고정 ??박스 공유 ?�만 ?�요).
- **??07-19 `d56aa05`(MASK-NAN-GRAD) ?�시???????�버 pull ?�수, ?�히 ??  ?�버??pull ???�a ?�행 금�?**(마스??3종이 가�?0.0?�로 ??기록??.
  pull ???�린 lr_search.csv(844?? 배치 ??`run_lr_search.py --exclude FEATHer`
  ?�발????resume??마스??3�?미착??모델 ?�충??
**?�음 ?�람 루프(루프 1)**: ?�·④a ?�료 ??`--validate`(32) ??`PYTHONUTF8=1
--summary` 출력 ?�달받아 LTSF 8???�동�?+ 07-06 diff 기록 + canonical mfg
?�키?�처(d8/k7/p12/B3) ?�확??+ λ 최종 ?�정 ??커밋 ??pull ????FEATHer LTSF
160 + ?�b FEATHer lr 발사.

**Positioning (settled):** Fresh **JMS** submission (TPAMI reject ??IoT-J ??JMS manufacturing pivot). Main table = **6 manufacturing datasets** (Steel ·
WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD), LTSF-8 demoted to a
generalization section, short-horizon PdM (CMAPSS/CMAPSS3/PMSM) **NOT in the
paper** (lr-banked shelf, `--group cmapss`). FEATHer's 453p > SparseTSF 41p is
justified by **ROBUSTNESS, not accuracy** (OFAT num_bands FLAT on clean).
Framing guard: NOT "more params = more robust" but "41p linear lacks headroom;
FEATHer's inductive bias buys graceful degradation at sub-1K". R3's
"long-term-on-edge is contrived" attack resolved via **A+C framing**: (A)
industrial edge NEEDS on-device long-horizon (air-gap/OT isolation, per-sensor
scale, outage autonomy ??our datasets are the evidence), (C) edge = param
budget, horizon = task, orthogonal axes; "why 1K" ??accuracy-per-param Pareto
frontier. 12-model baseline lineup unchanged.

**??CODE AUDIT (2026-07-14c) ??full-repo review done, findings in
`manuscript/notes/code_audit_2026-07-14.md`. ??THE CRITICAL BATCH IS FIXED
(same session, all validated end-to-end):**
- ??**P0-1 robustness override plumbing**: `run_robustness.py` dispatch now
  mirrors the training-time `get_method_defaults`+`get_dataset_overrides`
  merge ??`--model_overrides`; the worker gained the flag +
  `_apply_overrides()`; `gate_logging.py` applies the same merge itself.
  Verified live: trained ETTh1 d16/k3/p8/B4 ckpt loads + clean MSE
  reproduces training-time test MSE exactly.
- ??**P0-2 corruption seeding**: salted `hash((ft,sev))` ??  `utils/noise.corruption_seed()` (deterministic index-based). Verified:
  re-run reproduces corrupt-MSE to 6 decimals across processes.
- ??**P0-3 robustness resume**: `load_robust_set` counts DISTINCT
  (fault,severity) pairs; worker skips conditions already in the CSV
  (per-condition resume ??no duplicate rows on retry). Verified via
  truncate-and-refill test.
- ??**P1-1 spectral loss band order** fixed (`bands[S-1-s]` ??POINT owns the
  highest slice, LOW the lowest). Synthetic check: correct placement ??loss
  0.0, swapped ??2.0. Frozen LTSF OFAT rows kept (λ axis was flat).
- ??**P1-2 Weather split** ??standard 0.7/0.1/0.2 (legacy hard-coded branch
  + dead PM/PEMS_BAY/METR branches deleted). ALL pre-fix Weather rows are
  invalid ??purge + re-run (see server procedure below).
- ??**P1-3 RMSE** = sqrt(global MSE) in both workers (+ NaN-aware CORR/R2
  accumulation). Old rows: derive sqrt(MSE) at table time.
- ??**P2-1 worker exit code**: training worker now exits 1 if any seed run
  failed (kills the reformer-style silent-skip class). **P2-2 CSV header
  race**: atomic `open(mode="x")` create-with-header in both workers.
- ??**P2-3..8 + P3 batch (2026-07-14d) ??AUDIT FULLY CLOSED** (23-check
  smoke suite passed + all 4 orchestrators `--check` exit 0; details in the
  audit doc UPDATE 2 block):
  - P2-3 check_duplicates schema-detects the robust CSV (key extends with
    train_exp_tag/fault_type/severity ??`--fix` now safe on it). P2-4
    explicit FEATHer CLI arch flags beat `_DATASET_OVERRIDES` (flags default
    None; sweep behavior byte-identical). P2-5 best_state=None raises (no
    row/ckpt; exits via P2-1 path). P2-6 mCE drops no-ref + model-incomplete
    cells for ALL models (warns; emits n_cells). P2-7 ablation_table = one
    table per dataset (no raw-MSE pooling). P2-8 lr/hp `--summary` flag seed
    under-coverage; hp combo "complete" requires full seeds.
  - P3: prep `_fill_gaps` gap policy (linear interp ??~1 h, hold-last
    beyond ??no lookahead/linear bridging of outages; audited max-gap
    asserts); BlastFurnace winsorize bounds from the TRAIN region (also
    finally clips the late Tp4 ~3311 sentinel: hi 3313??55); EWELD D=14
    assert; metrics.py np.divide out= init; severity tables +
    corruption_seed ??torch-free `utils/severity.py` (noise.py re-exports,
    robust_summary shares the object, utils/__init__ now lazy PEP 562);
    FEATHer half-RevIN (means detached, stdev not) KEPT + documented as
    deliberate (changing it would invalidate all frozen OFAT rows); stale
    docstrings/comments swept.
  - ??**DATA REGENERATED** under the new gap policy ??cells changed:
    MetroPT3 16.6% / N2O 2.7% / WindSCADA 1.1% / BlastFurnace 0.9% /
    EWELD·Steel 0%. Pre-fix copies kept at `data/<name>/data.csv.preP3bak`.
    ??transfer the 4 changed data.csv to the server and purge/refill their
    lr_search rows (added to SERVER PROCEDURE below).

**??SERVER PROCEDURE after pulling the fix commit** (order matters):
```bash
git pull   # (at a natural pause; running workers finish on old code harmlessly)
# invalidated rows ??resume refills them:
python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply
# (07-14e 결정 ??λ ?�라?�스 부분퍼지�??��? FEATHer OFAT ?�면 ?�실??
# hp_search�??�째�?비우�?run_hp_search.py가 512???��? ?�탐??# (교정??spectral loss + ?��? Weather split ?�래?�서):
python tools/audit/purge_rows.py --csv results/hp_search.csv --model FEATHer --apply
# (P3 data regen, 2026-07-14d) transfer the 4 regenerated data.csv files
# from local (WindSCADA / BlastFurnace / N2O / MetroPT3 ??Steel/EWELD
# unchanged), THEN purge their lr rows so lr is re-selected on the new data
# (~960 of 2,160 lr runs; MetroPT3 changed 16.6% ??too big to caveat away):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,BlastFurnace,N2O,MetroPT3 --apply
# then just re-run the same sweep commands ??resume fills the gaps.
# Weather + lambda OFAT slices (hp_search) refill via run_hp_search.py;
# re-freeze the Weather row in _DATASET_OVERRIDES from --summary afterwards.
#
# (MASK-FIX, 2026-07-15) observed-only scoring ??transfer + re-refill:
#   1. transfer the 3 regenerated data.csv (WindSCADA / N2O / MetroPT3 ??now
#      hold-last, no interp) AND the 4 companion mask.csv (WindSCADA / N2O /
#      MetroPT3 / BlastFurnace) to data/<name>/.  (EWELD unchanged: 0 imputed.)
#   2. purge + refill lr for the 3 whose data changed (BF's 1-cell mask is
#      below noise -> its rows stay valid; EWELD unchanged):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,N2O,MetroPT3 --apply
#   (this SUPERSEDES the 07-14d BlastFurnace lr purge line above ??BF no longer
#    needs a refill under MASK-FIX; the other 3 do, for both the P3 regen and
#    the interp->hold+mask change.)
```

**Evidence state (verified from hp_search.csv)**: only **num_bands is
confirmed FLAT on clean** LTSF-8 (B2/B3/B4 mean MSE 0.4505/0.4418/0.4457).
Gate on/off clean effect UNTESTED (needs ablation sweep); ALL robustness
UNTESTED (sweep unrun). The robustness thesis = 1 confirmed fact + an unproven
bet ??the robustness sweep is the #1 risk.
??**λ-flat confound ???�소 방식 격상 (user 결정 2026-07-14e)**: λ ?�라?�스�??�는 ?�??**FEATHer OFAT 512???�면 ?�실??* ??모든 FEATHer HP가 교정??loss + ?��? Weather split ?�래?�서 ?�색?�어 provenance caveat ?�체가 ?�멸.
07-06 ?�결분과??diff�?기록??�?(?�게 ?�르�?�??�체가 버그 ?�향??증거).
canonical mfg ?�키?�처(d8/k7/p12/B3)???�집계로 ?�확??(sub-1K ?�약??지배적?�라 ?��? ?�상). Memory: project_lambda_flat_confound.md.
All audit-fix sites are tagged in-code ??`grep -rn "AUDIT-FIX"` lists every
fixed location with its finding ID.

**Server state**: 07-18b **2-?�버 ?�윕-분할** ??�??�버(Linux, 공유, GPU1)=
**??LTSF**(`fcst_results.csv`), ???�버(Windows, ?�용 GPU1, env `lee`)=
**??FEATHer OFAT**(`hp_search.csv`) + **?�a ?�조 lr**(`lr_search.csv`) ????(CSV ?�라 ?�시 ?�전; ????블록 + ROADMAP.md 보드).
**??07-20 ???�버 GPU ?�고+?�리**: env `lee`??torch가 `2.8.0+cpu`(CPU ?�용
빌드)�?깔려 ?�어 `cuda.is_available()=False` ???�·④a가 **?��? CPU�?* ?�아
GPU 0%. `pip install torch==2.5.1 --index-url .../cu121`�?교체(`2.5.1+cu121
True 1` ?�인). 로컬 `feather`?� ?�일 버전(병합 ?�합??. CPU�???OFAT/lr ?��?
**?�택???�윕?�라 ?�기 불요**(최종 ?�자??????GPU ?�실??. 발사 ?�인 = ?�커
`Using device: cuda:0` + `nvidia-smi`??python ?�로?�스(util ??? �?초소??모델?�라 ?�상). ?��? = `memory/reference_new_server_env_lee.md`.
**07-20 Ctrl+C ?�치 `37e4758`**(personal push): `utils/dispatch_queue.py` ???�일 Ctrl+C가 ?�윕 ?�리 ?�체 종료(stop ?�벤??+ in-flight ?�식 ?�들 추적 ??`subprocess.run`??Popen`, ?�커가 ??????집고 ?�던 ?�식 terminate, 고아 방�?).
**???�버 pull ?�요.** (기존??Ctrl+C가 죽인 ?�식??"?�패 ???�로 보고 ?�음 ?�을
?�워 �??�마??Ctrl+C ?�야 ?�음.)

**Local git**: main fully PUSHED to personal/ijaehun **through `f906d1a`**
(07-23). 07-20 ?�후 ?�시�? 그리???�구(`fbe077d` 그리?�확?��?f569e04` summary
pred_len ?�터·`ed4549e` ?�성-only+`--sort_by`·`e7924ee` `--per_horizon`) + **numeric
버그 3?�스**(`fb09aeb` dispatch Popen가?��?15eba56` mCE skipna·`f906d1a` RMSE=?�집계MSE).
**???�버 pull ?�요**(?�는 ?�윕 무영?? ?�음 발사/?�생?��???. Push policy:
**ijaehun (personal) ONLY**, not WDSLab, unless asked. 미커�?로컬 = ?�래??베이?�라??LSTM/GRU/TCN + run_classical.py + `baselines/__init__.py` M) + ?�커 ?�로�??�래�?--optimizer/--weight_decay) + 검?�수??E ????체인 ?�난 ??별도 커밋(?�으�?list_models=15�??�윕 ?�동 ?�장).

**??NEXT**
1. ??~~Apply audit fixes~~ ??ENTIRE audit closed 2026-07-14d (P0/P1 in
   cfd90c5, P2-3..8+P3 in the close-out commit). Nothing left in the audit
   doc.
2. **Server (07-18 ?�차 ?�일-GPU 모드 ??ROADMAP.md가 ?�일 ?�스)**: 3-?�시
   ??GPU 1 ?�차�??�환. 지�?**??LTSF ?�행 �?* ???�나�?**??OFAT ???�a lr**
   ?�나?? ?��? 체인: **?�·④a ?�료** ??`--validate` ??`--summary` ?�달 ??   (Claude) 8???�동�?+ 07-06 diff + canonical ?�확??+ λ ?�정 ??커밋 ??   pull ????FEATHer LTSF 160 + ?�b FEATHer lr ??lr `--summary` MERGE ??mfg
   �??�윕 1,440 (`--save_model`) ??robustness ??`robust_summary.py` +
   `gate_logging.py`.
3. **A+C Intro/Motivation draft** (`manuscript/drafts/`) ??needs air-gap /
   OT-isolation citations; scope to genuinely-local settings.
4. **Still-missing analysis code**: component ablation UNDER corruption
   (gate-off/band-off variants through the robustness sweep) + Pareto plot
   generator (MSE-vs-params frontier).
4b. **JMS ?�래??베이?�라??(구현?? 미커�?미활?�화)**: LSTM/GRU/TCN
   (`baselines/`, list_models 15) + XGBoost (`run_classical.py`). 커밋 ??   **??체인 ?�난 ??* ?�버 ?�성??15모델 ?�윕 + `run_classical.py --group
   mfg`). ARIMA/LightGBM·배포?�측(#1구멍)·SWIFT문단?� ?�속. ?�세 =
   ROADMAP.md "JMS ?�포지?�닝" + `memory/project_jms_baseline_repositioning.md`.
5. ??~~Push to ijaehun~~ ??done 2026-07-14d (through 59c85da).
6. Raw-data cleanup under `data/`: DECIDED skip (raw = regen safety net; drop
   PMSM raw first if disk pressure).

### Decision history (compact ??full blocks in `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-07-14.md`)

- **2026-07-23 (TSLib ?��? ?��????�이?�라??clean + LMS EAAI 발견 + xlsx ?�리)**:
  (1) **TSLib(thuml/Time-Series-Library) shallow clone�?eval/train/metric/data
  4�??��? ?��?*(?�브?�이?�트) ??**버그 0, ?�확???�심 ?��? ?�치**. ?�히 **metric
  scale = ?�규??공간(????--inverse 기본 off) ???�리 ?�자 문헌비교 가???�정**;
  best-val ?�택·split(ETT 12/4/4·Custom .7/.1/.2)·train-only Scaler·공유5모델 HP ?�치.
  ?�?�험 ?�탈 3�?val-loss ?�플가중평�?FEATHer val=L1/train drop_last=True=?�TSLib)
  ?��? ?�도?�·방?��??�·조치불?? DLinear/PatchTST ?�본 ?�성?�만 로컬 미확?�이??  **?�버 ?�행?�공+문헌범위 ?�자�?경험???�인**. clone=`C:/Users/lee/tslib_ref`(repo�?5.4M).
  (2) **LMS-AutoTSF = EAAI(Elsevier Q1) 2026 게재 ?�인**("?�confidence-aware??,
  arXiv 2412.06866?� ?�리?�린?? ??R4 방어 peer-reviewed 3?? **?�현=공식 기본 base
  `LMSAutoTSF`(V2 confidence-aware???�떤 forecasting ?�크립트??미배?? ???�당, ?�실??  불요**. ?��??�트�?venue 갱신. [[project-baseline-lineup]]. (3) LTSF ?�현결과�?  **`results/ltsf_baseline_results.xlsx`**(병합?�더+best강조+Coverage 2?�트)�??�리,
  throwaway probe CSV ??��(??_reference/`??TSLib clone 보�?). (4) **?�습?�커+?�이??  로딩+utils ???�일 버그?�트(?�브?�이?�트 2??**: ?�자 조용??망칠 버그 **0**
  (마스?�·metric?�적·windowing·scaling·CMAPSS·seed·losses밴드?�서·noise?�상?�학·
  severity?�드 ??clean). **?�버�?1�??�정·?�시 `fb09aeb`**: `dispatch_queue.gpu_worker`가
  `Popen` ?�외 ??job??조용???�롭(failed 미기�??�launch+wait try/except 가?�로
  failed 기록+계속. ???�버 pull 권장(?�는 ?�윕 무영?? ?�음 발사부??. ?�여=?�람?�인??  minor(CORR/R2 마스??배치가중·Scaler가 outage hold-last??물듦·Electricity timeF 계산??��).
  **(추�? ?�운??** config/모델 ?�러?�터(`baselines/__init__.py` override?�우??82?��?
  wrapper·FEATHer SPK/DTK/denorm)=**clean 버그0**; ?�운?�트림서 **2�????�정**:
  `15eba56` robust_summary **mCE skipna 버그**(fault�??�째 ?�락 모델??3축평균으�???��
  조작?�skipna=False+경고, robustness ?�드?�인?? + `f906d1a` main_table **RMSE=??집계MSE)**
  (mean?�MSE?�√meanMSE, Jensen 과소보고 ?�소·TSLib per-run ?�치·RMSE std??point?�라 ?�략,
  MSE/MAE?�만 ±std). **?�번 검증서 numeric 버그 3�??�음(dispatch·mCE·RMSE), robustness
  ?�윕 ?�직?�라 ?�험 무영?? paper ??뽑기 ???�제 ?�정.** lr-search가 period-tied 모델??  default period�???문서?�된 caveat(?�고 ??문장).
- **2026-07-21 (?�이?�셋 6�??�확??+ 베이?�라???�외?�유 ?�일 + CycleNet ?�정
  ???��? ?�스??결정, 코드·?�이?�·서�?변�?0)**: (1) **mfg-6 ?��?, MetroPT3
  KEEP** ???�롭 ?�보�?4???�고?�으???��??�보 ?��? ??PMSM=2Hz persistence-
  trivial / TEP=?��? / flotation=공정중복·?��? 07-06 ?�스 / ETT=generalization+
  robustness ?�중?�용 / Korean10=감사리스???�력?�러?�터중복). MetroPT3=코드
  no-op(?�롭?????�음)+?�전기계 ?�산?�래?��? 채우??�? **?�정기�? ?�레?�밍=
  "비트리비??persistence-감사)+?�양???�측, 결측=?�로?�콜-handled"**(??무결??  threshold·?�메?�순??금�? ??MetroPT3/WindSCADA가 ?�웃?�이?�라 ?�기모순).
  [[project-mfg6-metropt3-decision-0721]] + ?�이?�셋 **기각 로그**
  [[reference-dataset-reject-log]](?�이?�셋 ?�의 ?�개 ??먼�? ?�질 �?. (2)
  **베이?�라???�외?�유 self-consistency ?�스**(CLAUDE.md "Excluded from
  benchmark" 반영): S-Mamba "no MCU port"??*CUDA커널 ?�현불�?**, TimeCMA
  "5 orders"??*?�전?�습=?�른 ?�원?�래??*, SWIFT "18k"??*official code 미성??
  FITS tie**(DLinear 18k가 ?�에 ?�어 sub-1K ?�리 즉사), 방어문단 ?�작?? seq_len=
  96 "LTSF convention"??*배포?�로?�콜+MCU ?�력버퍼 바이?�예??*(lookback�??�일
  명시). **size???��? ?�외기�? ?�님 ??41p??.6M ?�팬??목적**. (3) **CycleNet/
  RLinear=cite-not-baseline**(중간-tier·미요�?�게?�팅; "period 모델?�라"???�기
  ??SparseTSF??period; ?�칙???�함가?�이???�구/1차스?�강?�면 즉시 escalation).
  (4) **robustness 방어?�험 ?�설�?*: ablation-under-corruption=**1�????�량
  confound-closer**(gate/band-off ??53p) / 2×2 ?�량?�왑=보강(SparseTSF period-
  ?�브?�·FEATHer num_bands?? clean?�확???��? 보고�?detune반론 봉합) / ?�중?�로?�콜
  (메인 96 + 부�?literature lookback, **SparseTSF@720 ?�수**) / int8=limitations
  ??문장(리비?? / **?��? 1�??�윕 ?�과?�인??게이??*. [[project-baseline-lineup]]
  [[project-robustness-justification]] 07-21 블록. **?�인??12·?�②???�윕 무영??*;
  체인 ???�래??4�?LSTM/GRU/TCN+XGBoost)�??�제 추�?. CLAUDE.md ?�스?�분 미커�?  (?�음 ?�고/문서 커밋???�봉).
- **2026-07-20c (??LTSF ?�윕 sanity-audit ???��? clean)**: ?�던 LTSF CSV ?�음.
  TPAMI Table 3(ETTh1/ETTh2) ?�현 MSE·MAE ?�부�?±0.01(FITS/TQNet ?�일�?
  PatchTST ?�재가 ??좋음=개선, DLinear/ETTh2=?�드불안???�성), Weather ?�효
  (07-14+ ?�실??, RMSE 07-15+�??�상(????sqrt(MSE) ?�도), HP?�싱 ?��?
  (7 per-dataset 공식 / 4 ?�일 default = ?�쩔 ???�고 ?�답), split ?��?
  (ETT 12/4/4, ?�머지 0.7/0.1/0.2), 공정??OK. **?�험 ?�계 문제?�음, ?��? �?  ????것뿐.** ?��? = `memory/project_ltsf_baseline_verification_0720.md`.
  **decimation-vs-native ?�험 = ????*(?�중주장 ?�님+slow-process????��
  리스????리비??카드; ?�트�?anti-decimation?� ?�용+?�다?�으�?봉합).
- **2026-07-20b (???�버 GPU ?�리 + Ctrl+C ?�치)**: ???�버 env `lee` torch가
  `2.8.0+cpu`???�·④a가 CPU로만 ?�던 �?발견 ??`torch==2.5.1+cu121` ?�설치로
  ?�리(`2.5.1+cu121 True 1`). CPU ?��? ?�택?�이???��?. `dispatch_queue.py`
  ?�일-Ctrl+C 종료 ?�치(`37e4758`, personal push, ???�버 pull). ???�버 ?�실??  = ?��??�A `run_hp_search.py --jobs_per_gpu 3` / ?��??�B `run_lr_search.py
  --exclude FEATHer --jobs_per_gpu 4`(RAM ?�유�?3/4 ?�커; ??Traffic 겹칠 ??  RAM 주시). ?��? = `memory/reference_new_server_env_lee.md`.
- **2026-07-20 (코드/?�로?�콜 검???�정 A~E ?�료)**: ?�브?�이?�트 검??A/B
  (`manuscript/notes/review_findings_2026-07-19.md`) 발견 10�??��? ?�정 ??  **A `1e078d7`**(#1 test/robustness ?�로카운?�→?�패·#2 pred-NaN 가?��?9
  robustness resume ?�손가?��?10 override ?�로?�콜�?블록리스?? + **B
  `c9ddd94`**(#3 ?�역 관�??� MSE/MAE 추정???�일: 배치-bs가중→nansum/finite-
  count, persistence·classical�??�치, 무마?�크 byte-identical) + **C·D
  `bbe6067`**(#4 main_table RMSE=?�MSE ?�별 ?�도·#5 `--seeds` ?�터+?�드??불일�?  경고·#6 robust_summary dw/μ_rel/AUC 공통-?� 가?? ?�버 ?�시. **E??로컬
  미커�?*(#7 run_classical ?�패??exit1·#8 XGBoost early-stop??진짜 val
  split?�로 ???�래??묶음�??�께 ?��?. 검�? Steel 0.4884/N2O 0.7333
  byte-identical ??**기존 ???�실??불필?? ?�·② 무영??*. ?��? �??�문 ?�술
  봉합 3�?lr-config 명시/FFT-period ?�크립트/wilcoxon ?��???.
- **2026-07-19 (MASK-NAN-GRAD, `d56aa05` ?�시??**: MASK-FIX ?�속 치명 버그
  발견·?�정 ??`_masked_loss`가 full-tensor�?`(output-target)` 먼�? 계산 ??  NaN ?��??�??autograd 그래??진입 ??backward `0*NaN=NaN` gradient ??  **마스??3�?N2O/WindSCADA/MetroPT3)?�서 �??�텝 가중치 ?�체 NaN 발산** ??  train/val/test 집계 `sum/max(1,cnt)=0`??**가�?0.0 기록**(P2-5 가?�도
  0.0=?�한?�라 ?�과). ?�정 (1) ?�덱???�행 `output[mask]-target[mask]`,
  (2) val_cnt==0?�NaN(P2-5가 ?�끄?�게 ?�패). 3�??�모??검�?N2O 0.0??.7333 /
  MetroPT3 0.6914 / Steel 0.4884 무회귀). **???�버 pull ???�a ?�개 ?�수**.
  ?�송??lr_search.csv??마스??�?GasTurbine 286??purge ?�료(1130??44).
  같�? ?? ??중간?��?(886???�린, �?문헌범위, DLinear/ETTh2 ?�드불안???�성)
  + ??TPAMI FEATHer ?�치 vs ??베이?�라??비교(ETT 1?�권 ?��?, DiPE ?�세,
  Electricity�??�점) ??`manuscript/notes/ltsf_midsweep_check_2026-07-19.md`.
- **2026-07-16**: (1) **베이?�라??5?? ?�드**(교수 결정: ?�출 ?�선 + 리비??  보강) ??`--num_seeds 3`, 코드 ?�정 ?�음, ??LTSF 3?�드 ?�발?? ?�는 ?�째�?3?�드,
  lr?� 2?�드 ?��?; 리비??top-up = `--num_seeds 5`(resume??2028/2029�?채�?).
  (2) **?�고 ?�스???�작??* ??`*_jms.md` 5�?abstract/intro/related/datasets/
  methodology) mfg-6 + A+C 고속-?�리(?�기�? + robustness 보수 + SWIFT + half-RevIN
  + ?�론 ?�림. (3) **고속 ?�레?�밍**: HR-SHM?� ?�재 문헌?�나 ?�리 ?�이??slow-
  process ???�기 ?�용�? EWELD=금속?�업 ?�력부??+극한기상 �?. Memory:
  project_manuscript_jms_drafts.md.
- **2026-07-15 (MASK-FIX)**: 결측 처리 ?�론????모든 �?**hold-last**(보간
  ?�거, ?�과?? + companion `mask.csv`�?imputed ?� ?�시 ??로더가 ?�답�?  NaN ??**loss/metric 관�?only 채점**(traffic/?�료/?�서 ?��?; 12모델 ?�일??  공정). 코드 5?�일(prep/data_loader/data_factory/run_forecast/metrics;
  robustness??공유 nan-aware metric ?�동 ?�속). 검�? 배선 ?�모??ALL PASS
  + ?�습 Steel 0.5648/MetroPT3(17.7%마스?? 0.7533. ?�측 결측�? MetroPT3
  17.67%/N2O 5.79%/WindSCADA 1.20%/BlastFurnace 1�?EWELD 0. **?�실??3�?*
  (MetroPT3/N2O/WindSCADA ??data가 interp?�hold). BF 1칸·EWELD 0 = 불필??
  ?�형보간 lookahead 공격�??�멸. ?��?: ROADMAP.md "MASK-FIX" 블록.
- **2026-07-15 (JMS ?�포지?�닝 + ?�래??베이?�라??구현)**: JMS 리뷰???�조
  ?�구?????�무 ?�뢰 ?�커�?**?�래??베이?�라??* 추�? + **"?�무�??�확·비배??
  vs ?��?�?sub-1K·배포)" 2�??�레?�밍** + **배포 ?�험=#1 구멍**(QEMU Layer-2
  skeleton ???�측) + ETT/Weather 강등(?��? 결정). **SWIFT(2501.16178)=?�용�?*
  (18.1k=sub-1K ?�님·FITS ?�?�·robustness 0·ICLR2025 withdrawn). **구현 ?�료
  (미커�?로컬)**: LSTM/GRU/TCN(nn.Module, `baselines/{LSTM,GRU,TCN}/`,
  list_models 15, 기존 worker�??�습, arch 고정+lr?�치) + XGBoost(별도
  `run_classical.py`, 채널?�립 horizon-feature 직접?�스?? 마스??aware,
  GPU, 고정config+early-stop, lr?�색X). ?�모???��? ?�한. **???�버 ?�성??  보류**(push?�면 ?�윕 3모델?? ??체인 ?�난 ???�도?�으�?. ARIMA/LightGBM
  미결. Memory: project_jms_baseline_repositioning.md.
- **2026-07-14d/e**: 감사 P2-3..8+P3 ?��? ?�정(23-check 검�? `59c85da`) ??  감사 종결; prep �??�책?�로 ?�이??4�??�생??MetroPT3 16.6% ?? +
  BlastFurnace winsorize train-?�역; **07-14e ?�용??결정: λ-?�라?�스
  ?�측???�??FEATHer OFAT 512 ?�면 ?�실??*(07-06 ?�결 8???��??�정,
  diff 기록); LTSF/lr?� `--exclude FEATHer`�??�발?? FEATHer �??�④b)?�
  ?�동�??? TSLib ?�조로 eval ?�로?�콜 ?�석 ?�인; push through 30f037d.
- **2026-07-14b**: `robust_summary.py` +4 metrics (μ_rel / AUC / mCE-vs-
  SparseTSF / degradation curve) + `gate_logging.py`, both synthetic-validated;
  R3 resolved (A+C, no short-term in paper); ROBUSTNESS_DATASETS
  GasTurbine?�EWELD = [Steel, WindSCADA, EWELD, ETTh1, Weather].
- **2026-07-14**: mfg lineup LOCKED at 6 (EWELD in; GasTurbine fully removed ??  broken, no revival); freq-tied period HPs data-driven in
  `_DATASET_OVERRIDES`; SCINet added-then-reverted; TEP/ziya07/AI4I/GasSensor
  vetted+rejected; PyTSF-MfG = cite for mfg-scarcity gap.
- **2026-07-13c**: short-horizon PdM DROPPED from paper (kept as shelf);
  mfg 3?? (+MetroPT3 +N2O +BlastFurnace-stress); Tier-2 loader prune
  (data_dict = 18 keys); push policy ??ijaehun only.
- **2026-07-13/b**: strategy pivot ??robustness (not accuracy) justifies
  params, protocol+citations in
  `manuscript/notes/robustness_justification_evidence.md`; server bugs fixed
  (reformer_pytorch import-time dep; --ngpu `torch.cuda.set_device`) ??  4 models were silently missing from CSVs, resume fills;
  `check_duplicates.py` added (fcst 769 rows, 0 dups).
- **2026-07-06/b**: OFAT 512 + combo 32 done ??FEATHer LTSF 8 rows FROZEN into
  `_DATASET_OVERRIDES` (raw-verified); dataset precedent audit ??all 6
  confirmed, "first to cast as edge LTSF" holds; wording guards in
  `manuscript/notes/dataset_precedent_audit.md`.
- **2026-07-03**: OFAT verdict ??canonical mfg arch stays d8/k7/p12/B3
  (d16/p6 win val but break sub-1K); `--ngpu` dynamic queue + `--jobs_per_gpu`
  added; ??FEATHer params scale with pred_len ??sub-1K claim is H=96-scoped.
- **2026-07-02/b**: PMSM dropped from main (2Hz trivial persistence), rebuilt
  for PdM; PHM-2018 vetted+rejected; protocol sequencing locked (arch from
  LTSF OFAT, lr-only on mfg, symmetric).
- **2026-06-25/26**: JMS manufacturing pivot (professor's call, from IoT-J);
  TEP dropped, C-MAPSS added as separate short-horizon section (later
  shelved); Word submission format.
- **2026-06-11..18**: eval protocol rewritten (best-val-epoch, test-once);
  per-dataset FEATHer tuning replaces single-config narrative; baseline LTSF
  sweep launched; tex symbol/citation fixes.

## Environment

- **Conda env**: `feather` at `C:\Users\lee\.conda\envs\feather\` ??Activate
  with `conda activate feather` before any commands. Has `torch 2.5.1+cu121`,
  `darts 0.37.1`, `reformer_pytorch 1.4.4` (for iTransformer's
  SelfAttention_Family).
- `mamba-ssm` is intentionally **not** installed (CUDA-toolchain build,
  out-of-scope for our edge-deployable benchmark).

## Repository structure

Layout rule (2026-08-24): **`utils/` = data + training machinery**, **`tools/` =
things a human runs to produce or inspect outputs** (paper artifacts, audits,
HP searches). Nothing under `tools/` is imported by the training path.

```
FEATHer/
?��??� run_forecast.py                 THE orchestrator ??main sweeps + --ablation_arm (--check, resume, dispatch)
?��??� setup_baselines.sh              git-clone baseline upstream repos (idempotent) + pip deps
?��??� scripts/benchmarks/run_forecast.py   worker (one model × dataset × pred_len × N seeds)
?��??� baselines/                      top-level model directory
??  ?��??� _import_helper.py           sys.modules isolation for upstream imports
??  ?��??� __init__.py                 registry (14) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
??  ?��??� FEATHer/FEATHer.py          our model (2 variants via channel_mode; see _FEATHER_MFG_N / _FEATHER_LTSF_CI)
??  ?��??� <13 baseline dirs>/wrapper.py + <Name>-main/ upstream clones (gitignored)
?��??� utils/
??  ?��??� prep_manufacturing.py       raw ??data/<name>/data.csv + mask.csv (9 recipes; data/ gitignored)
??  ?��??� data_factory.py             data_provider + data_select (lazy darts import; data_dict = 18 keys)
??  ?��??� data_loader.py              Dataset classes (ETT_hour/minute, Custom, CMAPSS)
??  ?��??� dispatch_queue.py           --ngpu dynamic job queue (one worker subprocess per GPU)
??  ?��??� losses.py                   spectral_separation_loss_scales (FEATHer-specific)
??  ?��??� metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
??  ?��??� seed.py                     set_seed() + parse_seed_list()
??  ?��??� timefeatures.py
?��??� tools/
??  ?��??� audit/                      check_progress · check_duplicates · purge_rows · results_table · migrate_checkpoints
??  ?��??� search/                     search_feather_{N_mfg,CIN_ltsf,L_ltsf}.py ??how the frozen configs were selected
??  ?��??� paper/                      table / figure generators
??      ?��??� merge_raw.py            per-sweep CSVs ??mfg_raw.csv / ltsf_raw.csv (applies the LTSF anchor rule)
??      ?��??� build_paper_tables.py   ??analysis/01,02,03_*.xlsx
??      ?��??� build_complexity_xlsx.py + measure_complexity.py   FFT-aware MACs/params ??analysis/04
??      ?��??� ablation_summary.py     --ablation_arm deltas vs the main-table row (??% = signal)
??      ?��??� compute_r2.py           post-hoc R2 = 1 ??MSE/Var(y_test) (stored R2 column is unusable)
??      ?��??� main_table.py           mean±std table + persistence column (md / latex / csv)
??      ?��??� wilcoxon.py             pairwise signed-rank vs FEATHer
??      ?��??� pareto_plot.py · energy_envelope.py               accuracy-per-resource figures
??      ?��??� persistence_baseline.py copy-last baseline under the exact worker protocol
?��??� deployment/                     board/ (C port + golden) · cortex_m3/ (estimator) · EDGE_EXPERIMENT.md
?��??� results/                        mfg_raw.csv · ltsf_raw.csv · complexity.csv · _archive/ (provenance) · checkpoints/
?��??� analysis/                       01??4 xlsx tables · RESULTS_STATUS.md · lit_review/ (gitignored)
?��??� _archive/                       retired-but-retrievable: robustness/ · ablation_v1/ · grids/ · classic_baselines/
?��??� data/                           raw + processed data (gitignored ??transfer to server manually)
?��??� manuscript/                     reviews, drafts, notes (gitignored; tex_workspace/ + drafts/*.md selectively tracked)
```

**Retired 2026-08-24** (all under `_archive/`, each with a README): the
robustness track (orchestrator + worker + `utils/noise.py` + `utils/severity.py`
+ `robust_summary.py` + `gate_logging.py` ??FEATHer degraded worst of the edge
tier, claim dropped); the 30 `FEATHer_<axis>_<variant>` ablation builders and
`ablation_table.py` (predate the residual MLP and channel mode, so they built
the pre-08 architecture); `run_grid_ltsf_v2.py` (superseded by `tools/search/`).
The OFAT/lr search orchestrators went earlier.

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos (+ reformer_pytorch dep) ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Manufacturing main table (default group; 6 datasets) ===
python run_forecast.py --check --exp_tag main             # 1,440 mfg runs (12×6×4×5)
python run_forecast.py --exp_tag main --save_model        # group=mfg by default; --save_model makes robustness ckpts
python run_forecast.py --group ltsf --exp_tag main --save_model   # LTSF-8 generalization (1,920)
python run_forecast.py --group cmapss --exp_tag main --save_model # PdM shelf (540; NOT in paper)
python run_forecast.py --group all  --exp_tag main --save_model   # mfg + ltsf = 3,360
python run_forecast.py --model FEATHer --save_model                # one model across group
python run_forecast.py --data ETTh1 --pred_len 96 --save_model     # one (data, horizon)
python run_forecast.py --exclude TimesNet,MDMLP_EIA                # skip heavy models
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke

# Multi-GPU: all four orchestrators take --ngpu N (dynamic queue, one worker
# subprocess per GPU; utils/dispatch_queue.py; no DDP) and --jobs_per_gpu M
# (ngpu×M workers, auto OMP/MKL pinning). ngpu=1 default = sequential.
python run_forecast.py --exp_tag main --save_model --ngpu 2 --jobs_per_gpu 3

# === lr search for the mfg main table (Option B; runs BEFORE the main sweep) ===
# 12 models × 5 lr × 9 datasets (6 mfg + 3 PdM shelf) × 2 horizons × 2 seeds = 2,160.
# Bypasses _DATASET_OVERRIDES (default arch) ??period overrides only affect the main sweep.
python run_lr_search.py --check
python run_lr_search.py --ngpu 2            # resumable
python run_lr_search.py --summary           # best lr per (method,dataset) ??MERGE into _DATASET_OVERRIDES
                                            # (merge into EXISTING period entries ??no duplicate keys!)

# === HP search (DONE ??configs frozen in _DATASET_OVERRIDES) ===
# Kept as reproducibility evidence for how the shipping configs were picked.
python tools/search/search_feather_N_mfg.py --summary     # mfg  ??ds16_k7_B4
python tools/search/search_feather_CIN_ltsf.py --summary  # LTSF ??per-dataset
# Windows gotcha: run --summary with PYTHONUTF8=1 PYTHONIOENCODING=utf-8 (em-dash vs cp949)

# === Component ablation (mfg; one component off per arm, on the SHIPPING model) ===
python run_forecast.py --ablation_arm all --group mfg --num_seeds 1   # 6 arms × 24 = 144
python run_forecast.py --ablation_arm gateuni,band2 --group mfg       # subset
python tools/paper/ablation_summary.py --per_dataset                  # verdict (??% = signal)
# arms: noresid / ci / norevin / gateuni / band2 / d4.  Rows are tagged
# abl_<arm> and routed to results/ablation.csv; the main sweep is the baseline
# and is never re-run.

# === Progress audit ===
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py

# === Paper artifact generators ===
python tools/paper/merge_raw.py                              # ??mfg_raw.csv / ltsf_raw.csv
python tools/paper/build_paper_tables.py                     # ??analysis/01,02,03_*.xlsx
python tools/paper/compute_r2.py --csv results/mfg_raw.csv   # post-hoc global R2
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer

# === Data regen (data/ is gitignored) ===
python utils/prep_manufacturing.py            # all 6 mfg recipes
python utils/prep_manufacturing.py EWELD      # one (BlastFurnace needs openpyxl)

# === Cortex-M3 edge cost estimator / QEMU (Layer 1 / Layer 2, WSL) ===
python -m deployment.cortex_m3.run --check
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build (IoT-J tex now legacy; JMS = Word via drafts/*.md) ===
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
| FEATHer (ours) | (JMS) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | EAAI (Elsevier) | 2026 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

(SCINet was briefly added 2026-07-13 then REMOVED ??reviewer only named it as
prior multi-scale work, not a baseline request; Related-Work differentiation
instead.) ??5 orders of magnitude (267 ??1.6M). Three recent peer-reviewed
SOTA (DiPE-Linear DASFAA'26, MDMLP-EIA AAAI'26, LMS-AutoTSF EAAI'26 ??the
last confirmed 2026-07-23 as journal-published, not arXiv-only) address R4's
"outdated baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation.
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures into uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** ??selective-scan **custom CUDA kernels** ??not fairly
  reproducible under our unified from-scratch protocol (AMP off, identical
  training loop); `mamba-ssm` intentionally not installed. *(NOT "no MCU
  port" ??the ceiling models, e.g. iTransformer 841k / MDMLP 1.6M, don't fit
  an MCU either; MCU-portability can't be the exclusion criterion.)*
- **TimeCMA** (AAAI 2025) ??frozen **pretrained GPT-2 features = a different
  resource class** (not trainable from scratch). *(NOT a size cutoff ??we KEEP
  MDMLP 1.6M as a ceiling; size is never the exclusion criterion, the 41p??.6M
  span IS the point.)*
- **SWIFT** (arXiv 2501.16178) ??**no mature official code** ??not fairly
  reproducible under our protocol; performance **ties FITS** (already in the
  table ??FITS represents that point). ICLR2025-withdrawn = secondary note
  only. *(NOT "18k not sub-1K" ??DLinear 18,624p is a benchmarked baseline.)*

Reviewer-defense paragraph (paper draft):
> "Model size is not our exclusion criterion ??our benchmark deliberately
> spans 41 to 1.6M parameters to map the capacity?�accuracy frontier. We
> exclude only architectures we cannot fairly reproduce under a unified
> from-scratch protocol ??Mamba-family models (S-Mamba) requiring custom
> CUDA kernels, and methods without mature official code (SWIFT) ??or that
> belong to a different resource class, namely LLM-augmented / pretrained
> forecasters (e.g., TimeCMA) depending on frozen external features rather
> than being trained from scratch. These are cited in Related Work but lie
> outside our reproducibility-controlled benchmark."

## Training protocol ??TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ??Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | Deployment protocol: unified across all models; justified by MCU input-buffer byte budget (720×D×4B vs 96×D×4B). Other HP from each paper's recipe ??lookback is the one deliberate override (state this to avoid the "published recipe vs 96 override" self-contradiction; SparseTSF@720 goes in an appendix rank-stability check). |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; selection = best-val epoch, test evaluated once on that model |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025??027 (3 seeds; �? in revision) | 07-16 triage: `--num_seeds 3` submit-first, top-up to 5 (`--num_seeds 5`) in revision ??3??, resume fills the 2; base_seed/exp_tag fixed. lr search stays `LR_SEEDS=[2025,2026]`. |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ?�️ Per-(method, dataset) overrides

| Axis | Source |
|---|---|
| `lr` | baselines: each paper's per-dataset script; mfg: `run_lr_search.py` (merge pending) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs | paper Table or official script; FEATHer LTSF-8 rows from OFAT (frozen) |
| Freq-tied (`cycle`, `period_len`, FEATHer `period`) | data-driven per mfg dataset (FFT-detected), same rule for all 3 period-models |

Routing: `_METHOD_DEFAULTS` + `_DATASET_OVERRIDES` in `baselines/__init__.py`;
orchestrator merges defaults + overrides + CLI, forwards `lr`/`loss` as worker
flags and the rest as `--model_overrides "k=v;k2=v2"`; worker's
`_apply_overrides()` sets each key on `args` before model construction.
FEATHer is tuned per-dataset like every baseline (symmetric fair-comparison
framing); canonical mfg arch = base d8/k7/p12/B3.

## Architecture (FEATHer model)

`baselines/FEATHer/FEATHer.py`, four components:

1. **Multi-band Decomposition** ??depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). `num_bands ??{2,3,4}`.
2. **DenseTemporalKernel (DTK)** ??project to latent ??depthwise causal conv
   ??project back. Shared across all bands.
3. **FFTFrequencyGate** ??FFT magnitude ??Conv1d ??softmax weights, one per
   band, adaptive per sample. Casts to float32 (cuFFT half-precision).
4. **SparsePeriodKernel (SPK)** ??period-aware head: reshape by period phases,
   shared linear projection, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss (FEATHer-only;
??band-order inversion bug ??audit P1-1 ??fix pending).

Key FEATHer args: `--d_state 8` (DTK latent), `--kernel_size 7`, `--period 12`
(must divide seq_len AND pred_len), `--num_bands 3`, `--lambda_spec 0.01`.

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics computed once on the best-val-epoch model. `exp_tag` partitions
experiments (`main`, `robust`, `ablation_dtk`, `smoke`, `lr_*`, `hp_*`).
Use `--results_csv PATH` to redirect verification runs.

## Datasets

Loaded via `utils/data_factory.py` (lazy darts import; `data_dict` = 18 keys
after the 2026-07-13c Tier-2 prune).

| Dataset | Source | Horizon set |
|---|---|---|
| **Steel · WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD** (MFG **main table**; BlastFurnace D=28 = stress tier) | local CSV (`utils/prep_manufacturing.py`) | [96, 192, 336, 720] |
| CMAPSS / CMAPSS3 / PMSM (PdM **shelf**, not in paper; unit-aware `Dataset_CMAPSS`) | local CSV | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (generalization) | darts | [96, 192, 336, 720] |
| nrel (kept per user) | local CSV | [96, 192, 336, 720] |

Robustness set: `run_robustness.py:ROBUSTNESS_DATASETS` =
**[Steel, WindSCADA, EWELD, ETTh1, Weather]** (EWELD's 5,741 extreme-weather
event windows = the only non-synthetic distribution-shift test).
EWELD gotcha: strong daily cycle ??MUST report a seasonal-naive baseline
(copy-last-day ~0.55). Each `data/<name>/` holds raw + cleaned `data.csv`;
regenerate with `python utils/prep_manufacturing.py [<name> ...]`
(BlastFurnace needs `openpyxl`). `data/` is gitignored ??transfer manually.

## Known issues / gotchas

- **??AUDIT (2026-07-14) ??FULLY CLOSED** (P0/P1/P2-1,2 in cfd90c5;
  P2-3..8 + P3 in the 07-14d close-out). Residual DATA facts to remember:
  pre-fix Weather rows and stored-RMSE values in old CSVs are stale (purge /
  derive sqrt(MSE) respectively); lr rows for WindSCADA/BlastFurnace/N2O/
  MetroPT3 predate the gap-policy data regen (purge+refill per SERVER
  PROCEDURE); `data/<name>/data.csv.preP3bak` = pre-fix data copies (delete
  once the server transition is confirmed).
- **`--ngpu N` + upstream hardcoded `.cuda()` (fixed 2026-07-13)**: workers
  call `torch.cuda.set_device(args.gpu)` at startup. Do NOT sed the upstream
  `baselines/*/*-main/` files (gitignored, re-cloned).
- **`reformer_pytorch` is an import-time dep of 4 baselines** (iTransformer,
  LMS_AutoTSF, TimeMixer, TimesNet). Missing ??those 4 die at import, write
  no CSV row, and look "not yet run" (failed workers currently exit 0 ??  audit P2-1). `setup_baselines.sh` installs `reformer_pytorch==1.4.4`.
- **AMP is forcibly disabled** in the worker. Re-enabling breaks FITS
  (complex grads) and TimesNet (half-precision cuFFT).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import (sys.modules collisions otherwise).
- **TQNet** receives `cycle_index = torch.zeros(B)` (our DataLoader has no
  per-sample cycle position) ??consistent across all batches.
- **R2 metric** can explode to extreme negatives on near-constant channels
  (SST??); MSE/MAE/RMSE unaffected; R2/CORR supplementary only.
- **FEATHer params scale as D²** (`in_proj`): sub-1K holds for D??4 only.
  Weather 1.4K / Electricity 115K / Traffic 776K / BlastFurnace(D=28) 1,986
  are framed as stress tests.
- **FEATHer params also scale with pred_len** (SPK backbone): quoted counts
  are H=96 values; at H=720 D??1 exceeds 1K. Sub-1K claim must be scoped to
  H=96 in the manuscript (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1??** ??infra (multi-seed, 12-model registry, 4-axis noise).
- **Phase 4** (running) main sweeps on server: mfg 1,440 + LTSF 1,920.
- **Phase 4b** (blocked on P0 fixes + mfg checkpoints) robustness sweep:
  12 models × 5 datasets × 4 horizons × 5 seeds × 20 conditions = 24,000 rows,
  inference-only from `--save_model` checkpoints.
- **Phase 5** paper artifacts under `tools/paper/` (+ still to write:
  ablation-under-corruption aggregation, Pareto plot).

## Paper context

- Target venue: **Journal of Manufacturing Systems (JMS)**, Elsevier/SME, Q1.
  Story: edge PdM/process-monitoring forecasting on machine-side MCUs;
  FEATHer = sub-1K Cortex-M3-deployable forecaster, robust to shop-floor
  sensor faults, validated across 6 industrial domains. Contributions:
  (1) sub-1K multiscale-frequency forecaster, (2) first edge/sub-1K mfg LTSF
  benchmark (6 domains), (3) sensor-fault robustness, (4) Cortex-M3
  deployment evidence.
- **Submission format: Word** (Elsevier "Your Paper Your Way"; no rigid
  template; Highlights 3?? bullets ??5 chars; Vancouver refs; CRediT +
  declarations). Draft in markdown under `manuscript/drafts/` ??paste into
  Word. Order: Intro (A+C framing) ??Datasets ??Related ??Method polish ??  results when sweeps land.
- Reuse map from `feather_iotj.tex`: Methodology + Theoretical Analysis
  ~as-is (trim theory); Intro/Related/Datasets need the manufacturing
  reframe; Results/Ablation/Robustness wait for sweep numbers.
- Wording guards: CMAPSS = "first to cast as a fixed-horizon multivariate
  sensor forecasting BENCHMARK" (not "first to forecast"); PMSM cite AIP
  Advances 16(4):045101; Steel all-channels departure stated explicitly;
  mfg-LTSF precedent = "first to cast as edge PdM LTSF"
  (`manuscript/notes/dataset_precedent_audit.md`).
- Reviewer-by-reviewer notes in memory; reject letter at
  `manuscript/notes/reject_mail.md`; robustness protocol citations at
  `manuscript/notes/robustness_justification_evidence.md`.
