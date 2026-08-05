# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

## Current status (2026-07-14e)

**⚡⚡⚡⚡⚡⚡ 2026-08-05 (배포집합 프레이밍 + L/N/CI 패밀리 + 천장스캔 발사 +
MixLinear 보드포팅 완료; 상세 = `memory/project_deployment_family_pivot_0805.md`):**
- **① 프레이밍 확정 = "MCU 배포가능 모델 중 최정확"** (경량tier 아님). DLinear는 배포표서
  **size로 자연탈락**(ETTh1 H720 = 139,680 params ×4B = **545KB > 128KB flash**, 전부 로그값).
  MCU params 상한 ≈ **2.7만(float32)** = (128KB−~15KB펌웨어−~1.5KB테이블)÷4B. robustness 제거
  재확인.
- **② FEATHer 패밀리 = 2축 4칸** (DLinear Linear/NLinear/DLinear 미러): 채널처리(mix/CI) ×
  비선형잔차(off/on) → **L**(base canonical)·**N**(mix+잔차, 저채널↑)·**CI**(ci_full=채널독립,
  **D무관 ~140~370 params sub-1K**)·**CIN**. **D²범인 = `in_proj=nn.Linear(eff_d,eff_d)`
  (FEATHer.py line338)**; CI가 이걸 제거(Electricity 125K→255). **CI는 params↓ 대신 MACs↑**
  (채널반복). 잔차 재설계: 저랭크(rank16)→단순 비선형 **D→2D→D**(천장스캔은 현 rank16). **★
  패밀리 2개(L/N) vs 3개(L/N/CI) = PENDING**, 게이트 = **mfg BlastFurnace(D=28)서 CI가
  FITS(1.005) 이기나** (지금 mix가 지는 유일 mfg). `--channel_mode {mix,ci_full}`(2330b9c)·
  `--use_residual_mlp` 이미 있고 orchestrator 둘다 forward(검증).
- **③ 천장스캔 발사 (서버, 대기중)**: 4칸 × 2loss(mse/l1) × 12데이터(mfg6+LTSF저채널6) × 4H ×
  **1seed = 384런**, 격리 CSV `results/fcst_FEATHer_results.csv`(WDSLab싱크), exp_tag
  `feather_{L,N,CI,CIN}`+`_l1`. 1seed로 천장(test-MSE) 본뒤 best만 3seed. **세션끝 64/384**
  (mfg L 완료, N 진행). **★ mfg=MSE 확정**(L: MSE학습 0.713 < MAE 0.744 재확인; MAE는 LTSF만).
  느림 = N2O/MetroPT3 대형 + oversubscription(MSE+MAE+lookback 2GPU) → **FORCE_OMP/동시sweep↓**.
- **④ R² 드롭**: mfg/LTSF 전부 -수조(r2_mean이 윈도우내부 T축분산으로 나눔 → near-constant
  채널 폭발). **MSE/MAE만**(문헌표준). 교수원하면 R²=1−MSE/Var(전역) 후처리(재실행불요,
  ≈1−MSE=순위동일).
- **⑤ 복잡도표 T5 = `analysis/04_complexity.xlsx` 완성**(베이스라인, FEATHer placeholder):
  **FFT-aware MACs 카운터** `tools/paper/measure_complexity.py`(matmul=FlopCounterMode÷2 +
  FFT=N·log2N; **DiPE 0→21K**; params CSV일치). `results/complexity.csv`(288). **★ FEATHer는
  MACs 최소 아님**(FEATHer-L 79K vs FITS 17K); 옛 참고표 "min 1.67M"=다른규약. **차별점 =
  보드 latency**(FITS 796ms > FEATHer 349ms = MACs적어도 실칩느림).
- **⑥ ★ MixLinear 보드포팅 완료 + x86 golden PASS** (배포집합 유일 구멍이었음, 49p=FEATHer
  직접 sub-1K 라이벌): `deployment/board/export_golden_mixlinear.py`(period24 SparseTSF대칭,
  staged **0 diff**) + `src/mixlinear_infer.{c,h}`(공용커널+복소linear; forward=mean-sub→shared
  conv1d(k25)잔차→segment→time(factored Lin2×2)→freq(len4 FFT+복소Lin4→2→4+iFFT-real)→mix
  α0.5) + golden 3헤더 + `test/test_mixlinear.c` + `_mixlinear_scalar.py`. **x86 golden PASS**
  (**WSL gcc 11.4 로컬**, 4단계 ~1e-7). **보드코드 6개 전부 완료**; 남은건 MixLinear 물리측정
  (main.c 4편집=`#define RUN_MIXLINEAR`+분기, BOARD_RUNBOOK 51-60 패턴) + FEATHer최종변형 재측정.
- **⑦ GRU/XGBoost = mfg만** (LTSF 스킵). **웹검증**: 제조PdM 논문 RF/XGBoost/LSTM/GRU 관행적
  포함(XGBoost 소규모mfg 강자+해석성) = practitioner-tier credibility + "generic RNN sub-1K
  안됨" 논거. **보드 안올림**(DLinear처럼 size배제). **공통축 = flash바이트**(XGBoost=트리라
  params N/A → KB로 비교). 튜닝 = GRU hidden{32,64,128}·XGBoost depth{4,6,8} mean-rank 1개 +
  val선택. 코드 미커밋(list_models15=스윕확장 트리거, 별도커밋).
- **⑧ 서버 CSV 정리**: `fcst_results.csv` FEATHer 192행 제거→아카이브 + 5seed여분(2028/2029)
  352행 분리→`_cleanup_20260805/`(백업+README 머지백법). 메인 = 베이스라인13 균일3시드 3429행.
  `baseline_organized.csv`+`organize_baseline_results.py`(재실행용). lookback `--exclude
  FEATHer,천장7` seq336/720 이어감.
- **▶ 미커밋 로컬**: MixLinear 포팅 전부(export/C/golden/test) + 복잡도도구(measure_complexity/
  build_complexity_xlsx) + organize_baseline_results + run_grid_ltsf_v2. **▶ 다음 세션**:
  천장스캔 완주 대기(CI/CIN 나오면 **패밀리 L/N vs L/N/CI 결정**) → 잔차 최종형 → 재튜닝 3seed →
  표(T1 라인업 flash축·T4 배포·01/03 갱신·Pareto) → MixLinear 보드측정 + GRU/XGBoost.
  [[project-deployment-family-pivot-0805]]

**⚡ 2026-08-04b (lit_review vault 전면정리 + baseline provenance/재현불가 정량증거;
상세 = `memory/project_litreview_vault_provenance_0804b.md`):** 실험 대기 중 **문서·분석
세션**(코드/서버 변경 0, gitignored vault).
- **산출**: `analysis/lit_review/experiments/`(신규) foundation 11개(protocol·lineup·
  datasets·**master_table**·**number_crosscheck**·**datasets_credibility** 등) +
  `papers/`에 baseline method 노트 **9개 신규**(iTransformer·PatchTST·TQNet·TimesNet·
  TimeMixer·MDMLP_EIA·LMS·CycleNet·DiPE) + 전 21편 `category::` 태그 + baseline PDF
  13개+TimeXer 다운(pdfs/ 22개, arXiv ID로 연도 확정).
- **★ lookback 2생태계**: L=96 native(TSLib 계보) vs **L=720 native(엣지-linear=
  SparseTSF/FITS/MixLinear/DiPE)** → uniform L=96가 **우리가 이겨야 할 sub-1K 라이벌을
  핸디캡** → 방어 = 배포 바이트예산 + **SparseTSF@720 부록 rank-stability**.
- **★ 복사-웹**(PDF 직접확인): MixLinear←FITS(byte-identical) · DLinear의 Transformer←
  FEDformer(캡션 자백) · DiPE비선형·CycleNet←iTf+TimeMixer · iTf/PatchTST←Zeng2022 ·
  TimesNet/LMS/TimeMixer=TSLib 재run(뿌리) · TQNet=인용 · MDMLP=불명.
- **★★ 재현불가 정량증거(방법론 최강 방어)**: **DLinear ETTh1@720, 전부 nominally L=96·
  상당수 같은 TSLib repo인데 0.513(TimeXer/신규클러스터)~0.695(TimesNet)=35% 스프레드.**
  최신 3편(LMS/TQNet/MDMLP)=서로 byte-identical(내부공유), TimeXer=endpoints만(부분중복,
  소스 아님); TimeXer·LMS 둘다 "TSLib 재run" 명시인데 값 다름 → **같은 repo 재run도
  재현 안 됨.** ⇒ 논문 방법론에 "공표숫자 직접비교 불가 → 통일 L=96 from-scratch 전면
  재현만이 공정" 문장으로 직행.
- **★ mfg-6 공신력 웹검증(4개 신규 vet)**: 전부 credible — Steel(UCI#851/BRI Q1)·
  WindSCADA(Zenodo/Wind Energy Q1)·BlastFurnace(**Data in Brief'25** DOI)·N2O(descriptor=
  arXiv, **앵커=Water Research'25**)·MetroPT3(**Nature Sci Data'22**/UCI#791)·EWELD(**Nature
  Sci Data'23**/Figshare). 아무도 96–720 LTSF 안씀→"제조 엣지 LTSF 최초" 유효. **우리
  사용=권고 정합**(BlastFurnace 29,602행 raw[200행 분류파일 아님]·MetroPT3 아날로그7만).
  논문 to-do: N2O→Water Research 인용·EWELD seasonal-naive 필수·MetroPT3 아날로그7+결측
  17% 마스킹 명시·Steel 전채널 공동예측 명시.
- **▶ 결정(코드 0)**: (1) **배포 실험 재프레이밍** — FEATHer는 최소·최속·최저에너지 다
  아님(SparseTSF가 이김) → **효율경쟁 프레이밍 금지, accuracy-per-resource Pareto로**:
  FEATHer가 **FITS/DiPE는 에너지×정확도 지배**(더 빠르고 더 정확), SparseTSF/MixLinear엔
  효율만 짐(걘 명확히 덜 정확). **헤드라인=정확도(sub-1K 최정확+MCU 탑재), 배포=feasibility
  +FITS/DiPE지배 보조.** (2) **더 작은 보드 구매 NO**(입력버퍼 모델무관·FEATHer 여전히
  크고 느림·SparseTSF 에너지 못이김=보드무관 → 애매함 안 고침; 배터리 각도만 원하면 그때
  L053R8 ULP 1개). (3) **라인업 추가 NO**(S4D/SegRNN/MambaSimple/Koopa: Mamba=원칙제외,
  나머진 Related Work 인용+escalation; 14 유지).
- **▶ 다음**: 서버 점검(CycleNet/MixLinear LTSF + baseline 룩백 스윕 `--exp_tag lookback`
  seq48/192/336/720 edge-6) + **FEATHer v2(MLP 640p) 보드 재측정**(정확도=v2 / 배포실측=
  canonical 453p = 불일치 구멍). [[project-litreview-vault-provenance-0804b]]

**⚡⚡⚡⚡⚡⚡ 2026-08-04 (JMS 배포-중심 피벗 + robustness 사망확정 + FEATHer-MLP v2;
상세 = `memory/project_jms_deployment_pivot_0804.md`):**
- **① robustness thesis 데이터로 사망**: `robust_feather_v2_mlp.csv` 6 mfg 완주 →
  FEATHer-MLP(canonical+MLP+RevIN)이 엣지-7 중 **꼴찌** — Dw(상대) 1.211 vs peer
  1.006~1.034 = **7/7**, corruptMSE(절대) 0.824 vs 0.688~0.749 = **7/7**(clean 좋아도
  못 구제). no_revin(EWELD 폭발 비보편)·robust-RevIN(median/MAD 실패)·center(N2O 악화)
  = **작동하는 보편 fix 없음.** → **JMS서 robustness 표/주장 통째 제거**(넣으면 "제일
  안 강건"이라 역효과; 주장 안 하면 안 테스트해도 됨). robustness = 향후 ② 정규화 동기로만.
- **② JMS = 배포-중심 피벗(확정)**: 기둥 = accuracy-per-param + Cortex-M3 실측. 파라미터
  정당화 "453p>41p는 robustness"(죽음) → **"정확도"**(FEATHer가 SparseTSF/MixLinear보다
  정확)로 전환.
- **③ 정확도 = FEATHer-MLP(v2) #2**: `feather_v2_mlp.csv` 완주(14데이터=6mfg+8LTSF,
  3seed, **MLP rank16 + loss mse**). mfg 엣지-7 OVERALL MSE: DLinear 0.668 > **FEATHer
  0.694(640p)** > CycleNet 0.720 > FITS 0.726 > DiPE 0.729 > MixLinear/SparseTSF 0.749.
  = **#2, 29배 큰 DLinear 빼곤 다 이김, Steel 1위(DLinear 제침).** DLinear 갭의 **79%가
  BlastFurnace 하나**(D=28 고채널, MLP 채널혼합 무익; CI가 레버 후보나 미검증).
- **④ 2-표 구조(DLinear 처리)**: 정확도표=**엣지-7 전부 유지**(DLinear 포함 — 빼면 "이기는
  놈 숨김"=리뷰어 즉살) / 배포표=**MCU 들어가는 것만**(F103RB 128KB flash): 실측 5개
  (persistence/SparseTSF/DiPE/FITS/FEATHer)+MixLinear(포팅 시). **DLinear 탈락**(H720서
  138K≈552KB>128KB = 물리적 미탑재 = dodging 아니라 발견).
- **⑤ 클래식 베이스라인**(GRU/XGBoost) 배포표 추가 = 제조 리뷰어 credibility. 규칙 =
  크리플링 금지(**budget 내 best config**) + **자연크기 병기**(조작 아님 증명 + "generic
  RNN은 sub-1K로 안 줄어듦"=우리 논거) + **XGBoost는 flash 바이트축**(트리라 params 무의미).
  코드 이미 있음(미커밋, [[project-jms-baseline-repositioning]]).
- **⑥ LTSF = FEATHer #5(약함)**: 0.406 vs FITS 0.376/SparseTSF 0.380/DiPE 0.380/DLinear
  0.399. MLP 넣어도 약함 = 제조 타겟 설계, LTSF 홈 아님. ⚠ LTSF선 accuracy-per-param도
  불리(SparseTSF 41p<640p인데 더 정확). **공정성 OK**(FEATHer LTSF per-dataset 튜닝됨,
  베이스라인보다 더). 근데 **재튜닝 동기 정당 = 이중 미스매치**(arch가 L1+MLP없이 튜닝됐는데
  지금 MSE+MLP). **목표="동급 일반화증명" not 이김**(07-22: arch서치 최적=degenerate·FITS
  못 이김); 리뷰어 "mfg 오버핏?" 방어용.
- **▶ 코드(미커밋 로컬, push 필요)**: `run_grid_ltsf_v2.py`(신규 — MLP+MSE-aware,
  **kernel≥3 강제**(kernel=1=DTK 죽여 선형화 degenerate 배제, FEATHer 정체성 유지),
  144config, 격리 CSV `results/grid_ltsf_v2.csv`, 기본 6 저채널 LTSF) + `build_paper_tables.py`
  (FEATHer 소스 canonical→feather_v2_mlp, params 453→640, **03_accuracy_ltsf.xlsx** 시트
  추가). **둘 다 baselines/__init__.py 안 건드림 = 스윕 자동확장 트리거 없음 = 안전 커밋**
  (서버 grid pull 위해 push 필요).
- **▶ 서버 상태**: **CycleNet·MixLinear LTSF 미실행**(baseline 나중 추가돼 mfg만 돎). ⚠
  orchestrator `--model`은 **단일만**(콤마 X, `--exclude`만 콤마) → 따로 발사:
  `python run_forecast.py --group ltsf --model CycleNet --exp_tag main --num_seeds 3
  --ngpu 2 --jobs_per_gpu 3`(그다음 MixLinear). 기본 `results/fcst_results.csv` resume.
  표 3개 갱신(`analysis/0{1,2,3}_*.xlsx`).
- **▶ 다음 세션**: (1) CycleNet/MixLinear LTSF 발사→표 재생성 (2) grid 도구 push→서버 pull→
  ETTh1 프로브 576런(1seed)→`--summary` 판정 (3) 게이트 통과 시 6개 확장→best per-dataset→
  `_DATASET_OVERRIDES` 갱신→3seed 최종 (4) 배포표(GRU/XGBoost budget-best+자연크기+바이트축)
  셋업. [[project-jms-deployment-pivot-0804]]

**⚡⚡⚡⚡⚡ 2026-08-03 (체계적 문헌리뷰 → 노벨티 진단 + 향후 방향; 상세 =
`memory/project_lit_review_novelty_0803.md`, vault = `analysis/lit_review/`):**
- **동기**: 사용자 "FEATHer=짬뽕(기존 트릭 조합), 방법 노벨티 없다" 우려 → **Obsidian
  vault 체계적 리뷰**(md+wikilink+graph, gitignored). PDF 1차출처 정독→구조화 노트 1편씩,
  **10편 완료**: SparseTSF·FITS·MixLinear·DLinear/NLinear·RevIN·OnTheRoleOfRevIN·FAN·
  FredNormer·SAN·TS-Fault·SensorFault-Bench.
- **★ 결론(냉정)**: **현 FEATHer 방법 노벨티 = 0.** 부품 전부 선점(center=NLinear·CI=
  DLinear·필터robust=FITS/MixLinear·FFT정규화=FAN). **FFT 게이트=부채**(07-31b 기여~0 +
  FAN이 원리적으로 함 → 사망테스트 대상). 조립은 노벨티 아님 = 사용자 직감 정확.
- **강건 벤치마크 2편이 선점(방법 아니라 진단도구=우리 방법노벨티 위협 X, 인용우군)**:
  TS-Fault(clean↔robust 반비례 ρ=−0.544) · **SensorFault-Bench(RevIN이 clean 돕고
  degradation 키움 — 우리 07-31b 진단을 benchmark로·우리 도메인 wind SCADA서 선점).**
- **정규화 수리 공간 포화**: FAN(NeurIPS'24)·FredNormer(ICLR'25심사)·SAN(NeurIPS'23) 다
  **clean 비정상성** 대상. **OnTheRoleOfRevIN**: "scale 제거=predictive context 버림" →
  **center의 N2O/EWELD clean 악화는 버그 아니라 알려진 트레이드오프**(center 만능수리❌,
  robust-RevIN이 일반해). **RobustTSF(ICLR'24)**: MAE robust·MSE non-robust 증명(07-30
  L1→MSE 재고 대상) + regime=**training-오염 vs 우리 inference-손상**=제일 깨끗한 차별화.
- **★ 유일하게 빈 방법 자리**: **손상 가능성 있는 입력 윈도우서 reversible mean/std를
  within-window robust 추정, scale(context) 유지, sub-1K.** 경계: 주파수❌(FAN/FredNormer/
  SAN)·미래예측❌(SAN)·across-dataset❌(FredNormer). 실패근거=median(50%miss breakdown)·
  center(scale버림)=이 자리가 어렵다=아직 빔.
- **▶ 결정 (사용자)**: **FEATHer = 이대로 실험 더 → JMS 제출**(재설계 안 함, 확정).
  **향후(다음 논문) 2갈래**: ①모델링 재설계("진단→아키텍처", 부품 사망테스트) + **②손상-
  robust 정규화**(model-agnostic=독립 방법논문 높은venue + ①의 노벨티 척추). 그림: FEATHer
  (JMS)→②(높은venue)→①(②품은 재설계). 산출=`analysis/lit_review/{FUTURE_WORK.md,
  gaps/SYNTHESIS_solution_approach.md, _MOC.md, papers/*.md}`.
- **▶ 모드 전환**: 리뷰 마무리. 실험(구 서버 FEATHer+MLP full rerun 등) 대기하며 pending
  처리 모드로. [[project-lit-review-novelty-0803]]

**⚡⚡⚡⚡ 2026-07-31b (정확도 착지 + MLP 실험 + RevIN robustness 진단; 상세 =
`memory/project_mlp_revin_diagnosis_0731.md`):**
- **① 정확도 착지 (엣지-7, mfg 완주 4 = Steel/WindSCADA/BlastFurnace/EWELD, 3seed×4H,
  전부 mse학습·mse채점):** FEATHer = **sub-1K 정확도 리더.** MSE 평균 **3위/7** (DLinear
  0.645[18.6K]·CycleNet 0.708[9.5K]만 앞, FEATHer 0.7095[453]) — **진짜 sub-1K(MixLinear
  49·SparseTSF 41·DiPE 267·FITS 1.2K) 전부 이김, MixLinear는 16셀 전부 승.** MAE 4위
  (MixLinear/SparseTSF는 이김). **07-30 L1/MSE confound 해소 확정** (val_loss 스케일로도
  mse 학습 검증). ⚠ MetroPT3/N2O 꼬리(FEATHer/MixLinear/CycleNet) 미완. DLinear 갭=**전부
  BlastFurnace(D=28)+WindSCADA(D=15) 고채널** — Steel/EWELD는 FEATHer 2위.
- **② 잔차 MLP 실험 (신규, `--use_residual_mlp/--mlp_rank` 기본 OFF, canonical
  byte-identical):** 게이트 표현 h에 **채널축 저랭크 비선형 잔차** `h+=W_up(GELU(W_down(h)))`
  (D→rank→D, +2·D·r; Steel 453→640@r16, **sub-1K 유지**). **clean MSE: Steel −7%·
  WindSCADA −8% (3seed), BlastFurnace ~0**(고채널은 채널혼합이 이질채널 섞어 무익). 추정
  FEATHer+MLP ~0.68 → **CycleNet 제치고 #2**. **robustness 중립**(Dw 본전, 절대오차는 개선).
- **③ ★ RevIN robustness 진단 (결정적, Steel 3-arm, 다 MLP-on):** canonical FEATHer가
  엣지 중 **robustness 최악**(Steel Dw~1.24 vs 라이벌 1.03~1.06; 최저 severity서 즉시 계단,
  impulse@0.05 Dw 2.17). **범인 확정 = RevIN, FFT 게이트 아님.** gate-off(uniform) Dw
  1.202→1.200(무변화) / **no_revin Dw 1.202→1.076(라이벌 급락)**. **게다가 no_revin이 clean
  MSE도 최고(Steel 0.477, −13%) = robustness+clean 이중 이득.** 메커니즘=손상 윈도우로
  mean/std 계산→impulse 스파이크가 통계 오염→정규화·역정규화 증폭. (사용자 RevIN 가설 적중,
  Claude 게이트 가설 오답.) **FFT 게이트는 clean/robust 기여 ~0 = 값어치 미검증**(gate ablation
  사상 최초로 돎).
- **전략 전환**: (a) **accuracy-first** 프레이밍(robustness-not-accuracy는 데이터로 역전됐으나
  RevIN-fix로 정당 부활 가능). (b) **노벨티 척추 = "센서고장 하 instance-norm(RevIN) 취약성
  진단 + robust-RevIN(median/MAD) fix"** — JMS(제조저널, ML노벨티 바 낮음)에 딱, 제조 문제와
  일치. (c) **DLinear 추월은 BlastFurnace 고채널 하나에 달림** — 근데 mfg는 **per-dataset
  튜닝 금지(전 모델 default config 대칭)** → 단일 canonical 아키텍처 변경으로 고채널 살리되
  저채널 안 망쳐야(어려움). **표준부품(FFN)으로 DLinear 쫓기=노벨티 0, 지양.** (d) FITS가
  robust한 이유=FFT를 **필터(low-pass cut_freq=seq_len//4)로 씀**(반응 아님)→고주파 손상 폐기.
- **▶ 다음 (RevIN이 핵심)**: WindSCADA/EWELD로 no_revin 재확인(Steel은 정상성 커서 제거 이득일
  수 있음, 비정상성 데이터선 clean 해칠 수도→**robust-RevIN이 일반해**) → **robust-RevIN(median/
  MAD) 변종 구현=논문 기여 핵심.** + MLP A/B EWELD/BlastFurnace 완주 + 정확도 MetroPT3/N2O 꼬리.
- **▶ 코드 (personal push): `ffbffde` MLP flag** + robustness MLP 배선 + **NaN-tolerant
  resume**(동시 append torn-row 픽스) + **`--no_revin`/`--gate_uniform` 진단 토글**(기본 OFF).
  격리 CSV: `feather_mlp_ab.csv`(정확도)·`robust_mlp_ab.csv`(robustness). robustness는
  **jobs_per_gpu 2 권장**(torn-row). robustness arm은 학습과 **flag 동일**해야(forward 일치).
  [[project-mlp-revin-diagnosis-0731]]

**⚡⚡⚡ 2026-07-31 대개편 (엣지-tier 재구조화; 상세 =
`memory/project_edge_tier_restructure_0731.md`):**
- **프레이밍**: JMS 유지(IoT-J 이동 X) + **IIoT 프레이밍만**(데이터 교체 X, "제조"→
  "산업 IoT 엣지" 서술만). **메인 비교 = 엣지-tier만, 천장(대형)=Pareto plot 점으로만**
  (삭제 X, 비교표 X → "왜 SOTA 없냐" 방어 + accuracy-per-param 유지). 엣지 정의 =
  "published default에서 자연히 작은" 모델(억지 축소=strawman 금지).
- **라인업 14**: 엣지-7 (SparseTSF 41·**MixLinear 49**·DiPE 267·**FEATHer 453**·FITS
  1.2K·CycleNet-linear 9.5K·DLinear 18.6K) + 천장-7 (PatchTST~MDMLP 35K~1.6M, **기존
  데이터 재사용·재실행 X**). **클래식(LSTM/GRU/TCN/XGBoost)=`_archive/` 아카이브**(리비전).
- **⚠ MixLinear(ICLR'26, 49p) = FEATHer 최대 위협** — dual time+freq, sub-1K SOTA 주장,
  FEATHer niche 직접 경쟁자. 정확도로 이기면 sub-1K 정확도 리더 주장 접고 robustness/배포로.
  **벤치마크 급함 = Exp1 첫 관문.** MixLinear/CycleNet 배선(주기=per-dataset FFT 미러
  SparseTSF/TQNet, lr=논문 스크립트 default 0.03/0.005; wrapper=debug print 억제+lpf clamp).
- **FEATHer = canonical(d8/k7/B3+FFT period) + loss l1→mse** (07-30 confound: mse학습이
  MSE표 이김). **전 14모델 train=mse/eval=mse = 공정.** arch/lr 서치(grid/OFAT/mfg-search/
  lr-search) 전부 `_archive/`. **공정성 감사 완료**(전 모델 논문 default).
- **robustness = 엣지-7 × mfg-6** (LTSF 앵커 ETTh1/Weather 드롭 — 정확도표 커버리지 일치),
  4 fault축(report서 quant 드롭). **체크포인트 = 모델별 폴더**(`checkpoints/<tag>/<model>/`,
  `migrate_checkpoints.py`로 2151개 이동 완료). **dispatch = 경량 먼저**(`_MODEL_DISPATCH_ORDER`).
- **최종 실험 9**: 메인(1 정확도L96·2 robustness·3 lookback 엣지7×mfg6×L{48,192,336}·
  4 ablation-clean 5축) / 가치(5 데이터효율cold-start·6 ablation-under-corruption·
  7 에너지envelope 계산) / 끝(8 Pareto·9 표). **FFT 게이트 해석성=버림.**
- **▶ 서버 상태**: **Exp1(mfg 정확도, 양GPU ngpu2)** + **Robustness(엣지7×mfg6, GPU0,
  FORCE_OMP=4)** 병행 RUNNING. Exp1 경량먼저라 MixLinear→FEATHer→CycleNet 순 체크포인트
  생성. robustness=체크포인트 있는 것만(멱등, 재실행으로 채움). ⚠ **박스 공유 시 FORCE_OMP
  필수**(각 orchestrator 자기워커수로 자동핀→동시=oversubscription→CPU스레싱→GPU 80%→40%).
- **실험 코드 = 전부 준비 완료** (board 포팅만 남음): `--train_frac`(데이터효율 `ff139fd`)
  + `pareto_plot.py`(`19c83d4`) + `energy_envelope.py`(`93f0009`) + `run_robustness
  --ablation_axis`(ablation-under-corruption `9df1b40`). lookback=`--seq_len` 반복,
  ablation-clean=기존 인프라. ⚠ **에너지 발견**: "연산=대기 1.5%=공짜"는 NUCLEO sleep
  (비대표) 기준; production STOP선 연산 90% 지배 → **진짜 논거="FEATHer 짧은 latency→
  배터리 오래감(latency≠params)"**. **남은 코드 = board C 포팅**(MixLinear 우선, config
  확정 후 WSL/보드 배치; 로컬 gcc 없음). **병행 to-do = 원고 리프레임**(*_jms.md 옛
  프레이밍→IIoT/엣지tier/14모델/MixLinear/mse). push `e22236f`~`9df1b40`(personal).
  체크 명령: `results_table.py --data Steel,WindSCADA,EWELD` / `robust_summary.py --csv
  results/robust_results.csv`. [[project-edge-tier-restructure-0731]]

**⚡⚡ 2026-07-30 발견 (L1/MSE 손실 confound — 상세 =
`memory/project_l1_mse_loss_confound_0730.md`):** mfg 서치 중간점검 중 사용자가
"FEATHer가 clean 정확도로 압도적으로 나쁘다" → 원인 진단 결과 **손실/지표 불일치**.
- **FEATHer만 `loss=l1`, 베이스라인 11개는 전부 `loss=mse`**(코드 `get_method_defaults`
  확인)인데 클린 정확도 표는 **MSE로 채점**. L1은 이상치에 robust → 큰 잔차를 방치 →
  스파이크 많은 공정신호(Steel 등)에서 그 오차가 **MSE에서 제곱돼 폭발**.
- **증거(완주분 Steel/WindSCADA/EWELD, 같은 예측·지표만 교체)**: FEATHer\*(셀별
  min-config 오라클) **MSE 채점 10.3/12(최하위)** ↔ **MAE 채점 4.3/12(sub-1K 1위** —
  FITS·DiPE·SparseTSF·DLinear·PatchTST·TQNet·MDMLP 전부 이김). 큰 모델(TimesNet/LMS)은
  어느 지표든 여전히 위. → "sub-1K가 500K-param을 클린으로 못 이긴다"는 유효, 바뀔 수
  있는 건 "**sub-1K tier 안에서 FEATHer 최강**"이라는 더 강한 정확도 주장.
- **파급**: ETTh1 그리드 "clean 8/9위" 결론도 MSE 기준이라 **같은 confound** → 07-28
  "FEATHer clean 못 이김" 재검토 대상(MSE 지표 착시 성분).
- **사용자 결정 = L1·MSE 둘 다 재학습 비교.** `search_feather_mfg.py`에 **`--loss
  {l1,mse}` 추가**(커밋 `eb91d39`, personal push): `mse`→`results/feather_mfg_search_mse.csv`
  자동 라우팅 + exp_tag `search_mse_*` 접두어 → 돌던 L1 서치(`feather_mfg_search.csv`,
  `search_ds*`)와 **CSV·resume키 완전 격리, 동시 실행 안전**(l1 경로/태그 무변경).
- **▶ 07-30 발사**: 구 서버 GPU0 `FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --loss mse --ngpu 1 --jobs_per_gpu 2`(324런). 판정 =
  `--loss mse --summary`. 두 서치 다 MSE·MAE 기록 → **4분면(L1학습/MSE학습 ×
  MSE표/MAE표)**으로 헤드라인 지표·손실 최종 결정, canonical은 3seed 최종화.
  ⚠ **MSE-학습 서치 결과 아직 없음**(방금 발사) — 위 10→4 역전은 L1-학습분 MAE
  재채점으로만 확인. [[project-l1-mse-loss-confound-0730]]
- **▶ 서버 상태 (07-30):** 구 서버 GPU1 = mfg 베이스라인 스윕 RUNNING(505/792 ≈64%;
  Steel/BF/WindSCADA 완주, EWELD 진행 중=heavy 3모델 꼬리[iTransformer 0·TimesNet 4·
  TimeMixer 9], N2O/MetroPT3 미착수). 구 서버 GPU0 = FEATHer mfg 서치 L1(135/324) +
  **MSE 서치 신규(0/324)**. LTSF `fcst_results.csv` 거의 완주(시드 혼재). 값 건강
  (NaN·zero·dup 0).

**⚡ 2026-07-28 결정 (제조-가치 실험 스코프 확정; 상세 =
`memory/project_mfg_value_experiments_0728.md`):** ETTh1 그리드 완주(360/360) →
오라클 표 확인 → **FEATHer clean 정확도로 못 이김 재확정**(test-peek 오라클 줘도
ETTh1 16칸 중 1칸만 1등; 공정 val-선택 9/13위). 셀링 = robustness + 배포. 그
배포 기둥을 "3D"로(돈다→오래 돈다→재튜닝 없이 돈다) 강화하려 ②③ 검토 → 결론:
- **③ one-config-fits-all-sensors = 폐기.** 제조는 `_DATASET_OVERRIDES`상 **전원
  논문 default config + period 모델만 FFT 자동주기** = 모두 "한 config" → FEATHer만
  강점 아님, 비교 우위 없음. (③-full=canonical vs per-dataset-최적 FEATHer 갭은
  제조 그리드 8.6k런이라 리뷰어 요구 시에만; 지금은 문장 하나로 강등.)
- **② 에너지/배터리 envelope = 채택** (제조-가치 1번). 실측(35mJ/추론·latency·
  활성전류 30mA) 계산만, **새 학습/보드 실험 불요.** ⚠ NUCLEO sleep 재측정 금지
  (11.4mA=ST-LINK 지배 비대표) → production bare-MCU **데이터시트 STOP(~14-24µA)**
  기준. 논문엔 **전자(배터리 수명 projection "코인셀 N년", sleep=데이터시트,
  projection 명시) + 후자(순수 실측 "35mJ/추론, duty서 연산 1.5%") 둘 다.** 핵심=
  latency≠param순→FEATHer가 FITS/DiPE보다 배터리 오래감→에너지×robustness Pareto.
- **① PdM 조기경보 리드타임 = NO-GO.** 구조적 anomaly detection(residual→경보)이라
  forecasting 정체성 흐리고 AD-baseline flank 열림 + small N(MetroPT 고장 3~4건) +
  라벨 파이프라인에 없음(원 논문에만). **동기/future-work 한 문장으로만.** 제조-가치=
  ② 에너지 + 기존 정확도/robustness/배포로 충분.
- **부수 확인**: seq_len=96·H={96,192,336,720}; 제조 대칭 프로토콜(전원 default+FFT
  period) 공정성 이슈 없음 = settled. **Results 흐름 확정**: §6.1 정확도(mfg+Pareto)
  →§6.2 robustness→§6.3 배포→§6.4 에너지(②)→§6.5 LTSF(강등)→§6.6 ablation.
- **산출물**: `ltsf_baseline_results.xlsx` FEATHer* grid-ceiling 행을 완주 ETTh1(360)
  +prelim h2/m1/m2 오라클로 갱신(min-MSE config MSE/MAE/CORR, 하이라이트 재계산).
- **▶ 서버 상태 (07-28, 3 스윕 동시):** **구 서버 GPU1 = mfg 베이스라인 스윕
  RUNNING** (792 / **245 done** ≈31%, FEATHer 제외 11모델; Steel 먼저·N2O/MetroPT3
  꼬리; Dropbox 0행은 싱크지연, LTSF ① 32/32 완주). **구 서버 GPU0 = FEATHer mfg
  arch 서치 RUNNING** (`FORCE_OMP=5 CUDA_VISIBLE_DEVICES=0 python
  search_feather_mfg.py --ngpu 1 --jobs_per_gpu 2`, 324런=27config×3ds×4H×1seed,
  격리 CSV `feather_mfg_search.csv`). **신 서버 = FEATHer LTSF 그리드 RUNNING** on
  ETTh1/h2/m1/m2+Exch/Weather. **CSV 3개 달라 동시 안전.**
- **▶ FEATHer mfg HP = per-dataset 튜닝 금지(대칭)·서치→단일 canonical 선택.** 서치
  1회로 (a) per-dataset best[진단+one-config-gap 부록] + (b) mean-rank canonical vs
  base d8/k7/B3 두 뷰. arch 27개(d_state{4,8,16}×kernel{3,5,7}×num_bands{2,3,4},
  k=1 제외), period=FFT·lr/λ 고정. 도구 `search_feather_mfg.py`(personal `b9c9ed7`).
  끝나면 `--summary`로 판정→확실히 나으면 canonical 교체, 아니면 유지. robustness
  Plan B/sub-1K tier표는 데이터 나온 뒤. [[project-mfg-value-experiments-0728]]

**⚡⚡ 2026-07-27b 결정 (mfg lr 서치 폐기 → default lr; 상세 =
`memory/project_mfg_lr_default_decision.md`):**
- **제조 lr 서치(④a baseline + ④b FEATHer)를 전면 폐기하고 각 모델의
  `_METHOD_DEFAULTS` lr을 씀.** 결정적 증거 = `lr_search.csv` 44개 mfg 셀에서
  default lr vs 5-lr 서치 best-lr의 val_loss 개선이 **중앙값 0.1%**(평균 0.4%,
  최대 6.4%=MDMLP·Steel 하나; >3% 1/44·>10% 0/44). best-lr이 default와 10배
  달라도 val_loss는 0.5%만 움직임 → mfg는 **loss 표면이 lr에 대해 평평**, 서치
  best는 노이즈.
- **⚠ BASELINE에만 적용, FEATHer는 제외.** 베이스라인 default는 각 논문 저자
  공표값이라 정당하지만, FEATHer는 우리 모델이라 **공표 default 없음**(1e-3은 우리가
  임의로 찍은 값) → "default 스킵" 논리 성립 안 함. FEATHer는 진행 중인 **그리드서치
  (ETTh1/h2/m1/m2+확장) 결과로 config 확정 후 사용자와 함께 보고 별도 발사**(arch=
  canonical mean-rank, period=FFT, lr=그리드기반). FEATHer는 sub-1K라 병목 아님.
- **효과: 베이스라인 mfg 본 스윕(11모델, `--exclude FEATHer`) 즉시 발사 가능.**
  `--num_seeds 3`(제출용; 나중에 `--num_seeds 5` resume=총 5시드 논문값, LTSF와 동일
  정책) + `--save_model`. 베이스라인 mfg lr은 `_DATASET_OVERRIDES`에 lr 없음→default
  폴백(period_len/cycle만 override), **코드 수정 불요**. heavy(iTransformer 0행·OOM
  딸꾹질) lr 대기 소멸.
- **LTSF는 무영향** — baseline은 각 논문 per-dataset 공표 lr 유지, FEATHer LTSF
  동결 OFAT 행 유지. 이 결정은 **mfg 축에만** 적용, [[project-open-lr-policy]]를
  mfg에서 대체.
- `lr_search.csv`는 폐기 아니라 **이 결정의 증거물**로 보존(사용자 결정: 삭제 X).
  LR_SEEDS 2→1 TEMP 핵(`5398d89`)은 이제 무의미(mfg lr 안 돎; correctness 무해,
  청소 시 [2025,2026] 원복). ⚠ SparseTSF default(0.02)는 서치 격자 밖이나 공표값이라 유지.
- **▶ 서버 상태 (07-27b 발사 — 양 서버 가동):**
  - **구 서버 = mfg 베이스라인 스윕 RUNNING** — `--exp_tag main --save_model
    --exclude FEATHer --num_seeds 3 --ngpu 1 --jobs_per_gpu 3` (11모델×6mfg×4H×3seed
    =792런 → `fcst_results.csv` mfg 행). seed 3=제출용, 나중 `--num_seeds 5` resume=
    총 5시드. `--save_model`=robustness 체크포인트.
  - **신 서버 = 그리드서치 RUNNING** — ETT(ETTh1 ~완주/h2·m1·m2 진행) + **Exchange→
    Weather 확장(병렬, ETT와 byte-identical 360격자; Electricity/Traffic은 느림+
    non-sub-1K라 스킵)**. **lr 서치 중지**(mfg lr 폐기로 무의미).
  - **산출물**: `results/ltsf_baseline_results.xlsx` 갱신 = 11 베이스라인(현 시드혼재
    seed평균) + **FEATHer\* grid-ceiling 행**(sub-1K 칸별 min-MSE=test-peek, ETTh1만
    신뢰/h2m1m2 preliminary/비-ETT 공백; 1등초록·2등노랑, FEATHer는 ETTh1@336만 1등
    =clean 절대정확도 승부 아님 재확인). ① LTSF 베이스라인 11모델 32/32 완주(시드혼재:
    4모델 5시드·2모델 24×5+8×3·5모델 3시드).
  - **다음 게이트**: mfg 792런 완주→robustness 스윕 / 그리드 완주→FEATHer config 판정
    (kernel=1 arch-degenerate 지배라 canonical 교체 X 예상)→FEATHer mfg+LTSF 발사.
- **▶ dispatch 순서 통일 (커밋 `67d69ea`→`3c4d27f`, personal push):** `run_forecast.py`
  `_DATA_DISPATCH_ORDER` — mfg+ltsf **통일**, 작은 데이터셋 먼저 큰 것 나중(tier≈행×채널)
  → 빠른 결과 먼저·느린 꼬리가 안 막음. mfg: Steel→BlastFurnace→WindSCADA→EWELD→
  **MetroPT3→N2O**(N2O 210k·MetroPT3 306k행이 최장); ltsf: Exchange→ETTh1/h2/m1/m2→
  Weather→**Electricity→Traffic**(채널 321·862가 최장). 이제 **dataset-major**(한 데이터셋
  11모델 다 돌고 다음). **구 서버 적용=돌던 mfg Ctrl+C(`37e4758`)→`git pull`→같은 명령
  재실행(resume+새 순서)**. `run_forecast.py`만 push(클래식 절대 제외).

**⚡ 2026-07-27 업데이트 (MCU 배포 실측 완료 — 트랙 단일 소스 =
`deployment/EDGE_EXPERIMENT.md`, 절차서 = `deployment/board/BOARD_RUNBOOK.md`):**
- **보드(NUCLEO-F103RB) 도착 → 5모델 실보드 실측 전부 완료** (기여 4번 "Cortex-M3
  배포 증거"가 추정→실측). CubeIDE 프로젝트 = `C:\Users\lee\STM32Cube_projects\
  blink_test`(repo 밖). golden 5모델 **bit-exact PASS**. **FEATHer @F103RB
  (p12/453, -O2 soft-float): 349ms / 총 flash 17.6KB(learnable 1,812B+twiddle
  768B=§8 실증) / ~35mJ/추론 / 활성 30mA·대기 11.4mA**. 레이턴시: pers 0.08 /
  STSF 47 / FEATHer 349 / FITS 796 / DiPE 1312 ms (**파라미터순≠레이턴시순**=논거;
  활성전류는 5모델 전수 29-31mA=모델무관 실증). ⚠ 배포 상세는 **CLAUDE.md에
  안 씀**(규칙) — 전부 `EDGE_EXPERIMENT.md` 07-27 로그. **남음 = FEATHer p24(489p)
  골든 재생성 후 재측정(서버 config 확정 시) + 논문 §7/§8 표·문단.**
  [[project-deployment-layer3-plan]]

**⚡ 2026-07-26/27 업데이트 (중간점검 + lr 속도핵; 상세 = `memory/project_grid_search_pivot_0722.md`):**
- **서버 상태(현재)**: 구 서버 = ① LTSF(`fcst_results.csv`) 계속. 신 서버 =
  ETTh1 그리드(`grid_search.csv`) + ④a 제조 lr(`lr_search.csv`) 동시. CSV는
  `Dropbox\WDSLab\05. 실험\FEATHer\results\`로 싱크돼 로컬서 열람 가능.
- **① LTSF 커버리지**: 11모델 거의 완주(iTransformer만 Traffic/Weather 미완,
  TimesNet 2셀). ⚠ 시드 혼재 — 6모델 5시드(옛)·5모델 3시드(07-16 이후). 최종표
  전 시드 정책 확정 필요.
- **그리드 커버리지**: **ETTh1 95.1%(1,370/1,440)**, 남은 70 = `grid_ds8_k9_*`
  꼬리(정렬상 마지막이라 정상). ETTh2/m1/m2 = 방금 시작(~33 config). 사용자
  **순차 재실행**(ETTh1→h2→m1→m2, `foreach($d){run_grid_search.py --data $d …}`).
- **ETTh1 부분판정(95%·1시드)**: 형식상 "grid BEATS base(val)"지만 **승자가
  arch-degenerate** — base d8/k7/p12/B3 = sub-1K 272중 val **160위(중위)**, best
  sub-1K `grid_ds16_k1_p4_B3`(981p) Δval+0.0165·MSE−3.6%, **top8이 kernel=1
  지배**(DTK=temporal kernel 죽임). clean정확도는 시그니처 arch 뭉갤 때만↑ →
  **robustness-not-accuracy 증거지 canonical 교체 근거 아님, 재동결 금지**. 공정
  표(val선택) FEATHer = ETTh1 **9/13위**·ETTh2 **10/12위**(≈SparseTSF), 오라클
  (test-peek)=ETTh1 1위지만 반칙. 셀링 = accuracy-per-param + robustness.
- **④a 제조 lr = 72%(237/330 mfg셀)**: 가벼운 7모델 완료, **무거운 4개
  (iTransformer 0·TimeMixer/TimesNet Steel만·TQNet 부분)** 남음(=OOM 딸꾹질
  모델들). 느림원인 = heavy × big-ROW(N2O 210k·MetroPT3 306k 행; 채널은 작음
  6–28) × 50ep × 5lr × 2seed.
- **⚡ lr 속도핵 (커밋 `5398d89`, personal push)**: `LR_SEEDS 2→1` (**TEMP —
  lr 끝나면 `[2025, 2026]` 원복 필수**) + `run_lr_search.py --model
  iTransformer,TimeMixer,TimesNet,TQNet --num_epochs 20 --jobs_per_gpu 1` =
  **4~6배**. **안전근거 = lr은 선택도구**(출력=best-lr만→`_DATASET_OVERRIDES`,
  최종숫자는 ⑤서 50ep/3seed 재산출 → ep/seed 컷이 보고값 무영향, per-cell
  독립이라 불공정 X). 1seed는 pull로 받게 커밋 필요(미커밋은 pull 안 됨).
  **⚠ lr `--summary` 후 LR_SEEDS 원복 + heavy best-lr이 격자끝(0.0001/0.01)
  아닌지 확인.** ⚠ 미커밋 로컬 여전 = 클래식 베이스라인 + 워커 프로브 플래그.

**⚡⚡ 2026-07-22 업데이트 (이번 세션 — HP 방향전환; 상세 = `memory/project_grid_search_pivot_0722.md`):**
- **사람 루프 1 실행됨**: ② OFAT `--validate`→`--summary` → **LTSF 8행 재동결**
  (커밋 `0768e9b` + push ijaehun; validate가 raw combo 4/8 기각 → base+1축 폴백).
  **λ flat 재확인 → confound CLOSED**; **canonical mfg d8/k7/p12/B3 유지**.
  (CLAUDE.md 07-21 텍스트도 이 커밋에 동봉됨.)
- **TPAMI 갭 발견 (정직)**: 재동결 FEATHer가 ETTh1/h2 **clean 8/9위**
  (SparseTSF 69p·DiPE 511p한테 짐), 옛 TPAMI FEATHer보다 **~5-7% 나쁨**,
  **SPK@720 장기예측 주장 재현 안 됨**(같은 L=96). thesis(robustness-not-
  accuracy)와 모순은 아니나 "ETT clean 승리/​@720 강세" 주장은 **은퇴**.
- **프로브로 원인 규명 (로컬 ETTh1, 3시드)**: lr(0.01 불안정)·loss(MSE가 L1보다
  나쁨)·**AdamW+wd+lr0.01(TPAMI 정확재현)=0.410 더 나쁨** → **어떤 training
  recipe도 TPAMI 0.373 회복 못 함** → 옛 TPAMI ETT = **legacy 아티팩트**(현
  동결이 이 축들엔 near-best). 워커에 `--optimizer/--weight_decay` 프로브
  플래그 추가(backward-compat, **로컬 미커밋**).
- **결정: OFAT → 조인트 그리드서치** (유일 남은 레버 = arch 상호작용).
  **grid > Optuna**(작은 이산공간+초소형 모델+방어성+병렬). **불가침 = val 선택**
  (test-peek 금지). **공정성 = 모든 모델 per-dataset 튜닝이라 대칭**(OFAT는 안
  버림 — 리뷰어 R2#23 민감도분석 + num_bands/λ flat 증거로 여전히 쓰임).
  **2층**: LTSF=per-dataset 그리드 / MFG=**mean-rank canonical 1개**(엣지
  "one sub-1K config가 모든 센서에" 스토리 강화; period=per-dataset FFT, lr=lr서치).
- **`run_grid_search.py` 제작·push** (`0f6dd63`+`0c2576a`): arch 4축 전수,
  λ/lr/loss 고정, `--seeds`(1시드 선택), resumable/`--check`/`--summary`.
  `results/grid_search.csv`. **0722b 그리드 확장(`fbe077d`, personal)**: 극단-소형
  코너로 넓힘 — d_state{**2**,4,8,16}·kernel{**1**,3,5,7,9}·period{**2,4**,6,8,12,24}
  ·num_bands{2,3,4}=**360조합**, GRID_PREDS=**[96,192,336,720]**(period 전부
  gcd=48 나눠 드롭 0). 극단코너(d_state2/k1/p2/H720) 스모크 유한(kernel=1은 DTK만,
  FFT게이트 kernel=5 하드코딩; period=1은 SPK degenerate라 제외). 커밋=
  run_grid_search.py **단독**(클래식/프로브 미커밋 유지).
- **RUNNING = go/no-go (신 서버 ETTh1 그리드)**: **③ FEATHer 중지됨** → ETTh1
  그리드가 **④a 베이스라인 mfg lr(`lr_search.csv`)과 동시** 실행 중 (CSV 달라 안전,
  단일 GPU라 느림·RAM 주시). **0722b로 파일럿=1,440런**(360×4H×1시드). ③의 ETTh1/h2
  행은 `fcst_feather_ltsf.csv`에 **현동결 레퍼런스로 보존**(비교용, 삭제 금지).
  구 서버 = ① LTSF(`fcst_results.csv`) 계속. 로컬 = 우리 실험 없음. **서버 재발사**=
  옛 그리드 Ctrl+C→`git pull personal main`→`run_grid_search.py --data ETTh1
  --pred_len 96,192,336,720 --seeds 2025 --jobs_per_gpu 1`; **resume가 옛 H96 144행
  유지**(옛144⊂새360)·나머지만 채움(purge 불요). `--summary`도 같은 `--pred_len/
  --seeds`+`PYTHONUTF8=1`. **판정**: **grid≫base(val)** → LTSF-8+mfg-6 확장·재동결·
  ③⑤ 재실행(loop-1 재오픈). **base ties/wins** → OFAT 충분, 현 동결 유지·robustness.
- ⚠ **신 서버 ④a lr 딸꾹질(사용자 보류)**: MDMLP_EIA/MetroPT3/H96가 CUDA
  `illegal memory access`로 죽음(async라 `.item()`/`empty_cache`에 뜬 것뿐; 그
  프로세스 컨텍스트만 오염, 그리드는 별 프로세스라 무관). 유력=단일32GB GPU를
  그리드+④a 최중량(MDMLP 1.6M)이 나눠 써서 메모리 압박. 가짜행 없음(exit1+
  MASK-NAN-GRAD 가드→missing 유지, resume 재시도). **나중에 그 셀만 격리+
  `CUDA_LAUNCH_BLOCKING=1`로 재현**(단독통과=메모리→④a jobs 줄이기 / 단독실패=
  모델·데이터 버그).
- ⚠ **미커밋 로컬**: 클래식 베이스라인(list_models 15) + 워커 프로브 플래그
  (--optimizer/--weight_decay). 클래식은 여전히 push 금지(스윕 자동확장 트리거).
  (07-23: throwaway `results/probe_*.csv` + repro표 CSV 삭제, LTSF 재현결과는
  `results/ltsf_baseline_results.xlsx`로 정리 — MSE/MAE/COR 시점별 + Coverage 시트.)

**△ 2026-07-16 업데이트 (요약 — 상세는 아래 블록/Decision history):**
- **베이스라인 5→3 시드 (교수 결정)**: `--exclude FEATHer` 스윕을 `--num_seeds 3`
  으로 낮춰 제출 우선, **리비전에서 `--num_seeds 5`로 보강**(3시드=[2025,2026,
  2027]⊂5시드라 resume이 남은 2개만 채움; base_seed/exp_tag 불변 필수). **① LTSF는
  3시드로 재발사됨.** ⑤ 제조 본 스윕은 FEATHer 포함 통째로 `--num_seeds 3`(메인
  테이블 12모델 시드 일관성). ④a lr은 원래 2시드 고정이라 무관. **코드 수정
  없음**(CLI 플래그; 다운스트림 도구 전부 `nunique` 동적 집계, wilcoxon은 셀 넘어
  풀링이라 3시드도 유의성 충분). ② OFAT 선택은 3시드로 하지 말 것(안정성).
- **원고 텍스트 파트 재작성 (서버 무관)**: `manuscript/drafts/*_jms.md` 5개
  (abstract/intro/related/datasets/methodology+theory)를 mfg-6 + A+C 고속-"다리"
  프레이밍(동기만; 우리는 slow-process 평가) + **robustness 보수 서술** + SWIFT
  차별화 + half-RevIN + 이론 트림(R8 placeholder Theorem 3/4/5 제거)으로 갱신.
  옛 `0X_*_draft.md`는 IoT-J 아카이브. 상세: `memory/project_manuscript_jms_drafts.md`.
  남은 텍스트: Conclusion 리프레임 + Appendix 증명 2개 + Algorithm 1 검토.
- **고속 프레이밍 판정**: HR-SHM(고속 센싱→다스텝 예측, decimation이 이벤트를
  aliasing) = 실재 문헌이나 우리 데이터는 slow-process → **동기 인용만**, 우리
  평가 regime이라 주장 금지("horizon=step count" 다리로 봉합). EWELD = 금속산업
  전력**부하**(≠"극한기상 부하") + 극한기상 이벤트 창.

**⚡ 지금 서버 상태 (07-18b 2-서버 스윕-분할 — 체인의 단일 소스는 `ROADMAP.md`):**
**전용 GPU 1개짜리 신 서버 확보 → 스윕별 CSV 분리 이용해 병렬 분배**(사용자
결정). ①②④a는 CSV가 달라(`fcst_results`/`hp_search`/`lr_search`) 동시 안전.
- **구 서버 (Linux, 공유 128GB, GPU1)** = **① LTSF** (`FORCE_OMP=5
  CUDA_VISIBLE_DEVICES=1 python run_forecast.py --group ltsf … --ngpu 1
  --jobs_per_gpu 2 --exclude FEATHer --num_seeds 3`). `fcst_results.csv`.
- **신 서버 (Windows, 전용 32GB, GPU 1개, env `lee`, `C:\Users\user\Desktop\
  FEATHer`)** = **② FEATHer OFAT + ④a 제조 lr 둘 다** (CSV 다름 → 동시 안전).
  구 서버서 멈춰있던 hp를 `results/hp_search.csv` 복사해 이사(resume). 단일 GPU
  라 큐 옵션 최소: **② OFAT = `python run_hp_search.py`(jobs 1 유지 — elec/traffic
  워커가 RAM 무거움)**, **④a lr = `python run_lr_search.py --exclude FEATHer
  --jobs_per_gpu 3`(mfg는 RAM 가벼워 병렬 3배; GPU 10%·RAM ~8GB 여유라 이게 속도
  레버)**. RAM은 ② OFAT가 Traffic(862ch) 잡을 때만 조심(`free`). ④a엔 mfg
  data/ + `lr_search.csv` 전송 필요(뱅크 Steel/EWELD/BF resume).
- **⚠ 신 서버 `bash`는 bash 4+ 아님** — `setup_baselines.sh`의 `declare -A`가
  조용히 실패("division by 0" → clone 0개인데 "All present" 가짜 성공). **11개
  upstream repo는 cmd에서 수동 `git clone --depth 1 <url> baselines/<N>/<N>-main`**
  (URL = `setup_baselines.sh`의 REPOS). reformer_pytorch는 이미 설치됨. 안 하면
  베이스라인이 upstream import에서 죽음(예: DiPE_Linear `No module 'timeprophet'`).
- **⚠ 규칙: 같은 CSV는 한 박스에서만.** `lr_search.csv`=신 서버만(구 서버 lr 금지),
  `hp_search.csv`=신 서버만, `fcst_results.csv`=구 서버만. 최종엔 각 CSV를 한
  results/로 모으고 `check_duplicates.py` 검증. `FORCE_OMP` env = 커밋 `880c83e`
  (worker당 OMP/MKL 스레드 고정 → 박스 공유 시만 필요).
- **⚠ 07-19 `d56aa05`(MASK-NAN-GRAD) 푸시됨 → 양 서버 pull 필수, 특히 신
  서버는 pull 전 ④a 실행 금지**(마스크 3종이 가짜 0.0으로 또 기록됨).
  pull 후 클린 lr_search.csv(844행) 배치 → `run_lr_search.py --exclude FEATHer`
  재발사 → resume이 마스크 3종+미착수 모델 재충전.
**다음 사람 루프(루프 1)**: ②·④a 완료 → `--validate`(32) → `PYTHONUTF8=1
--summary` 출력 전달받아 LTSF 8행 재동결 + 07-06 diff 기록 + canonical mfg
아키텍처(d8/k7/p12/B3) 재확인 + λ 최종 판정 → 커밋 → pull → ③ FEATHer LTSF
160 + ④b FEATHer lr 발사.

**Positioning (settled):** Fresh **JMS** submission (TPAMI reject → IoT-J →
JMS manufacturing pivot). Main table = **6 manufacturing datasets** (Steel ·
WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD), LTSF-8 demoted to a
generalization section, short-horizon PdM (CMAPSS/CMAPSS3/PMSM) **NOT in the
paper** (lr-banked shelf, `--group cmapss`). FEATHer's 453p > SparseTSF 41p is
justified by **ROBUSTNESS, not accuracy** (OFAT num_bands FLAT on clean).
Framing guard: NOT "more params = more robust" but "41p linear lacks headroom;
FEATHer's inductive bias buys graceful degradation at sub-1K". R3's
"long-term-on-edge is contrived" attack resolved via **A+C framing**: (A)
industrial edge NEEDS on-device long-horizon (air-gap/OT isolation, per-sensor
scale, outage autonomy — our datasets are the evidence), (C) edge = param
budget, horizon = task, orthogonal axes; "why 1K" → accuracy-per-param Pareto
frontier. 12-model baseline lineup unchanged.

**⚑ CODE AUDIT (2026-07-14c) — full-repo review done, findings in
`manuscript/notes/code_audit_2026-07-14.md`. ✅ THE CRITICAL BATCH IS FIXED
(same session, all validated end-to-end):**
- ✅ **P0-1 robustness override plumbing**: `run_robustness.py` dispatch now
  mirrors the training-time `get_method_defaults`+`get_dataset_overrides`
  merge → `--model_overrides`; the worker gained the flag +
  `_apply_overrides()`; `gate_logging.py` applies the same merge itself.
  Verified live: trained ETTh1 d16/k3/p8/B4 ckpt loads + clean MSE
  reproduces training-time test MSE exactly.
- ✅ **P0-2 corruption seeding**: salted `hash((ft,sev))` →
  `utils/noise.corruption_seed()` (deterministic index-based). Verified:
  re-run reproduces corrupt-MSE to 6 decimals across processes.
- ✅ **P0-3 robustness resume**: `load_robust_set` counts DISTINCT
  (fault,severity) pairs; worker skips conditions already in the CSV
  (per-condition resume — no duplicate rows on retry). Verified via
  truncate-and-refill test.
- ✅ **P1-1 spectral loss band order** fixed (`bands[S-1-s]` — POINT owns the
  highest slice, LOW the lowest). Synthetic check: correct placement → loss
  0.0, swapped → 2.0. Frozen LTSF OFAT rows kept (λ axis was flat).
- ✅ **P1-2 Weather split** → standard 0.7/0.1/0.2 (legacy hard-coded branch
  + dead PM/PEMS_BAY/METR branches deleted). ALL pre-fix Weather rows are
  invalid → purge + re-run (see server procedure below).
- ✅ **P1-3 RMSE** = sqrt(global MSE) in both workers (+ NaN-aware CORR/R2
  accumulation). Old rows: derive sqrt(MSE) at table time.
- ✅ **P2-1 worker exit code**: training worker now exits 1 if any seed run
  failed (kills the reformer-style silent-skip class). **P2-2 CSV header
  race**: atomic `open(mode="x")` create-with-header in both workers.
- ✅ **P2-3..8 + P3 batch (2026-07-14d) — AUDIT FULLY CLOSED** (23-check
  smoke suite passed + all 4 orchestrators `--check` exit 0; details in the
  audit doc UPDATE 2 block):
  - P2-3 check_duplicates schema-detects the robust CSV (key extends with
    train_exp_tag/fault_type/severity — `--fix` now safe on it). P2-4
    explicit FEATHer CLI arch flags beat `_DATASET_OVERRIDES` (flags default
    None; sweep behavior byte-identical). P2-5 best_state=None raises (no
    row/ckpt; exits via P2-1 path). P2-6 mCE drops no-ref + model-incomplete
    cells for ALL models (warns; emits n_cells). P2-7 ablation_table = one
    table per dataset (no raw-MSE pooling). P2-8 lr/hp `--summary` flag seed
    under-coverage; hp combo "complete" requires full seeds.
  - P3: prep `_fill_gaps` gap policy (linear interp ≤ ~1 h, hold-last
    beyond — no lookahead/linear bridging of outages; audited max-gap
    asserts); BlastFurnace winsorize bounds from the TRAIN region (also
    finally clips the late Tp4 ~3311 sentinel: hi 3313→855); EWELD D=14
    assert; metrics.py np.divide out= init; severity tables +
    corruption_seed → torch-free `utils/severity.py` (noise.py re-exports,
    robust_summary shares the object, utils/__init__ now lazy PEP 562);
    FEATHer half-RevIN (means detached, stdev not) KEPT + documented as
    deliberate (changing it would invalidate all frozen OFAT rows); stale
    docstrings/comments swept.
  - ⚠ **DATA REGENERATED** under the new gap policy — cells changed:
    MetroPT3 16.6% / N2O 2.7% / WindSCADA 1.1% / BlastFurnace 0.9% /
    EWELD·Steel 0%. Pre-fix copies kept at `data/<name>/data.csv.preP3bak`.
    → transfer the 4 changed data.csv to the server and purge/refill their
    lr_search rows (added to SERVER PROCEDURE below).

**▶ SERVER PROCEDURE after pulling the fix commit** (order matters):
```bash
git pull   # (at a natural pause; running workers finish on old code harmlessly)
# invalidated rows — resume refills them:
python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply
# (07-14e 결정 — λ 슬라이스 부분퍼지를 대체) FEATHer OFAT 전면 재실행:
# hp_search를 통째로 비우고 run_hp_search.py가 512런 전부 재탐색
# (교정된 spectral loss + 표준 Weather split 아래에서):
python tools/audit/purge_rows.py --csv results/hp_search.csv --model FEATHer --apply
# (P3 data regen, 2026-07-14d) transfer the 4 regenerated data.csv files
# from local (WindSCADA / BlastFurnace / N2O / MetroPT3 — Steel/EWELD
# unchanged), THEN purge their lr rows so lr is re-selected on the new data
# (~960 of 2,160 lr runs; MetroPT3 changed 16.6% — too big to caveat away):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,BlastFurnace,N2O,MetroPT3 --apply
# then just re-run the same sweep commands — resume fills the gaps.
# Weather + lambda OFAT slices (hp_search) refill via run_hp_search.py;
# re-freeze the Weather row in _DATASET_OVERRIDES from --summary afterwards.
#
# (MASK-FIX, 2026-07-15) observed-only scoring — transfer + re-refill:
#   1. transfer the 3 regenerated data.csv (WindSCADA / N2O / MetroPT3 — now
#      hold-last, no interp) AND the 4 companion mask.csv (WindSCADA / N2O /
#      MetroPT3 / BlastFurnace) to data/<name>/.  (EWELD unchanged: 0 imputed.)
#   2. purge + refill lr for the 3 whose data changed (BF's 1-cell mask is
#      below noise -> its rows stay valid; EWELD unchanged):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,N2O,MetroPT3 --apply
#   (this SUPERSEDES the 07-14d BlastFurnace lr purge line above — BF no longer
#    needs a refill under MASK-FIX; the other 3 do, for both the P3 regen and
#    the interp->hold+mask change.)
```

**Evidence state (verified from hp_search.csv)**: only **num_bands is
confirmed FLAT on clean** LTSF-8 (B2/B3/B4 mean MSE 0.4505/0.4418/0.4457).
Gate on/off clean effect UNTESTED (needs ablation sweep); ALL robustness
UNTESTED (sweep unrun). The robustness thesis = 1 confirmed fact + an unproven
bet → the robustness sweep is the #1 risk.
⚠ **λ-flat confound → 해소 방식 격상 (user 결정 2026-07-14e)**: λ 슬라이스만
재는 대신 **FEATHer OFAT 512런 전면 재실행** — 모든 FEATHer HP가 교정된
loss + 표준 Weather split 아래에서 탐색되어 provenance caveat 자체가 소멸.
07-06 동결분과의 diff를 기록할 것 (크게 다르면 그 자체가 버그 영향의 증거).
canonical mfg 아키텍처(d8/k7/p12/B3)도 재집계로 재확인 (sub-1K 제약이
지배적이라 유지 예상). Memory: project_lambda_flat_confound.md.
All audit-fix sites are tagged in-code — `grep -rn "AUDIT-FIX"` lists every
fixed location with its finding ID.

**Server state**: 07-18b **2-서버 스윕-분할** — 구 서버(Linux, 공유, GPU1)=
**① LTSF**(`fcst_results.csv`), 신 서버(Windows, 전용 GPU1, env `lee`)=
**② FEATHer OFAT**(`hp_search.csv`) + **④a 제조 lr**(`lr_search.csv`) 둘 다
(CSV 달라 동시 안전; 위 ⚡ 블록 + ROADMAP.md 보드).
**⚠ 07-20 신 서버 GPU 사고+수리**: env `lee`에 torch가 `2.8.0+cpu`(CPU 전용
빌드)로 깔려 있어 `cuda.is_available()=False` → ②·④a가 **전부 CPU로** 돌아
GPU 0%. `pip install torch==2.5.1 --index-url .../cu121`로 교체(`2.5.1+cu121
True 1` 확인). 로컬 `feather`와 동일 버전(병합 정합성). CPU로 돈 OFAT/lr 행은
**선택용 스윕이라 폐기 불요**(최종 숫자는 ③/⑤ GPU 재실행). 발사 확인 = 워커
`Using device: cuda:0` + `nvidia-smi`에 python 프로세스(util 낮은 건 초소형
모델이라 정상). 세부 = `memory/reference_new_server_env_lee.md`.
**07-20 Ctrl+C 패치 `37e4758`**(personal push): `utils/dispatch_queue.py` —
단일 Ctrl+C가 스윕 트리 전체 종료(stop 이벤트 + in-flight 자식 핸들 추적 →
`subprocess.run`→`Popen`, 워커가 새 잡 안 집고 돌던 자식 terminate, 고아 방지).
**양 서버 pull 필요.** (기존엔 Ctrl+C가 죽인 자식을 "실패 잡"으로 보고 다음 잡을
띄워 매 런마다 Ctrl+C 해야 했음.)

**Local git**: main fully PUSHED to personal/ijaehun **through `f906d1a`**
(07-23). 07-20 이후 푸시분: 그리드 도구(`fbe077d` 그리드확장·`f569e04` summary
pred_len 필터·`ed4549e` 완성-only+`--sort_by`·`e7924ee` `--per_horizon`) + **numeric
버그 3픽스**(`fb09aeb` dispatch Popen가드·`15eba56` mCE skipna·`f906d1a` RMSE=√집계MSE).
**양 서버 pull 필요**(도는 스윕 무영향, 다음 발사/표생성부터). Push policy:
**ijaehun (personal) ONLY**, not WDSLab, unless asked. 미커밋 로컬 = 클래식
베이스라인(LSTM/GRU/TCN + run_classical.py + `baselines/__init__.py` M) + 워커 프로브
플래그(--optimizer/--weight_decay) + 검토수정 E — 현 체인 끝난 뒤 별도 커밋(섞으면
list_models=15로 스윕 자동 확장).

**▶ NEXT**
1. ✅ ~~Apply audit fixes~~ — ENTIRE audit closed 2026-07-14d (P0/P1 in
   cfd90c5, P2-3..8+P3 in the close-out commit). Nothing left in the audit
   doc.
2. **Server (07-18 순차 단일-GPU 모드 — ROADMAP.md가 단일 소스)**: 3-동시
   → GPU 1 순차로 전환. 지금 **① LTSF 실행 중** → 끝나면 **② OFAT → ④a lr**
   하나씩. 남은 체인: **②·④a 완료** → `--validate` → `--summary` 전달 →
   (Claude) 8행 재동결 + 07-06 diff + canonical 재확인 + λ 판정 → 커밋 →
   pull → ③ FEATHer LTSF 160 + ④b FEATHer lr → lr `--summary` MERGE → mfg
   본 스윕 1,440 (`--save_model`) → robustness → `robust_summary.py` +
   `gate_logging.py`.
3. **A+C Intro/Motivation draft** (`manuscript/drafts/`) — needs air-gap /
   OT-isolation citations; scope to genuinely-local settings.
4. **Still-missing analysis code**: component ablation UNDER corruption
   (gate-off/band-off variants through the robustness sweep) + Pareto plot
   generator (MSE-vs-params frontier).
4b. **JMS 클래식 베이스라인 (구현됨, 미커밋/미활성화)**: LSTM/GRU/TCN
   (`baselines/`, list_models 15) + XGBoost (`run_classical.py`). 커밋 후
   **현 체인 끝난 뒤** 서버 활성화(15모델 스윕 + `run_classical.py --group
   mfg`). ARIMA/LightGBM·배포실측(#1구멍)·SWIFT문단은 후속. 상세 =
   ROADMAP.md "JMS 재포지셔닝" + `memory/project_jms_baseline_repositioning.md`.
5. ✅ ~~Push to ijaehun~~ — done 2026-07-14d (through 59c85da).
6. Raw-data cleanup under `data/`: DECIDED skip (raw = regen safety net; drop
   PMSM raw first if disk pressure).

### Decision history (compact — full blocks in `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-07-14.md`)

- **2026-07-23 (TSLib 정밀 대조 — 파이프라인 clean + LMS EAAI 발견 + xlsx 정리)**:
  (1) **TSLib(thuml/Time-Series-Library) shallow clone로 eval/train/metric/data
  4축 정밀 대조**(서브에이전트) → **버그 0, 정확성 핵심 전부 일치**. 특히 **metric
  scale = 정규화 공간(둘 다 --inverse 기본 off) → 우리 숫자 문헌비교 가능 확정**;
  best-val 선택·split(ETT 12/4/4·Custom .7/.1/.2)·train-only Scaler·공유5모델 HP 일치.
  저위험 이탈 3개(val-loss 샘플가중평균/FEATHer val=L1/train drop_last=True=옛TSLib)
  전부 의도적·방어가능·조치불요. DLinear/PatchTST 원본 생성자만 로컬 미확인이나
  **서버 실행성공+문헌범위 숫자로 경험적 확인**. clone=`C:/Users/lee/tslib_ref`(repo밖 5.4M).
  (2) **LMS-AutoTSF = EAAI(Elsevier Q1) 2026 게재 확인**("…confidence-aware…",
  arXiv 2412.06866은 프리프린트) → R4 방어 peer-reviewed 3편. **재현=공식 기본 base
  `LMSAutoTSF`(V2 confidence-aware는 어떤 forecasting 스크립트도 미배선) → 정당, 재실행
  불요**. 레지스트리 venue 갱신. [[project-baseline-lineup]]. (3) LTSF 재현결과를
  **`results/ltsf_baseline_results.xlsx`**(병합헤더+best강조+Coverage 2시트)로 정리,
  throwaway probe CSV 삭제(→`_reference/`에 TSLib clone 보관). (4) **학습워커+데이터
  로딩+utils 전 파일 버그헌트(서브에이전트 2회)**: 숫자 조용히 망칠 버그 **0**
  (마스킹·metric누적·windowing·scaling·CMAPSS·seed·losses밴드순서·noise손상수학·
  severity시드 다 clean). **실버그 1개 수정·푸시 `fb09aeb`**: `dispatch_queue.gpu_worker`가
  `Popen` 예외 시 job을 조용히 드롭(failed 미기록)→launch+wait try/except 가드로
  failed 기록+계속. 양 서버 pull 권장(도는 스윕 무영향, 다음 발사부터). 잔여=사람확인용
  minor(CORR/R2 마스킹 배치가중·Scaler가 outage hold-last에 물듦·Electricity timeF 계산낭비).
  **(추가 라운드)** config/모델 클러스터(`baselines/__init__.py` override라우팅 82키·6
  wrapper·FEATHer SPK/DTK/denorm)=**clean 버그0**; 다운스트림서 **2개 더 수정**:
  `15eba56` robust_summary **mCE skipna 버그**(fault축 통째 누락 모델이 3축평균으로 랭킹
  조작→skipna=False+경고, robustness 헤드라인표) + `f906d1a` main_table **RMSE=√(집계MSE)**
  (mean√MSE→√meanMSE, Jensen 과소보고 해소·TSLib per-run 일치·RMSE std는 point이라 생략,
  MSE/MAE에만 ±std). **이번 검증서 numeric 버그 3개 잡음(dispatch·mCE·RMSE), robustness
  스윕 아직이라 실험 무영향, paper 표 뽑기 전 선제 수정.** lr-search가 period-tied 모델을
  default period로 돎=문서화된 caveat(원고 한 문장).
- **2026-07-21 (데이터셋 6종 재확인 + 베이스라인 제외사유 통일 + CycleNet 판정
  — 전부 텍스트/결정, 코드·데이터·서버 변경 0)**: (1) **mfg-6 유지, MetroPT3
  KEEP** — 드롭 후보로 4턴 숙고했으나 대체 후보 전부 패(PMSM=2Hz persistence-
  trivial / TEP=시뮬 / flotation=공정중복·이미 07-06 패스 / ETT=generalization+
  robustness 삼중사용 / Korean10=감사리스크+전력클러스터중복). MetroPT3=코드
  no-op(드롭된 적 없음)+회전기계 자산클래스가 채우는 축. **선정기준 프레이밍=
  "비트리비얼(persistence-감사)+다양성+실측, 결측=프로토콜-handled"**(⚠ 무결성
  threshold·도메인순도 금지 — MetroPT3/WindSCADA가 아웃라이어라 자기모순).
  [[project-mfg6-metropt3-decision-0721]] + 데이터셋 **기각 로그**
  [[reference-dataset-reject-log]](데이터셋 논의 재개 시 먼저 던질 것). (2)
  **베이스라인 제외사유 self-consistency 픽스**(CLAUDE.md "Excluded from
  benchmark" 반영): S-Mamba "no MCU port"→**CUDA커널 재현불가**, TimeCMA
  "5 orders"→**사전학습=다른 자원클래스**, SWIFT "18k"→**official code 미성숙+
  FITS tie**(DLinear 18k가 표에 있어 sub-1K 논리 즉사), 방어문단 재작성, seq_len=
  96 "LTSF convention"→**배포프로토콜+MCU 입력버퍼 바이트예산**(lookback만 통일
  명시). **size는 절대 제외기준 아님 — 41p–1.6M 스팬이 목적**. (3) **CycleNet/
  RLinear=cite-not-baseline**(중간-tier·미요청·게이팅; "period 모델이라"는 폐기
  — SparseTSF도 period; 원칙상 포함가능이라 요구/1차스윕강하면 즉시 escalation).
  (4) **robustness 방어실험 재설계**: ablation-under-corruption=**1차 등-용량
  confound-closer**(gate/band-off ≈453p) / 2×2 용량스왑=보강(SparseTSF period-
  노브↑·FEATHer num_bands↓, clean정확도 유지 보고로 detune반론 봉합) / 이중프로토콜
  (메인 96 + 부록 literature lookback, **SparseTSF@720 필수**) / int8=limitations
  한 문장(리비전) / **전부 1차 스윕 효과확인에 게이팅**. [[project-baseline-lineup]]
  [[project-robustness-justification]] 07-21 블록. **라인업 12·①②④ 스윕 무영향**;
  체인 후 클래식 4종(LSTM/GRU/TCN+XGBoost)만 실제 추가. CLAUDE.md 텍스트분 미커밋
  (다음 원고/문서 커밋에 동봉).
- **2026-07-20c (① LTSF 스윕 sanity-audit — 전부 clean)**: 돌던 LTSF CSV 훑음.
  TPAMI Table 3(ETTh1/ETTh2) 재현 MSE·MAE 대부분 ±0.01(FITS/TQNet 동일값,
  PatchTST 현재가 더 좋음=개선, DLinear/ETTh2=시드불안정 특성), Weather 유효
  (07-14+ 재실행), RMSE 07-15+만 정상(옛 행 sqrt(MSE) 유도), HP소싱 표준
  (7 per-dataset 공식 / 4 단일 default = 어쩔 수 없고 정답), split 표준
  (ETT 12/4/4, 나머지 0.7/0.1/0.2), 공정성 OK. **실험 설계 문제없음, 남은 건
  덜 돈 것뿐.** 세부 = `memory/project_ltsf_baseline_verification_0720.md`.
  **decimation-vs-native 실험 = 안 함**(하중주장 아님+slow-process라 역풍
  리스크 → 리비전 카드; 인트로 anti-decimation은 인용+톤다운으로 봉합).
- **2026-07-20b (신 서버 GPU 수리 + Ctrl+C 패치)**: 신 서버 env `lee` torch가
  `2.8.0+cpu`라 ②·④a가 CPU로만 돌던 것 발견 → `torch==2.5.1+cu121` 재설치로
  수리(`2.5.1+cu121 True 1`). CPU 행은 선택용이라 유지. `dispatch_queue.py`
  단일-Ctrl+C 종료 패치(`37e4758`, personal push, 양 서버 pull). 신 서버 재실행
  = 터미널A `run_hp_search.py --jobs_per_gpu 3` / 터미널B `run_lr_search.py
  --exclude FEATHer --jobs_per_gpu 4`(RAM 여유로 3/4 워커; ② Traffic 겹칠 때
  RAM 주시). 세부 = `memory/reference_new_server_env_lee.md`.
- **2026-07-20 (코드/프로토콜 검토 수정 A~E 완료)**: 서브에이전트 검토 A/B
  (`manuscript/notes/review_findings_2026-07-19.md`) 발견 10건 전부 수정 —
  **A `1e078d7`**(#1 test/robustness 제로카운트→실패·#2 pred-NaN 가드·#9
  robustness resume 파손가드·#10 override 프로토콜축 블록리스트) + **B
  `c9ddd94`**(#3 전역 관측-셀 MSE/MAE 추정량 통일: 배치-bs가중→nansum/finite-
  count, persistence·classical과 일치, 무마스크 byte-identical) + **C·D
  `bbe6067`**(#4 main_table RMSE=√MSE 행별 유도·#5 `--seeds` 필터+시드수 불일치
  경고·#6 robust_summary dw/μ_rel/AUC 공통-셀 가드) 서버 푸시. **E는 로컬
  미커밋**(#7 run_classical 실패시 exit1·#8 XGBoost early-stop을 진짜 val
  split으로 — 클래식 묶음과 함께 대기). 검증: Steel 0.4884/N2O 0.7333
  byte-identical → **기존 행 재실행 불필요, ①·② 무영향**. 남은 것=논문 서술
  봉합 3건(lr-config 명시/FFT-period 스크립트/wilcoxon 상대화).
- **2026-07-19 (MASK-NAN-GRAD, `d56aa05` 푸시됨)**: MASK-FIX 후속 치명 버그
  발견·수정 — `_masked_loss`가 full-tensor로 `(output-target)` 먼저 계산 →
  NaN 타깃 셀이 autograd 그래프 진입 → backward `0*NaN=NaN` gradient →
  **마스크 3종(N2O/WindSCADA/MetroPT3)에서 첫 스텝 가중치 전체 NaN 발산** →
  train/val/test 집계 `sum/max(1,cnt)=0`이 **가짜 0.0 기록**(P2-5 가드도
  0.0=유한이라 통과). 수정 (1) 인덱싱 선행 `output[mask]-target[mask]`,
  (2) val_cnt==0→NaN(P2-5가 시끄럽게 실패). 3중 스모크 검증(N2O 0.0→0.7333 /
  MetroPT3 0.6914 / Steel 0.4884 무회귀). **양 서버 pull 후 ④a 재개 필수**.
  전송용 lr_search.csv는 마스크3종+GasTurbine 286행 purge 완료(1130→844).
  같은 날: ① 중간점검(886행 클린, 값 문헌범위, DLinear/ETTh2 시드불안정=특성)
  + 옛 TPAMI FEATHer 수치 vs 새 베이스라인 비교(ETT 1위권 유지, DiPE 우세,
  Electricity만 약점) → `manuscript/notes/ltsf_midsweep_check_2026-07-19.md`.
- **2026-07-16**: (1) **베이스라인 5→3 시드**(교수 결정: 제출 우선 + 리비전
  보강) — `--num_seeds 3`, 코드 수정 없음, ① LTSF 3시드 재발사, ⑤는 통째로 3시드,
  lr은 2시드 유지; 리비전 top-up = `--num_seeds 5`(resume이 2028/2029만 채움).
  (2) **원고 텍스트 재작성** — `*_jms.md` 5개(abstract/intro/related/datasets/
  methodology) mfg-6 + A+C 고속-다리(동기만) + robustness 보수 + SWIFT + half-RevIN
  + 이론 트림. (3) **고속 프레이밍**: HR-SHM은 실재 문헌이나 우리 데이터=slow-
  process → 동기 인용만. EWELD=금속산업 전력부하(+극한기상 창). Memory:
  project_manuscript_jms_drafts.md.
- **2026-07-15 (MASK-FIX)**: 결측 처리 정론화 — 모든 갭 **hold-last**(보간
  제거, 인과적) + companion `mask.csv`로 imputed 셀 표시 → 로더가 정답만
  NaN → **loss/metric 관측-only 채점**(traffic/의료/센서 표준; 12모델 동일→
  공정). 코드 5파일(prep/data_loader/data_factory/run_forecast/metrics;
  robustness는 공유 nan-aware metric 자동 상속). 검증: 배선 스모크 ALL PASS
  + 학습 Steel 0.5648/MetroPT3(17.7%마스킹) 0.7533. 실측 결측률: MetroPT3
  17.67%/N2O 5.79%/WindSCADA 1.20%/BlastFurnace 1칸/EWELD 0. **재실행 3개**
  (MetroPT3/N2O/WindSCADA — data가 interp→hold). BF 1칸·EWELD 0 = 불필요.
  선형보간 lookahead 공격면 소멸. 세부: ROADMAP.md "MASK-FIX" 블록.
- **2026-07-15 (JMS 재포지셔닝 + 클래식 베이스라인 구현)**: JMS 리뷰어=제조
  연구자 → 실무 신뢰 앵커로 **클래식 베이스라인** 추가 + **"실무층(정확·비배포)
  vs 엣지층(sub-1K·배포)" 2층 프레이밍** + **배포 실험=#1 구멍**(QEMU Layer-2
  skeleton → 실측) + ETT/Weather 강등(이미 결정). **SWIFT(2501.16178)=인용만**
  (18.1k=sub-1K 아님·FITS 타이·robustness 0·ICLR2025 withdrawn). **구현 완료
  (미커밋/로컬)**: LSTM/GRU/TCN(nn.Module, `baselines/{LSTM,GRU,TCN}/`,
  list_models 15, 기존 worker로 학습, arch 고정+lr서치) + XGBoost(별도
  `run_classical.py`, 채널독립 horizon-feature 직접다스텝, 마스킹-aware,
  GPU, 고정config+early-stop, lr탐색X). 스모크 전부 유한. **⚠ 서버 활성화
  보류**(push하면 스윕 3모델↑; 현 체인 끝난 뒤 의도적으로). ARIMA/LightGBM
  미결. Memory: project_jms_baseline_repositioning.md.
- **2026-07-14d/e**: 감사 P2-3..8+P3 전부 수정(23-check 검증, `59c85da`) →
  감사 종결; prep 갭 정책으로 데이터 4종 재생성(MetroPT3 16.6% 등) +
  BlastFurnace winsorize train-영역; **07-14e 사용자 결정: λ-슬라이스
  재측정 대신 FEATHer OFAT 512 전면 재실행**(07-06 동결 8행 대체 예정,
  diff 기록); LTSF/lr은 `--exclude FEATHer`로 재발사, FEATHer 몫(③④b)은
  재동결 후; TSLib 대조로 eval 프로토콜 정석 확인; push through 30f037d.
- **2026-07-14b**: `robust_summary.py` +4 metrics (μ_rel / AUC / mCE-vs-
  SparseTSF / degradation curve) + `gate_logging.py`, both synthetic-validated;
  R3 resolved (A+C, no short-term in paper); ROBUSTNESS_DATASETS
  GasTurbine→EWELD = [Steel, WindSCADA, EWELD, ETTh1, Weather].
- **2026-07-14**: mfg lineup LOCKED at 6 (EWELD in; GasTurbine fully removed —
  broken, no revival); freq-tied period HPs data-driven in
  `_DATASET_OVERRIDES`; SCINet added-then-reverted; TEP/ziya07/AI4I/GasSensor
  vetted+rejected; PyTSF-MfG = cite for mfg-scarcity gap.
- **2026-07-13c**: short-horizon PdM DROPPED from paper (kept as shelf);
  mfg 3→6 (+MetroPT3 +N2O +BlastFurnace-stress); Tier-2 loader prune
  (data_dict = 18 keys); push policy → ijaehun only.
- **2026-07-13/b**: strategy pivot — robustness (not accuracy) justifies
  params, protocol+citations in
  `manuscript/notes/robustness_justification_evidence.md`; server bugs fixed
  (reformer_pytorch import-time dep; --ngpu `torch.cuda.set_device`) —
  4 models were silently missing from CSVs, resume fills;
  `check_duplicates.py` added (fcst 769 rows, 0 dups).
- **2026-07-06/b**: OFAT 512 + combo 32 done → FEATHer LTSF 8 rows FROZEN into
  `_DATASET_OVERRIDES` (raw-verified); dataset precedent audit — all 6
  confirmed, "first to cast as edge LTSF" holds; wording guards in
  `manuscript/notes/dataset_precedent_audit.md`.
- **2026-07-03**: OFAT verdict — canonical mfg arch stays d8/k7/p12/B3
  (d16/p6 win val but break sub-1K); `--ngpu` dynamic queue + `--jobs_per_gpu`
  added; ⚠ FEATHer params scale with pred_len → sub-1K claim is H=96-scoped.
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
├── setup_baselines.sh              git-clone baseline upstream repos (idempotent) + pip deps
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         worker (one model × dataset × pred_len × N seeds)
│       └── run_robustness.py       robustness worker (Gauss/miss/impulse/quant sweep)
├── baselines/                      top-level model directory
│   ├── _import_helper.py           sys.modules isolation for upstream imports
│   ├── __init__.py                 registry (12 main + 30 ablation) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
│   ├── FEATHer/                    our model (FEATHer.py + ablation/{multiscale,dtk,gating,head,complexity}.py)
│   └── <11 baseline dirs>/wrapper.py + <Name>-main/ upstream clones (gitignored)
├── utils/
│   ├── seed.py                     set_seed() + parse_seed_list()
│   ├── data_factory.py             data_provider + data_select (lazy darts import; data_dict = 18 keys)
│   ├── data_loader.py              Dataset classes (ETT_hour/minute, Custom, CMAPSS)
│   ├── dispatch_queue.py           --ngpu dynamic job queue (one worker subprocess per GPU)
│   ├── losses.py                   spectral_separation_loss_scales (FEATHer-specific)
│   ├── metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
│   ├── noise.py                    4-axis corruption (gauss/miss/impulse/quant)
│   ├── severity.py                 torch-free severity grids + corruption_seed (single source)
│   └── timefeatures.py
├── tools/
│   ├── prep_manufacturing.py       reproducible raw→data.csv for the 6 MFG datasets (tracked; data/ gitignored)
│   ├── audit/check_progress.py     read results CSV, print coverage + mean±std preview
│   ├── audit/check_duplicates.py   resume-key dup audit + --fix (schema-aware: robust CSV key auto-extends)
│   └── paper/                      Phase 5 table/figure generators
│       ├── main_table.py           5-seed mean±std table + persistence column (md / latex / csv)
│       ├── wilcoxon.py             pairwise signed-rank vs FEATHer
│       ├── ablation_table.py       5 axes × 4 horizons aggregator (clean accuracy only)
│       ├── robust_summary.py       μ_rel / AUC / mCE-vs-SparseTSF / degradation curves + heatmaps
│       ├── gate_logging.py         FFT gate-under-corruption logging (per-band weights, CSV+PNG)
│       └── persistence_baseline.py copy-last baseline under the exact worker protocol
├── deployment/cortex_m3/           Layer-1 edge-cost estimator + qemu/ Layer-2 skeleton (WSL)
├── results/                        fcst_results.csv · robust_results.csv · edge_estimates.csv · checkpoints/<exp_tag>/
├── data/                           raw + processed data (gitignored — transfer to server manually)
└── manuscript/                     reviews, drafts, notes (gitignored; tex_workspace/ + drafts/*.md selectively tracked)
```

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
# Bypasses _DATASET_OVERRIDES (default arch) — period overrides only affect the main sweep.
python run_lr_search.py --check
python run_lr_search.py --ngpu 2            # resumable
python run_lr_search.py --summary           # best lr per (method,dataset) → MERGE into _DATASET_OVERRIDES
                                            # (merge into EXISTING period entries — no duplicate keys!)

# === FEATHer OFAT HP search (LTSF generalization only; DONE, rows frozen) ===
python run_hp_search.py --check / --summary / --validate
# Windows gotcha: run --summary with PYTHONUTF8=1 PYTHONIOENCODING=utf-8 (em-dash vs cp949)

# === Robustness sweep (after mfg-main checkpoints exist) ===
# (audit fully closed 2026-07-14d — only blocker left is the mfg-main
# checkpoints themselves)
python run_robustness.py --check            # 1,200 ckpt × 20 cond = 24,000 rows
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --model FEATHer --train_exp_tag main

# === Ablation sweep (5 axes × 30 FEATHer variants; scope extends to mfg) ===
python run_forecast.py --ablation_axis all --data ETTh1 --exp_tag ablation
# axes: ms (15) / gate (4) / dtk (4) / head (4) / complexity (3)

# === Progress audit ===
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py      # schema-aware (robust CSV key auto-extends)

# === Paper artifact generators (Phase 5) ===
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer
python tools/paper/ablation_table.py --axis ms --format latex
python tools/paper/robust_summary.py --output_dir manuscript/figures/
python tools/paper/gate_logging.py --data Steel --pred_len 96 --ckpt <path>

# === Data regen (data/ is gitignored) ===
python tools/prep_manufacturing.py            # all 6 mfg recipes
python tools/prep_manufacturing.py EWELD      # one (BlastFurnace needs openpyxl)

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

(SCINet was briefly added 2026-07-13 then REMOVED — reviewer only named it as
prior multi-scale work, not a baseline request; Related-Work differentiation
instead.) → 5 orders of magnitude (267 → 1.6M). Three recent peer-reviewed
SOTA (DiPE-Linear DASFAA'26, MDMLP-EIA AAAI'26, LMS-AutoTSF EAAI'26 — the
last confirmed 2026-07-23 as journal-published, not arXiv-only) address R4's
"outdated baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation.
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures into uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** — selective-scan **custom CUDA kernels** → not fairly
  reproducible under our unified from-scratch protocol (AMP off, identical
  training loop); `mamba-ssm` intentionally not installed. *(NOT "no MCU
  port" — the ceiling models, e.g. iTransformer 841k / MDMLP 1.6M, don't fit
  an MCU either; MCU-portability can't be the exclusion criterion.)*
- **TimeCMA** (AAAI 2025) — frozen **pretrained GPT-2 features = a different
  resource class** (not trainable from scratch). *(NOT a size cutoff — we KEEP
  MDMLP 1.6M as a ceiling; size is never the exclusion criterion, the 41p–1.6M
  span IS the point.)*
- **SWIFT** (arXiv 2501.16178) — **no mature official code** → not fairly
  reproducible under our protocol; performance **ties FITS** (already in the
  table → FITS represents that point). ICLR2025-withdrawn = secondary note
  only. *(NOT "18k not sub-1K" — DLinear 18,624p is a benchmarked baseline.)*

Reviewer-defense paragraph (paper draft):
> "Model size is not our exclusion criterion — our benchmark deliberately
> spans 41 to 1.6M parameters to map the capacity–accuracy frontier. We
> exclude only architectures we cannot fairly reproduce under a unified
> from-scratch protocol — Mamba-family models (S-Mamba) requiring custom
> CUDA kernels, and methods without mature official code (SWIFT) — or that
> belong to a different resource class, namely LLM-augmented / pretrained
> forecasters (e.g., TimeCMA) depending on frozen external features rather
> than being trained from scratch. These are cited in Related Work but lie
> outside our reproducibility-controlled benchmark."

## Training protocol — TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ✅ Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | Deployment protocol: unified across all models; justified by MCU input-buffer byte budget (720×D×4B vs 96×D×4B). Other HP from each paper's recipe — lookback is the one deliberate override (state this to avoid the "published recipe vs 96 override" self-contradiction; SparseTSF@720 goes in an appendix rank-stability check). |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; selection = best-val epoch, test evaluated once on that model |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025–2027 (3 seeds; ⤴5 in revision) | 07-16 triage: `--num_seeds 3` submit-first, top-up to 5 (`--num_seeds 5`) in revision — 3⊂5, resume fills the 2; base_seed/exp_tag fixed. lr search stays `LR_SEEDS=[2025,2026]`. |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ⚠️ Per-(method, dataset) overrides

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

1. **Multi-band Decomposition** — depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). `num_bands ∈ {2,3,4}`.
2. **DenseTemporalKernel (DTK)** — project to latent → depthwise causal conv
   → project back. Shared across all bands.
3. **FFTFrequencyGate** — FFT magnitude → Conv1d → softmax weights, one per
   band, adaptive per sample. Casts to float32 (cuFFT half-precision).
4. **SparsePeriodKernel (SPK)** — period-aware head: reshape by period phases,
   shared linear projection, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss (FEATHer-only;
⚠ band-order inversion bug — audit P1-1 — fix pending).

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
| **Steel · WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD** (MFG **main table**; BlastFurnace D=28 = stress tier) | local CSV (`tools/prep_manufacturing.py`) | [96, 192, 336, 720] |
| CMAPSS / CMAPSS3 / PMSM (PdM **shelf**, not in paper; unit-aware `Dataset_CMAPSS`) | local CSV | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (generalization) | darts | [96, 192, 336, 720] |
| nrel (kept per user) | local CSV | [96, 192, 336, 720] |

Robustness set: `run_robustness.py:ROBUSTNESS_DATASETS` =
**[Steel, WindSCADA, EWELD, ETTh1, Weather]** (EWELD's 5,741 extreme-weather
event windows = the only non-synthetic distribution-shift test).
EWELD gotcha: strong daily cycle → MUST report a seasonal-naive baseline
(copy-last-day ~0.55). Each `data/<name>/` holds raw + cleaned `data.csv`;
regenerate with `python tools/prep_manufacturing.py [<name> ...]`
(BlastFurnace needs `openpyxl`). `data/` is gitignored — transfer manually.

## Known issues / gotchas

- **⚑ AUDIT (2026-07-14) — FULLY CLOSED** (P0/P1/P2-1,2 in cfd90c5;
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
  LMS_AutoTSF, TimeMixer, TimesNet). Missing → those 4 die at import, write
  no CSV row, and look "not yet run" (failed workers currently exit 0 —
  audit P2-1). `setup_baselines.sh` installs `reformer_pytorch==1.4.4`.
- **AMP is forcibly disabled** in the worker. Re-enabling breaks FITS
  (complex grads) and TimesNet (half-precision cuFFT).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import (sys.modules collisions otherwise).
- **TQNet** receives `cycle_index = torch.zeros(B)` (our DataLoader has no
  per-sample cycle position) — consistent across all batches.
- **R2 metric** can explode to extreme negatives on near-constant channels
  (SST≈0); MSE/MAE/RMSE unaffected; R2/CORR supplementary only.
- **FEATHer params scale as D²** (`in_proj`): sub-1K holds for D≤14 only.
  Weather 1.4K / Electricity 115K / Traffic 776K / BlastFurnace(D=28) 1,986
  are framed as stress tests.
- **FEATHer params also scale with pred_len** (SPK backbone): quoted counts
  are H=96 values; at H=720 D≥11 exceeds 1K. Sub-1K claim must be scoped to
  H=96 in the manuscript (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1–3** ✅ infra (multi-seed, 12-model registry, 4-axis noise).
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
  template; Highlights 3–5 bullets ≤85 chars; Vancouver refs; CRediT +
  declarations). Draft in markdown under `manuscript/drafts/` → paste into
  Word. Order: Intro (A+C framing) → Datasets → Related → Method polish →
  results when sweeps land.
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
