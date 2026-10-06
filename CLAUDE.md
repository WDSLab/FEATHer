# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
지난 상태 블록은 `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-10-01.md`(06월~09-30) · `..._2026-10-06.md`(10-01~10-04 실험
실행 기록)에 원문 그대로 있다 (평소엔 읽을 필요 없음 — 세션별 상세는 메모리 파일).

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) = 기계 옆 MCU
(NUCLEO-F103RB: 128 KB flash / 20 KB SRAM, FPU 없음)에서 **장기(H 96~720) 다채널 예측**을 하는
296-파라미터 모델 (Steel D=6, H=96). 논문 = **Journal of Manufacturing Systems (JMS)** 신규 투고
(TPAMI reject → IoT-J → JMS).

**주장의 범위**: 칩에 실제로 들어가는 모델들 중에서 비교한다. 최신 경량 모델은 전부 채널을 독립으로 다루는데,
제조 데이터에서는 채널을 섞는 게 이득이고, FEATHer는 섞기를 MCU 메모리 안에 들어가게 만들었다
(칩당 43채널). flash 가 H에 거의 무관(18.1→18.5 KB, H96→720; 파라미터는 MetroPT3 340→444 — 옛 "1.008배"는 동결 전 모델 값, 쓰지 말 것)해서 긴 H로 갈수록 칩 안 비교에서 유리해진다.
판정 기준은 파라미터가 아니라 **실측 flash·SRAM·보드 지연시간**.

## Current status (2026-10-06) — 실험 끝, 원고 글쓰기 단계

**모든 실험 끝(보드 재생 포함). 표 T1–T13 + Fig.1 이 v5 에 다 채워짐(노란 `?` 없음). 남은 것 = 글쓰기(1~5장 본문 비어 있음).**
상세 = 메모리 `project_session_1006.md`(10-02~10-06 전체 + HANDOFF) · `project_fable_review_1005.md`(독립 검토·결정) ·
`project_board_replay_1004.md`(보드 재생) · `project_ablation_plan_1001.md`(ablation·새 아이디어 기록).

- **원고 파일**: `FEATHer_JMS_v1~v4.docx` = 사용자가 검토·수정한 사본 (**덮어쓰기 금지**). 빌드 대상 = **v5**
  ← `python manuscript/version/_build/make.py` (`_build/template_merged.docx` → `build_docx.py` + `tables_cf.py`).
  **사용자가 Word 에서 고치면 빌더에 옮길 것** — 빌드 전 직전 버전과 글자 diff. 렌더 확인 = Word COM → PDF → pdftoppm.
  제목 *FEATHer: Long-horizon multi-sensor forecasting on machine-side microcontrollers for manufacturing systems*, 저자 5명
  (공동1저자 † / 공동교신 *). 소속·교신 표기 = 제출된 CF-JEPA R2(`CF-JEPA/manuscript/revision_r2_20260928/submission/
  20261004_EM_submitted/CFJEPA_revised_FV.docx`)와 같은 형식(학과 줄 + 대학 줄, "*Correspondence:").
- **표 (v5, 10-06)**: T1 경량 모델 설계 · T2 데이터셋(+ "H = 720 ahead" 열: 12 h~30 d) · **T3 칩 비용 · T4/T5 정확도(9모델)** (10-06 번호 변경: 비용이 정확도 앞, 흐름 = `manuscript/drafts/analysis_flow.md`; 메모리 파일의 "T5 칩 비용"·"T3/T4 정확도"는 옛 번호) ·
  Fig.1 · T6 칩 안 종합(7모델; + Dataset rank 행, 서버급 7모델 상한 문장) · **ablation T7 MLP·섞기(+Ch/chip) / T8 헤드(H열:
  MSE + 가중치 KB) / T9 게이트 / T10 밴드+스펙트럼 손실** (CF-JEPA 형식: 데이터 6열, 평균 ± std, **굵게 없음**; RevIN ablation 없음)
  · **T11 입력 길이**(7모델 × L96/192/336) · **T12 칩당 채널 vs 입력 길이(9모델, L96 실측 / L>96 상한)** · **T13 LTSF**.
  **표 서식 규칙(사용자)**: 모든 표 폭 = 본문 폭 8844 twip, **행 높이 정확히 0.5 cm**(넘치면 잘림 → 렌더 확인 후 라벨 줄이기),
  **글자 전부 8 pt**(위첨자 기호만 7 pt). 데이터 이름 = "**industrial datasets**"(Ind. = 제조 공정 3 + 산업 설비·공정 3).
- **▶ 다음 (글쓰기 순서)**: 3장 방법(구조 그림, SPK→SparseTSF 귀속, **"칩에 맞춘 설계" 철학: 메모리가 모델을 고른다 →
  입력 L 이 SRAM 을 정하고 출력 H 는 SRAM 무관 → L96 이유(T11·T12)**, 밴드·게이트 = 칩 예산 안의 다중 규모 입력부) → 4장(보드
  재생 문단 + 표/문단 결정) → 5장(초안 `drafts/sec5_implications.md` 있음; [보드 재생] 빈칸 채우기) → 한계점 → 서론(뼈대
  `drafts/sec1_intro_outline.md`, 10-02 조사 반영 필요) → 2장 검토(`drafts/sec2_related.md`) → 초록 → 노란 표시·참고문헌.
  빌더에 drafts md 영어 줄 삽입 코드는 **아직 없음**. 커밋은 요청 시, personal 만.
- **글쓰기 방식**: `manuscript/drafts/secN_*.md` 에 **영어 한 문장 + 한국어 한 줄(`> `)**. 7월 `*_jms.md`·arXiv 프리프린트 문장 재사용 금지.
  인용 = `[저자연도]` + 파일 끝 검증 서지(메모리 `reference_related_work_1001.md`, `reference_problem_validation_1002.md`).
- **⚠ 우리 프리프린트 arXiv 2601.11350**: 옛 구조·숫자 사용 금지, "real board (LM3S6965EVB)" 는 실제 QEMU → 서론에서 인용 +
  "이번이 첫 실보드 실측" 명시, arXiv v2 정정은 공저자와 결정. 코드 공개 = `github.com/WDSLab/FEATHer`.
- **확정된 프레이밍 (10-04)**: "기계 옆 칩에서는 **메모리**가 모델을 고르고(파라미터·속도 아님), 그 메모리 안에서 정확도를 올리는
  열쇠는 **센서 섞기**다. 경량 모델은 비용 때문에 안 섞는다(CMoS 인용) → 우리는 섞기를 20 KB 에 넣었다." 기여 순서 = ① 칩 위
  실측 비교(9모델, 같은 C 규칙) ② FEATHer(섞는 유일한 sub-1K, 43 ch/chip, H 무관 크기) ③ 산업 데이터 근거 ④ 실무 가이드(5장 본문).
  새 부품 시도 3개(근거리/원거리 헤드 A, 예측 후 섞기 B, 센서 고장 활용) 모두 실패 → **부품 추가 그만**.
- **미결 작은 결정**: `data/*/data.csv.premask` 3개(~110 MB) 삭제 여부 · 10-01~10-06 작업분 커밋·push 여부(personal).

### 정리 (10-01, 사용자 승인) + 코드 공개 전 할 일
- **삭제함**: `__pycache__`(85파일) · `deployment/board/build/`(make 가 다시 만듦) · `tex_workspace` LaTeX 부산물 ·
  `data/*/data.csv.preP3bak` 5개(133 MB, 7월 수정 전 백업). **남은 백업**: `data/*/data.csv.premask`(MASK-FIX 전) — 미결.
- **옮김**: 9월 dev 중간 파일 45개 + `_devlogs` → `results/_archive/dev_2026-09/`(README 에 파일별 출처; 도구 기본 경로는
  아직 `results/` 라 재실행 시 `--csv` 로 경로 지정) · 옛 초안 15개 → `manuscript/drafts/_old/`(재사용 금지) ·
  `run_config_sweep.py` → `_archive/`. **제자리 유지**: `deployment/BOARD_RESULTS_20260824.md`(펌웨어 바닥 10.4 KB 출처, 코드
  주석 4곳이 가리킴)·`BOARD_SESSION_PLAN.md`, `board_footprint_recompute_*`(Table 7 의 31채널 근거), `probes`, `*_r2.csv`.
- **코드 공개(WDSLab/FEATHer) 전에 할 일 — 지금은 안 함, 검증 필요**: ① `FEATHer.py` 의 기각된 연구 옵션 제거(gate_mix, head2,
  mix_listen/lag/zero_init, input/input_raw 섞기, band_mode fft, spk_hidden, gate_mode 변형 등) — 제거 후 동결 모델 파라미터·
  출력이 바이트 단위로 같은지 확인 ② 보드 예전 포트 정리: 옛 938p `feather_infer` + `test_golden` + `export_golden.py`(다른
  export 스크립트가 import 함), 기각된 `feather2in` 포트 — Makefile 수정 + WSL `make test` 재실행 ③ `deployment/cortex_m3`
  (QEMU, 프리프린트가 쓴 평가)는 기록용으로 보관 ④ `tools/dev` 도구들의 기본 경로 정리.

### 확정 숫자 (원고에 쓰는 값)
- **mfg (3시드, 6데이터 × 4H)**: FEATHer-mix OVERALL **0.6990** · FEATHer-CI 0.7361. **섞기 이득 = +5.9% 하나로 통일 (10-05: 24칸 per-cell % 변화의 평균; H별 +4.8/+5.6/+6.8/+6.2 는 같은 규칙; 평균끼리 나눈 +5.3% 는 쓰지 말 것). 검정 = 셀 MSE 쌍 one-sided Wilcoxon (T6·T11 동일). 파라미터는 D 에 따라 296~1,924 (H96).**
  옛 R16(452p) 0.6943 → R4로 +0.66% 손해 대신 칩당 31 → **43채널**.
- **LTSF (8데이터 × 4H = 32칸, 경량 9개)**: FEATHer-CI 평균 MSE **0.3753** / MAE **0.3662** = 둘 다 최저, 그러나
  **MSE 1등 0칸**(MAE 2칸) — **평균만 좋음**. FITS 0.3759 와 0.16% 차. MSE 평균순위 3.81(1위 STSF-MLP 3.56), MAE 3.50(1위).
  **32칸 모두 6위 이내 = 하위 3위에 한 번도 안 든 유일한 모델**(칸별 1등과의 평균 차 5.2% vs FITS 5.3%; DLinear 는 1등 11칸이지만
  최악 45%, STSF-MLP 52%). 1:1 MSE: SparseTSF 29–3 · MixLinear 27–5 · DiPE 19–13 · CycleNet 21–11 · FITS 22–10(p 0.13 보정 전;
  09-30 Holm 0.25) · DLinear 18–14(p 0.12) · **STSF-MLP 11–21 짐**. mix 는 LTSF 에서 CI 보다 나쁨.
- **Ablation (H96, 6데이터, 변화 = 데이터별 % 변화의 평균)**: 섞기 끄기 +4.8% · 잔차 MLP 끄기 +3.8% · R16 −0.4%(31채널) ·
  게이트 제거 +0.8 · 균등 0.0 · 고정 학습 비율(FFT 없음) +0.1 · 밴드 1개 −0.1. **값을 하는 부품 = 섞기와 잔차 MLP 뿐.**
  (평균끼리 나눈 4.3/3.4% 는 다른 규칙 — 쓰지 말 것.)
- **섞기 효과**: H별 +4.8/+5.6/+6.8/+6.2 (단조 아님 → "긴 H에서 더 크다"; T7 주석), WindSCADA +0.6 → +12.3.
  LTSF(T12 마지막 줄): ETT −1.7~−4.9, Exchange −10.9 vs Weather +6.7, Electricity +5.3, Traffic +2.6. ⚠ 채널 수만으로는 설명 안 됨
  (Steel 6채널은 +13%) → "공정으로 묶인 제조 센서에서 이득"이 더 맞지만 사후 설명. 두 변종을 모두 싣는 것이 방어.
- **보드 (`results/board_latency_20260929.csv`, 27칸 PASS)**, 7채널 H96/H720 ms: SparseTSF 47/90 · MixLinear
  87/901 · CycleNet 157/✗ · **FEATHer-mix 161/208** · FEATHer-CI 202/250 · FITS 274/5,896 · DLinear 315/✗ ·
  STSF-MLP 434/1,445 · DiPE 727/12,735. 전류 8모델 29.6~30.0 mA (CycleNet 미측정) → 에너지 = 3.3 V × 30 mA × 지연.
- **칩당 채널**: mix 43 · CI 48 · FITS 47~49 · CycleNet 50. flash 초과: DLinear 146/255/546 KB (H192/336/720),
  CycleNet 135/281 KB (H336/720).
- **H=720 칩 안**: FEATHer-mix 가장 정확(0.808), 더 빠른 건 SparseTSF 뿐(7% 덜 정확).
- **입력 길이 (T11, 정식 3시드, 칩 7모델)**: MSE 평균순위 L96 1.50 / L192 2.21 / L336 2.71 = 셋 다 1등; 평균 MSE 0.699/0.667/0.647
  (L336 2등 FITS·DiPE 0.650). MAE: L96 1등, L192 FITS 와 공동 1등, **L336 3등**(DiPE 2.71). 자기 개선 L336 −7.8% vs 베이스라인
  −10~11%. 섞기 이득 L96 +5.9%(23/24, p<0.001) → L192 +4.1%(20/24, p<0.001) → L336 +2.7%(18/24, **p=0.045**).
- **T12 칩당 채널 vs L (MetroPT3 D=7)**: 모든 모델 ≈ 49 → 24 → 14 → 6 (L96 실측, 이후 상한); DLinear L≥192 면 H96 에서도 flash ✗;
  CycleNet L≥336 ✗; FITS L720@H96·L≥336@H720 ✗; FEATHer-mix 자기 버퍼 기준 43/18/7/0. **출력 H 는 SRAM 무관**(보드 재생 RAM 동일).
- **Dataset rank (T6, 6데이터 단위)**: MSE FEATHer-mix 1.33(FITS 2.83), MAE FITS 와 공동 2.17; Friedman p<0.001/0.002.
- **정확도 상한**: 서버급 7모델(75 K~4.0 M params, 칩 불가) MSE 0.643~0.685, FEATHer-mix 0.699 = 2.0~8.6% 위.
- **연산 수 vs 보드 지연 (11칸, `results/ops_vs_latency_20261006.csv`)**: 파라미터 r² 0.000, 연산 수(보드 구현대로 변환 비용) r² 0.96.
- **보드 실시간 재생 (MetroPT3 s2025, 1시간마다)**: H96 1,022회·H720 1,012회 전부 보드 MSE = PyTorch(0.542156 / 0.613206),
  지연 평균 161 / 211 ms(입력에 따라 ±10%; T3 의 208 은 입력 하나), 계산 비율 0.004 / 0.006%, 계산 전력 0.38 / 0.50 J/일,
  펌웨어 14.3 / 14.7 KB flash, 7.55 KB RAM. ⇒ 현장 주기에서 속도는 선택 기준 아님 → 정확도·메모리가 고름.
- **헤드 (T8)**: 선형 헤드 −5.5/−4.4/−2.9/−1.9%(더 정확) 이지만 H336 131 KB·H720 273 KB 로 칩 ✗ → SPK 유지(DLinear 와 같은 칩 기준).
- **가이드 (5장 본문)**: H96 은 DLinear(MSE)·CycleNet(MAE) 가 더 정확·칩 OK; H≥192 는 FEATHer-mix(H192 MAE 만 CycleNet);
  ≤43 ch + H≥192 → FEATHer-mix, 44~48 ch → CI 모델 또는 칩 2개.
- **필터 없이(경량 8개, 24칸)**: MSE 순위 DLinear 1.83 > 우리 2.21 · MAE CycleNet 2.04 > 우리 2.96(1등 칸은 우리 9로 최다)
  → 두 지표 모두 2위 안은 우리뿐. MSE 로 우리를 이기는 가장 작은 모델 = DLinear (flash 4.6배 H96 ~ 30배 H720; T3 값 83.5/18.1, 546/18.5 KB).

### 주장 금지 (반복 확인됨)
- "DLinear를 이긴다" — MSE 11–13. DLinear는 모든 H에서 더 정확, H≥192에서 칩에 못 들어갈 뿐.
- "H가 길수록 (DLinear 대비) 격차가 커진다" — 틀림. DLinear 의 MSE 우위가 오히려 커짐(3.4→6.2%); 커지는 건 비용 차이.
- int8/행렬 합치기 실험 다시 제안 금지 (09-29 결정: 베이스라인 재최적화 안 함, 한계점 한 문장).
- "LTSF 1위" / "LTSF 에서 가장 정확" → "on par with the best lightweight models" (+ 쓸 수 있음: "32칸 모두 하위 3위 밖인 유일한 모델").
- "현장 의사결정(정비·에너지)이 개선된다" — 평가 안 함(Steel 피크·MetroPT3 조기경보 둘 다 기각됨; 지표 바꿔 재시도 = p-해킹) →
  "어디에 쓰이는지"는 글로 설명, "나아졌다"는 한계점/향후 과제.
- 게이트 표현(10-04 사용자): 숫자와 함께 "평균 오차를 0.3% 줄인다"(약간)로 씀 — "크게 개선" 금지. 긴 H 부호는 −0.5/+1.3/−0.4 다 적음.
  밴드·스펙트럼 손실은 "정확도를 ±1% 안에서 유지" — **"효과 없음" 이라고 쓰지 말고 기여 순위로**(섞기 5.9 · MLP 3.8 > 게이트 0.3 > 밴드).
  밴드 3개 > 2개는 6/6 데이터(+1.0%, p 0.03). 밴드·게이트는 **해석 도구로도 주장 금지**(게이트 비중이 시드마다 바뀜).
- "섞기가 센서 고장에 강하다" — 반대(고장을 퍼뜨림). "보드 재생 = 실설비 설치" 아님(저장 데이터 재현).
- 모양 지표(CORR 등) 주장 — 사전 규칙으로 기각, MSE/MAE 만.
- 순위를 말할 땐 항상 **어느 풀인지** 명시 (9모델 전체면 DLinear MSE 1등 13 > 우리 10).
- "제조 TSF 벤치마크 최초" — RCIM(arXiv 2411.17499)이 선점. "최초"는 MCU 배포·전 채널 동시·경량 tier 로 좁힐 것.

### 한계점에 넣을 문장
(10-06) DLinear·int8 은 한계점이 아니라 **방법 절 한 줄**: "모든 모델 float32 같은 규칙, 양자화는 모든 모델에 적용 가능한 별도
최적화라 범위 밖" · CycleNet 위상 인덱스 0 고정 · H 길이 변환은 전
모델 직접식 · CycleNet 전류 미측정 · 재계산 허용(정규화+밴드)은 FEATHer-mix 포트의 예외 · L=96 고정(긴 룩백
네이티브 모델에 불리 — Traffic/Electricity 격차 +56~73%) · Steel 학습 불안정(원인 미규명) · 입력 길이 ≤ 336
(L720 은 7채널 입력 버퍼만으로 SRAM 초과, 모든 모델 제외) · SPK 주기 P 는 L·H 를 나눠야 해서 ≤ 48 (Steel/EWELD 실제 96) ·
현장 의사결정 지표 미평가(향후 실설비 적용) · mfg=mix / LTSF=CI 변종 선택(두 변종 모두 보고) · 1초보다 짧은 간격(진동 등)이면 속도가
중요해짐 · 연구 범위 = 가장 싼 20 KB 급 칩(1 MB 급이면 큰 모델도 들어감) · **(10-05 사용자) 센서 고장: 한 센서가 멈추면 섞기가 그 고장을 연결된 센서로 퍼뜨림 (멀쩡한 센서 MSE Steel +8.9%, 평균 +2.0%; CI 는 0) → 배포 시 고장 감지 후 처리 필요** (results/dev_sensor_dropout.csv).

### JMS 글쓰기 방향 (10-01)
- **JMS 가 좋아할 것(우리 근거)**: ① 기계 옆 실제 칩 실측 + 보고 정확도 = 배포 C 코드 ② 실무자용 선택 가이드(센서 수·H →
  이 칩에서 고를 모델, 칩당 43채널) ③ 제조 공정 센서는 묶여 있어 섞기가 이득인데 경량 모델은 전부 안 섞음 ④ 실제 산업 데이터 6종
  ⑤ 긴 H = 현장 판단 시간(H720 = MetroPT3 12시간 ~ BlastFurnace 30일) ⑥ 클라우드 없이 현장 독립 동작(OT 격리, 인용 필요)
  ⑦ "파라미터로 칩 비용을 고르면 틀린다". 비용은 돈 아니라 칩 등급·칩당 센서 수까지만.
- **약점**: 현장 판단 개선 증거 없음 · 실설비 설치 없음 · 방법 새로움 약함(밴드·게이트 효과 0) · DLinear 더 정확(int8 질문).
- **5장 쓸 때**: 데이터별 "H720 = 실제 몇 시간/며칠 → 어떤 현장 판단" 표(측정 간격으로 계산) · 배포 가이드(결정표 데이터로) ·
  서론에서 섞기를 공정 센서 결합과 연결. LTSF 는 방어용("손해 안 봄") + 변종 선택 문단을 먼저 씀.
- **억셉 판단(10-01, 데이터 아닌 판단)**: 지금(원고 잘 마무리 전제) 리젝 40~50%, 나머지 대부분 메이저. 최대 변수 = 입력 길이 실험
  (FEATHer 도 비슷하게 좋아지면 30%대, 베이스라인이 앞지르면 크게 악화). 바로 억셉은 기대 안 함; 목표 = 메이저 리비전.

### 용어
**channel** (feature 금지). 제목·키워드의 "multi-sensor" 는 유지. SPK 헤드는 **SparseTSF에 귀속** 명시(방법 절).

## Architecture (FEATHer) — FROZEN (09-04 구조, 09-23 mlp_rank 4 재동결)

`baselines/FEATHer/FEATHer.py`, 레지스트리 `_FEATHER_FROZEN` (`baselines/__init__.py`). 변경 시 스윕 재실행 필요.

```bash
--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none --gate_mode banded --mlp_rank 4
mfg:  --channel_mode mix          LTSF: --channel_mode ci_full   (overrides 로 자동)
```
```
RevIN -> 3 bands: conv k=3 | conv k=5 | AvgPool(4)+interp -> banded FFT gate -> weighted sum
      -> residual MLP (D->4->GELU->D) -> SPK head -> RevIN^-1
```
- DTK·`in_proj` 는 측정으로 제거(클래스는 ablation 용으로 남음). `--d_state`/`--kernel_size` 는 죽은 인자.
- `--band_kernels 3,5` 는 같은 구조지만 RNG 흐름이 바뀌어 초기화가 다름 → 항상 `--point_kernel` 형태.
- banded 게이트 = 대역별 에너지 비율 → `Linear(S,S)`, 0 초기화(균등에서 시작).
- 잔차 MLP = 유일한 비선형. `ci_full` 에선 1→4→1.
- Loss = MSE + 0.01 × spectral-separation (λ 올리면 단조 악화, 사실상 꺼진 항).
- 파라미터: mfg-mix Steel 296 / CI 76 · LTSF ETTh1 CI 298 / mix 454 (merge_raw 가 num_params 로 검증).

## Environment & gotchas

- **Python**: `C:\Users\lee\.conda\envs\feather\python.exe` (torch 2.5.1+cu121, darts, reformer_pytorch; Py 3.9).
  bash 의 `python` 은 Windows 스토어 스텁 → 경로 직접. docx 스킬 스크립트는 Py≥3.10 →
  `/c/ProgramData/anaconda3/python.exe`.
- **Dropbox 폴더에서 `rmtree` 금지** (반쯤 지워짐; golden/ 한 번 복구). 파일 단위로.
- Bash heredoc 안의 `\n`/`\u` 이스케이프가 깨짐 → 스크립트는 Write 로 파일 작성 후 실행.
- Windows 출력: `PYTHONUTF8=1` (Weather 'µ', em-dash 로 cp949 사망).
- 원고 렌더 확인: PowerShell Word COM `SaveAs2(pdf, 17)` → `pdftoppm` (LibreOffice 없음).
- 보드: WSL gcc 10.3 (footprint 측정) vs CubeIDE GCC 14.3 (보드 빌드). WFI 로 잠든 칩은 connect-under-reset.
  CubeIDE 프로젝트는 repo 밖 `C:\Users\lee\STM32Cube_projects\blink_test`.
- AMP 강제 OFF (FITS 복소 grad, TimesNet half cuFFT). upstream 은 반드시 `_import_helper.isolated_import`.
  `baselines/*/*-main/` upstream 클론은 gitignore — 수정 금지, `setup_baselines.sh` 로 재클론.
- R2/CORR 은 보조 지표. 저장된 R2 열은 쓸 수 없음(`tools/paper/compute_r2.py` 로 사후 계산).
- Git: **personal(ijaehun) 에만 push**, 커밋은 요청 시에만.

## Repository structure

`utils/` = 데이터·학습 기계, `tools/` = 사람이 돌려 산출물을 만드는 것 (학습 경로에서 import 안 함).

```
run_forecast.py                   오케스트레이터 (--check, resume, --ngpu/--jobs_per_gpu, --ablation_arm)
scripts/benchmarks/run_forecast.py 워커 (모델 × 데이터 × H × N시드)
baselines/__init__.py             레지스트리(15) + _METHOD_DEFAULTS + _DATASET_OVERRIDES + _FEATHER_FROZEN
baselines/<Name>/wrapper.py       upstream Model 을 forward(x)->(B,H,D) 로 감쌈
utils/                            prep_manufacturing (raw -> data.csv + mask.csv) · data_factory · data_loader
                                  · dispatch_queue · losses · metrics · seed
tools/audit/                      check_progress · check_duplicates · purge_rows · results_table · scan_divergence
tools/search/                     동결 config 를 고른 기록 (재현용)
tools/dev/                        bs512 빠른 dev 루프(run_dev) + 진단 도구 (README 참고)
tools/paper/                      merge_raw · table_main/table_mfg/table_deploy · fig_pareto_board · wilcoxon
                                  · ablation_summary · measure_complexity · compute_r2 · persistence_baseline
deployment/board/                 C 포트 src/*_infer.{c,h} 9종 · export_golden_*.py · test/ (make test)
                                  · measure_footprint.py · board_run.py (빌드->플래시->읽기 자동) · README(공통 C 규칙)
manuscript/version/               FEATHer_JMS_v1/v2.docx + _build/ (원고 빌더)
manuscript/notes/                 dataset_precedent_audit · reject_mail · 아카이브 (gitignored)
results/  analysis/  data/        결과 CSV · 표 · 데이터 (data/ 는 gitignore, 서버로 수동 전송)
deployment/board/ (10-06 추가)    main.c.stream(실시간 재생 펌웨어) · stream_flash.py · replay.py (--sim / --port COM3, pyserial)
_archive/                         은퇴 코드: robustness · ablation_v1 · grids · classic_baselines · lr/hp search
```

**C 포트 9종**: feather2 / feather2r (mix, 보드는 재계산 스케줄 feather2r) · feather2ci · dlinear · cyclenet ·
stsf · stsf_mlp · mixlinear · dipe · fits. **공통 규칙 ①**("원래 계산은 그대로, 메모리는 정리만"): 안 섞는 모델은
채널 하나씩 · 출력은 끊을 수 있는 단위로 바로 방출 · 작업공간 공유 · 재계산 금지(FEATHer-mix 의 정규화+밴드
재계산만 명시적 예외). 96점 FFT 는 전 모델 같은 빠른 커널(`fea_rfft96_fast`). export 가 `<PFX>_H_GEN`·
`FEA2_R_GEN` 을 가중치 헤더에 스탬프 → `.c` 가 `_Static_assert` (호라이즌/랭크 불일치 = 컴파일 실패).
골든 가중치는 전부 랜덤(blind 골든 방지).

## Results files

| 파일 | 내용 |
|---|---|
| `results/mfg_baselines.csv` · `ltsf_baselines.csv` | 베이스라인 14 × 데이터 × 4H × 3시드 (얼림) |
| `results/feather_mfg_candR4{,_ci}.csv` · `feather_ltsf_candR4{,_mix}.csv` | FEATHer 정식 런 (mix/CI) |
| `results/feather_{mfg,ltsf}.csv` · `{mfg,ltsf}_raw.csv` | `tools/paper/merge_raw.py` 출력 (raw = 둘을 이어붙임) |
| `results/ablation.csv` | `--ablation_arm` 런 (`abl_*`) |
| `results/board_footprint.csv` | ARM 크로스컴파일 flash/SRAM (모델 × D × H) |
| `results/board_latency_20260929.csv` · `board_current_20260929.csv` | 보드 실측 |
| `results/complexity.csv` | 파라미터·연산 수 (⚠ FEATHer 행은 동결 전 구조 — 연산 수는 아래 ops_vs_latency 사용) |
| `results/{feather,feather_ci,stsf_mlp}_mfg_lookback_L{192,336}.csv` | 입력 길이 정식 (T11; 베이스라인은 `_archive/baselines/fcst_results_baselines.csv` exp_tag lookback) |
| `results/sram_vs_lookback_20261006.csv` · `ops_vs_latency_20261006.csv` | T12 계산 · 연산 수 vs 보드 지연 |
| `results/board_replay_MetroPT3_H{96,720}_s2025_board_every60.csv` | 보드 실시간 재생 (예측 1회당 1행) |
| `results/board_footprint_rsweep_1004.csv` | R2/R4/R8 크로스컴파일 (T7 Ch/chip) |
| `results/dev_*.csv` | dev 실험(입력 길이 dev, 아이디어 A `dev_nearfar`, 센서 고장 `dev_sensor_dropout`) — 원고 숫자 아님 |
| `results/_archive/` | 옛 값(452p 판, 빠른 FFT 전, CycleNet 전 등) — provenance |

CSV 키: `exp_tag, model, data, pred_len, seq_len, seed` + MSE/MAE/RMSE/CORR/R2/val_loss/best_epoch/num_params.
결과 CSV 의 FEATHer 이름 = `FEATHer_mix` / `FEATHer_CI` (footprint CSV 는 mix 를 `FEATHer` 로 씀).

## Model registry (15)

| Model | Venue | ETTh1 H96 params |
|---|---|---|
| SparseTSF | ICML'24 | 41 |
| MixLinear | ICLR'26 | 49 |
| FEATHer (CI / mix) | ours | 298 / 454 |
| DiPE_Linear | DASFAA'26 | 339 |
| SparseTSF_MLP | ICML'24 | 1,181 |
| FITS | ICLR'24 | 1,200 |
| CycleNet | NeurIPS'24 | 9,480 |
| DLinear | AAAI'23 | 18,624 |
| PatchTST | ICLR'23 | 35,171 |
| TimeMixer | ICLR'24 | 75,497 |
| TQNet | ICML'25 | 95,240 |
| LMS_AutoTSF | EAAI'26 | 181,318 |
| TimesNet | ICLR'23 | 605,479 |
| iTransformer | ICLR'24 | 841,568 |
| MDMLP_EIA | AAAI'26 | 1,671,446 |

경량 tier(칩 비교 대상) = 위 7개(SparseTSF ~ DLinear) + FEATHer. 나머지 = 정확도 천장 참고.
**제외(관련연구에만)**: S-Mamba(커스텀 CUDA 커널, 통일 프로토콜 재현 불가) · TimeCMA(사전학습 GPT-2 = 다른 자원
클래스) · SWIFT(공식 코드 미성숙, FITS 와 동률). **크기는 제외 기준이 아님** (41p~1.7M 폭이 핵심).

## Training protocol

- 전 모델 공통: seq_len **96** (MCU 입력 버퍼 근거, 유일한 의도적 변경) · batch 32 · 50 epoch · patience 10 ·
  Adam + CosineAnnealing · AMP off · **3시드 2025–2027** · best-val epoch 에서 test 1회 · 같은 split ·
  **loss = MSE (FEATHer 포함)**.
- 모델별: LTSF lr·구조 = 각 논문 공식 스크립트 (TFB 방식 `_DATASET_OVERRIDES`). mfg = 각 모델 default lr
  (lr 서치는 중앙값 0.1% 개선뿐이라 폐기). 주기 관련 HP(period/cycle/period_len) = 데이터별 FFT 검출, 3모델 같은 규칙.
  DLinear ETTh2 lr = 5e-3 (공식 5e-2 는 lradj=type1 짝이라 우리 스케줄러에서 발산; val 로만 재선택).
- 결측: hold-last 채움 + `mask.csv` → loss/지표는 관측값만 (MetroPT3 17.7% · N2O 5.8% · WindSCADA 1.2%).

## Datasets

- **mfg-6 (메인)**: Steel · WindSCADA · BlastFurnace(D=28) · N2O · MetroPT3 · EWELD — `utils/prep_manufacturing.py`.
- **LTSF-8 (일반화)**: ETTh1/h2/m1/m2 · Weather · Exchange · Electricity · Traffic (darts).
- H = {96, 192, 336, 720}. 후보 데이터 논의 시 먼저 메모리 `reference_dataset_reject_log.md` 확인.

## Common commands

```bash
# 원고 빌드 (v5 를 씀; v1~v4 덮어쓰기 금지)
python manuscript/version/_build/make.py

# 결과 병합 / 표 / 그림
python tools/paper/merge_raw.py
python tools/paper/table_deploy.py
python tools/paper/fig_pareto_board.py
python tools/paper/ablation_summary.py --per_dataset

# 스윕 (resume; 재실험 필요할 때만)
python run_forecast.py --check --exp_tag main
python run_forecast.py --model FEATHer --group mfg --exp_tag main --ngpu 1 --jobs_per_gpu 2
python run_forecast.py --ablation_arm gateuni,band1 --group mfg --num_seeds 3

# 보드 / C
cd deployment/board && make test                    # WSL, 골든 전부
python deployment/board/measure_footprint.py --dims 1,4,7,8,16,32 --horizons 96,192,336,720
python deployment/board/board_run.py --plan         # 보드 측정 계획표 (새 칸은 board_prep.EXTRA_PLAN 에)
python deployment/board/stream_flash.py --H 96 --ckpt results/checkpoints/candR4/FEATHer/MetroPT3_H96_L96_s2025.pth
python deployment/board/replay.py --data MetroPT3 --H 96 --port COM3 --every 60   # --sim = 보드 없이 검증

# 감사
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py
```
