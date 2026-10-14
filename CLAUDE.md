# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
지난 상태 블록은 `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-01.md`(06월~09-30) · `..._2026-10-06.md`(10-01~10-04 실험
실행 기록) · `..._2026-10-06b.md`(10-06 오전 "실험 끝, 글쓰기 단계" 블록) · `..._2026-10-07.md`(10-06 저녁 블록) · `..._2026-10-07b.md`(10-07 기록: 공식 대조·Fable 검토·파일 정리) · `..._2026-10-14.md`(10-13 블록: 칩 우선 규칙·3시드·재설계 1라운드)에 원문 그대로 있다 (평소엔 읽을 필요 없음 — 세션별
상세는 메모리 파일). **사용자에게는 항상 한글로 답한다.**

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) = 기계 옆 MCU
(NUCLEO-F103RB: 128 KB flash / 20 KB SRAM, FPU 없음)에서 **장기(H 96~720) 다채널 예측**을 하는
296-파라미터 모델 (Steel D=6, H=96). 논문 = **Journal of Manufacturing Systems (JMS)** 신규 투고
(TPAMI reject → IoT-J → JMS).

**주장의 범위**: 칩에 실제로 들어가는 모델들 중에서 비교한다. 최신 경량 모델은 전부 채널을 독립으로 다루는데(한 센서 안의
시간 패턴만 봄), 칩에서는 메모리 때문에 과거를 길게 못 보므로 **옆 센서를 보는 것(섞기)**이 이득이고, FEATHer는 섞기를 MCU
메모리 안에 들어가게 만들었다 (칩당 43채널). 철학 = "과거를 길게 못 보면 옆 센서를 본다"(10-06 v2). flash 가 H에 거의 무관(18.1→18.5 KB, H96→720; 파라미터는 MetroPT3 340→444 — 옛 "1.008배"는 동결 전 모델 값, 쓰지 말 것)해서 긴 H로 갈수록 칩 안 비교에서 유리해진다.
판정 기준은 파라미터가 아니라 **실측 flash·SRAM·보드 지연시간**.
⚠ **10-14 변경**: 메인 비교 = 크기를 맞춘 비교(1K판·칩 최대판, 아래 상태 블록), "옆 센서를 본다"(입력 길이) 주장은 메인에서 뺌, "296 파라미터"는 장점으로 내세우지 않음.

## Current status (2026-10-14 저녁) — 크기 맞춘 비교 규칙 확정(후보 · 시도 9번) + 코드 대정리 끝, 다음 = 실험 실행

**한 줄**: 메인 비교 = **크기 맞춘 비교**(1K판 · 칩 최대판). 10-14 에 후보 규칙을 확정하고(아래) 코드를 크게 정리했다(값·결과 동일 확인).
**새 세션 시작점 = 메모리 `project_session_1014.md` 의 HANDOFF.** (이전 블록 원문 = `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-14b.md`.)

- **크기 맞춘 비교 규칙 (10-14 사용자 결정, `configs/experiments/size.yaml`)**: 저자 설정에서 출발해 **폭·깊이·구간** 하이퍼파라미터만,
  (가) 하나로 예산 채움 / (나) 하나를 최소로 두고 다른 하나로 채움, 나머지 저자 값, 폭 1·저자와 같음·6데이터 중 하나라도 불가 → 제외.
  **모든 모델 시도 9번**(후보 수 × lr 점: 3×3 · 2×5 · 1×9), 고르기 = 시드 2025 · **H96 + H720** 검증, 데이터별 순위 평균. 모든 시도 기록 =
  `results/analysis/size_selection_audit.csv`. SparseTSF·MixLinear 도 주기로 1K 에 맞춤. DiPE = 1K판 참고 줄(최소 1,809)·최대판 정식.
  1K 후보 = `docs/size_candidates.json`(`tools/results/size_candidates.py`): CMoS 3 · PhaseFormer 3 · SMLP 2 · TimeBase 2 · FITS/SparseTSF/MixLinear/DiPE 1.
  **아직 없음**: FEATHer 후보(같은 규칙으로 정할 것) · 칩 최대판 후보(실제 빌드 필요). 용어 = "크기 하이퍼파라미터"(다이얼 금지).
- **코드 정리 (10-14, 전부 값·결과 동일 확인)**: 하이퍼파라미터 → `configs/<모델>.yaml`(19개), 실험 정의 → `configs/experiments/*.yaml`(ablation·control·size) ·
  결과 줄에 `hparams` 열(값이 바뀐 칸은 run.py 가 다시 돌림) · FEATHer 1,644줄 → 5파일 ~310줄(`FEATHer.py` 실제 모델, `ablation.py`, `variants.py`) ·
  `train.py` 697 → ~300줄(규칙 밖 옵션 삭제: oracle/adamw/train_frac/dump_preds) · `data_factory`·`data_loader` 표 방식 · `metrics.TestScores` ·
  주석 정리 88파일 · 옛 코드 = `_archive/`(FEATHer_full_20261014, board_legacy, cortex_m3, redesign_1012, tools_20261014). 10-12 재설계는 종료(채택 없음).
  검증: FEATHer 18설정 가중치·출력 동일, 14데이터 배열 동일, run.py 계획 동일, 옛/새 train.py 실제 실행 결과 동일(FEATHer 2·SparseTSF·CMoS), WSL make test 15포트 PASS.
- **버그 수정 (10-14)**: 분 단위 데이터의 시간 특징 주기 `"m"`(pandas = 월) → `"min"`. 시간 특징을 쓰는 건 iTransformer 뿐 → 그 96칸을
  `results/_archive/itransformer_month_marks_20261014.csv` 로 옮김(재실행 필요: `--stage main` 60 + `--stage ltsf` 36).
- **빠진 칸 (10-14 저녁)**: main 132(PhaseFormer cand:Weather 72 + iTransformer 60) · ltsf 36(iTransformer) · size_select 1K 887 · lookback(서버, 저자 설정).
- **서버**: 10-14 오전부터 옛 방식 lr 고르기(1K→max) 실행 중이었음 → 새 코드 push 후 멈추고 `git pull` + `pip install pyyaml` 확인 후 새 방식으로 재시작.
  옛 방식 결과 중 후보 ① × lr 3개 × H96/H720 칸은 같은 config_id 라 재사용됨.
- **사용자 결정 (10-13~14)**: 파라미터 수는 장점 아님 · 1K 기준 · lr 재선택 · 입력 길이 주장 메인에서 뺌 · 논문 간결화(본문 = 설정 표 · 크기 맞춘 정확도 2판 ·
  정확도 vs flash 그림 · 칩 비용 · 작은 ablation) · 저자 설정 결과·옛 FEATHer·4K 는 본문에 안 넣음 · "결과 보고 비교 방식 바꿨다" 걱정 다시 꺼내지 말 것.
- **Git**: 10-14 정리분 = 커밋 "Code cleanup: configs/*.yaml, size-matched candidates, FEATHer split ..." (personal push). 커밋은 요청 시, personal 만. ⚠ 보드 빌드는 한 번에 하나(golden/ 공유).

### 코드 공개 전 할 일 · 제자리 유지 파일
(10-01·10-07 파일 정리 기록 = `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-07b.md`)
- **제자리 유지 (옮기지 말 것)**: `deployment/BOARD_RESULTS_20260824.md`(펌웨어 바닥 10.4 KB 출처, 코드 주석 4곳이 가리킴)·`BOARD_SESSION_PLAN.md`, `board_footprint_recompute_*`(Table 7 의 31채널 근거), `probes`, `*_r2.csv`.
- **코드 공개(WDSLab/FEATHer) 전에 할 일**: ①·③ **10-14 완료** — `FEATHer.py` 기각 옵션 제거(부품 `gate.py`·`heads.py` 로 분리,
  29설정×H2 가중치·출력·밴드 동일 확인), 옛 전체 버전 = `_archive/FEATHer_full_20261014/`, `deployment/cortex_m3` → `_archive/cortex_m3`,
  옛 보드 export(`export_golden.py`·`export_batch.py`·`export_golden_feather2in.py`) → `_archive/board_legacy/` (emit_array 는
  `deployment/board/golden_io.py`). ② 도 10-14 완료: 옛 C 포트(`feather_infer`, `feather2in`, `test_batch`)와 골든 →
  `_archive/board_legacy/{src,test,golden}`, Makefile 정리, WSL `make test` 15개 포트 PASS. ④ `tools/dev` 정리 완료
  (`load_worker` → `tools/dev/worker.py`). 남은 것: `analysis/diagnostics/*`(옛 옵션 사용, 기록용) · README.md(옛 실행기 안내).

### 확정 숫자 (원고에 쓰는 값)
- ⚠ **10-14**: 아래는 **저자 설정·동결 FEATHer 기준(이제 부록)** 숫자다. 메인(크기 맞춘 비교) 숫자는 3시드 후 새로 계산. 섞기 · 입력 길이 · 메모리 일치 숫자는 메인 주장에서 빠짐.
- ⚠ **10-12**: 아래 베이스라인 관련 숫자·순위·"칩 안 가장 정확"·"LTSF 경량 평균 최저" 주장은 **재실험 전 값 — 쓰지 말 것**. 현재 값 = 위 상태 블록 + 메모리 `project_session_1012.md`.
- **(10-07 외부 리뷰 반영, 초안에 적용됨)**: ① "연산 수가 지연의 96 % 설명" **삭제**(FFT 기준 연산 수 r² 0.44; 직접 DFT 기준만 0.956) → 남는 주장 = "파라미터 수 r² 0.00" + 모델 예시 ·
  "파라미터가 flash 를 못 맞힌다" → "**한 예측 범위에서 보고된** 파라미터 수로는 못 맞힌다" ② **검정 단위 = 데이터(6쌍, H 평균, 단측 Wilcoxon 최소 p 0.016)**, 칸 수·시드 변동은 보조:
  섞기 이득 6/6 p 0.016(시드 5.7~6.2 %) · L192 5/6 p 0.031 · **L336 5/6 p 0.16(데이터 단위 비유의)** · front-none 5/6 p 0.031 · SparseTSF+통로 5/6 p 0.047 · STSF-MLP+통로 4/6 비유의 ·
  통로 on SparseTSF 6/6 · on STSF-MLP 5/6 · noresid 6/6 · band2 6/6 p 0.016 (스크래치패드 `dataset_level_stats.py`) ③ 5.9 % 는 항상 "**빼면 5.9 % 오른다**"(분모 mix) ④ mix–CI 차이는 섞기만이 아님
  (CI 는 필터·게이트도 채널별) → 6.1 한계점에 명시, **대조 arm "mix 모드 + 채널별 1→4→1" 72런 필요**(미실행) ⑤ 가이드 = "후보를 좁힌다 + 최종은 설비 검증 데이터" ⑥ "천 개 미만 유일" → "가장 작은 섞는 모델(296~1,924)";
  "경량 전부 CI" → "DLinear 보다 작은 경량 전부"(SAMformer 예외) ⑦ L>96 칩당 채널은 **상한**("최대") ⑧ 동일 메모리 우위 = 보간 → **데이터별 메모리 일치 비교로 교체 완료**(`_archive/tools_20261014/paper/memory_matched.py` → `results/analysis/memory_matched_20261007.csv`):
  각 변종이 D 에서 감당하는 최장 L 이 두 변종 **동일**(6·7ch → L336, 14ch → L192, 28ch → L96), 그 L 에서 **mix 가 6/6 데이터(p 0.016)·24/24 칸 우세**
  (BlastFurnace +1.0 · N2O +2.7 · EWELD +2.7 · MetroPT3 +3.2 · WindSCADA +5.0 · Steel +12.7 %, 평균 4.5 %). CI 가 유리한 구간 = CI 만 창을 한 번 더 늘리는 **8~13·19~24채널**(6데이터 평균 약 1.4 %, 해당 데이터 없음).
  "43채널 3.7 % / 18채널 0.7 %" 보간 값은 **더 이상 쓰지 않음**. Table 12 에 메모리 일치 열 추가 필요(빌더 미반영).
- **(10-07 숫자 검토로 정정, 초안에 반영됨)**: 밴드+게이트 파라미터 = 10D+12(Steel D=6 → **72**, MetroPT3 → 82) · 선형 헤드 H336 = **128 KB**(KiB; 131 은 십진) ·
  칩당 채널 CI 모델 = **42~50**(DiPE H720 42) · 밴드 3 > 2 의 p = **0.016(단측)**(0.03 은 양측) · 섞기 이득 최소 데이터 = **MetroPT3·BlastFurnace 약 1 % 이하**(BlastFurnace ≈0 아님) ·
  L720 입력 창 7채널 = 20,160 B 로 20,480 B **미만**(스택·버퍼까지 더해야 초과 — "입력 창만으로 초과" 문구 금지) · 서버급 대비 파라미터 **두~네 자릿수** ·
  LTSF 섞기 데이터별 값은 **셀별 % 규칙**으로 ETT −1.4~−5.2 · Exchange −8.4 · Weather +7.7 · Electricity +5.3 · Traffic +2.8(아래 "평균끼리" 값과 다름 — 본문은 셀별 규칙) ·
  BlastFurnace 파라미터 1,924 는 **H96**(H720 1,950) · ops→지연 r² 0.96 은 `ops_dft` 열(`ops_fft` 0.44) — 투고 전 확인 · 센서 고장 2.0/8.9 % 는 **창 절반(k=48) 고정** 값.
- ⚠ **10-07**: 베이스라인 설정을 프로토콜대로 고쳐 다시 돌리므로 **베이스라인 관련 숫자(LTSF 전부, mfg 순위·승패, T11)는 재실행 후 다시 계산**. FEATHer 자체 숫자·보드·ablation 은 그대로.
- ⚠ **10-06**: 아래 칩 풀 순위(T6 "7모델 MSE·MAE 1등")는 **새 베이스라인 넣기 전** 값. CMoS 포함 시(시드 2025) MSE 1등 칸
  우리 14 · CMoS 10, 평균순위 CMoS 1.71 · 우리 2.17, MAE CMoS 1.38 · 우리 3.17 → 3시드 확정 후 T6 다시 계산.
- **뒤 보기 vs 옆 보기 (10-06, 철학의 핵심 증거, 24칸 평균 MSE (칩당 센서))**: CI L96 0.736 (48) · L192 0.690 (24) · L336 0.657 (13);
  mix L96 **0.699 (43)** · L192 0.667 (18) · L336 0.647 (7). 같은 센서 수 보간: 43개 3.7 %, 18개 0.7 % 옆 보기 우세.
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
- **보드 (`results/board/board_latency_20260929.csv`, 27칸 PASS)**, 7채널 H96/H720 ms: SparseTSF 47/90 · MixLinear
  87/901 · CycleNet 157/✗ · **FEATHer-mix 161/208** · FEATHer-CI 202/250 · FITS 274/5,896 · DLinear 315/✗ ·
  STSF-MLP 434/1,445 · DiPE 727/12,735. 전류 13모델 29.2~30.0 mA (10-07 전부 실측) → 에너지 = 3.3 V × 30 mA × 지연.
  새 모델·후보별 값 = `results/board/board_latency_candidates_20261007.csv` (판정된 후보 줄을 쓸 것).
- **칩당 채널**: mix 43 · CI 48 · FITS 47~49 · CycleNet 50. flash 초과: DLinear 146/255/546 KB (H192/336/720),
  CycleNet 135/281 KB (H336/720).
- **H=720 칩 안**: FEATHer-mix 가장 정확(0.808), 더 빠른 건 SparseTSF 뿐(7% 덜 정확). ⚠ 10-07: 새 베이스라인 전 값 — TimeBase(항상 더 빠름)·CMoS 로 깨질 가능성 큼, 재실행 후 다시 씀.
- **입력 길이 (T11, 정식 3시드, 칩 7모델)**: MSE 평균순위 L96 1.50 / L192 2.21 / L336 2.71 = 셋 다 1등; 평균 MSE 0.699/0.667/0.647
  (L336 2등 FITS·DiPE 0.650). MAE: L96 1등, L192 FITS 와 공동 1등, **L336 3등**(DiPE 2.71). 자기 개선 L336 −7.8% vs 베이스라인
  −10~11%. 섞기 이득 L96 +5.9%(23/24, p<0.001) → L192 +4.1%(20/24, p<0.001) → L336 +2.7%(18/24, **p=0.045**).
- **T12 칩당 채널 vs L (MetroPT3 D=7)**: 모든 모델 ≈ 49 → 24 → 14 → 6 (L96 실측, 이후 상한); DLinear L≥192 면 H96 에서도 flash ✗;
  CycleNet L≥336 ✗; FITS L720@H96·L≥336@H720 ✗; FEATHer-mix 자기 버퍼 기준 43/18/7/0. **출력 H 는 SRAM 무관**(보드 재생 RAM 동일).
- **Dataset rank (T6, 6데이터 단위)**: MSE FEATHer-mix 1.33(FITS 2.83), MAE FITS 와 공동 2.17; Friedman p<0.001/0.002.
- **정확도 상한**: 서버급 7모델(75 K~4.0 M params, 칩 불가) MSE 0.643~0.685, FEATHer-mix 0.699 = 2.0~8.6% 위.
- **연산 수 vs 보드 지연 (11칸, `results/analysis/ops_vs_latency_20261006.csv`)**: 파라미터 r² 0.000, 연산 수(보드 구현대로 변환 비용) r² 0.96.
- **보드 실시간 재생 (MetroPT3 s2025, 1시간마다)**: H96 1,022회·H720 1,012회 전부 보드 MSE = PyTorch(0.542156 / 0.613206),
  지연 평균 161 / 211 ms(입력에 따라 ±10%; T3 의 208 은 입력 하나), 계산 비율 0.004 / 0.006%, 계산 전력 0.38 / 0.50 J/일,
  펌웨어 14.3 / 14.7 KB flash, 7.55 KB RAM. ⇒ 현장 주기에서 속도는 선택 기준 아님 → 정확도·메모리가 고름.
- **헤드 (T8)**: 선형 헤드 −5.5/−4.4/−2.9/−1.9%(더 정확) 이지만 H336 131 KB·H720 273 KB 로 칩 ✗ → SPK 유지(DLinear 와 같은 칩 기준).
- **가이드 (5장 본문)**: H96 은 DLinear(MSE)·CycleNet(MAE) 가 더 정확·칩 OK; H≥192 는 FEATHer-mix(H192 MAE 만 CycleNet);
  ≤43 ch + H≥192 → FEATHer-mix, 44~48 ch → CI 모델 또는 칩 2개.
- **필터 없이(경량 8개, 24칸)**: MSE 순위 DLinear 1.83 > 우리 2.21 · MAE CycleNet 2.04 > 우리 2.96(1등 칸은 우리 9로 최다)
  → 두 지표 모두 2위 안은 우리뿐. MSE 로 우리를 이기는 가장 작은 모델 = DLinear (flash 4.6배 H96 ~ 30배 H720; T3 값 83.5/18.1, 546/18.5 KB).

### 주장 금지 (반복 확인됨)
- (10-06) "칩 안에서 가장 정확" — CMoS 가 MAE 로 앞섬 → "MSE 1등 칸 최다 + H≥192 우세" (3시드 확정 후 문구 확정).
- (10-06) "센서 연동이 강할수록 섞기 이득이 크다" — 사전 규칙 실패(ρ 0.36) · "시간차(선행) 반응을 잡는다" — 9월 mix_lag 이
  시차 최대 EWELD 에서도 이득 0 → "함께 움직임"까지만 · "옆 센서가 과거를 대신한다" → "**일부** 대신한다" ·
  4개 요약 = 물리적 공정 요인(미확인) · "1천 개 이하로 섞는 모델은 우리뿐" 은 유지 가능(F-Net 은 약 18K 로 확인됨).
- "DLinear를 이긴다" — MSE 11–13. DLinear는 모든 H에서 더 정확, H≥192에서 칩에 못 들어갈 뿐.
- "H가 길수록 (DLinear 대비) 격차가 커진다" — 틀림. DLinear 의 MSE 우위가 오히려 커짐(3.4→6.2%); 커지는 건 비용 차이.
- int8/행렬 합치기 실험 다시 제안 금지 (09-29 결정: 베이스라인 재최적화 안 함, 한계점 한 문장).
- "LTSF 1위" / "LTSF 에서 가장 정확" → "on par with the best lightweight models" (+ 쓸 수 있음: "32칸 모두 하위 3위 밖인 유일한 모델").
- "현장 의사결정(정비·에너지)이 개선된다" — 평가 안 함(Steel 피크·MetroPT3 조기경보 둘 다 기각됨; 지표 바꿔 재시도 = p-해킹) →
  "어디에 쓰이는지"는 글로 설명, "나아졌다"는 한계점/향후 과제.
- 게이트 표현(10-04 사용자): 숫자와 함께 "평균 오차를 0.3% 줄인다"(약간)로 씀 — "크게 개선" 금지. 긴 H 부호는 −0.5/+1.3/−0.4 다 적음.
  밴드·스펙트럼 손실은 "정확도를 ±1% 안에서 유지" — **"효과 없음" 이라고 쓰지 말고 기여 순위로**(섞기 5.9 · MLP 3.8 > 게이트 0.3 > 밴드).
- **(10-07 사용자 결정) 밴드 + 게이트 = "한 묶음, 네 H 모두 약 1 %"** 로 주장(대조군: front-none 대비 +0.99 %, 17/24, p<0.001, H별 1.6/0.5/0.7/1.2).
  따로따로 숫자 금지(부호가 뒤집힘) · 메커니즘 주장 금지 · 2장 빈틈으로 세우지 않음(1장 기여 ② 둘째 문장 + 3장/T9–T10 만) ⇒ **동결 유지, 재포팅·재실험 없음**.
  문구 = `drafts/_old/sec1_intro_outline_v0_1002.md` P8-②. 2장 = 안내 문단 + 2.4 Positioning(G1 칩 실측 없음 · G2 섞는 경량 모델 없음 · G3 제조 전 채널 실측 없음).
  밴드 3개 > 2개는 6/6 데이터(+1.0%, p 0.03). 밴드·게이트는 **해석 도구로도 주장 금지**(게이트 비중이 시드마다 바뀜).
- "섞기가 센서 고장에 강하다" — 반대(고장을 퍼뜨림). "보드 재생 = 실설비 설치" 아님(저장 데이터 재현).
- 모양 지표(CORR 등) 주장 — 사전 규칙으로 기각, MSE/MAE 만.
- 순위를 말할 땐 항상 **어느 풀인지** 명시 (9모델 전체면 DLinear MSE 1등 13 > 우리 10).
- "제조 TSF 벤치마크 최초" — RCIM(arXiv 2411.17499)이 선점. "최초"는 MCU 배포·전 채널 동시·경량 tier 로 좁힐 것.

### 한계점에 넣을 문장
(10-06) DLinear·int8 은 한계점이 아니라 **방법 절 한 줄**: "모든 모델 float32 같은 규칙, 양자화는 모든 모델에 적용 가능한 별도
최적화라 범위 밖" · CycleNet 위상 인덱스 0 고정 · H 길이 변환은 전
모델 직접식 · 재계산 허용(정규화+밴드)은 FEATHer-mix 포트의 예외 · L=96 고정(긴 룩백
네이티브 모델에 불리 — Traffic/Electricity 격차 +56~73%) · Steel 학습 불안정(원인 미규명) · 입력 길이 ≤ 336
(L720 은 7채널 입력 버퍼만으로 SRAM 초과, 모든 모델 제외) · SPK 주기 P 는 L·H 를 나눠야 해서 ≤ 48 (Steel/EWELD 실제 96) ·
현장 의사결정 지표 미평가(향후 실설비 적용) · mfg=mix / LTSF=CI 변종 선택(두 변종 모두 보고) · 1초보다 짧은 간격(진동 등)이면 속도가
중요해짐 · 연구 범위 = 가장 싼 20 KB 급 칩(1 MB 급이면 큰 모델도 들어감) · **(10-05 사용자) 센서 고장: 한 센서가 멈추면 섞기가 그 고장을 연결된 센서로 퍼뜨림 (멀쩡한 센서 MSE Steel +8.9%, 평균 +2.0%; CI 는 0) → 배포 시 고장 감지 후 처리 필요** (results/dev/dev_sensor_dropout.csv).

### JMS 글쓰기 방향 (10-01)
- **JMS 가 좋아할 것(우리 근거)**: ① 기계 옆 실제 칩 실측 + 보고 정확도 = 배포 C 코드 ② 실무자용 선택 가이드(센서 수·H →
  이 칩에서 고를 모델, 칩당 43채널) ③ 제조 공정 센서는 묶여 있어 섞기가 이득인데 경량 모델은 전부 안 섞음 ④ 실제 산업 데이터 6종
  ⑤ 긴 H = 현장 판단 시간(H720 = MetroPT3 12시간 ~ BlastFurnace 30일) ⑥ 클라우드 없이 현장 독립 동작(OT 격리, 인용 필요)
  ⑦ "파라미터로 칩 비용을 고르면 틀린다". 비용은 돈 아니라 칩 등급·칩당 센서 수까지만.
- **약점**: 현장 판단 개선 증거 없음 · 실설비 설치 없음 · 방법 새로움 약함(밴드·게이트는 한 묶음 약 1 %) · DLinear 더 정확(int8 질문).
- **5장 쓸 때**: 데이터별 "H720 = 실제 몇 시간/며칠 → 어떤 현장 판단" 표(측정 간격으로 계산) · 배포 가이드(결정표 데이터로) ·
  서론에서 섞기를 공정 센서 결합과 연결. LTSF 는 방어용("손해 안 봄") + 변종 선택 문단을 먼저 씀.
- **억셉 판단(10-01, 데이터 아닌 판단)**: 지금(원고 잘 마무리 전제) 리젝 40~50%, 나머지 대부분 메이저. 최대 변수 = 입력 길이 실험
  (FEATHer 도 비슷하게 좋아지면 30%대, 베이스라인이 앞지르면 크게 악화). 바로 억셉은 기대 안 함; 목표 = 메이저 리비전.

### 용어
**channel** (feature 금지). 제목·키워드의 "multi-sensor" 는 유지. SPK 헤드는 **SparseTSF에 귀속** 명시(방법 절).
(10-07) **forecast 계열만** (predict 는 predictive maintenance·RUL prediction·참고문헌 제목·"파라미터가 지연을 맞힌다"는 통계적 의미에만).
전체 통일표 = `manuscript/notes/terminology.md` (새 용어 쌍은 거기에 추가).

## Architecture (FEATHer) — FROZEN (09-04 구조, 09-23 mlp_rank 4 재동결)

`baselines/FEATHer/FEATHer.py`(조립 + 허용 옵션표 OPTIONS) · `gate.py` · `heads.py` · `ablation.py`(부품 바꾸는 ablation 4종) · `variants.py`(대조 실험 변형 2종) · `wrapper.py`, 설정 `configs/FEATHer.yaml` (frozen 블록). 클래스 기본값 = 동결 모델(10-14). 변경 시 스윕 재실행 필요.
(10-13) 재설계 옵션 `--agg_shared 1`(다듬기 필터 공유) · `--head_seg s`(SPK 구간 길이를 주기에서 분리; 0 = 주기) — 기본 꺼짐 = 동결 모델 바이트 동일. 크기 맞춘 비교의 FEATHer = 동결 + 이 둘.

```bash
--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none --gate_mode banded --mlp_rank 4
mfg:  --channel_mode mix          LTSF: --channel_mode ci_full   (overrides 로 자동)
```
```
RevIN -> 3 bands: conv k=3 | conv k=5 | AvgPool(4)+interp -> banded FFT gate -> weighted sum
      -> residual MLP (D->4->GELU->D) -> SPK head -> RevIN^-1
```
- DTK·`in_proj` 는 측정으로 제거(ablation 용 `dtk_mode=mlp`·`use_in_proj` 만 남음). 10-14 에 기각 옵션 전부 삭제(`--kernel_size` 등).
- `--band_kernels 3,5` 는 같은 구조지만 RNG 흐름이 바뀌어 초기화가 다름 → 항상 `--point_kernel` 형태.
- banded 게이트 = 대역별 에너지 비율 → `Linear(S,S)`, 0 초기화(균등에서 시작).
- 잔차 MLP = 유일한 비선형. `ci_full` 에선 1→4→1.
- Loss = MSE + 0.01 × spectral-separation (λ 올리면 단조 악화, 사실상 꺼진 항).
- 파라미터: mfg-mix Steel 296 / CI 76 · LTSF ETTh1 CI 298 / mix 454 (merge_raw 가 num_params 로 검증).
- (10-06) P1 옵션 `factor_head` / `factor_src` 는 10-14 삭제(실험 안 함, `_archive/FEATHer_full_20261014/` 에 있음). `--front none` 은 대조 실험이 써서 남음.

## Environment & gotchas

- **Python**: `C:\Users\lee\.conda\envs\feather\python.exe` (torch 2.5.1+cu121, darts, reformer_pytorch; Py 3.9).
  bash 의 `python` 은 Windows 스토어 스텁 → 경로 직접. docx 스킬 스크립트는 Py≥3.10 →
  `/c/ProgramData/anaconda3/python.exe`.
- **Dropbox 폴더에서 `rmtree` 금지** (반쯤 지워짐; golden/ 한 번 복구). 파일 단위로.
- Bash heredoc 안의 `\n`/`\u` 이스케이프가 깨짐 → 스크립트는 Write 로 파일 작성 후 실행.
- Git Bash 에서 `wsl.exe` 에 `/mnt/...` 경로를 넘길 땐 `MSYS_NO_PATHCONV=1` (안 하면 `C:/Program Files/Git/mnt/...` 로 바뀜).
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
run.py                       유일한 실행 입구 — 프로토콜이 요구하는 칸 중 결과에 없는 것만 실행
                                  (--stage ltsf / ltsf_select / select / main / lookback / ablation / control, --save_model)
scripts/train.py                  학습 코드 (모델 × 데이터 × H × 시드들 → 결과 행). run.py 가 부름 (10-07 이름 변경)
tools/results/                    protocol(후보·설정·ablation·대조군 정의, main_rows/KEY/안전 쓰기) · merge_incoming · select_verdict
                                  (_archive/: build_raw·reorganize_results = 10-07 한 번만 쓴 이전 도구, 다시 실행 금지)
docs/                             EXPERIMENT_PROTOCOL.md (실험 규칙) · MODEL_FINAL.md (FEATHer 동결 구조) · ENV.md (환경·
                                  베이스라인 커밋) · chosen_{mfg,ltsf}.json (고른 설정, 커밋함)
_archive/run_forecast_orchestrator.py   옛 실행기 (10-07 은퇴; run 이 같은 명령을 만드는 것 61칸 대조 확인)
baselines/__init__.py             모델 목록 MODELS (이름 -> 폴더/wrapper) + 설정 읽기 함수 4개 (10-14 정리)
configs/<Model>.yaml              모델별 하이퍼파라미터 (default + 데이터별 + "<데이터>@<H>"); 값은 제자리에서 고침, 덮어쓰기 줄 금지
configs/experiments/*.yaml        실험 정의: ablation(항목·플래그·H) · control · size(크기 비교 후보·lr 점); 10-12 재설계는 끝남(채택 없음) → _archive/redesign_1012/; protocol.py 가 읽기만 함
baselines/<Name>/wrapper.py       upstream Model 을 forward(x)->(B,H,D) 로 감쌈
utils/                            prep_manufacturing (raw -> data.csv + mask.csv) · data_factory · data_loader
                                  · dispatch_queue · losses · metrics · seed
tools/audit/                      check_progress · check_duplicates · purge_rows · results_table · scan_divergence
tools/dev/                        bs512 빠른 dev 루프(run_dev) + 진단 도구 (README 참고)
tools/paper/                      merge_raw · table_main/table_mfg/table_deploy · fig_pareto_board · wilcoxon
                                  · ablation_summary · measure_complexity · compute_r2 · persistence_baseline
deployment/board/                 C 포트 src/*_infer.{c,h} 13종 · export_golden_*.py (--config 후보) · test/ (make test)
                                  · measure_footprint.py · board_run.py (빌드->플래시->읽기 자동) · README(공통 C 규칙)
manuscript/version/               FEATHer_JMS_v1/v2.docx + _build/ (원고 빌더)
manuscript/notes/                 dataset_precedent_audit · reject_mail · 아카이브 (gitignored)
results/  analysis/  data/        결과 CSV · 표 · 데이터 (data/ 는 gitignore, 서버로 수동 전송)
deployment/board/ (10-06 추가)    main.c.stream(실시간 재생 펌웨어) · stream_flash.py · replay.py (--sim / --port COM3, pyserial)
_archive/                         은퇴 코드: robustness · ablation_v1 · grids · classic_baselines · lr/hp search
```

**C 포트 13종**: feather2 / feather2r (mix, 보드는 재계산 스케줄 feather2r) · feather2ci · dlinear · cyclenet ·
stsf · stsf_mlp · mixlinear · dipe · fits · (10-07) cmos · timebase · phaseformer · samformer(섞는 모델; 새 4개는
고른 설정 `docs/chosen_mfg.json` 을 board_prep 이 자동으로 읽음, 보드 계획 28~37번). **공통 규칙 ①**("원래 계산은 그대로, 메모리는 정리만"): 안 섞는 모델은
채널 하나씩 · 출력은 끊을 수 있는 단위로 바로 방출 · 작업공간 공유 · 재계산 금지(FEATHer-mix 의 정규화+밴드
재계산만 명시적 예외). 96점 FFT 는 전 모델 같은 빠른 커널(`fea_rfft96_fast`). export 가 `<PFX>_H_GEN`·
`FEA2_R_GEN` 을 가중치 헤더에 스탬프 → `.c` 가 `_Static_assert` (호라이즌/랭크 불일치 = 컴파일 실패).
골든 가중치는 전부 랜덤(blind 골든 방지).

## Results files (2026-10-07 재정리 — 상세 `results/README.md`)

| 파일 | 내용 |
|---|---|
| `results/raw_mfg.csv` · `raw_ltsf.csv` | 정식 결과 (role = main/reference/lookback/ablation/control, config_id = 설정 이름) |
| `results/select_{mfg,ltsf}.csv` | 설정 고르기 결과 (고른 설정은 git 에 올라가는 `docs/chosen_{mfg,ltsf}.json`) |
| `results/_incoming/` | 새 실행 조각 (run.py 출력, exp_tag = `role|config_id`) → merge_incoming 으로 합침 |
| `results/board/` | 보드·칩 측정 (footprint, latency_20260929, **latency_candidates_20261007**(후보별), current(13모델), replay, rsweep, recompute) |
| `results/analysis/` | complexity, ops_vs_latency, sram_vs_lookback, coupling, r2, kpi, linear_ceiling |
| `results/dev/` | dev 실험 (원고 숫자 아님) |
| `results/_archive/pre_raw_20261007/` | raw 를 만든 옛 CSV (mfg_baselines, feather_*candR4*, mfg_raw …) — **원고 빌더가 아직 읽음** |
| `results/_archive/legacy_rows_20261007.csv` | 규칙과 설정이 달라 raw 에서 뺀 행 (지우지 않음) |

CSV 키: `role, config_id, exp_tag, model, data, pred_len, seq_len, seed` + MSE/MAE/RMSE/CORR/R2/val_loss/best_epoch/num_params.
결과 CSV 의 FEATHer 이름 = `FEATHer_mix` / `FEATHer_CI` (footprint CSV 는 mix 를 `FEATHer` 로 씀).

## Model registry (15 + 10-06 추가 4 = 19)

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

경량 tier(칩 비교 대상) = 위 7개(SparseTSF ~ DLinear) + FEATHer + 아래 10-06 추가 4개(SAMformer 는 H96 만 칩 안). 나머지 = 정확도 천장 참고.
**10-06 추가 (레지스트리 19개)**: CMoS(ICML'25, 7채널 H96 939 → H720 6,009; 안 쓰는 매핑 1개 포함, 배포 783~4,839) ·
TimeBase(ICML'25, 58~240) · PhaseFormer(ICLR'26, 1,500~1,630) — 셋 다 채널 독립 · SAMformer(ICML'24, 21.7K~82K, **섞음**).
래퍼 = `baselines/<Name>/wrapper.py`, 공식 저장소는 `setup_baselines.sh` 가 받음(`*-main`, gitignore).
**제외(관련연구에만)**: S-Mamba(커스텀 CUDA 커널, 통일 프로토콜 재현 불가) · TimeCMA(사전학습 GPT-2 = 다른 자원
클래스) · SWIFT(공식 코드 미성숙, FITS 와 동률; 여전히 프리프린트) · F-Net(IEEE TAI 2026, 약 18K·섞음, TF·무라이선스) ·
LiPFormer·LightMTS·FilterNet·TinyCast·RhyMix 등(10-06 조사, 메모리 `project_cmos_baseline_1006.md`). **크기는 제외 기준이 아님**.

## Training protocol — `docs/EXPERIMENT_PROTOCOL.md` 가 기준 (10-07)

- 전 모델 공통: L **96**(입력 길이 실험 48·192·336) · batch 32 · 50 epoch · patience 10 · Adam + Cosine · clip 1.0 · AMP off ·
  3시드 2025–2027 · best-val epoch 에서 test 1회 · loss = 관측값 MSE + 방법 고유 항.
- 설정: LTSF·제조 주기 = `configs/<모델>.yaml` (10-14 YAML 로 이전, 값 동일 확인) · mfg = `tools/results/protocol.py`
  `CANDIDATES` + `resolve()`(후보 설정 + mfg 주기류 값) → 고른 설정 `docs/chosen_mfg.json` · `docs/chosen_ltsf.json`.
- 워커 훅: `_aux_loss`(TimeBase) · `_train_loss` / `_epoch_start` / `_test_start`(DiPE) · SAMformer SAM · `--result_model_name`.
- 예외: DLinear ETTh2 lr 5e-3(공식 0.05 발산, 검증 재선택) · 공식 없는 LTSF 칸 = 그 데이터 검증으로 후보 선택(`ltsf_select`).
- 결측: hold-last 채움 + `mask.csv` → loss/지표는 관측값만 (MetroPT3 17.7% · N2O 5.8% · WindSCADA 1.2%).

## Datasets

- **mfg-6 (메인)**: Steel · WindSCADA · BlastFurnace(D=28) · N2O · MetroPT3 · EWELD — `utils/prep_manufacturing.py`.
- **LTSF-8 (일반화)**: ETTh1/h2/m1/m2 · Weather · Exchange · Electricity · Traffic (darts).
- H = {96, 192, 336, 720}. 후보 데이터 논의 시 먼저 메모리 `reference_dataset_reject_log.md` 확인.
- (10-07 정리) 논문에서 뺀 데이터(CMAPSS·CMAPSS3·PMSM·nrel)와 `data.csv.premask` 백업 3개 = `data/_shelf/`.

## Common commands

```bash
# 크기 맞춘 비교 (메인; 규칙 configs/experiments/size.yaml)
python tools/results/size_candidates.py --write          # 1K 후보 값 -> docs/size_candidates.json
python run.py --stage size_select --tier 1k --check      # 후보 x lr (모델당 9번) x H96/H720 x 시드 2025
python tools/results/size_verdict.py --write             # (후보, lr) 선택 -> docs/chosen_size.json + size_selection_audit.csv
python run.py --stage size --ngpu 1 --jobs_per_gpu 3     # 고른 것 x 4H x 3시드
python tools/results/size_dials.py --write [--tier 1k]   # 칩 빌드 결과 -> docs/size_dials.json (최대판 후보 만들 때 입력)
python tools/paper/observe_structure.py                  # 학습 구간 관찰
# 10-07 실험 관리 (docs/EXPERIMENT_PROTOCOL.md)
python run.py --stage select --check            # ltsf / ltsf_select / select / main / lookback; --check = 빠진 칸만 보기
python run.py --stage select --ngpu 1 --jobs_per_gpu 3
python tools/results/merge_incoming.py               # _incoming 조각 -> raw_*.csv / select_mfg.csv (--archive 로 조각 보관)
python tools/results/select_verdict.py --write       # 고르기 판정 -> docs/chosen_mfg.json (--group ltsf -> docs/chosen_ltsf.json), 커밋할 것
python tools/paper/make_result_html.py               # 결과표 HTML -> results/result.html (merge 후마다)
python deployment/board/chip_verdict.py [--candidates]  # 칸 단위 칩 판정 (보드 없이, WSL 빌드)
python deployment/board/board_run.py --stream          # 가장 빠듯한 칸을 실제 채널 수로 보드 실측 (10-12)

# 원고 빌드 (v5 를 씀; v1~v4 덮어쓰기 금지)
python manuscript/version/_build/make.py

# 결과 병합 / 표 / 그림
python tools/paper/merge_raw.py
python tools/paper/table_deploy.py
python tools/paper/fig_pareto_board.py
python tools/paper/ablation_summary.py --per_dataset   # raw_mfg.csv 의 role ablation

# ablation · 대조군 · 보드용 체크포인트 (run.py 에 통합, 10-07)
python run.py --stage ablation --check
python run.py --stage control --ngpu 1 --jobs_per_gpu 3
python run.py --stage main --models FEATHer_mix --data MetroPT3 --save_model   # 보드 export 용 저장

# 보드 / C
cd deployment/board && make test                    # WSL, 골든 전부
python deployment/board/measure_footprint.py --dims 1,4,7,8,16,32 --horizons 96,192,336,720
python deployment/board/board_run.py --plan         # 보드 측정 계획표 (새 칸은 board_prep.EXTRA_PLAN 에)
python deployment/board/board_run.py --candidates   # 후보 설정별 지연 (10-07, 77칸 끝남; 이어서 실행)
python deployment/board/stream_flash.py --H 96 --ckpt results/checkpoints/candR4/FEATHer/MetroPT3_H96_L96_s2025.pth
python deployment/board/replay.py --data MetroPT3 --H 96 --port COM3 --every 60   # --sim = 보드 없이 검증

# 감사
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py

```
