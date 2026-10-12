# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
지난 상태 블록은 `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-01.md`(06월~09-30) · `..._2026-10-06.md`(10-01~10-04 실험
실행 기록) · `..._2026-10-06b.md`(10-06 오전 "실험 끝, 글쓰기 단계" 블록) · `..._2026-10-07.md`(10-06 저녁 블록) · `..._2026-10-07b.md`(10-07 기록: 공식 대조·Fable 검토·파일 정리)에 원문 그대로 있다 (평소엔 읽을 필요 없음 — 세션별
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

## Current status (2026-10-12) — 공정한 재실험 결과: FEATHer 순위 하락, 재설계 검토 중

**한 줄**: 베이스라인 공정성 문제(FEATHer 만 mfg 에서 설정을 고름 · 베이스라인 설정 출처가 제각각 · 7개 모델이 공식 설정과
다름)를 찾아 **실험 규칙을 새로 정했다 → `docs/EXPERIMENT_PROTOCOL.md`**. 결과는 `results/raw_{mfg,ltsf}.csv` 로 통합했고
`run.py` 가 빠진 칸만 돌린다. **새 세션 시작점 = 메모리 `project_session_1007.md`(HANDOFF)**.
(이전 상태 블록 원문 = `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-07.md` · `..._2026-10-07b.md`.)

- **규칙 요약 (상세 = 프로토콜 §5)**: 베이스라인 설정 = 저자가 공개한 값만(최종 스크립트 + 저자 기본 설정; 빈칸용 argparse 값
  제외, 조합 금지) · L96 공식 스크립트 우선(CycleNet lr 0.01) · LTSF = 데이터(·H)별 공식 ·
  **mfg = 후보(저자 데이터별 설정 묶음) 중 6데이터 검증 손실 평균 순위 1등 하나를 모든 데이터·모든 L 에** (시드 2025, 테스트 안 봄) ·
  **공식 설정 없는 LTSF 14칸도 같은 방식(그 데이터 검증, `--stage ltsf_select`)** · **칩 판정은 칸 단위**(후보를 미리 거르지 않음,
  넘는 칸만 "칩 밖" — DLinear 와 같은 취급) · 건강 검사(best epoch ≤2, 0부터 셈 / 발산; LTSF 는 실패 시 lr 0.1·0.01배 검증 재선택) ·
  입력 길이로 정의 안 되는 값은 최대값(MixLinear lpf, FITS cut_freq ≤ L/2) · 주기류는 규칙(FFT ≤48 / 하루 주기) · 학습 방법은 통일(Huber·에폭·FITS train_mode)
  하되 방법 고유 손실 항은 유지(FEATHer 스펙트럼, TimeBase 직교, DiPE 주파수 비율+라우터 온도, SAMformer SAM).
- **10-07 사용자 결정**: FEATHer 동결 유지, mfg·LTSF 다시 안 돌림(LTSF 주기 = 7월 LTSF 검증 선택 → 사실대로 공개) ·
  대조군 판정 = 규칙 2(밴드·게이트 약 1 %, "작은 보정"으로 낮춰 씀; 핵심 = 섞기 통로 — 섞기 이득이 바탕과 무관하게
  SparseTSF +6.7 · SparseTSF-MLP +4.2 · FEATHer +5.9 %, 데이터 패턴도 같음) · **L = 48 추가**(판정 규칙 = 프로토콜 §2:
  섞기 이득 > +5.9 % & p<0.05 일 때만 추세 주장), L720 안 함(입력 창만으로 SRAM 초과) · 서버급 7개 = L96 참고 표(mfg+LTSF, mfg 는 기본 설정 명시) ·
  CycleNet 은 모든 L 에 lr 0.01 하나 · SparseTSF-MLP 는 자기 공식 lr(선형 lr 빌려 쓰던 것 폐기).
- **쓰지 말 문장 (10-07 사용자 결정)**: FEATHer 개발 중 테스트를 본 결정(R4·채널 방식)과 선택 시드 공개 문장은 쓰지 않음 —
  단 "FEATHer 포함 모든 모델을 검증으로 골랐다" 같은 사실과 다른 문장은 금지, "베이스라인 설정은 검증 손실로 골랐다"까지만.
- **⚠ 10-12 핵심 (새 세션 시작점 = 메모리 `project_session_1012.md` HANDOFF)**: 베이스라인을 프로토콜대로 고른 설정으로 다시 돌리니
  **FEATHer-mix 는 더 이상 가장 정확하지 않다.** mfg 평균 MSE(L96, 3시드): DLinear 0.656 · CMoS 0.682 · PhaseFormer 0.683 · DiPE 0.696 ·
  **FEATHer-mix 0.699** · SAMformer 0.708 · FITS 0.709 · SMLP 0.722 · CycleNet 0.724 · FEATHer-CI 0.736 · MixLinear/SparseTSF 0.749 · TimeBase 0.843
  (우리 숫자는 그대로, 베이스라인이 좋아짐: PhaseFormer −6.4 · DiPE −4.5 · CMoS/FITS −2.3 %). 모든 칸 칩 안 = FEATHer·SparseTSF·SMLP·DiPE·MixLinear·TimeBase;
  CMoS H720 ✗ · FITS H720·28ch ✗ · CycleNet H≥336 ✗ · DLinear H≥192 ✗ · SAMformer H96 D≤7 만 · PhaseFormer 전부 ✗.
  DiPE(Electricity: 전문가 4·RevIN 끔·주파수 손실 0.3) 가 MSE 22/24 칸 우세(BlastFurnace −10~−17 %, WindSCADA −10 %), 우리는 H720 N2O·EWELD(DiPE 붕괴)만;
  MAE 는 우리 14/24; 속도 DiPE 751 ms / 약 12.9 s vs 우리 161 / 208 ms. **H720 칩 안 1등은 유지(0.808)**. LTSF(부분): FEATHer-CI 12개 중 7등.
  대조군: 섞기 통로를 붙이면 SparseTSF +6.7 %(6/6, p 0.016) · SMLP +4.6 %(p 0.031) · `_mixM` +6.5/+4.8 %(p 0.078) → 정규화 탓 아님;
  SparseTSF+통로 ≈ FEATHer-mix(≈0.70) → 값을 하는 건 섞기 통로. **사용자가 FEATHer 재설계(검증으로만 선택) 검토 중** — 방향은 사용자와 하나씩.
- **10-09~12 확정**: `docs/chosen_mfg.json`(6c5109a) · `docs/chosen_ltsf.json`(e681cfe) · 안정성 가설 사전 등록 검사 **통과**(ρ −0.675, p 0.004;
  `docs/prereg_stability_1009.md`, "같은 공정이라 섞기 이득"은 반증됨) · 프로토콜 6번 수정(d3904c7, 사용자 A: 발산만 실패, best epoch ≤2 는 데이터 특성 →
  재실험·LTSF lr 재선택 없음, 사후 수정으로 공개) · 섞기만의 효과 사전 등록(62bf8ec) + `--mix_sep`(45c9322, 동결 모델 바이트 동일).
- **10-12 오후 (세션 73e9858d, 상세 = 메모리 `project_session_1012.md` 끝부분)**:
  ① **mixsep 판정 = 5.9 % 를 M = 2.1 % 로 교체**(사전 등록 규칙; 5/6 데이터, p 0.047) — 섞기 자체 몫 ≈ 1/3, 나머지 채널별 용량. 원고 미반영.
  ② **칸 단위 칩 판정 완료**(`deployment/board/chip_verdict.py` → `results/board/chip_verdict_20261012.csv`, 펌웨어 전체 빌드, 실제 D, 고른 설정):
  24칸 전부 = FEATHer-mix/CI · SparseTSF · MixLinear · TimeBase · DiPE · SMLP; CMoS 18 · CycleNet 12 · DLinear 6 · FITS 5 · SAMformer 5 · PhaseFormer 0.
  C 포트 주기 48 지원(WindSCADA·BlastFurnace; MixLinear L/P 비제곱일 때 범위 밖 읽기 수정 — P24 영향 없음), golden P48 24/24 · `make test` 15/15.
  ③ **LTSF 1,920칸 완료**: FEATHer-CI 경량 13개 중 5등(0.375, 1등 칸 0) — "LTSF 평균 최저" 폐기; 전 칸 칩 안 7개 풀에서는 평균 1등(SMLP 와 0.1 %).
  ④ **베이스라인 칩 제약 선택 검토 중**(사용자): 후보 설정마다 칩 판정이 달라짐(PhaseFormer ETTh2·CMoS 비-Traffic·FITS 비-Weather 는 들어감) →
  `chip_verdict.py --candidates` 실행 중 → "24칸 전부 칩 안 후보 중 검증 1등"으로 다시 뽑기(사후 변경으로 공개). 바뀌는 모델(CMoS·PhaseFormer·FITS 후보)만
  main 시드 2026·27 + lookback 재실행 → 서버 lookback 은 이 3개 빼고 실행 중.
  ⑤ **재설계 진행**(사용자 (가) 선택): `docs/prereg_redesign_1012.md`(34da661) — 정규화 후보 B 평균만 · C RevIN 끔 · D 수준 학습 · E 섞기에 통계,
  1단계 96런 로컬 실행 중(새벽 2시쯤) → `tools/results/redesign_verdict.py --stage 1`(검증만) → 2단계 lr → 3단계 3시드. 다음 후보 = 헤드(청크 4·8·12, 크로네커;
  선행 = MixLinear·SegTSF·LightTS → "정확도 부품"으로만). "요약된 긴 과거"는 평가 방식을 바꿔야 해서 **사용자가 기각**.
  서버 결과 = Dropbox `WDSLab/05. 실험/FEATHer/results/` → 로컬 `results/_incoming/` → `merge_incoming.py`(로컬 raw 가 원본). ⚠ `deployment/board/golden/` 은 gitignore — git checkout 으로 복구 불가, 파일 복사로 백업.
- **다음**: 칩 제약 설정 확정 → 바뀐 모델 추가 실행 → 재설계 1·2·3단계 → 헤드 실험 → 보드 한 번(아슬아슬한 칸·P48·재설계) → 표 도구를 raw 로 → 주장 다시 쓰기.
- **보드 (10-07 저녁, 끝남)**: 새 C 포트 4개(CMoS·TimeBase·PhaseFormer·SAMformer) + 기존 4개(FITS·MixLinear·DiPE·SMLP)를
  후보 설정 대응으로(기본 export 는 바이트 동일). `make test` 14개 PASS. **모든 후보 구조의 지연**을 미리 잼 →
  `results/board/board_latency_candidates_20261007.csv`(77칸, MetroPT3 7채널 / H720 은 1채널 ×7) — 판정 후 보드 다시 안 꽂음.
  SparseTSF·DLinear·CycleNet 은 후보 구조가 같아 9/29 값 그대로. 9/29 값 재현 확인(SMLP 434 · DiPE 725 · FITS 274 · MixLinear 87).
  SAMformer 368 ms(H96 D7 만 칩 안). **전류 13개 모델 전부 실측** 29.2~30.0 mA(CycleNet·새 4개 추가).
  ⚠ FEATHer-mix 는 가장 빠르지 않음: H96 161 ms 중간(TimeBase 22~67 · CMoS 27~623 후보별 · SparseTSF 47 · MixLinear 72~87 더 빠름),
  H720 208 ms 상위권(SparseTSF 90 · TimeBase 90~270 · CMoS ETTh1 141). TimeBase = 비용 면 최대 위협 → 주장은 칩 안 정확도로.
- **원고 상태 (10-07 저녁)**: `manuscript/drafts/` = 초록 + 1~6장 초안 **완성**(영어 문장 + 한국어 줄, `¶` = 문단 끝; 리뷰 에이전트 2회 + 외부 리뷰 반영;
  베이스라인 걸린 숫자 `[TBD]`, 투고 전 확인 `[CHECK]`). 글쓰기 기준 = `manuscript/notes/analysis_flow.md`·`terminology.md`. 외부 리뷰에서 남은 실험 2개 = ① 섞기만 분리 arm
  (mix 모드 + 채널별 1→4→1, 72런; `_mixM` 2개 함께) ② front-none C 포트·보드 측정 + persistence 행. 빌더 TODO = Table 12 메모리 일치 열 · T6 칸별 풀 · 검정 단위 데이터(6쌍).
  빌더·표 도구는 아직 `results/_archive/pre_raw_20261007/` 의 옛 CSV 를 읽음(재정리 전후 원고 글자 동일 확인).
  v5 재빌드 안 함. 프레이밍 v2 = "과거를 길게 못 보면 옆 센서를 본다". 표 서식(8844 twip, 행 0.5 cm, 8 pt) · v1~v4 덮어쓰기 금지 ·
  drafts 형식(영어 한 문장 + 한국어 한 줄) · 프리프린트 주의 · 용어 channel 은 그대로.
- **Git**: 10-07 `06b089e`(프로토콜·실행기·결과 재정리·래퍼) · `2f76303`(Fable 반영: 칸 단위 칩·LTSF 선택) personal push.
  그 뒤 작은 수정 커밋 · `a6f2af9`(보드: 새 포트 4개 + 후보 대응 export) personal push. 커밋은 요청 시, personal 만.

### 코드 공개 전 할 일 · 제자리 유지 파일
(10-01·10-07 파일 정리 기록 = `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_2026-10-07b.md`)
- **제자리 유지 (옮기지 말 것)**: `deployment/BOARD_RESULTS_20260824.md`(펌웨어 바닥 10.4 KB 출처, 코드 주석 4곳이 가리킴)·`BOARD_SESSION_PLAN.md`, `board_footprint_recompute_*`(Table 7 의 31채널 근거), `probes`, `*_r2.csv`.
- **코드 공개(WDSLab/FEATHer) 전에 할 일 — 지금은 안 함, 검증 필요**: ① `FEATHer.py` 의 기각된 연구 옵션 제거(gate_mix, head2,
  mix_listen/lag/zero_init, input/input_raw 섞기, band_mode fft, spk_hidden, gate_mode 변형 등) — 제거 후 동결 모델 파라미터·
  출력이 바이트 단위로 같은지 확인 ② 보드 예전 포트 정리: 옛 938p `feather_infer` + `test_golden` + `export_golden.py`(다른
  export 스크립트가 import 함), 기각된 `feather2in` 포트 — Makefile 수정 + WSL `make test` 재실행 ③ `deployment/cortex_m3`
  (QEMU, 프리프린트가 쓴 평가)는 기록용으로 보관 ④ `tools/dev` 도구들의 기본 경로 정리.

### 확정 숫자 (원고에 쓰는 값)
- ⚠ **10-12**: 아래 베이스라인 관련 숫자·순위·"칩 안 가장 정확"·"LTSF 경량 평균 최저" 주장은 **재실험 전 값 — 쓰지 말 것**. 현재 값 = 위 상태 블록 + 메모리 `project_session_1012.md`.
- **(10-07 외부 리뷰 반영, 초안에 적용됨)**: ① "연산 수가 지연의 96 % 설명" **삭제**(FFT 기준 연산 수 r² 0.44; 직접 DFT 기준만 0.956) → 남는 주장 = "파라미터 수 r² 0.00" + 모델 예시 ·
  "파라미터가 flash 를 못 맞힌다" → "**한 예측 범위에서 보고된** 파라미터 수로는 못 맞힌다" ② **검정 단위 = 데이터(6쌍, H 평균, 단측 Wilcoxon 최소 p 0.016)**, 칸 수·시드 변동은 보조:
  섞기 이득 6/6 p 0.016(시드 5.7~6.2 %) · L192 5/6 p 0.031 · **L336 5/6 p 0.16(데이터 단위 비유의)** · front-none 5/6 p 0.031 · SparseTSF+통로 5/6 p 0.047 · STSF-MLP+통로 4/6 비유의 ·
  통로 on SparseTSF 6/6 · on STSF-MLP 5/6 · noresid 6/6 · band2 6/6 p 0.016 (스크래치패드 `dataset_level_stats.py`) ③ 5.9 % 는 항상 "**빼면 5.9 % 오른다**"(분모 mix) ④ mix–CI 차이는 섞기만이 아님
  (CI 는 필터·게이트도 채널별) → 6.1 한계점에 명시, **대조 arm "mix 모드 + 채널별 1→4→1" 72런 필요**(미실행) ⑤ 가이드 = "후보를 좁힌다 + 최종은 설비 검증 데이터" ⑥ "천 개 미만 유일" → "가장 작은 섞는 모델(296~1,924)";
  "경량 전부 CI" → "DLinear 보다 작은 경량 전부"(SAMformer 예외) ⑦ L>96 칩당 채널은 **상한**("최대") ⑧ 동일 메모리 우위 = 보간 → **데이터별 메모리 일치 비교로 교체 완료**(`tools/paper/memory_matched.py` → `results/analysis/memory_matched_20261007.csv`):
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
- (10-06) P1 옵션 `--factor_head 1` / `--factor_src low` / `--front none` (공유 인자 예측; **기본 꺼짐 = 동결 모델 바이트 동일**,
  검증 스크립트 결과 13/13 동일). 실험은 보류.

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
baselines/__init__.py             레지스트리(15) + _METHOD_DEFAULTS + _DATASET_OVERRIDES + _FEATHER_FROZEN
baselines/<Name>/wrapper.py       upstream Model 을 forward(x)->(B,H,D) 로 감쌈
utils/                            prep_manufacturing (raw -> data.csv + mask.csv) · data_factory · data_loader
                                  · dispatch_queue · losses · metrics · seed
tools/audit/                      check_progress · check_duplicates · purge_rows · results_table · scan_divergence
tools/search/                     동결 config 를 고른 기록 (재현용)
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
- 설정: LTSF = `baselines/__init__.py` `_DATASET_OVERRIDES`(10-07 수정 블록 포함, (모델, 데이터, H) 키) · mfg = `tools/results/protocol.py`
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
# 10-07 실험 관리 (docs/EXPERIMENT_PROTOCOL.md)
python run.py --stage select --check            # ltsf / ltsf_select / select / main / lookback; --check = 빠진 칸만 보기
python run.py --stage select --ngpu 1 --jobs_per_gpu 3
python tools/results/merge_incoming.py               # _incoming 조각 -> raw_*.csv / select_mfg.csv (--archive 로 조각 보관)
python tools/results/select_verdict.py --write       # 고르기 판정 -> docs/chosen_mfg.json (--group ltsf -> docs/chosen_ltsf.json), 커밋할 것

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

# 연동 강도 분석 (10-06, 사전 규칙 실패 기록)
python tools/paper/coupling_vs_mixing.py
```
