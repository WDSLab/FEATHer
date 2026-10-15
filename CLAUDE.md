# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
**사용자에게는 항상 한글로 답한다.** 지난 상태 블록·옛 확정 숫자·한계점 문장·JMS 글쓰기 방향 원문 =
`manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_*.md` (최신 = `..._2026-10-14c.md`, 10-14 밤 정리 직전 전체). 평소엔 읽을 필요 없음 —
세션별 상세는 메모리 파일.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) = 기계 옆 MCU
(NUCLEO-F103RB: 128 KB flash / 20 KB SRAM, FPU 없음)에서 **장기(H 96~720) 다채널 예측**을 하는 경량 모델.
논문 = **Journal of Manufacturing Systems (JMS)** 신규 투고 (TPAMI reject → IoT-J → JMS).

- **메인 비교 (10-13~14 결정) = 크기를 맞춘 비교**: 1K판(칩 저장 숫자 ≤ 1,000) · 칩 최대판(실제 펌웨어가 128 KB / 20 KB 안).
  저자 설정 결과·옛 FEATHer·4K 는 부록/미사용. 판정 기준 = 실측 flash·SRAM·보드 지연시간.
- **파라미터 수는 장점으로 내세우지 않음** · "옆 센서를 본다"(입력 길이) 주장은 메인에서 뺌 (L96 은 SRAM 표로 정당화).
- 논문은 간결하게: 본문 = 설정 표 · 크기 맞춘 정확도 2판 · 정확도 vs flash 그림 · 칩 비용 · 작은 ablation.

## Current status (2026-10-15) — 1K판 고르기 서버 실행 중 · 칩판 추첨 로컬 실행 중

**새 세션 시작점 = 메모리 `project_session_1014.md` 의 HANDOFF.** 규칙 원문 = `configs/experiments/size.yaml` +
`docs/EXPERIMENT_PROTOCOL.md` 결정 기록 "2026-10-14 밤" (a)~(q).

- **방식**: 저자 설정은 LTSF 에 맞춘 값 → **모든 모델을 제조 데이터에서 같은 규칙으로 재튜닝**(선례 Brigato TMLR'26 ·
  ProbTS · Melis ICLR'18 · Lucic NeurIPS'18; 공개 코드 대조 = Brigato NoChamps · Gorishniy rtdl; Fable 검토 2회).
  모델마다 **40개 전부 무작위**(고정 시드, 실행 전 커밋, 저자 설정 · 동결 FEATHer 를 출발점으로 넣지 않음).
  공통 범위: lr 1e-5~1e-1 로그 · dropout {0,.05,.1,.2} · 스위치 켬/끔 · 비율 {.1….9} · 주기 {2…48}(탐색) · 구간 {1…48};
  모델별 목록은 단위가 그 모델에만 있는 것뿐. 크기 = 목표 크기(상한의 U(0,1] 비율) + 크기 하이퍼파라미터 하나를 그 이하
  최대로, 나머지 무작위(개수 = 1 ~ 다른 값을 가장 작게 둘 때 들어가는 최대, 로그 균등). 유효성 = 만들어지고 4개 H 모두 돌아감.
- **고르기**: 40개 × 6데이터 × H96·H720 · 시드 2025, **15 epoch / patience 3**(Brigato 탐색 예산) → **데이터마다** 검증 순위
  평균으로 하나 → 정식 4 H × 3시드는 **50 / 10**(테스트 1회). 설정 하나로 4개 H(칩 배포 조건).
- **두 판 (모두 상한, 4개 H 모두)**: 1K판 = 학습 숫자 ≤ 1,000(숫자 세기, 고른 48개만 WSL 빌드) · 칩판 = 실제 펌웨어가 칩에
  (상한 = (128 KB − 가중치 외 flash)/4, 후보마다 H720 실제 빌드, "칩에 담을 수 있는 최선" — 작은 모델도 포함).
  제외: 1K = DiPE · DLinear · CycleNet · SAMformer, 칩판 = DLinear · CycleNet · SAMformer.
- **FEATHer**: 두 트랙 FEATHer_mix · FEATHer_CI(별개 모델, 두 줄 항상 다 실음), 섞기 MLP 는 두 트랙 모두 켬 고정,
  탐색 공간·조합은 베이스라인과 같은 커밋에 확정, **실행은 베이스라인 뒤**(run.py 는 이름을 줘야 FEATHer 실행).
- **진행 (10-15 낮)**: 1K 조합 커밋(`docs/size_trials.json`) → 서버 `size_select --tier 1k`(3,360칸, 짧은 학습으로 재시작,
  약 3일 예상; 서버 결과 조각에 10-13 옛 행이 섞여 있음 → merge 전에 치울 것). C 포트 확장(3e05eea: 탐색 공간의 모든 구조 값
  빌드 가능, make test 50 PASS 그대로 + check_dials 43). 칩판 추첨 로컬 실행 중(`size_trials.py --write --tier chip`,
  빌드 약 10초, 8~10시간; 그동안 다른 보드 빌드 금지).
- **LTSF**: 새 실험 없음. 기존 저자 설정 결과로 짧게(경량 8개 + FEATHer 두 트랙; DLinear · CycleNet · SAMformer · 큰 모델 제외).
- **안 하는 것**: 부록 없음 · 부록용 추가 실험 없음(PhaseFormer cand:Weather 72 · iTransformer 재실행 96 · lookback 전부 안 함).
  공정성은 본문 한 문단 + 조합표·모든 시도 결과는 공개 저장소 파일로만.
- **나중에**: 논문 표 도구 다시 쓰기(`tools/paper/*` 는 10-07 이전 CSV 를 읽음) · README.md · `analysis/diagnostics/*`(기록용).
- **사용자 결정 (다시 꺼내지 말 것)**: "결과 보고 비교 방식 바꿨다" 걱정 · int8/행렬 합치기 실험 · 사전 등록 문서.
- **Git**: 커밋은 요청 시, **personal(ijaehun) 에만 push**. ⚠ 보드 빌드는 한 번에 하나(golden/ 공유).

### 아직 유효한 측정값 (모델 설정과 무관)
- **보드 실시간 재생 (MetroPT3 s2025, 동결 FEATHer-mix)**: H96 1,022회·H720 1,012회 전부 보드 MSE = PyTorch, 지연 평균 161 / 211 ms,
  계산 비율 0.004 / 0.006 %, 펌웨어 14.3 / 14.7 KB flash, 7.55 KB RAM ⇒ 현장 주기에서 속도는 선택 기준 아님.
- **전류** 13모델 29.2~30.0 mA (10-07 실측) → 에너지 = 3.3 V × 30 mA × 지연. 파라미터 수 → 보드 지연 r² 0.00.
- 펌웨어 바닥 10.4 KB = `deployment/BOARD_RESULTS_20260824.md`.

### 주장 금지 (반복 확인됨)
- "DLinear 를 이긴다" · "H 가 길수록 DLinear 대비 격차가 커진다"(커지는 건 비용 차이) · "LTSF 1위" → "on par with the best lightweight".
- "현장 의사결정(정비·에너지)이 개선된다" — 평가 안 함 → 한계점/향후 과제. "보드 재생 = 실설비 설치" 아님(저장 데이터 재현).
- "센서 연동이 강할수록 섞기 이득이 크다"(ρ 0.36 실패) · "시간차 반응을 잡는다" · "섞기가 센서 고장에 강하다"(반대: 고장을 퍼뜨림, 한계점).
- 밴드·게이트: 따로따로 숫자 금지, 메커니즘·해석 도구 주장 금지 → "한 묶음, 약 1 %". "효과 없음" 대신 기여 순위로.
- 모양 지표(CORR 등) 주장 금지 — MSE/MAE 만. 순위를 말할 땐 **어느 풀인지** 명시.
- "제조 TSF 벤치마크 최초" — RCIM(arXiv 2411.17499) 선점 → "최초"는 MCU 배포·전 채널 동시·경량 tier 로 좁힘.
- 옛 FASTer 0.373(test-peeking) 인용 금지.

### 용어
**channel** (feature 금지). 제목·키워드 "multi-sensor" 유지. SPK 헤드는 **SparseTSF 에 귀속**. **forecast 계열만**
(predict 는 predictive maintenance·RUL·참고문헌 제목·통계적 의미에만). "크기 하이퍼파라미터"/"size hyperparameter"("다이얼" 금지).
전체 통일표 = `manuscript/notes/terminology.md`.

### 제자리 유지 (옮기지 말 것)
`deployment/BOARD_RESULTS_20260824.md`(코드 주석 4곳이 가리킴)·`BOARD_SESSION_PLAN.md`, `board_footprint_recompute_*`, `probes`, `*_r2.csv`.

## Architecture (FEATHer) — FROZEN (09-04 구조, 09-23 mlp_rank 4 재동결)

`baselines/FEATHer/FEATHer.py`(조립 + 허용 옵션표 OPTIONS) · `gate.py` · `heads.py` · `ablation.py`(부품 바꾸는 ablation 4종) ·
`variants.py`(대조 실험 변형 2종) · `wrapper.py`, 설정 `configs/FEATHer.yaml`. 클래스 기본값 = 동결 모델. 변경 시 스윕 재실행 필요.
크기 맞춘 비교용 옵션 `--agg_shared 1`(다듬기 필터 공유) · `--head_seg s`(SPK 구간 길이; 0 = 주기) — 기본 꺼짐 = 동결 모델 바이트 동일.

```bash
--no_in_proj --num_bands 3 --point_kernel 3 --dtk_mode none --gate_mode banded --mlp_rank 4
mfg:  --channel_mode mix          LTSF: --channel_mode ci_full   (overrides 로 자동)
```
```
RevIN -> 3 bands: conv k=3 | conv k=5 | AvgPool(4)+interp -> banded FFT gate -> weighted sum
      -> residual MLP (D->4->GELU->D) -> SPK head -> RevIN^-1
```
- `--band_kernels 3,5` 는 RNG 흐름이 바뀌어 초기화가 다름 → 항상 `--point_kernel` 형태.
- banded 게이트 = 대역별 에너지 비율 → `Linear(S,S)`, 0 초기화. 잔차 MLP = 유일한 비선형(`ci_full` 에선 1→4→1).
- Loss = MSE + 0.01 × spectral-separation. 파라미터: mfg-mix Steel 296 / CI 76 · LTSF ETTh1 CI 298 / mix 454.

## Environment & gotchas

- **Python**: `C:\Users\lee\.conda\envs\feather\python.exe` (torch 2.5.1+cu121; Py 3.9). bash 의 `python` 은 Windows 스토어 스텁 → 경로 직접.
  docx 스킬 스크립트는 Py≥3.10 → `/c/ProgramData/anaconda3/python.exe`.
- **Dropbox 폴더에서 `rmtree` 금지** (반쯤 지워짐). 파일 단위로.
- Bash heredoc 안의 `\n`/`\u` 이스케이프가 깨짐 → 스크립트는 Write 로 파일 작성 후 실행.
- Git Bash 에서 `wsl.exe` 에 `/mnt/...` 경로를 넘길 땐 `MSYS_NO_PATHCONV=1`. Windows 출력: `PYTHONUTF8=1`.
- 원고 렌더 확인: PowerShell Word COM `SaveAs2(pdf, 17)` → `pdftoppm` (LibreOffice 없음).
- 보드: WSL gcc 10.3 (footprint) vs CubeIDE GCC 14.3 (보드 빌드). WFI 로 잠든 칩은 connect-under-reset.
  CubeIDE 프로젝트는 repo 밖 `C:\Users\lee\STM32Cube_projects\blink_test`.
- AMP 강제 OFF (FITS 복소 grad, TimesNet half cuFFT). upstream 은 반드시 `_import_helper.isolated_import`.
  `baselines/*/*-main/` upstream 클론은 gitignore — 수정 금지, `setup_baselines.sh` 로 재클론.
- R2/CORR 은 보조 지표. 저장된 R2 열은 쓸 수 없음(`tools/paper/compute_r2.py` 로 사후 계산).

## Repository structure

`utils/` = 데이터·학습 기계, `tools/` = 사람이 돌려 산출물을 만드는 것 (학습 경로에서 import 안 함).

```
run.py                       유일한 실행 입구 — 프로토콜이 요구하는 칸 중 결과에 없는 것만 실행 (hparams 열이 바뀐 칸도 다시)
                                  (--stage ltsf / ltsf_select / select / main / lookback / ablation / control / size_select / size)
scripts/train.py                  학습 코드 (모델 × 데이터 × H × 시드들 → 결과 행). run.py 가 부름
configs/<Model>.yaml              모델별 하이퍼파라미터 (default + 데이터별 + "<데이터>@<H>"); 값은 제자리에서 고침
configs/experiments/*.yaml        실험 정의: ablation · control · size; protocol.py 가 읽기만 함
baselines/__init__.py             모델 목록 MODELS + 설정 읽기 함수;  baselines/<Name>/wrapper.py = upstream 을 forward(x)->(B,H,D) 로
tools/results/                    protocol(후보·설정·main_rows/KEY) · merge_incoming · select_verdict · size_trials(40개 추첨) · size_verdict · size_hparams
tools/paper/                      merge_raw · table_* · fig_pareto_board · wilcoxon · ablation_summary · make_result_html · observe_structure …
tools/audit/ · tools/dev/         진행·중복 점검 / bs512 빠른 dev 루프
utils/                            prep_manufacturing · data_factory · data_loader · dispatch_queue · losses · metrics · seed
docs/                             EXPERIMENT_PROTOCOL.md (실험 규칙·결정 기록) · MODEL_FINAL.md · ENV.md · chosen_{mfg,ltsf,size}.json · size_candidates.json
deployment/board/                 C 포트 src/*_infer.{c,h} · export_golden_*.py · test/ (make test) · measure_footprint · board_run · chip_verdict
                                  · main.c.stream / stream_flash.py / replay.py (실시간 재생)
manuscript/version/               원고 docx + _build/ (v1~v4 덮어쓰기 금지)
_archive/                         은퇴 코드 (FEATHer_full_20261014, board_legacy, cortex_m3, redesign_1012, tools_20261014 …)
```

**C 포트 13종**: feather2 / feather2r (보드는 재계산 스케줄 feather2r) · feather2ci · dlinear · cyclenet · stsf · stsf_mlp · mixlinear ·
dipe · fits · cmos · timebase · phaseformer · samformer. **공통 규칙 ①**("원래 계산은 그대로, 메모리는 정리만"): 안 섞는 모델은
채널 하나씩 · 출력은 나오는 대로 방출 · 작업공간 공유 · 재계산 금지(FEATHer-mix 정규화+밴드 재계산만 예외). 96점 FFT 는 전 모델
같은 커널(`fea_rfft96_fast`). export 가 `<PFX>_H_GEN` 등을 스탬프 → `.c` 가 `_Static_assert`. 골든 가중치는 전부 랜덤.

## Results files (상세 `results/README.md`)

| 파일 | 내용 |
|---|---|
| `results/raw_mfg.csv` · `raw_ltsf.csv` | 정식 결과 (role = main/reference/lookback/ablation/control/size…, config_id, hparams) |
| `results/select_{mfg,ltsf}.csv` | 설정 고르기 결과 (고른 설정 = `docs/chosen_*.json`, 커밋함) |
| `results/_incoming/` | 새 실행 조각 → `merge_incoming` 으로 합침 |
| `results/board/` | 보드·칩 측정 (footprint, latency, candidates, current, replay …) |
| `results/analysis/` | 분석 산출물 (size_selection_audit 등) · `results/dev/` = dev 실험(원고 숫자 아님) |
| `results/_archive/` | 옛 CSV (pre_raw_20261007 = 원고 빌더가 아직 읽음, legacy_rows, itransformer_month_marks) |

CSV 키: `role, config_id, exp_tag, model, data, pred_len, seq_len, seed` + MSE/MAE/RMSE/CORR/R2/val_loss/best_epoch/num_params.
FEATHer 이름 = `FEATHer_mix` / `FEATHer_CI` (footprint CSV 는 mix 를 `FEATHer` 로 씀).

## Models (레지스트리 19개)

- **경량 tier (칩 비교 대상)**: SparseTSF(ICML'24) · MixLinear(ICLR'26) · DiPE_Linear(DASFAA'26) · SparseTSF_MLP · FITS(ICLR'24) ·
  CycleNet(NeurIPS'24) · DLinear(AAAI'23) · CMoS(ICML'25) · TimeBase(ICML'25) · PhaseFormer(ICLR'26) · SAMformer(ICML'24, 섞음) + FEATHer.
  DLinear·CycleNet·SAMformer 는 4개 H 모두 들어가는 크기로 줄일 수 없어 1K판·칩판 모두 제외(부록 저자 설정 결과로만, 10-14 밤).
  DiPE 는 1K판 제외 · 칩판 포함.
- **정확도 천장 참고(칩 불가)**: PatchTST · TimeMixer · TQNet · LMS_AutoTSF · TimesNet · iTransformer · MDMLP_EIA.
- **제외(관련연구만)**: S-Mamba · TimeCMA · SWIFT · F-Net · LiPFormer·LightMTS·FilterNet·TinyCast·RhyMix 등. 크기는 제외 기준 아님.

## Training protocol — `docs/EXPERIMENT_PROTOCOL.md` 가 기준

- 전 모델 공통: L **96** · batch 32 · 50 epoch · patience 10 · Adam + Cosine · clip 1.0 · AMP off · 3시드 2025–2027 ·
  best-val epoch 에서 test 1회 · loss = 관측값 MSE + 방법 고유 항.
- 주기류 값은 규칙으로 계산(고르는 대상 아님). 결측 = hold-last + `mask.csv` → loss/지표는 관측값만.
- 워커 훅: `_aux_loss`(TimeBase) · `_train_loss` / `_epoch_start` / `_test_start`(DiPE) · SAMformer SAM.

## Datasets

- **mfg-6 (메인)**: Steel · WindSCADA · BlastFurnace(D=28) · N2O · MetroPT3 · EWELD — `utils/prep_manufacturing.py`.
- **LTSF-8 (일반화)**: ETTh1/h2/m1/m2 · Weather · Exchange · Electricity · Traffic. H = {96, 192, 336, 720}.
- 후보 데이터 논의 시 먼저 메모리 `reference_dataset_reject_log.md` 확인. 뺀 데이터 = `data/_shelf/`.

## Common commands

```bash
# 크기 맞춘 비교 (메인)
python tools/results/size_trials.py --write [--tier chip] # 40개 추첨 -> docs/size_trials.json (실행 전 커밋; chip = 실제 빌드)
python run.py --stage size_select --tier 1k --check      # 후보 x lr x H96/H720 x 시드 2025 (--check = 빠진 칸만 보기)
python tools/results/size_verdict.py --write             # -> docs/chosen_size.json + size_selection_audit.csv
python run.py --stage size --ngpu 1 --jobs_per_gpu 3     # 고른 것 x 4H x 3시드
# 그 밖의 단계
python run.py --stage main --check                       # ltsf / ltsf_select / select / main / lookback / ablation / control
python tools/results/merge_incoming.py                   # _incoming -> raw_*.csv
python tools/results/select_verdict.py --write
python tools/paper/make_result_html.py                   # 결과표 HTML -> results/result.html
# 보드 / C
cd deployment/board && make test                         # WSL, 골든 전부
python deployment/board/chip_verdict.py [--candidates]   # 칸 단위 칩 판정 (보드 없이)
python deployment/board/board_run.py --stream            # 가장 빠듯한 칸을 실제 채널 수로 보드 실측
python deployment/board/replay.py --data MetroPT3 --H 96 --port COM3 --every 60   # --sim = 보드 없이
# 감사
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py
```
