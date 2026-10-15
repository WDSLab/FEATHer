# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
**사용자에게는 항상 한글로, 아주 쉬운 말로, 결정은 하나씩 답한다.** 옛 상태 블록 원문 = `manuscript/notes/_archive/CLAUDE_STATUS_ARCHIVE_*.md`
(평소엔 읽지 않음). 세션 상세 = 메모리 `project_session_1015.md`.

## Project

FEATHer = 기계 옆 MCU(NUCLEO-F103RB: 128 KB flash / 20 KB SRAM, FPU 없음)에서 **장기(H 96~720) 다채널 예측**을 하는 경량 모델.
논문 = **Journal of Manufacturing Systems (JMS)** 신규 투고. 간결하게: 본문 = 설정 표 · 크기 맞춘 정확도 2판 · 정확도 vs flash 그림 ·
칩 비용 · 작은 ablation. **부록 없음.** 칩 비용은 실측 flash · SRAM · 보드 지연으로만 말한다(파라미터 수 → 지연 r² 0.00).

## Current status (2026-10-15)

규칙 원문 = `configs/experiments/size.yaml` + `docs/EXPERIMENT_PROTOCOL.md` 결정 기록 "2026-10-14 밤" (a)~(r).

- **메인 = 크기 맞춘 비교, 모든 모델을 제조 데이터에서 같은 규칙으로 재튜닝** (저자 설정은 LTSF 에 맞춘 값).
  선례 Brigato TMLR'26 · ProbTS · Melis ICLR'18 · Lucic NeurIPS'18 (공개 코드 대조: Brigato NoChamps · Gorishniy rtdl).
  - 모델마다 **40개 전부 무작위**(고정 시드, 실행 전 커밋 `docs/size_trials.json`; 저자 설정 · 동결 FEATHer 를 출발점으로 넣지 않음).
  - 공통 범위: lr 1e-5~1e-1 로그 · dropout {0,.05,.1,.2} · 스위치 켬/끔 · 비율 {.1…,.9} · 주기 {2…48} · 구간 {1…48}.
    크기 = 상한의 U(0,1] 비율을 목표로 크기 하이퍼파라미터 하나를 채움, 나머지 무작위. 유효성 = 4개 H 모두 만들어지고 돌아감.
  - 고르기 = 40개 × 6데이터 × H96·H720 · 시드 2025 · **15 epoch / patience 3** → 데이터마다 검증 순위 평균으로 하나 →
    정식 = 4 H × 3시드 · **50 / 10** (테스트 1회). 설정 하나로 4개 H.
- **두 판 (상한, 4개 H 모두)**: 1K판 = 학습 숫자 ≤ 1,000 · 칩판 = 실제 펌웨어가 칩에(상한 = (128 KB − 가중치 외 flash)/4,
  후보마다 H720 실제 빌드). 제외: 1K = DiPE · DLinear · CycleNet · SAMformer, 칩판 = DLinear · CycleNet · SAMformer.
- **FEATHer**: 두 트랙 FEATHer_mix · FEATHer_CI(별개 모델, 두 줄 항상 다 실음, 섞기 MLP 켬 고정). 조합은 이미 확정,
  **실행은 베이스라인 뒤**(run.py 는 `--models FEATHer_mix,FEATHer_CI` 를 줘야 실행).
- **진행**: 서버(DESKTOP-K390KQ4) 1K 고르기 3,360칸 실행 중(짧은 학습, 약 3일; 서버 결과 조각에 10-13 옛 행이 섞여 있음 →
  merge 전에 치울 것). 로컬 칩판 추첨 실행 중(`size_trials.py --write --tier chip`, 모델당 약 45분, 10-15 밤 끝; 그동안 보드 빌드 금지).
  C 포트는 탐색 공간의 모든 구조 값을 지원(3e05eea).
- **LTSF**: 새 실험 없음. 기존 저자 설정 결과(`raw_ltsf.csv`)로 짧게, 경량 8개 + FEATHer 두 트랙만. 결과: FEATHer-CI 는 MSE 6위 /
  MAE 4위(10개 중), 1등 칸 0 — 작은 모델(SparseTSF · MixLinear · TimeBase)은 크게 이기고, 7~29배 큰 모델(FITS · CMoS ·
  PhaseFormer · SMLP)에는 짐. "더 작으면서 더 정확한 모델 없음". **넣을지 · 어떻게(정확도 vs 크기 그림) 보여줄지는 메인 결과 후 결정.**
  예전 "LTSF 평균 최저"는 10-07 설정 수정 · CMoS/PhaseFormer 추가 전 값(쓰지 말 것).
- **안 하는 것**: 부록 · 부록용 실험(PhaseFormer cand:Weather 72 · iTransformer 재실행 · lookback) · int8/행렬 합치기 ·
  사전 등록 문서 · "결과 보고 비교 방식 바꿨다" 걱정 다시 꺼내기.
- **Git**: 커밋은 요청 시, **personal(ijaehun) 에만 push**. 보드 빌드는 한 번에 하나(golden/ 공유).

### 아직 유효한 측정값
- 보드 실시간 재생(MetroPT3, 동결 FEATHer-mix): 보드 MSE = PyTorch, 지연 161 / 211 ms (H96/H720), 펌웨어 14.3 / 14.7 KB, RAM 7.55 KB.
- 전류 13모델 29.2~30.0 mA → 에너지 = 3.3 V × 30 mA × 지연. 펌웨어 바닥 10.4 KB = `deployment/BOARD_RESULTS_20260824.md`(옮기지 말 것).

### 주장 금지
- "DLinear 를 이긴다" · "LTSF 1위/가장 정확" → "on par with the best lightweight" · 파라미터 수로 속도·비용 주장.
- "현장 의사결정이 개선된다"(평가 안 함) · "보드 재생 = 실설비 설치" · "섞기가 센서 고장에 강하다"(반대) ·
  "센서 연동이 강할수록 섞기 이득이 크다" · 밴드 · 게이트 따로 숫자 · CORR 등 모양 지표 · 순위는 어느 풀인지 명시.
- "제조 TSF 벤치마크 최초"(RCIM 2411.17499 선점) · 옛 FASTer 0.373(test-peeking).

### 용어
**channel**(feature 금지) · "multi-sensor" 유지 · SPK 헤드는 SparseTSF 에 귀속 · forecast 계열만 · **하이퍼파라미터**("다이얼" ·
"손잡이" 금지). 통일표 = `manuscript/notes/terminology.md`.

## Architecture (FEATHer, frozen)

`baselines/FEATHer/FEATHer.py`(조립 + OPTIONS) · `gate.py` · `heads.py` · `ablation.py` · `variants.py` · `wrapper.py`, 설정
`configs/FEATHer.yaml`. 클래스 기본값 = 동결 모델. 크기 비교용 옵션(`head_seg`, `agg_shared` 등)은 기본 꺼짐 = 동결 모델 동일.
```
RevIN -> 3 bands: conv k=3 | conv k=5 | AvgPool(4)+interp -> banded FFT gate -> weighted sum
      -> residual MLP (D->4->GELU->D) -> SPK head -> RevIN^-1          (mfg: mix, LTSF: ci_full)
```
Loss = MSE + 0.01 × spectral-separation. `--band_kernels` 대신 항상 `--point_kernel`(RNG 흐름).

## Environment & gotchas

- Python `C:\Users\lee\.conda\envs\feather\python.exe` (torch 2.5.1+cu121, Py 3.9); bash `python` = 스토어 스텁. `PYTHONUTF8=1`.
- Dropbox 폴더에서 `rmtree` 금지. heredoc 안 `\n` 이스케이프 깨짐 → 스크립트는 Write 로. WSL 경로엔 `MSYS_NO_PATHCONV=1`.
- 보드: WSL gcc(footprint, make test) vs CubeIDE GCC 14.3(보드). CubeIDE 프로젝트 = `C:\Users\lee\STM32Cube_projects\blink_test`.
- AMP OFF. upstream 은 `_import_helper.isolated_import`, `baselines/*/*-main/` 수정 금지(`setup_baselines.sh`).

## Repository

```
run.py                      유일한 실행 입구 (빠진 칸만 실행; --stage ... size_select / size, --tier 1k|chip, --check)
scripts/train.py            학습 워커           configs/<Model>.yaml · configs/experiments/{ablation,control,size}.yaml
tools/results/              protocol · merge_incoming · select_verdict · size_trials(추첨) · size_verdict(고르기) · size_hparams
tools/audit/                check_progress · check_duplicates · check_size_trials(추첨 점검)
tools/paper/                표·그림 도구 (10-07 이전 CSV 를 읽음 → 새로 써야 함)
docs/                       EXPERIMENT_PROTOCOL.md · size_trials.json · chosen_{mfg,ltsf,size}.json · MODEL_FINAL · ENV
deployment/board/           C 포트 13종 src/*_infer.* · export_golden_*.py · make test · check_dials.py · chip_verdict · board_run · replay
results/                    raw_{mfg,ltsf}.csv · select_*.csv · _incoming/ · board/ · analysis/ · _archive/
```
C 포트 공통 규칙 ①: 원래 계산 그대로, 메모리만 정리(FEATHer-mix 재계산만 예외). export 가 `*_GEN` 스탬프 → `.c` 가 `_Static_assert`.
CSV 키 `role, config_id, model, data, pred_len, seq_len, seed` (+ hparams). FEATHer 결과 이름 = `FEATHer_mix` / `FEATHer_CI`.

## Models · data · protocol

- 경량: SparseTSF · SparseTSF_MLP · MixLinear · DiPE_Linear · FITS · CMoS · TimeBase · PhaseFormer (+ DLinear · CycleNet · SAMformer:
  크기 비교 제외) + FEATHer. 큰 참고 모델 7개(PatchTST 등)는 이제 쓰지 않음.
- mfg-6: Steel · WindSCADA · BlastFurnace(D=28) · N2O · MetroPT3(가장 큼, 계산 41%) · EWELD. LTSF-8: ETTh1/2 · ETTm1/2 · Weather ·
  Exchange · Electricity · Traffic. H = {96, 192, 336, 720}. 데이터 논의 전 메모리 `reference_dataset_reject_log.md`.
- 학습 공통: L 96 · batch 32 · Adam + cosine · clip 1.0 · 3시드 2025–2027 · best-val epoch 에서 test 1회 · 관측값만 채점.

## Commands

```bash
python tools/results/size_trials.py --write [--tier chip]    # 40개 추첨 (실행 전 커밋)
python tools/audit/check_size_trials.py [--tier chip]         # 추첨 점검
python run.py --stage size_select --tier 1k --check           # 고르기 계획 (서버: --ngpu 2 --jobs_per_gpu 6)
python tools/results/size_verdict.py --write                  # 데이터마다 고르기 -> docs/chosen_size.json
python run.py --stage size --tier 1k                          # 정식 4 H x 3시드
python tools/results/merge_incoming.py                        # _incoming -> raw_*.csv
cd deployment/board && make test                              # WSL, C = PyTorch 대조 (python check_dials.py = 새 값 43개)
```
