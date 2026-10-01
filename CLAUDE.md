# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
지난 상태 블록(06월~09-30)은 `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-10-01.md` 에 원문 그대로 있다
(평소엔 읽을 필요 없음 — 세션별 상세는 메모리 파일).

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) = 기계 옆 MCU
(NUCLEO-F103RB: 128 KB flash / 20 KB SRAM, FPU 없음)에서 **장기(H 96~720) 다채널 예측**을 하는
296-파라미터 모델 (Steel D=6, H=96). 논문 = **Journal of Manufacturing Systems (JMS)** 신규 투고
(TPAMI reject → IoT-J → JMS).

**주장의 범위**: 칩에 실제로 들어가는 모델들 중에서 비교한다. 최신 경량 모델은 전부 채널을 독립으로 다루는데,
제조 데이터에서는 채널을 섞는 게 이득이고, FEATHer는 섞기를 MCU 메모리 안에 들어가게 만들었다
(칩당 43채널). 파라미터가 H에 거의 무관(1.008배)해서 긴 H로 갈수록 칩 안 비교에서 유리해진다.
판정 기준은 파라미터가 아니라 **실측 flash·SRAM·보드 지연시간**.

## Current status (2026-10-01) — 원고 글쓰기 단계

**표 T1–T9 + Fig.1 사용자 확정(T6 는 10-01 v3 로 수정). 남은 실험 = 입력 길이 L192·336 (서버, 아래 "진행 중"). 그 외엔 본문.**
상세 = 메모리 `project_session_0930.md`(표 최종안) · `project_session_0929.md`(보드·빠른 FFT).

- **원고 파일**: `FEATHer_JMS_v1.docx`·`v2.docx` = 사용자가 검토한 사본 (**덮어쓰기 금지**).
  현재 빌드 대상 = `manuscript/version/FEATHer_JMS_v3.docx` ← `python manuscript/version/_build/make.py` 한 줄
  (사용자가 새 버전을 원하면 `make.py` 의 DEFAULT_OUT 을 올림)
  (`_build/template_merged.docx` → `build_docx.py` + `tables_cf.py`). 서식 = 사용자 CF-JEPA KBS R2 논문.
  1·2·3·5장 본문과 참고문헌은 노란 자리표시. 노란 표시 = CRediT 역할, 코드 공개 주소, AI 선언, 연구비.
  제목 *FEATHer: Long-horizon multi-sensor forecasting on machine-side microcontrollers for manufacturing
  systems*, 저자 5명(공동1저자 † / 공동교신 *).
- **표 순서(확정)**: T1 경량 모델 설계(Domain/Nonlinear/Mixes channels — FEATHer 행만 Yes) · T2 데이터셋 ·
  T3/T4 제조 정확도(순수 정확도, 9모델 전체 굵게, 숫자 비우지 않음) · T5 칩 비용(flash 4H, 넘치는 가중치도 숫자로)
  · Fig.1(`tools/paper/fig_pareto_board.py`, 오른쪽 위 = 좋음) · **T6 칩 안 종합 비교(10-01 v3: 24칸 전부 칩에
  들어가는 7모델만 — DLinear·CycleNet 열 제외, 주석에 두 모델 사실 + DLinear flash 4.6~30배; FEATHer-mix MSE 순위
  1.50·1등 18/24, MAE 2.17·17/24 = 둘 다 1등)** · T7 부품 ablation · T8 채널 섞기 분석 · T9 LTSF.
- **▶ 다음**: 4장 결과 본문(표 순서대로) 또는 3장 구조 그림 → 3장 방법 → 1장 서론 → 2장 관련연구 → 초록 →
  노란 표시 확인 → CF-JEPA 잔재 정리 → 커밋(요청 시, personal 만; 미커밋 다수).

### ▶ 진행 중 (10-01): 입력 길이(룩백) 확인 — 서버에서 실행, 로컬은 코드만
- **왜**: 베이스라인 룩백(`results/_archive/baselines/fcst_results_baselines.csv`, exp_tag `lookback`, 경량 6개 ×
  mfg-6 × 4H × 3시드, STSF-MLP 없음)에서 입력이 길면 전부 좋아짐: L96→192 −5~6%, →336 −9~11%.
  **L192 에서 FITS 0.681 · DiPE 0.683 · CycleNet 0.682 가 우리 L96 0.699 보다 좋음.** 현재 FEATHer 룩백 결과는 없음.
- **범위**: L192·336 만. L720 은 모든 모델 제외 (7채널 입력 버퍼 20,160 B + 스택 552 B > SRAM 20,480 B; 한계점 한 문장).
  LTSF 룩백은 안 함.
- **0단계 완료**: 데이터별 주기(24/48)가 192·336 을 나눔 · 시험 구간 예측 목표 시점은 L 무관(border1 = 시작 − L) ·
  FEATHer·STSF-MLP 를 L96/192/336 × H96/720 × 6데이터로 CPU 생성·순전파 OK, L96 파라미터 24칸이 정식 CSV 와 일치.
  `tools/dev/run_dev.py` 에 arm `L96`/`L192`/`L336` 추가 (미커밋 → 서버 반영 필요).
- **서버 명령** (윈도우면 `PYTHONUTF8=1`, 공유 박스면 `FORCE_OMP`):
  ```bash
  # 1단계 dev (216런): 판단 = 베이스라인만큼(−5~6% / −9~11%) 좋아지나
  python tools/dev/run_dev.py --arms L96,L192,L336 --data Steel,WindSCADA,BlastFurnace,N2O,MetroPT3,EWELD --pred_len 96,720 --csv results/dev_lookback.csv --prefix lb
  python tools/dev/dev_summary.py --csv results/dev_lookback.csv --prefix lb --ref L96 --pred_len 96   # 와 --pred_len 720
  # 2단계 정식 (각 72런, 모델·L 마다 CSV 따로)
  python run_forecast.py --model FEATHer --group mfg --seq_len 192 --exp_tag lookback --num_seeds 3 --results_csv results/feather_mfg_lookback_L192.csv --ngpu 1 --jobs_per_gpu 3
  python run_forecast.py --model FEATHer --group mfg --seq_len 336 --exp_tag lookback --num_seeds 3 --results_csv results/feather_mfg_lookback_L336.csv --ngpu 1 --jobs_per_gpu 3
  python run_forecast.py --model SparseTSF_MLP --group mfg --seq_len 192 --exp_tag lookback --num_seeds 3 --results_csv results/stsf_mlp_mfg_lookback_L192.csv --ngpu 1 --jobs_per_gpu 3
  python run_forecast.py --model SparseTSF_MLP --group mfg --seq_len 336 --exp_tag lookback --num_seeds 3 --results_csv results/stsf_mlp_mfg_lookback_L336.csv --ngpu 1 --jobs_per_gpu 3
  ```
- **사전 판단 기준**: 같은 L 에서 FEATHer-mix 가 칩 7모델 중 MSE 평균순위 1등 유지 → 같은-L 비교표 추가, 주장 유지 /
  베이스라인이 앞지르면 → 숨기지 않고 싣고 프레이밍(입력 길이 = 정확도↔SRAM 교환) 사용자와 결정.
- **이후**: 3단계 = L × 모델 표 (T6 와 같은 7모델) · 4단계 칩 비용 = 일단 계산 추정(채널당 입력 버퍼 L×4 B + L96 실측),
  C 포트는 96점 FFT 고정이라 L≠96 실측은 필요할 때만.

### ▶ Ablation 마무리 계획 (10-01, 사용자 합의 · 실험은 서버에서, 아직 실행 안 함)
- **지금 T7(부품 한 표) + T8(섞기 a/b/c) → 부품별 별도 번호 표 5개**로 교체. 열은 단순하게(파라미터 · MSE · 변화%[데이터별
  변화의 평균, + = 나빠짐] · 필요 시 칩당 채널/flash). H96 · mfg-6 · 3시드 기준.
  - **A 잔차 MLP·채널 섞기**: 지금(R4) / R2 / R8 / R16(−0.4%, 31채널) / 섞기 끔(= FEATHer-CI, +4.8%, 48채널) /
    MLP 없음(+3.8%). 섞기는 MLP 안에서 일어남(`nl_down` 이 채널 축 D→R) — CI 도 1→R→1 MLP 있음. CI 의 R 변화는 안 함(표 단순화).
  - **B 예측 헤드**: SPK vs 일반 선형(`headlin`), H 4개 + H별 파라미터·flash. ⚠ 8월 옛 모델에선 선형 헤드가 3.6% 더 정확 →
    그러면 "정확도 조금 ↔ flash 증가 막음" 교환으로 보고. SPK 는 SparseTSF 귀속 명시.
  - **C 주파수 게이트**: FFT / 없음 +0.8 / 균등 0.0 / 고정 학습(+0.1) — 있음(H96). BlastFurnace 만 2~3%.
  - **D 밴드 분해**: 3 / 2 / 1개(−0.1%, 250p). 효과 없음 → 리뷰어 위험 + 사용자 규칙(동률이면 뺀다)과 충돌.
  - **E 정규화·손실**: RevIN 끔(`norevin`), 스펙트럼 손실 λ=0(`spec0`).
  - 표 밖: H별 섞기 효과 → 본문 / LTSF 섞기 효과 → LTSF 절 / SPK 주기 P 제약 → 한계점.
- **서버 실행 순서**(`run_forecast.py --ablation_arm …` 은 arm 이 이미 있음; R 은 `--mlp_rank`): ① 입력 길이(위, 288)
  ② B `headlin` 4H (72) ③ C·D 긴 H: `gatenone`·`band1` H192~720 (108) + `band2` H96 (18) ④ A R2·R8 H96 (36)
  ⑤ E `norevin`·`spec0` H96 (36). 칩당 채널 수(R2·R8·MLP 없음)는 학습 없이 크로스컴파일.
- **사전 판단 기준**: B 선형 헤드가 더 정확 → 교환 관계로 보고 / C·D 긴 H 에서 평균 ≥1% 또는 데이터 2개 이상 ≥2% 나빠짐 →
  "긴 H 에서 효과"로 유지, 아니면 사용자 결정(정직 보고 vs 제거; 9/28 미결 "밴드·게이트 뺄지"와 같은 문제) /
  E λ=0 동률 → 보고할지 뺄지 사용자 결정 / RevIN 은 그대로 보고.
- 실험 후: 원고 빌더(`tables_cf.py`)에 표 A~E 추가, 지금 T7·T8 대체.

### 확정 숫자 (원고에 쓰는 값)
- **mfg (3시드, 6데이터 × 4H)**: FEATHer-mix OVERALL **0.6990** · FEATHer-CI 0.7361 (섞기 끄면 +5.3%).
  옛 R16(452p) 0.6943 → R4로 +0.66% 손해 대신 칩당 31 → **43채널**.
- **LTSF (8데이터)**: FEATHer-CI 0.3753 / mix 0.3806. 평균은 최저지만 **MSE 1등 0칸**, FITS와 p 0.25.
- **T7 ablation (H96, 6데이터)**: 섞기 끄기 +4.8% · 잔차 MLP 끄기 +3.8% · 게이트 제거 +0.8 · 균등 0.0 ·
  고정 학습 비율(FFT 없음) +0.1 · 밴드 1개 −0.1. **값을 하는 부품 = 섞기와 잔차 MLP 뿐.**
- **T8 섞기**: H별 +4.8/+5.6/+6.8/+6.2 (단조 아님 → "긴 H에서 더 크다"), WindSCADA +0.6 → +12.3.
  LTSF는 채널 수로 갈림: ETT −1.7~−4.9, Exchange −10.9 vs Weather +6.7, Electricity +5.3, Traffic +2.6.
- **보드 (`results/board_latency_20260929.csv`, 27칸 PASS)**, 7채널 H96/H720 ms: SparseTSF 47/90 · MixLinear
  87/901 · CycleNet 157/✗ · **FEATHer-mix 161/208** · FEATHer-CI 202/250 · FITS 274/5,896 · DLinear 315/✗ ·
  STSF-MLP 434/1,445 · DiPE 727/12,735. 전류 8모델 29.6~30.0 mA (CycleNet 미측정) → 에너지 = 3.3 V × 30 mA × 지연.
- **칩당 채널**: mix 43 · CI 48 · FITS 47~49 · CycleNet 50. flash 초과: DLinear 146/255/546 KB (H192/336/720),
  CycleNet 135/281 KB (H336/720).
- **H=720 칩 안**: FEATHer-mix 가장 정확(0.808), 더 빠른 건 SparseTSF 뿐(7% 덜 정확).
- **필터 없이(경량 8개, 24칸)**: MSE 순위 DLinear 1.83 > 우리 2.21 · MAE CycleNet 2.04 > 우리 2.96(1등 칸은 우리 9로 최다)
  → 두 지표 모두 2위 안은 우리뿐. MSE 로 우리를 이기는 가장 작은 모델 = DLinear (flash 4.6배 H96 ~ 30배 H720; T5 값 83.5/18.1, 546/18.5 KB).

### 주장 금지 (반복 확인됨)
- "DLinear를 이긴다" — MSE 11–13. DLinear는 모든 H에서 더 정확, H≥192에서 칩에 못 들어갈 뿐.
- "H가 길수록 (DLinear 대비) 격차가 커진다" — 틀림. DLinear 의 MSE 우위가 오히려 커짐(3.4→6.2%); 커지는 건 비용 차이.
- int8/행렬 합치기 실험 다시 제안 금지 (09-29 결정: 베이스라인 재최적화 안 함, 한계점 한 문장).
- "LTSF 1위" → "on par with the best lightweight models".
- "FFT 게이트가 도움이 된다" → "고정 비율과 같거나 1% 미만으로 약간 낫다". 게이트는 정체성 이유로 유지(사용자 결정).
- 모양 지표(CORR 등) 주장 — 사전 규칙으로 기각, MSE/MAE 만.
- 순위를 말할 땐 항상 **어느 풀인지** 명시 (9모델 전체면 DLinear MSE 1등 13 > 우리 10).
- "제조 TSF 벤치마크 최초" — RCIM(arXiv 2411.17499)이 선점. "최초"는 MCU 배포·전 채널 동시·경량 tier 로 좁힐 것.

### 한계점에 넣을 문장
DLinear 행렬 합치기/int8 (베이스라인 재최적화 안 함, 규칙 ①) · CycleNet 위상 인덱스 0 고정 · H 길이 변환은 전
모델 직접식 · CycleNet 전류 미측정 · 재계산 허용(정규화+밴드)은 FEATHer-mix 포트의 예외 · L=96 고정(긴 룩백
네이티브 모델에 불리 — Traffic/Electricity 격차 +56~73%) · Steel 학습 불안정(원인 미규명).

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
| `results/complexity.csv` | 파라미터·연산 수 |
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
# 원고 빌드 (v2 만 덮어씀)
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

# 감사
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py
```
