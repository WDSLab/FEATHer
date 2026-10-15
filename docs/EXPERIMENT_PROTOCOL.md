# 실험 프로토콜 (2026-10-07 확정)

이 문서는 FEATHer 논문의 모든 학습 실험이 **어떤 데이터·길이·설정으로, 어떤 규칙에 따라** 돌았는지를 정한다.
규칙은 결과를 보기 전에 정했고, 결과가 나빠져도 되돌리지 않는다. 코드에서의 위치는 각 절 끝에 적었다.

- 근거 자료: `manuscript/drafts/_records/baseline_official_configs.md` (공식 스크립트를 줄 단위로 읽은 표, 값마다 file:line)
- 모델 구조: `MODEL_FINAL.md` · 결과 파일: `results/README.md` · 돌릴 목록: `manuscript/notes/run_queue_1007.md`

---

## 1. 데이터

| 묶음 | 데이터 | 분할 | 역할 |
|---|---|---|---|
| 산업 (mfg) | Steel · WindSCADA · BlastFurnace · MetroPT3 · N2O · EWELD (6~28채널, 1분~1시간) | 시간순 70/10/20 % | 주 실험 |
| LTSF | ETTh1 · ETTh2 · ETTm1 · ETTm2 · Weather · Exchange · Electricity · Traffic | ETT 12/4/4개월, 나머지 70/10/20 % | 일반화 확인 |

- 전처리: `utils/prep_manufacturing.py` (연속 측정 채널만, 일정 시간 격자, BlastFurnace 는 학습 구간 기준 0.5~99.5 % 자르기).
- 결측: 과거 값으로 채워 입력에 쓰고(맨 앞 구간만 뒤 값), **손실·지표에서는 제외** (`mask.csv`).
- 표준화: 채널별로 학습 구간의 평균·표준편차. 지표는 표준화된 값에서 계산.

## 2. 입력 길이 L · 예측 길이 H

| 실험 | 데이터 | L | H | 시드 |
|---|---|---|---|---|
| 메인 | mfg 6 + LTSF 8 | 96 | 96·192·336·720 | 2025·2026·2027 |
| 입력 길이 | mfg 6 | **48·192·336** | 96·192·336·720 | 3 |
| 설정 고르기 | mfg 6 | 96 | 96·192·336·720 | 2025 |
| 설정 고르기 (공식 설정 없는 LTSF 15칸 중 후보가 2개 이상인 14칸; CycleNet Exchange 는 후보 1개라 바로 사용) | 해당 LTSF 데이터 | 96 | 96·192·336·720 | 2025 |

- L = 96: 칩(20 KB SRAM)에 채널마다 입력 창을 두어야 하는 조건에서 칩당 수십 채널이 가능한 길이. 유일한 의도적 변경.
- **L = 720 은 하지 않는다**: 7채널 입력 창만으로 SRAM 을 넘어 어떤 모델도 칩에 올릴 수 없다(범위 밖).
- 입력 길이 실험은 칩에 들어가는 범위에서 "과거를 늘려 주면 따라오나"를 본다. LTSF 에서는 하지 않는다. 대상 = FEATHer-mix·CI +
  칩 안 경쟁자 8개 + DLinear · CycleNet (옛 표 11 과 같은 풀; SAMformer 는 H96 만 칩에 들어가 제외). 모든 L 에 L = 96 에서 고른 설정 하나.
- **L = 48 판정 (결과 보기 전 10-07 고정)**: 섞기 이득(FEATHer-mix vs CI, 24칸 % 변화 평균)이 L = 96 의 +5.9 % 보다 크고
  단측 Wilcoxon p < 0.05 이면 "과거가 짧을수록 옆 보기 이득이 커진다"고 쓰고, 아니면 추세 주장은 하지 않는다.

## 3. 공통 학습 절차 (모든 모델 동일)

batch 32 · 최대 50 epoch · patience 10 · Adam + CosineAnnealing(T_max 50) · gradient clip 1.0 · AMP 끔 ·
검증 손실 최저 epoch 에서 테스트 1회 · 손실 = 관측값 MSE.

- **방법 고유의 학습 항은 유지**: FEATHer 스펙트럼 항(λ 0.01) · TimeBase 직교 항 · DiPE 시간/주파수 손실 비율과
  전문가 라우터 온도 일정 · SAMformer 의 SAM 최적화.
- **학습 방법 선택은 통일(공식과 달라도)**: Huber 손실(PhaseFormer), 에폭 수·lradj·batch, FITS train_mode 2 등.
- 코드: `scripts/train.py` (모델 연결 지점 `_epoch_start` / `_train_loss` / `_test_start`).

## 4. 모델과 역할

| 역할 | 모델 | 설정 고르기 | 칩 비교 |
|---|---|---|---|
| 우리 | FEATHer-mix · FEATHer-CI | 아니오 (동결) | ○ |
| 칩 안 경쟁자 | SparseTSF · SparseTSF-MLP · MixLinear · DiPE · FITS · CMoS · TimeBase · PhaseFormer | 예 | ○ |
| 긴 H 에서 칩 밖 경량 | DLinear(H≥192 flash 초과) · CycleNet(H≥336) · SAMformer(H≥192) | 예 (CycleNet 은 후보 1개) | 표에는 싣고 칩 안 순위에서는 제외 |
| 참고 (정확도 상한) | PatchTST · TimeMixer · TQNet · LMS-AutoTSF · TimesNet · iTransformer · MDMLP-EIA | 아니오 (mfg 기본 설정, LTSF 공식) | ✗ |
| 대조군 | SparseTSF(+MLP) + 섞기 통로, FEATHer 밴드·게이트 제거 | 아니오 — **바탕 모델이 고른 설정**(SparseTSF / SparseTSF-MLP / FEATHer-mix), config_id `ctrl:<arm>@<바탕 설정>` | — |

참고 모델은 L = 96 으로 학습한다 (7개 중 6개는 원 논문 기본 입력 길이가 96, PatchTST 만 336).

## 5. 설정 규칙

1. **출처**: 베이스라인 설정은 저자가 공개한 값만 쓴다 — 데이터별 **최종** 스크립트(`*_best.sh` 등; 탐색용 스크립트의
   값은 쓰지 않음) + 저자가 기본 설정으로 내놓은 값(설정 파일, PyTorch 클래스 기본값·예제). 모든 스크립트가 덮어쓰는
   argparse 빈칸용 기본값은 쓰지 않는다. 값을 조합하거나 새로 만들지 않는다.
2. **L = 96 용 공식 설정이 있으면 그것** (CycleNet Linear-Input-96: lr 0.01).
3. **LTSF**: 데이터(·H)별 공식 설정. **공식 최종 설정이 없는 칸**(스크립트가 없거나 탐색 스크립트만 있음 — Exchange 9개 모델,
   MixLinear ETTm1·ETTm2·Weather, FITS Electricity·Traffic, DiPE Traffic)은 mfg 와 같은 방식으로 고른다: 그 모델의 공개 설정
   (4번의 후보 목록)을 **그 데이터의 검증 손실**(H 별 순위의 평균, 시드 2025)로 비교해 하나를 쓴다. 후보가 하나뿐이면 그것
   (CycleNet). 주기류 값은 그 데이터의 값. (10-07 변경: 처음에는 "ETTh1 설정"이었으나 근거가 약해 바꿈.)
4. **mfg**: 후보 = 저자의 데이터별 설정 묶음(H별 설정이면 H별 묶음 통째로) + 저자 기본 설정. 같은 설정은 하나로 합친다.
   후보마다 6데이터 × 4H × 시드 2025 로 학습하고, **데이터별 검증 손실(4H 평균) 순위의 6데이터 평균이 1등인 후보 하나**를
   6데이터·모든 L 에 쓴다. 테스트 값은 보지 않는다. 동점은 파라미터가 적은 쪽, 다음은 ETTh1 후보.
5. **칩 판정은 칸 단위** (10-07 Fable 검토 후 변경): 후보를 칩 크기로 미리 거르지 않는다. 모든 모델이 정확도(검증 손실)만으로
   설정을 고르고, 고른 설정이 어떤 칸(데이터 × H)에서 칩(128 KB flash / 20 KB SRAM)을 넘으면 그 칸만 "칩 밖"으로 표시한다
   — DLinear · CycleNet · SAMformer 와 같은 취급. 칩 안 순위는 지금처럼 24칸 모두 칩에 들어가는 모델로 매긴다.
   **수정 (2026-10-12, main 결과를 본 뒤 — 사후 변경, 논문에 밝힘)**: mfg 설정은 **칩에 들어가는 칸이 가장 많은 공개 설정 중**
   4번 규칙(검증 순위)으로 고른다. 칩 판정 = `deployment/board/chip_verdict.py --candidates`(모든 후보 × 24칸 펌웨어 빌드, 실제 채널
   수). 이유: 정확도 우선 선택이 칩 안 비교에서 일부 모델을 통째로 빼서(PhaseFormer Traffic 0/24 칸, 공개 ETTh2 설정이면 21/24)
   "칩 안에서 가장 정확" 비교를 우리에게 유리하게 기울였다(Fable 검토 권고). 이 변경은 우리 순위를 낮춘다(CMoS 가 H720 에도 들어옴).
   바뀐 모델 = CMoS Traffic → Weather (24/24) · FITS Weather → ETTm1 (23/24) · PhaseFormer Traffic → ETTh2 (21/24); 나머지는
   이미 24/24 이거나(SMLP·DiPE·MixLinear·TimeBase, 그대로 1등) 모든 공개 설정의 구조가 같다(DLinear·CycleNet·SAMformer).
   정확도 우선 설정의 결과(옛 main 행, config_id 그대로 raw 에 남음)는 표 주석 한 줄로 밝힌다. LTSF 는 공식 설정 그대로.
6. **건강 검사**: 고르기에서 best epoch ≤ 2 인 칸이 절반을 넘거나 값이 없는/발산한 후보는 제외. 정식 3시드에서 시드 2개
   이상이 발산하거나 best epoch ≤ 2 면 다음 후보. "성능이 나쁘다"는 바꿀 이유가 아니다.
   **LTSF 는 후보가 하나라** 실패한 (모델, 데이터)에 한해 lr 만 공식값의 0.1배 · 0.01배 중 **검증 손실**로 다시 고르고
   논문에 예외로 밝힌다 (선례: DLinear ETTh2 공식 lr 0.05 가 우리 스케줄러에서 발산 → 검증으로 5e-3 재선택).
   best epoch 는 0부터 센다 (≤ 2 = 처음 세 epoch 안).
   **수정 (2026-10-11, 결과를 본 뒤 — 논문에 밝힘)**: 정식 3시드 단계의 "best epoch ≤ 2 → 다음 후보 / lr 재선택"은 쓰지 않고,
   **발산(값 없음·무한대)만 실패**로 본다. 근거(main 결과, `P.main_rows`): best epoch ≤ 2 가 시드 2개 이상인 칸이 mfg 60/480 ·
   LTSF 91/584 로 FEATHer·서버급 포함 거의 모든 모델에 있고, 모델이 아니라 데이터별로 몰린다(BlastFurnace 27/60, ETTm2·ETTh*);
   같은 칸에서 일찍 멈춘 시드가 끝까지 학습한 시드보다 나쁘지 않다(시험 MSE mfg +0.09 %, LTSF −2.33 %); 걸린 칸이 풀 안에서
   하위에 몰리지 않는다. ⇒ 학습 실패가 아니라 검증 손실이 초반에 최저인 데이터 특성(조기 종료가 제 역할을 한 것). 모든 모델에
   같은 규칙이 적용되므로 설정 변경·재실험 없음(LTSF FITS ETTm2 · DiPE ETTm1 · MixLinear ETTh2 lr 재선택도 하지 않음).
   고르기 단계의 후보 제외(셀 절반 초과)는 실행 전에 정한 대로 이미 적용했고 그대로 둔다(DiPE mfg ETTh1, TimeBase LTSF-Exchange Weather).
7. **입력 길이가 정의를 넘는 값**은 그 입력 길이가 허용하는 최대값: MixLinear lpf ≤ ⌈L/주기⌉, FITS 자르는 주파수 ≤ L/2
   (공식 계산식 `(L // base_T + 1) × H_order + 10`).
8. **주기류 값은 규칙으로 계산** (고르는 대상이 아님):
   - 주기가 L·H 를 나눠야 하는 모델(FEATHer · SparseTSF · SparseTSF-MLP · MixLinear · TimeBase · PhaseFormer, CMoS 주기 힌트):
     학습 구간 FFT 의 2~48 스텝 최대 피크 → 허용값 {2,3,4,6,8,12,16,24,48} 중 가장 가까운 값.
   - 주기 성분을 빼는 모델(CycleNet · TQNet)과 FITS 기준 주기: 하루 샘플 수(같은 규칙의 값).
9. **FEATHer**: 동결 설정 하나. 데이터별로 다른 값은 주기뿐(mfg = 8번 규칙, LTSF = 아래 6절).

| mfg 데이터 | FFT 주기 (≤48) | 하루 주기 (CycleNet·FITS) |
|---|---|---|
| Steel | 24 | 96 |
| WindSCADA | 48 | 144 |
| BlastFurnace | 48 | 24 |
| MetroPT3 | 24 | 298 |
| N2O | 24 | 288 |
| EWELD | 24 | 96 |

코드: LTSF 설정 = `configs/<모델>.yaml` (10-14 부터; 이전 `baselines/__init__.py`) · mfg 후보·해석 = `tools/results/protocol.py`
(`CANDIDATES`, `resolve()`) · 판정 = `tools/results/select_verdict.py`.

## 6. FEATHer 설정과 개발 이력 (공개할 사실)

| 설정 | 값 | 정한 방법 |
|---|---|---|
| 밴드 구성 | 3개 {k3, k5, 풀링} | mfg **검증** 손실로 6개 후보 비교 — 0.28 % 안에 몰려 실패율·파라미터로 동점 처리 |
| 섞기 폭 R | 4 | 칩당 43채널을 위해. 미리 정한 허용(평균 +2 %, 데이터별 +5 %) 안에서 mfg 결과로 확인(+0.66 %, 정확도를 잃는 방향) |
| 스펙트럼 손실 λ | 0.01 | 개발 중 결과 (올리면 단조 악화) |
| lr | 1e-3 | 탐색하지 않음 |
| 채널 방식 | mfg = mix, LTSF = CI | 결과를 보고 정함 → **두 변종 모두 보고** |
| LTSF 주기·예외 | 데이터별 (ETTh1 6 …), Electricity·Traffic lr 0.005 (Electricity λ 0) | 7월 LTSF **검증** 탐색 값. LTSF 는 모든 모델이 데이터별 설정을 쓰므로 그대로 둔다 |

설정을 바꿔 얻는 정확도 차이: FEATHer 구조 격자 0.28 %, 데이터별 선택 0.2504 vs 0.2501, 베이스라인 lr 5개 탐색 중앙값 0.1 %.

## 7. 모델별 설정 요약

| 모델 | LTSF (공식) | mfg 후보 |
|---|---|---|
| SparseTSF | lr 0.02/0.03, 주기 | ETTh1(0.02) · ETTh2(0.03) |
| SparseTSF-MLP | 자기 lr (0.001~0.05), Traffic hidden 512 | ETTh1 · ETTh2 · Weather · Electricity · Traffic |
| DLinear | lr (ETTm2 H 별, Weather = 기본 1e-4); ETTh2 = 6번 예외 | ETTh1 · ETTm1 · ETTm2 · Exchange · Electricity · Traffic |
| CycleNet | lr 0.01 (L96 스크립트), 주기 | 1개 (L96 설정 = `cand:ETTh1`) |
| DiPE | 전문가 수·손실 비율·RevIN (데이터별) | ETTh1 · ETTh2 · ETTm2 · Weather · Electricity · 기본 |
| MixLinear | alpha·lpf (H 별), lr | ETTh1 · ETTh2 · Electricity · Traffic |
| FITS | H_order (H 별)·기준 주기·채널별 | ETTh1 · ETTm1 · Weather |
| CMoS | 데이터 × H 별 전부 | ETTh1 · ETTh2 · ETTm1 · ETTm2 · Weather · Electricity · Traffic · 기본 |
| TimeBase | 데이터 × H 별 | ETTh1 · ETTh2 · ETTm1 · ETTm2 · Weather · Electricity · Traffic |
| PhaseFormer | 데이터 × H 별 (hidden 폭 포함) | ETTh1 · ETTh2 · ETTm1 · ETTm2 · Weather · Electricity · Traffic |
| SAMformer | lr, rho 0.7 | ETTh1 · Weather · 기본(rho 0.5) |

## 8. 결과 관리

- `results/raw_mfg.csv` · `raw_ltsf.csv` = 정식 결과, `select_{mfg,ltsf}.csv` = 고르기 결과, 고른 설정 = `docs/chosen_{mfg,ltsf}.json`
  (git 에 올림 — 공개 저장소로 재현 가능).
- 모든 행에 `role`(main / reference / lookback / ablation / control / select)과 `config_id`
  (frozen · official · official-exc · ref-default · cand:<DS> · cand:default · ctrl:<arm>).
- `run.py --stage {ltsf, ltsf_select, select, main, lookback, ablation, control}` 가 계획과 결과를 비교해 **빠진 칸만** 돌린다
  (결과 조각 → `results/_incoming/`, `tools/results/merge_incoming.py` 로 합침).
- 규칙과 설정이 다른 옛 결과는 `results/_archive/legacy_rows_20261007.csv` (지우지 않음).
- 표에 쓸 행은 `tools/results/protocol.main_rows()` 로 고른다 (모델마다 고른/공식 설정만, 실패한 실행 제외). raw 에는 안 고른
  후보의 행도 남아 있으므로 직접 평균내지 않는다. 실패(발산)한 실행은 MSE 가 비어 있고 best_epoch = −1 인 행으로 남는다.

## 9. 논문에 밝힐 것

- 학습 절차 통일, lr·구조는 저자 값; mfg 는 모든 모델이 **검증 손실로 설정 하나**를 고름(후보 = 저자 공개 설정, 1~8개).
- FEATHer 개발 이력(6절) · LTSF 에서 FEATHer 주기가 검증 선택이라는 것.
- 예외: DLinear ETTh2 lr · 공식 설정 없는 데이터 = ETTh1 설정 · 입력 길이 최대값 처리(MixLinear lpf, FITS) ·
  CycleNet 은 입력 길이별 공식 설정이 있으나 하나로 통일 · Huber/에폭/FITS train_mode 통일 · 참고 모델 기본 설정.
- 공식 저장소 특이점: SparseTSF linear 스크립트가 모델 종류를 넘기지 않음(논문대로 선형 사용) · DiPE.sh 의 데이터 경로
  복사 실수(주석의 의도대로 설정 사용) · CMoS 주기 힌트는 L 보다 크면 효과 없음.

## 10. 변경 이력

- 2026-10-14 (버그 수정, 사용자 결정): 분 단위 데이터(ETTm1·ETTm2·Weather·Steel·WindSCADA·MetroPT3·N2O·EWELD)의 시간 특징
  주기가 `"m"` 이었는데 pandas 는 이를 "월"로 읽어 시간 특징이 '몇 월'뿐이었다 → `"min"`(분)으로 수정. 시간 특징을 쓰는 모델은
  iTransformer(참고 모델) 하나뿐이라 그 96칸(제조 5 × LTSF 3 데이터)만 다시 돌린다; 옛 줄은
  `results/_archive/itransformer_month_marks_20261014.csv` 로 옮김. FEATHer·경량 베이스라인은 시간 특징을 쓰지 않아 영향 없음.

- 2026-10-14 (코드 정리, 값 변화 없음): 모델 하이퍼파라미터를 `baselines/__init__.py` 에서 `configs/<모델>.yaml` 로 이전
  (덮어쓰기 없이 실제로 쓰이던 최종값만; 전후 설정·모델 가중치 전부 동일 확인). 결과 줄마다 `hparams` 열(그 칸이 쓴 하이퍼
  파라미터 전체 JSON)을 기록하고, `run.py` 는 기록된 값이 지금 YAML 값과 다른 칸을 다시 돌린다(10-14 이전 줄은 기록 없음 = 완료로 봄).
  실험 정의(ablation · 대조 · 재설계 · 크기 비교 후보)도 `configs/experiments/*.yaml` 로 이전(읽힌 정의 동일 확인). 한 번도 돌리지 않은
  ablation 6개(readd_dtk · readd_inproj · headmlp · headconv · gatesm · specHi)와 그것만 쓰던 FEATHer 코드는 삭제(원본 `_archive/FEATHer_full_20261014/`).
  10-12 재설계는 종료(채택 없음, 2단계 중단): `--stage redesign`·revin_mode(mean/learn_level)·mix_stats 삭제, 정의·판정 도구는 `_archive/redesign_1012/`.

- 2026-10-14 (사용자 결정, 결과 보기 전): 크기 맞춘 비교의 **후보를 여러 개로**. 크기 하이퍼파라미터는 폭 · 깊이 · 구간 종류만
  (`protocol.SIZE_KNOBS`), (가) 하나로 예산을 채움 / (나) 하나를 최소로 두고 다른 하나로 채움, 나머지는 저자 값, 폭 1 · 저자와 같음 ·
  6데이터 중 하나라도 안 되는 후보는 제외 (`tools/results/size_candidates.py` → `docs/size_candidates.json`). 모든 모델 시도 수
  동일(후보 수 × lr 점: 3×3, 2×5, 1×9), 고르기 = 시드 2025 · **H96 + H720** 검증 손실의 데이터별 순위 평균
  (`size_verdict.py`, 모든 시도 기록 `results/analysis/size_selection_audit.csv`). SparseTSF · MixLinear 도 주기로 1K 에 맞춤(10-14 저녁 취소, 아래).
  DiPE 는 1K 안에 들어가는 후보가 없어(최소 1,809) 1K판 참고 줄(순위 제외), 최대판은 정식. 설정 하나를 6데이터에 공통으로 쓰는
  것은 저자 논문(데이터별 설정)보다 엄격한 조건. 칩 최대판 후보는 실제 빌드로 따로 만듦. 10-13 단일 다이얼 결과는 후보 ① 로 재사용.

- 2026-10-14 저녁 (사용자 결정, 결과 보기 전): **주기는 크기 하이퍼파라미터가 아니다.** 주기는 8번 규칙으로 정한 데이터 값이고,
  다른 값(16 · 12 · 4)이나 입력 길이와 같은 값(96 = 구간 하나)은 모델이 보는 방식 자체를 바꾼다(MixLinear 는 lpf 잘림도 바뀜).
  → 주기를 바꾸는 후보 삭제: SparseTSF-MLP·TimeBase "주기 96 + 채움", PhaseFormer "주기 16 + 채움". SparseTSF · MixLinear 는
  크기를 바꿀 곳이 주기뿐이라 **저자 크기 그대로**(1K 예산 미만: 저장 숫자 79~145 · 133~185, 결과와 함께 표기; 10-14 의
  "주기로 1K 에 맞춤" 취소). 후보 수 = CMoS 3 · PhaseFormer 2 · 나머지 1 → 시도 9~10번 그대로(1K 계획 876칸).
  **(같은 날 밤 아래 결정으로 대체됨.)**

- 2026-10-14 밤 (사용자 결정, 결과 보기 전) — **크기 맞춘 비교 = 모든 모델을 제조 데이터에서 같은 방식으로 다시 튜닝.**
  이유: 저자 설정은 저자 데이터(LTSF)에 맞춘 값이라, 제조 데이터에서는 어느 모델에도 맞춘 값이 아니다. 선례 = Brigato et al.
  TMLR 2026(시계열, 모든 모델 · 모든 데이터 재튜닝, 공통 탐색 공간, 40회, 검증), ProbTS(공통 lr 격자), Melis ICLR'18(파라미터
  예산 + 같은 탐색), Lucic NeurIPS'18(모든 모델 같은 lr 범위); 반대 선례(기본값만) = RCIM 2411.17499 · Monash.
  규칙 (`configs/experiments/size.yaml`, 조합 = `docs/size_trials.json`, 실행 전 커밋):
  ① 무작위 탐색 40회/모델(Optuna 가 아닌 무작위 = 조합을 실행 전에 확정·공개 가능), t01 = 앞서 고른 저자 설정(+ 데이터 FFT 주기,
  예산 초과면 첫 크기 키로 채움) → 튜닝 결과가 검증에서 저자 설정보다 나쁠 수 없음. ② 공통 범위: lr 1e-5~1e-1 로그(Adam 기본
  1e-3 의 ×/100, 모든 모델 같음 — 저자 lr 기준 배수 방식 폐기), dropout {0, .05, .1, .2}, 스위치 켬/끔, 0~1 비율 {.1 … .9},
  **주기 {2,3,4,6,8,12,16,24,48} 도 탐색**(앞의 "주기는 크기 아님" 결정을 뒤집음: 이제 검증이 고르므로 망가진 값은 탈락하고,
  FFT 규칙에 대한 질문이 사라지며, SparseTSF·MixLinear 도 예산을 쓸 수 있다). 단위가 그 모델에만 있는 값(TimeBase 직교 손실
  가중치)만 모델별 목록. ③ 크기: 시도마다 크기 키 하나를 무작위로 채움 키로, 나머지 크기 키는 무작위(개수 = 1~들어갈 수 있는
  최대값 로그 균등), 채움 값은 데이터별로 예산 이하 최대 모델(Melis 의 고정 예산과 같은 방식); 6데이터에 안 들어가거나 중복이면
  다시 뽑음(횟수 기록). ④ 같은 40개를 6데이터 × H96·H720 × 시드 2025 로 돌리고 **데이터마다** 두 H 의 검증 순위 평균으로 하나
  (`size_verdict.py`; "6데이터 공통 설정 하나" 폐기), 발산만 실패, 테스트는 고른 뒤 1회. ⑤ 고른 lr 이 범위 끝 10배 구간에 데이터
  절반 이상이면 그 모델만 그쪽으로 10배 넓혀 같은 횟수 추가. ⑥ 학습 절차(L96 · batch 32 · 50 epoch · patience 10 · Adam+cosine ·
  clip 1.0)는 고정. ⑦ 스위치 분류·안 쓰는 키(이유)·모든 시도 기록은 부록(`size_hparams.csv`, `size_selection_audit.csv`).
  FEATHer 는 베이스라인을 다 정한 뒤 같은 규칙·같은 횟수로(마지막).
  **같은 날 밤 Fable 독립 검토 후 수정 (사용자 결정, 결과 보기 전)**: (a) 예산은 **상한**이지 목표가 아니다 — ③의 "예산 이하
  최대로 채움" 대신 시도마다 **목표 크기를 (0, 상한] 에서 균등하게** 뽑고 그 이하 최대로 채운다(꽉 채우기는 손잡이가 주기
  하나뿐인 SparseTSF 를 39개 모두 주기 12 = lr 만 다른 시도로 만들고, 작은 모델이 나을 기회를 없앴다). 크기도 검증이 고르고,
  실제 크기는 표에 적는다. "근처(±%)" 허용은 쓰지 않음(특정 모델 숫자를 보고 폭을 정하게 됨). (b) lr 상한 0.1 → **1** (→ (o)에서 다시 0.1)
  (TimeBase 저자 lr 0.05~0.4 를 처음부터 포함; ⑤의 넓히기 규칙은 안전장치로 유지). (c) FITS 자르는 주파수 ≤ L/2 = 48,
  MixLinear lpf ≤ ⌈96/주기⌉(모델이 잘라 버리는 값은 뽑지 않음). (d) 최대판 → **칩판**: 상한 = 실제 펌웨어가 칩에 들어감(H720 ·
  데이터별 채널 수로 빌드), 목표 크기 (0, 칩 한계] — 작은 모델도 포함("이 칩이 담을 수 있는 최선"), 나머지 절차는 1K 와 같음.
  (e) **FEATHer 탐색 공간도 베이스라인 조합표와 같은 커밋에 미리 정한다**(실행만 마지막); 이번 비교를 위해 추가한 FEATHer
  손잡이(head_seg · agg_shared)는 논문에 밝힌다. (f) FEATHer 개발 이력(제조 데이터로 시험한 설정 수 · 각 결정의 근거, 일부는
  테스트를 본 결정)을 부록에 있는 그대로 공개한다. (g) 보고: 다시 뽑은 횟수, 고른 시도와 2·3등의 검증 차이, 시도 수별 최고 검증
  곡선(Dodge 2019; 필수), 1K = 학습되는 숫자만 셈(칩 비용은 칩판), 저자 설정이 예산 초과면 yaml 첫 크기 키로 채움, lr 2자리
  반올림, DiPE 는 저자 설정도 예산 초과라 "튜닝 ≥ 저자 설정" 보장 없음.
  (h) **칩 확인 방식**: 1K판 = 40개 후보는 숫자 세기만(학습 숫자 ≤ 1,000 은 약 4 KB 라 flash 는 문제없음), **고른 설정
  48개(8모델 × 6데이터)만 WSL 빌드**로 flash · SRAM 확인(SRAM 은 숫자 개수와 따로라서) — 안 들어가면 그 칸을 "칩 밖"으로 표시
  (§5.5 칸 단위 판정과 같음). 칩판 = 40개 후보 **전부 WSL 빌드**(칩에 들어감이 상한). 실제 **보드 실측(지연 · 에너지)은 고른
  설정만**. 빌드는 한 번에 하나(golden/ 공유).
  (i) **판의 상한을 못 지키는 모델은 그 판에서 제외**(돌리지 않음): DiPE 는 가장 작게 해도 1,809개라 1K판 제외(참고 줄도
  없음, 논문에 한 줄), 칩판에는 40개 시도로 정식 포함.
  (l) **①의 t01(저자 설정) 폐지 — 40개 전부 무작위, 참고 실험도 없음.** 같은 규칙이면 FEATHer 의 t01 = 동결 설정인데, 그건 이
  제조 데이터로 개발한 설정이라 우리만 좋은 출발점을 받는다(베이스라인 t01 은 LTSF 에 맞춘 설정). 선행(Grinsztajn · McElfresh)이
  기본 설정을 첫 시도로 넣은 건 제안 모델이 없는 벤치마크 논문이라 이 비대칭이 없었다. "베이스라인을 약하게 만들지 않았나"는
  시도 수별 최고 검증 곡선으로 답한다. 그래서 "튜닝 ≥ 저자 설정" 보장 문구는 쓰지 않는다.
  (m) **FEATHer 탐색 공간(베이스라인 결과 전에 확정, 같은 커밋)**: 두 트랙 FEATHer_mix · FEATHer_CI 를 별개 모델로(SparseTSF /
  SparseTSF_MLP 처럼), 채널 모드는 트랙이 고정, 두 줄을 항상 다 싣는다(칸마다 좋은 쪽을 고르지 않음). 분류는 베이스라인에서 같은
  종류를 다룬 방식 그대로: 주기 = 공통 주기, head_seg = 공통 구간, mlp_rank · point_kernel = 크기(짝수 커널은 돌지 않아 탈락),
  num_bands = 목록 {1,2,3}, use_norm · agg_shared = 스위치, use_residual_mlp = **켬 고정**(mix 트랙의 유일한 채널
  섞기 = 트랙의 정체성; 탐색하자 크기 때문에 mix 40개 중 28개가 섞기 없이 뽑혀 두 트랙 모두 켬 고정, 두 트랙 차이 = 섞기 하나), 게이트 = 스위치 {banded, none}(DiPE 가중치
  스위치와 같은 종류), 선형 헤드 = 스위치 {spk, linear}(어느 판에도 안 들어가 자동 탈락), lambda_spec = 모델별 목록
  {0, .001, .01, .1}(0 = 끔), lr = 공통. front · mix_sep = 대조 실험용 다른 모델이라 제외. FEATHer 는 run.py 에서 이름을 줘야만
  실행(베이스라인 뒤). 동결 설정(mix 1K 기준 Steel 400 · BlastFurnace 1,368)은 출발점으로 넣지 않는다.
  (n) **실험 축소 (사용자)**: 부록용 추가 실험은 하지 않는다 — PhaseFormer cand:Weather main 72칸 · iTransformer 재실행 96칸
  (시간 특징 버그) · lookback(입력 길이) 전부 실행하지 않음. 저자 설정 결과는 이미 있는 것만(DLinear 등 제외 이유 한 문장).
  **부록 없음**: 공정성 근거는 본문 방법 절 한 문단(같은 범위 · 40개 무작위 · 데이터마다 검증)으로 쓰고, 조합표 · 모든 시도
  결과 · 분류표는 공개 코드 저장소의 파일(docs/size_trials.json, results/analysis/size_*.csv)로만 둔다. (g)의 보고 항목도 같은 방식.
  (j) **두 판 모두 "4개 H 모두" 상한 안이어야 한다**(데이터마다 설정 하나로 4개 H 를 돌리므로; 논문 주제 = 장기 예측).
  → DLinear(최소 약 14만 개, flash H192 부터 초과) · CycleNet(H336 부터 초과) · SAMformer(H96 만) 는 크기 하이퍼파라미터로
  줄일 수 없어 **두 판 모두 제외**. "짧은 H 에서는 칩에 들어가고 더 정확할 수 있다"는 본문 한 문장 + 부록 저자 설정 결과로 밝힌다.
  (k) 조합 유효성 = 모델이 만들어지고 **가짜 입력으로 실제로 돌아감**(출력 모양 · 유한값); 첫 추첨에서 CMoS 커널 103(> 입력 96)이
  만들어지기만 하고 돌지 못하는 조합으로 뽑혀 추가. 값이 결과를 바꾸지 않는 범위는 자름: CMoS 커널 ≤ 96, 보폭 ≤ 96 − 커널 + 1
  (그 이상은 conv 출력 1칸으로 같은 모델). 출력 값의 유한성은 보지 않음(DiPE 는 학습 전 초기 출력이 유한하지 않아, 보면
  크기 하이퍼파라미터가 통째로 탈락함 — 첫 확인 때 발견). 다른 값에 따라 상한이 정해지는 개수(MixLinear lpf ← 주기, CMoS
  보폭 ← 커널)는 그 값이 정해진 뒤 1 ~ 상한에서 로그 균등으로 뽑음(처음엔 가장 긴 주기 기준 상한 2 로 묶여 lpf 가 1~2 만 나왔음).
  개수형의 "들어갈 수 있는 최대값" = 다른 모든 탐색 값(크기 · 스위치 · 목록)을 모델이 가장 작아지는 값에 두고 계산(처음엔
  스위치를 저자 값에 두어 FEATHer-mix 섞기 폭 상한이 4 로 잘못 나왔고, 다음엔 MLP 스위치까지 꺼져 폭이 크기를 안 바꾸는 상태
  에서 65 가 나왔음) — 다른 값은 차례로 "그 개수가 여전히 크기를 바꾸는 값 중 가장 작은 모델"에 둔다 → FEATHer-mix 15 · CI 306.
  (o) **공개 코드 대조 후 (Brigato NoChamps · Gorishniy rtdl-revisiting-models)**: lr 범위 1e-5 ~ **1e-1** 로 되돌림 — 1 이면
  로그 균등이라 시도의 약 40% 가 1e-2 초과(두 코드 모두 탐색하지 않는 구역, Brigato 코드 상한 = 1e-2), 0.1 이면 약 20%. 하한 1e-5 =
  두 코드 · Lucic 와 같고 CMoS 저자 lr 2e-5 를 포함. 같은 점(인용 가능): 모든 모델 같은 lr 범위 · 40회 · 탐색 시드 1 / 정식 3 ·
  학습 절차와 batch 고정 · 기본 설정을 시도에 넣지 않음 · 스위치 켬/끔 탐색 · 크기 맞추려 키를 묶는 규칙. 다른 점: Brigato 는 H 마다
  고름 — 우리는 설정 하나로 4개 H(칩 배포 조건). 탐색 중 크기는 H720 만 재고, 받아들인 시도만 4개 H 전부 확인(속도).
  (p) **칩판 추첨 (2026-10-15)**: 1K 와 같은 규칙, 상한만 데이터별 칩 한계. 한계 = (128 KB − 그 모델의 가중치 외 flash) / 4 를
  저장 숫자로 환산(가중치 외 flash = 저자 설정 H720 실제 빌드 1번 − 4 × 저장 숫자). 목표 = 각 데이터 한계의 같은 비율(U(0,1]).
  개수는 2배씩 키운 뒤 반으로 나눠 찾음(크기가 값과 함께 커지는 개수만, 1K 판은 그대로). 받아들인 시도마다 6데이터 × H720 실제
  펌웨어 빌드(chip_verdict.one_cell: flash ≤ 128 KB · SRAM ≤ 20 KB), 안 들어가면 다시 뽑음(chip_off 로 셈). 4개 H 빌드는 고른
  설정만. 추첨 시드는 1K 와 다르게("|chip"). C 포트는 탐색 공간의 모든 구조 값을 지원(3e05eea).
  (q) **고르기 단계만 짧은 학습 (2026-10-15, 사용자, 1K 고르기 84칸 진행 중 결정)**: 서버 속도 시간당 4~9칸(MetroPT3 = 전체 계산의
  약 41%) → 1K 고르기만 8~9일. 고르기 실행(role sizesel)은 **15 epoch · patience 3**(Brigato 코드의 탐색 예산과 같음), 고른 설정의
  정식 실행(role size, 4 H × 3시드)은 프로토콜 그대로 50 / 10 → 메인 표 숫자의 학습 조건은 바뀌지 않음. 이 때문에 고르기 실행을
  정식 실행의 시드 2025 칸으로 재사용하던 규칙은 폐지(정식은 4 H × 3시드 전부 실행). 50/10 으로 이미 돈 84칸은 hparams 에
  budget 이 없어 다시 돌고, size_verdict 는 현재 예산의 행만 읽음.
  (r) **LTSF (2026-10-15, 사용자)**: 새 실험 없음. 이미 있는 저자 공식 설정 결과(raw_ltsf.csv, 8데이터 × 4 H × 3시드)를 짧게
  보여준다. 모델 = 메인과 같은 경량 8개(SparseTSF · SparseTSF-MLP · MixLinear · DiPE · FITS · CMoS · TimeBase · PhaseFormer) +
  FEATHer_mix · FEATHer_CI. DLinear · CycleNet · SAMformer 와 큰 모델 7개는 넣지 않음(iTransformer 36칸 재실행 불필요).
  크기를 맞추지 않은 저자 설정 비교라는 점을 밝히고, 표현은 "on par with the best lightweight models"까지(LTSF 1위 금지).

- 2026-10-13 (사용자 결정): **메인 비교 = 크기를 맞춘 비교**(1K판 · 칩 최대판). 각 베이스라인은 고른 저자 설정에서 크기
  다이얼 하나만 바꾸고(CMoS 구간 길이 · SparseTSF-MLP 은닉 폭 · TimeBase 기저 수 · PhaseFormer 잠재 폭 · DiPE 전문가 수 ·
  FITS 주파수 수), lr 은 {×0.1, ×1, ×10} 중 검증으로 다시 고른다(FEATHer 도 같은 규칙). 크기 = 칩에 저장하는 숫자(복소수 ×2).
  1K판 = 데이터마다 다이얼 하나, H 4개 모두 ≤ 1,000. 칩 최대판 = 데이터마다 H 4개 모두 칩에 들어가는 가장 큰 값(실제 빌드).
  다이얼이 없는 모델(MixLinear 등)은 원래 크기. 입력 길이 실험도 이 방식으로 다시 설계. 저자 설정 결과는 부록. 설정 표기
  `cand:<src>+<key>=<value>`. 근거: Fable 검토(lr 재선택 · 다이얼 민감도 · 공통 칸 순위 · 결정 기록 공개).
- 2026-10-13: C 포트 출력 방식 통일(베이스라인 8개도 값이 나오는 대로 내보냄, 비트 동일) → PhaseFormer 공개 설정 4개가 24/24 →
  5.5 규칙으로 **PhaseFormer ETTh2 → Weather**(검증 순위 1.00). 결과를 본 뒤의 변경, 경쟁자를 늘리는 방향 → 공개.

- 2026-10-12 (main 결과 후, Fable 검토): mfg 설정 = "칩에 가장 많이 들어가는 공개 설정 중 검증 1등"(5절 5번 수정).
  CMoS · FITS · PhaseFormer 설정 변경, 시드 2025 고르기 행을 main 으로 재사용, 시드 2026 · 2027 와 입력 길이는 새로 실행.

- 2026-10-07 (Fable 검토 후): 공식 설정 없는 LTSF 칸 = 검증 손실로 고르기(`--stage ltsf_select`), LTSF 학습 실패 시 lr 0.1·0.01배
  검증 재선택, best epoch 0부터 셈, 판정 도구는 후보가 다 끝나야 기록하고 고른 설정의 시드 2025 결과를 메인에 재사용.
- 2026-10-07 (Fable 검토 후): 칩 필터를 후보 단위 → 칸 단위로 변경 (CMoS·PhaseFormer 후보 복원, FITS 는 원래 미적용).
- 2026-10-07: 이 문서 작성. 공식 스크립트 재대조로 7개 모델의 LTSF 설정 수정, mfg 설정 고르기 도입, L = 48 추가,
  결과 파일 재정리. (이전: mfg 는 모델 기본 lr, LTSF 는 일부 모델이 공식과 달랐음.)
