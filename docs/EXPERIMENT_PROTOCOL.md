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

코드: LTSF 설정 = `baselines/__init__.py` (`_DATASET_OVERRIDES`, 10-07 수정 블록) · mfg 후보·해석 = `tools/results/protocol.py`
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
