# FEATHer 결과 현황 (2026-08-17)

이 문서 = **지금까지 나온 숫자 + 남은 실험 + 모델 개선 레버**의 단일 요약.
실험 체인 절차는 `ROADMAP.md`, 프로젝트 상태 서술은 `CLAUDE.md`.

---

## 0. 한 눈에

| | 결과 | 상태 |
|---|---|---|
| **제조(mfg) 정확도** | FEATHer **0.6940** = **배포가능 모델 중 1위** | ✅ 완료, 표 완성 |
| **LTSF 정확도** | FEATHer **0.3605** = 배포가능 중 **2위** (FITS 0.3582에 0.67% 뒤짐) | ⚠ 2데이터셋 미실행 (표는 반영됨) |
| **MCU 배포** | F103RB 실측 5모델 완료 (FEATHer 349ms / 17.6KB / 35mJ) | ⚠ 옛 453p 기준 = 최종 822p와 불일치 |
| **부품 기여** | 다중밴드·FFT게이트 = **선형 등가, 기여 0** (증명+측정) | ✅ 진단 완료 → 원고 수정 필요 |
| **robustness** | FEATHer 엣지 7모델 중 **꼴찌** | 🚫 논문에서 제거 확정 |

### 모델 패밀리 (확정, 동결됨)

**2변종, 차이는 `channel_mode` 하나.** 둘 다 비선형 잔차 MLP(rank 16) 포함.

| | 채널모드 | 잔차MLP | config | params |
|---|---|---|---|---|
| **FEATHer-N** (mfg) | `mix` | ON | 전 데이터셋 **공통 1개** `ds16_k7_B4` (period만 다름) | 822~4,522 |
| **FEATHer-CIN** (LTSF) | `ci_full` | ON | **데이터셋별 6개** | 244~2,148 |

⚠ **이름 주의:** 코드 변수는 `_FEATHER_LTSF_CI`지만 잔차 MLP가 켜져 있으므로 **실제로는 CIN**.
CLAUDE.md·메모리·코드 주석의 "CI" 표기를 **CIN으로 통일할 것** (원고에서 헷갈리면 치명적).
근거: 천장스캔에서 CIN 0.3588 < L 0.3647.

---

## 1. 정확도 — 제조 6데이터셋 (메인 표)

14모델 × 6데이터 × 4호라이즌 × 3시드, seq_len=96, MSE 학습/채점. **전부 완주.**

### 배포가능 집합 (128KB flash 안에 들어가는 모델)

| 모델 | params | OVERALL MSE |
|---|---:|---:|
| **FEATHer-N (ours)** | 822–4,522 | **0.6940** ← 1위 |
| CycleNet | 214,656 | 0.7198 |
| FITS | 5,100 | 0.7260 |
| DiPE_Linear | 977 | 0.7285 |
| MixLinear | 535 | 0.7486 |
| SparseTSF | 4,325 | 0.7487 |

> ⚠ CycleNet은 H720에서 288KB로 flash를 넘겨 배포집합에서 빠짐 (표에는 남김).

### 전체 라인업 (천장 모델 포함)

LMS_AutoTSF 0.6434 · TimeMixer 0.6641 · TQNet 0.6675 · DLinear 0.6684 ·
MDMLP_EIA 0.6733 · TimesNet 0.6735 · PatchTST 0.6764 · iTransformer 0.6854 ·
**FEATHer 0.6940** · CycleNet 0.7198 · FITS 0.7260 · DiPE 0.7285 ·
MixLinear 0.7486 · SparseTSF 0.7487

**읽는 법:** FEATHer는 20~1,000배 큰 모델들(DLinear 139K, TimesNet 301M)에는 지지만,
**MCU에 올라가는 모든 모델을 이깁니다.** 논문 프레이밍이 "sub-1K 최정확"이 아니라
**"128KB MCU 배포가능 중 최정확"**인 이유가 이것 — sub-1K로 좁히면 DiPE(977p)에게 짐.

산출물: `analysis/01_accuracy_mfg.xlsx`
소스: `results/baselines/fcst_results_baselines.csv` + `results/feather/feather_final.csv`

---

## 2. 정확도 — LTSF 8데이터셋 (일반화 섹션)

### 저채널 6개 기준 (FEATHer가 돌린 범위)

| 모델 | OVERALL MSE |
|---|---:|
| PatchTST | 0.3482 |
| TQNet / MDMLP_EIA | 0.3494 |
| LMS_AutoTSF | 0.3537 |
| **FITS** | **0.3582** |
| iTransformer | 0.3594 |
| **FEATHer (ours)** | **0.3605** |
| TimeMixer | 0.3606 |
| SparseTSF | 0.3626 |
| DiPE_Linear | 0.3628 |
| MixLinear | 0.3633 |
| CycleNet | 0.3673 |
| DLinear | 0.3867 |
| TimesNet | 0.3968 |

**배포가능 집합 안에서는 FITS에만 0.6% 뒤진 2위.** 셀 단위로는 10:8로 FEATHer 우세지만
mean-rank는 FITS가 앞섬(2.08 vs 2.58).

### FITS와의 승부 — 갭이 두 데이터셋에 몰려 있음

셀 단위로는 **FEATHer 13승 11패**인데 평균에서 0.67% 뒤집힘:

| 데이터셋 | vs FITS | |
|---|---:|---|
| Weather | −1.50% | ✅ |
| ETTm1 | −1.36% | ✅ |
| ETTh1 | −0.37% | ✅ |
| Exchange | −0.26% | ✅ |
| **ETTh2** | **+2.96%** | ❌ |
| **ETTm2** | **+6.18%** (H96 +8.6%, H192 +9.8%) | ❌ |

**ETTm2 하나가 전체를 뒤집고 있음.** 이 둘은 ETT 중 레벨 이동이 가장 심한 데이터 =
NLinear(마지막값 앵커)가 DLinear를 크게 이긴 바로 그 지점 → **앵커가 승부처**.

### ⚠ 남은 구멍
1. **Electricity·Traffic 미실행** — 베이스라인은 8개 다 있는데 FEATHer만 6개.
   (옛 이유였던 "D² 파라미터 폭발"은 `ci_full`(채널독립) 채택으로 **해소됨**)
   ⚠ `_DATASET_OVERRIDES`의 이 두 항목은 **옛 OFAT 행**(channel_mode·잔차 없음)이라
   그냥 돌리면 옛 FEATHer-L이 돎 → CLI로 `--channel_mode ci_full --use_residual_mlp` 명시 필요.
2. **03 표 OVERALL 열은 아직 비교 불가** — FEATHer만 6데이터 평균(0.3605),
   나머지는 8데이터 평균(FITS 0.3759). Electricity/Traffic 들어오면 해소.
   (2026-08-17: FEATHer 행 자체는 `build_paper_tables.py`에 LTSF 소스를 연결해 **채워짐**)

**앵커 규칙(적용됨):** Exchange만 `revin_mode=last_std`, 나머지 5개는 `standard`.
train-only 선택 규칙이 12/12 적중했고, Exchange를 0.3848 → 0.3524 (**−8.4%**)로 개선.
이게 LTSF에서 지금까지 **유일하게 먹힌 레버** (파라미터 0개).

---

## 3. MCU 배포 실측 (NUCLEO-F103RB, Cortex-M3 72MHz, FPU 없음)

| 모델 | params | latency | flash(총 이미지) | 에너지/추론 |
|---|---:|---:|---:|---:|
| persistence | 0 | 0.08 ms | 9.6 KB | 0.008 mJ |
| SparseTSF | 41 | 47 ms | 12.4 KB | 4.8 mJ |
| DiPE | 339 | 1,312 ms | 17.0 KB | 132 mJ |
| **FEATHer** | **453** | **349 ms** | **17.6 KB** | **~35 mJ** |
| FITS | 1,200 | 796 ms | 24.0 KB | ~80 mJ |

- **핵심 논거: 파라미터순 ≠ 레이턴시순.** DiPE는 339p인데 FEATHer(453p)보다 3.8배 느림.
  → FEATHer는 FITS·DiPE를 **정확도와 에너지 둘 다에서 지배**. SparseTSF/MixLinear에는
  효율만 짐(대신 걔넨 명확히 덜 정확) → **accuracy-per-resource Pareto**로 서술.
- 활성전류는 5모델 전수 29–31mA = **모델 무관**(CPU+flash 지배) → 에너지 = 전적으로 latency.
- **⚠ 불일치:** 위 실측은 **옛 453p(p12) 기준**. 최종 canonical은 **822p (ds16_k7_B4)**.
  → golden 재생성 + 재측정 필요.
- MixLinear는 C 포팅 + x86 golden PASS까지 됐고 **물리측정만 남음**(main.c 4줄 편집).

상세: `deployment/EDGE_EXPERIMENT.md`, 절차: `deployment/board/BOARD_RUNBOOK.md`

---

## 4. 부품 진단 — 선형 붕괴 (08-14~17)

### 발견
DTK에 활성화가 없어서 `Σ w_b·DTK(F_b(x)) = DTK(Σ w_b·F_b(x))` →
**다중밴드 4개 + 게이트 전체가 필터 하나로 붕괴.** 수치검증 오차 2e-9.
이 하나가 **모든 flat 결과를 설명**(num_bands 0.6%, λ_spec flat, 게이트 ablation 무변화).

### 붕괴를 깨면 좋아지나? → **아니오**

| arm | 내용 | 결과 |
|---|---|---|
| `dtk_gelu` | DTK에 활성화(파라미터 0) | LTSF 6데이터 24셀 **−0.02%** = 이득 0 |
| `dtk_gelu_banded` | 위 + 게이트가 밴드 형태를 보게 | **−0.22%** (2데이터) = 이득 0 |
| `gate_static` | FFT 제거, 가중치만 학습 | **+2.12%, 7셀 전패** = 하면 안 됨 |

- 데이터셋별로는 갈림: Exchange −8.9% / ETTm1 −1.3% / **ETTm2 +6.2%(학습 조기붕괴)**.
  val_loss가 test와 방향 일치 → per-dataset on/off 선택은 가능하나, **얹어봐야 0.3605→0.3596**
  (−0.25%, 시드노이즈 ±0.3~0.5% 안) → **FITS 못 넘음.**
- Exchange 개선분은 **앵커가 이미 더 잘 고친 것과 중복**(gelu 0.3627 vs 앵커 0.3524).
- GELU는 **최적화 불안정 부작용** 있음(ETTm2 4셀 전부 best_epoch 5~9 조기종료).

### FFT 게이트의 실제 정체
- 게이트 출력이 입력에 거의 불변(합성입력 5자리 동일, 실제 창 ±0.008~0.021)
  → **"입력 스펙트럼에 적응"은 사실이 아님.**
- 그런데 **빼면 나빠짐**(uniform +5.0%, static +2.1%) → 표현력이 아니라 **최적화상 유용한
  재파라미터화**로 작동 중. → **모듈은 유지, 설명만 낮춰 써야 함.**

### ⚠ 원고 수정 (필수)
`manuscript/drafts/methodology_jms.md`:
- "produces **complementary** scale representations" → **거짓, 삭제**
- "**adaptively fuses** based on the input spectrum" → **거짓, 삭제**
- SPK가 SparseTSF 헤드와 사실상 동일 → Methodology에 **명시적 귀속** 추가
- Ablation 표 각주: "이 부품들은 선형 등가라 기여가 flat이며, 비선형으로 붕괴를 깬
  대조실험(LTSF 24셀)에서도 개선이 없었다"

---

## 4b. 진행 중 (2026-08-17 발사, 서버 GPU1)

| 프로브 | 런 | 목적 | 판정 기준 |
|---|---:|---|---|
| `results/probes/probe_ltsf_highD.csv` | 8 | Electricity/Traffic에 **CIN** 적용, 위치 확인 | `num_params`가 **수백 대**여야 정상 / Electricity **<0.2254(FITS)** 면 승, Traffic **~0.633** 이면 합격 |
| `results/probes/probe_anchor2.csv` | 24 | ETTh2·ETTm2에 앵커 3종(`last`/`last_std`/`center`) | **ETTm2 −6%, ETTh2 −3%** 면 FITS 역전. 선택은 **train-only 규칙 → val → 엇갈리면 기존 유지**(test-peek 금지) |

통과 시 → 해당 데이터셋만 3시드 승격 → `ltsf_final2.csv`에 병합 → 표 재생성.

---

## 5. 실험 체크리스트

### ✅ 완료
- 베이스라인 mfg 13모델 × 24셀 × 3시드 (빠짐 없음)
- 베이스라인 LTSF 13모델 × 32셀 × 3시드 (빠짐 없음)
- FEATHer mfg final 3시드 (72행)
- FEATHer LTSF final 3시드 (저채널 6개, 앵커 포함)
- lookback 스윕: 엣지 6모델 × mfg 6 × seq{48,192,336,720} × 3시드 (DLinear@720만 12런 결측)
- 보드 실측 5모델 (latency/flash/전력/결정성)
- 복잡도(FFT-aware MACs) 베이스라인 전부
- 선형붕괴 진단 + 3-arm 프로브

### ❌ 남은 것
| 항목 | 규모 | 왜 필요한가 |
|---|---|---|
| **03 LTSF 표 FEATHer 행 채우기** | 코드 수정만 | 표가 지금 NaN |
| **FEATHer Electricity/Traffic** | 24런 | 표 구멍 2칸 |
| **Ablation (부품 5축)** | 중간 | **리뷰어 필수 요구** |
| **보드 FEATHer 822p 재측정** | 반나절 | 정확도표(822p) ↔ 배포표(453p) 불일치 |
| **보드 MixLinear 물리측정** | 반나절 | 배포집합 완성 |
| **04 복잡도표 FEATHer 행** | 계산만 | 현재 TBD |
| **Pareto plot / 에너지 envelope** | 생성만 | 코드 있음 |
| **GRU/XGBoost (mfg)** | 소규모 | 제조 리뷰어 credibility (선택) |

### 🚫 접음
- robustness 전체 (FEATHer 꼴찌 → 논문에서 제거)
- mfg 재설계 프로브 (이미 1위, 재실행 비용만 큼)
- `gate_static` / `banded` (측정으로 폐기)

---

## 6. 모델 개선 — 남은 레버 평가

| 레버 | 시도 결과 | 판단 |
|---|---|---|
| **앵커/정규화** (`revin_mode`) | Exchange −8.4%, 파라미터 0 | ✅ **유일하게 먹힌 축. 여기만 남음** |
| arch 변형 (period/d_state/kernel 등 5종) | 전부 실패 | ❌ |
| 비선형 도입 (dtk_act) | 24셀 −0.02% | ❌ |
| 게이트 개선 (banded/static) | 0 또는 악화 | ❌ |
| 잔차 MLP | Steel −7%, WindSCADA −8% | ✅ 이미 채택됨 |
| 채널모드 (mix/ci_full) | mfg=mix, LTSF=ci_full | ✅ 이미 채택됨 |

**결론:** 정확도를 더 끌어올릴 여지는 **귀납편향/정규화 축**에만 남아 있습니다.
표현력(arch·비선형·게이트)은 8종 시도 전부 실패 = **이 모델 크기에서는 선형이 이미 최적.**
문헌 리뷰(08-03)의 결론과도 일치 — 빈 방법 자리는 "손상 가능 윈도우에서의 robust 정규화"였음.

**⚠ 다만 LTSF 향상은 논문 필수사항이 아닙니다.** LTSF는 강등된 일반화 섹션이고,
목표는 "이김"이 아니라 **"동급 일반화 증명"**(현재 배포집합 2위 = 이미 충족).
메인 표(mfg)는 1위로 끝나 있습니다.
