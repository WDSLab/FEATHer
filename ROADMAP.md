# FEATHer 실험 로드맵 (JMS) — 2026-07-02 확정

> 서버의 두 실험(FEATHer OFAT 512런, LTSF 베이스라인 1,760런)이 끝난 뒤의
> 진행 순서. 세부 배경은 `CLAUDE.md` 상태 블록(2026-07-02/02b) 참고.
> 각 단계 사이의 사람 루프: **summary 가져오기 → 붙여넣기 → 커밋 → 서버 pull**.

## 현재 상태 보드 (2026-07-06 기준)

### A. 서버 실험 (자동 진행 — 기다리면 됨)

| # | 실험 | 단계 | 규모 | 상태 |
|---|---|---|---|---|
| 1 | FEATHer 콤보 검증 (`run_hp_search.py --validate --ngpu 2`) | ① | 32 | ✅ 완료 → summary 받음, 8행 붙여넣기 완료 (2026-07-06) |
| 2 | LTSF 베이스라인 잔여 (`run_forecast.py --group ltsf ... --ngpu 2`) | 일반화표 | 1,760 중 잔여 | 🔄 진행 중 |
| 3 | 제조 lr 서치 (`run_lr_search.py --ngpu 2`) | ③ | 1,440 | 🔄 진행 중 (summary 대기) |
| 4 | FEATHer LTSF 본실험 | ② | 160 | ▶ 실행 가능 (오버라이드 들어감; pull 후 발사) |
| 5 | 제조 본 스윕 (`--save_model`) | ⑤ | 720+540 | ⏳ #3 → 붙여넣기 후 |
| 6 | Robustness (추론) | ⑥ | 스왑 후 확정 | ⏳ #5 + SML 결정 |
| 7 | Ablation (30변형 × mfg) | ⑦ | ⑤때 확정 | ⏳ 체인 맨뒤 |

페이스 확인 (플래그 필수):
```bash
python run_hp_search.py --validate --check
python run_forecast.py --check --group ltsf --exp_tag main
python run_lr_search.py --check
```

### B. 결과 도착 시 (사람 루프는 딱 2번)

| 순서 | 트리거 | 사용자 | 그다음 (Claude/서버) |
|---|---|---|---|
| 1 | #1 완료 | ~~`run_hp_search.py --summary` 출력 전달~~ | ✅ LTSF 8행 붙여넣기+커밋 완료 (2026-07-06) → pull → #4 |
| 2 | #3 완료 | `run_lr_search.py --summary` 출력 전달 | lr 72칸 붙여넣기+커밋 → pull → #5 |
| 3 | #5 완료 | `check_progress.py --exp_tag main` 확인 | #6 + #7 발사 |
| 4 | 전부 완료 | `results/*.csv` 로컬 복사 | 표 생성 체인 (C) |

### C. 표 생성 체인 (전부 로컬, Claude가 함)

| 순서 | 커맨드 | 산출물 | 준비 상태 |
|---|---|---|---|
| 1 | `main_table.py --metric MSE/MAE --format latex` | 메인+PdM표, 퍼시스턴스 열 자동 | ✅ 통합·테스트 완료 (07-03) |
| 2 | `wilcoxon.py --reference FEATHer` | 유의성 | 기존 |
| 3 | `robust_summary.py` | 노이즈 히트맵+표 | 기존 |
| 4 | `ablation_table.py` | ablation 표 | 기존 |
| 5 | `deployment/cortex_m3/run.py` | 엣지 비용 (mfg 행) | 기존 |

퍼시스턴스 수치는 계산 완료: `results/persistence_baseline.csv` (21행,
프로토콜 동일 파이프라인; `tools/paper/persistence_baseline.py`).
루프 작업 보드는 `.claude/kanban.json`.

## 두 실험이 끝나면 손에 쥐는 것

1. **FEATHer OFAT 512런 완료** → `results/hp_search.csv`
   (LTSF 데이터셋별 FEATHer 아키텍처 성적표)
2. **LTSF 베이스라인 1,760런 완료** → `results/fcst_results.csv`의 LTSF 행들
   (generalization 테이블의 베이스라인 몫은 이걸로 끝. 더 돌릴 것 없음)

---

## ① OFAT 결과 정리 → FEATHer 설정 확정 (반나절, Claude가 함)

```bash
python run_hp_search.py --summary
```

출력에서 두 가지를 뽑는다:

- **LTSF 데이터셋별 승자 설정** → `_DATASET_OVERRIDES`의 FEATHer LTSF 행 8개
- **8개 데이터셋 횡단 집계** → 제조에 들고 갈 **단일 표준 아키텍처** 확정
  (num_bands 3 vs 4 여기서 종결; tex ~485행 "B=4 산업 기본값" 문장과 정합 확인)
- 붙여넣고 커밋 → 서버 pull

**①-진행 (2026-07-03): 횡단 집계 완료 — 표준 아키텍처 = base 유지
(d8/k7/p12/B3).** d_state=16(7/8 승)·period=6(일관 승)은 성능상 이기지만
sub-1K 예산 초과로 기각(d16 @D=14/H96=1,146p; p6 @H720=2,638p); k/λ/B는
flat — **num_bands 3 vs 4 종결: 완전 flat → 3 유지, tex "B=4 산업" 문장
근거 없음**. 단, LTSF 8행의 OFAT 추천 조합은 미학습 조합이라 동결 보류 →
서버에서 조합 검증 후 확정:

```bash
python run_hp_search.py --validate   # 데이터셋별 추천 조합 실행 (≤32런, 몇 시간)
python run_hp_search.py --summary    # 최종 판정: 조합 채택 or 최고 관측 설정 fallback
```

**①-완료 (2026-07-06): `--validate` 32런 완주 → `--summary` 최종 판정 받음
→ FEATHer LTSF 8행을 `_DATASET_OVERRIDES`에 붙여넣기 완료.** 7개 데이터셋은
combo가 rank 1.00으로 채택, Exchange만 combo(rank 7.5)가 단일축 승자
`hp_d_state_16`(rank 3.5)에 밀려 fallback(설계대로: 관측된 최고 설정 보장).
파라미터 검증: d_state=16이어도 D≤14 세트는 H=96에서 sub-1K 유지
(ETT 643~803, Exchange 690); H=720 초과는 base도 겪는 기존 SPK pred_len
스케일링 → sub-1K 주장은 계속 H=96 스코프. 8행은 **일반화 섹션용**이며 제조
메인표 표준 아키텍처는 base(d8/k7/p12/B3) 그대로 동결.

검증이 끝난 `--summary`의 paste 블록이 최종 LTSF 8행. 블록 관계:
- **③ lr 서치는 전부(1,440런) 지금 시작 가능** — 표준 아키텍처가 base로
  확정됐으므로 FEATHer 120런도 더 기다릴 이유 없음 (LTSF 조합 검증은 제조
  쪽과 무관).
- **② FEATHer LTSF 160런만 검증 완료를 기다림** — per-dataset LTSF 행이
  `_DATASET_OVERRIDES`에 들어가야 돌 수 있으므로, `--validate` → `--summary`
  → 붙여넣기 → 커밋 → 서버 pull 후 실행.

## ② FEATHer의 LTSF 본 실험 160런 (하루 안팎)

LTSF 스윕에서 FEATHer만 빠져 있었음(HP 없어서 블록). ①에서 설정이 들어갔으니:

```bash
python run_forecast.py --model FEATHer --group ltsf --exp_tag main --save_model
```

8데이터셋 × 4호라이즌 × 5시드 = 160런 → **generalization 테이블(1,920런) 완성**.

## ③ 제조 lr 서치 1,440런 (며칠)

```bash
python run_lr_search.py          # 12모델 × 5lr × 6데이터셋 × 2호라이즌 × 2시드
```

①에서 표준 아키텍처가 확정된 상태이므로 FEATHer 포함 전부 한 번에.
(GPU가 남으면 ②와 ③ 동시 실행 가능 — 서로 다른 CSV라 안 섞임.)

## ④ lr 결과 붙여넣기 (한나절, Claude가 함)

```bash
python run_lr_search.py --summary
```

→ 72칸(12모델 × 6데이터셋) lr을 `_DATASET_OVERRIDES` 제조 행에 붙여넣고
커밋 → 서버 pull.

## ⑤ 제조 본 스윕 1,260런 (GPU 며칠) — 논문 메인 숫자

```bash
python run_forecast.py --exp_tag main --save_model                  # 메인 720
python run_forecast.py --group cmapss --exp_tag main --save_model   # PdM 540
```

- `--save_model` 필수 — ⑥ robustness가 이 체크포인트를 재사용
- 진행률: `python tools/audit/check_progress.py --exp_tag main`

## ⑥ Robustness 스윕 (추론만이라 빠름)

사전 작업(Claude): `run_robustness.py:ROBUSTNESS_DATASETS`를 제조 데이터셋으로
교체 (이때 SML 운명 결정). 그 다음:

```bash
python run_robustness.py --train_exp_tag main --exp_tag robust
```

## ⑦ Ablation (FEATHer 변형 30종 × 제조 데이터셋)

```bash
python run_forecast.py --ablation_axis all --data Steel --exp_tag ablation
# WindSCADA, CMAPSS 등 정확한 스코프는 ⑤ 끝날 때쯤 확정
```

ms축 결과 = num_bands 논거 + 원고 "B=4 산업 기본값" 문장 정리 근거.

## ⑧ 로컬 마무리 (서버 안 씀, Claude가 함)

- 퍼시스턴스 베이스라인 행을 표 생성기(`tools/paper/main_table.py`)에 추가
- 엣지 비용 추정 재계산 (`deployment/cortex_m3/run.py`, 제조 데이터셋 행)
- 표/그림 생성: `main_table.py` / `wilcoxon.py` / `robust_summary.py` /
  `ablation_table.py`

## ⑨ 원고

숫자가 채워지는 대로 Word 원고에 반영. Intro 초안은
`manuscript/drafts/intro_jms.md`(최종 스코프 반영 완료), Related/Datasets
리프레임은 ③~⑤ 도는 동안 병행 가능.

---

## 한눈에

```
[지금 돌고 있음]  OFAT 512 ──┐      LTSF 베이스라인 1,760 ──→ (끝나면 그대로 완료)
                            ▼
① OFAT summary → FEATHer 설정 확정 (LTSF행 + 제조 표준 아키텍처)
                            ▼
② FEATHer LTSF 160런   ∥   ③ 제조 lr 서치 1,440런     ← 동시 가능
                            ▼
④ lr summary → _DATASET_OVERRIDES 붙여넣기
                            ▼
⑤ 제조 본 스윕 720+540 (--save_model)
                            ▼
⑥ Robustness (추론)   +   ⑦ Ablation
                            ▼
⑧ 표/그림/엣지  →  ⑨ 원고 완성
```

## 대기 중 체크리스트 (서버)

- [ ] `git pull` (커밋 `05efecc` 이상)
- [ ] 6개 `data.csv` 복사: Steel / GasTurbine / WindSCADA / CMAPSS /
      **CMAPSS3(새 폴더)** / **PMSM(재생성본, 22,216행)**
- [ ] `python run_lr_search.py --check` → **1,440** 확인
- [ ] `python run_hp_search.py --check` → OFAT 진행률 확인
- [ ] LTSF 스윕 재개가 필요해지면 `--group ltsf` 필수
      (피벗 후 기본 그룹이 mfg로 바뀜)
