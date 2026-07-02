# FEATHer 실험 로드맵 (JMS) — 2026-07-02 확정

> 서버의 두 실험(FEATHer OFAT 512런, LTSF 베이스라인 1,760런)이 끝난 뒤의
> 진행 순서. 세부 배경은 `CLAUDE.md` 상태 블록(2026-07-02/02b) 참고.
> 각 단계 사이의 사람 루프: **summary 가져오기 → 붙여넣기 → 커밋 → 서버 pull**.

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
