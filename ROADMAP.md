# FEATHer 실험 로드맵 (JMS) — 2026-07-14d 갱신

> 코드 감사(P0~P3) **전부 종결**, purge+resume 체제로 전환된 이후의 진행
> 순서. 세부 배경은 `CLAUDE.md` 상태 블록(2026-07-14d)과
> `manuscript/notes/code_audit_2026-07-14.md` 참고.
> 각 단계 사이의 사람 루프: **summary 가져오기 → 붙여넣기(로컬) → 커밋 →
> 서버 pull**. 푸시는 ijaehun(personal)만.

## 현재 상태 보드 (2026-07-14d 기준)

### A. 서버에서 지금 돌고 있는 것 (2026-07-14 발사 완료)

| # | 실험 | 커맨드 | 채우는 것 | 상태 |
|---|---|---|---|---|
| 1 | LTSF 스윕 재개 | `run_forecast.py --group ltsf --exp_tag main --save_model --ngpu 2` | 베이스라인 잔여 + **Weather 전 모델 재충전(P1-2)** + **FEATHer LTSF 160런 재충전(P1-1, 동결 HP로 첫 유효런)** + ETTh1/Weather robustness 체크포인트 | 🔄 진행 중 |
| 2 | 제조 lr 서치 재개 | `run_lr_search.py --ngpu 2` | 잔여 + **FEATHer 행 재충전(P1-1)** + **WindSCADA/BlastFurnace/N2O/MetroPT3 전 모델 재충전(P3 데이터 재생성)** | 🔄 진행 중 |

⚠ **전제 확인**: 위 두 개가 "재충전"이 되려면 발사 전에 purge 6건 + 4개
data.csv 전송이 서버에서 끝나 있어야 함. 확인법 — `--check`에서 Weather/
FEATHer/4개 데이터셋 셀이 missing으로 잡히면 OK:

```bash
python run_forecast.py --check --group ltsf --exp_tag main
python run_lr_search.py --check
```

### B. 서버에서 다음에 발사할 것 (지금 가능)

| # | 실험 | 커맨드 | 규모 | 비고 |
|---|---|---|---|---|
| 3 | hp_search 재충전 | `python run_hp_search.py --ngpu 2` | Weather OFAT 슬라이스 + **λ축 4태그×8데이터셋×2H×2시드=128런** | 전부 초소형 FEATHer 런 — A와 동시 실행 무방 (다른 CSV) |

### C. 결과 도착 시 사람 루프 (순서대로)

| 순서 | 트리거 | 하는 일 |
|---|---|---|
| 1 | #3 완료 | `PYTHONUTF8=1 python run_hp_search.py --summary` → **Weather 행 재동결** + **λ-confound 최종 판정** (flat이 진짜였는지; 판정 바뀌면 동결 8행의 lambda_spec 갱신) → 로컬 커밋 → 서버 pull. λ 판정은 `memory/project_lambda_flat_confound.md`에도 기록 |
| 2 | #2 완료 | `python run_lr_search.py --summary` → lr을 `_DATASET_OVERRIDES`의 **기존 period 엔트리에 MERGE (키 중복 금지!)** → 커밋 → pull. `(incomplete)`/`(seeds n/2)` 플래그 붙은 셀은 동결 전에 마저 채울 것 |
| 3 | 1+2 끝 | ⑤ 제조 본 스윕 발사 (아래) |

## ⑤ 제조 본 스윕 — 논문 메인 숫자 (GPU 며칠)

```bash
python run_forecast.py --group mfg --exp_tag main --save_model --ngpu 2   # 1,440런
```

- 12모델 × 6데이터셋(Steel/WindSCADA/BlastFurnace/MetroPT3/N2O/EWELD) × 4호라이즌 × 5시드
- `--save_model` 필수 — ⑥ robustness가 이 체크포인트를 재사용
- 진행률: `python tools/audit/check_progress.py --exp_tag main`
- (PdM 셸프 540런은 논문 밖 — 리뷰어가 요구할 때만 `--group cmapss`)

## ⑥ Robustness 스윕 — 논문 #1 기둥 (추론만, 상대적으로 빠름)

체크포인트 준비 조건: ⑤(mfg 3종) + A#1(ETTh1/Weather)이 `--save_model`로
완주. ROBUSTNESS_DATASETS = [Steel, WindSCADA, EWELD, ETTh1, Weather] (설정 완료).

```bash
python run_robustness.py --check                                  # 1,200 ckpt × 20조건 = 24,000행
python run_robustness.py --train_exp_tag main --exp_tag robust --ngpu 2
```

감사로 잡은 P0 버그들(override 미전달/시딩/resume) 전부 수정·검증된 상태 —
발사 블로커 없음. per-condition resume이라 중간에 죽어도 그냥 재실행.

## ⑦ Ablation (⑤와 겹쳐 돌려도 됨 — 다른 exp_tag)

```bash
python run_forecast.py --ablation_axis all --data Steel --exp_tag ablation --ngpu 2
# 스코프(전체 mfg vs 대표 데이터셋)는 ⑤ 진행 보며 확정
```

ms축 결과 = num_bands 논거. **추가로 아직 코드가 없는 것 2개** (Claude 작업,
서버 기다리는 동안 로컬에서 선행 가능):

- [ ] **corruption 하 component ablation**: gate-off/band-off 변형을
      robustness 스윕에 통과시키는 배선 (robustness가 "FFT 게이트 덕분"임을
      보이는 직접 증거 — 리뷰어 선제 방어)
- [ ] **Pareto plot 생성기**: MSE-vs-params 프런티어 (R3 "why 1K" 답변용)

## ⑧ 표/그림 생성 체인 (전부 로컬, Claude가 함)

`results/*.csv`를 로컬로 복사한 뒤:

| 순서 | 커맨드 | 산출물 |
|---|---|---|
| 1 | `main_table.py --metric MSE --format latex` | 메인표 (퍼시스턴스 열 자동; **EWELD는 seasonal-naive ~0.55도 병기**) |
| 2 | `wilcoxon.py --reference FEATHer` | 유의성 |
| 3 | `robust_summary.py --output_dir manuscript/figures/` | μ_rel/AUC/mCE-vs-SparseTSF/열화곡선+히트맵 |
| 4 | `gate_logging.py --data Steel --pred_len 96 --ckpt <path>` | 게이트 해석 그림 |
| 5 | `ablation_table.py` | ablation 표 (데이터셋별 분리 출력) |
| 6 | `deployment/cortex_m3/run.py` | 엣지 비용 (mfg 행) |

주의: 감사 이전에 쓰인 CSV의 RMSE 열은 편향값 → 표에서는 sqrt(MSE)로 유도
(main_table은 이미 그렇게 함).

## ⑨ 원고 (서버 도는 동안 병행 — 지금 가장 병목)

- [ ] **A+C Intro/Motivation 초안** (`manuscript/drafts/`) — air-gap/OT
      isolation 인용 필요; "진짜 로컬" 설정으로 스코프
- [ ] Related/Datasets 제조 리프레임 (워딩 가드:
      `manuscript/notes/dataset_precedent_audit.md`)
- [ ] Method: half-RevIN(means만 detach)을 서술할 거면 코드 그대로 정확히
      기술 (FEATHer.py에 deliberate 주석 있음)
- [ ] 갭-필 정책 서술 추가: "≤1h 선형보간, 그 이상 hold-last(미래 미참조)"
      + BlastFurnace winsorize train-영역 (prep_manufacturing.py 주석 참고)
- 숫자 자리는 ⑧이 채움. 제출 포맷: Word (Elsevier YPYW)

---

## 한눈에

```
[지금 돌고 있음]  ① LTSF 재개(Weather+FEATHer 재충전, --save_model)
                 ② lr 서치 재개(FEATHer+4데이터셋 재충전)
[지금 발사 가능]  ③ hp_search 재충전 (Weather OFAT + λ축 128런)
                            ▼
  사람 루프 1: hp --summary → Weather 행 재동결 + λ 판정 → 커밋 → pull
  사람 루프 2: lr --summary → lr 72칸 MERGE → 커밋 → pull
                            ▼
⑤ 제조 본 스윕 1,440 (--save_model)     ∥  ⑦ ablation (겹쳐 가능)
                            ▼
⑥ Robustness 24,000행 (추론)  +  corruption-ablation/Pareto 코드
                            ▼
⑧ 표/그림/엣지  →  ⑨ 원고 완성
[상시 병행] ⑨ Intro/Related 초안 — 서버와 무관, 지금 제일 병목
```

## 리스크 메모

- **robustness 스윕 = 논문 논지의 #1 리스크**: "num_bands clean-flat"은
  확인된 사실이지만 "corruption에서 FEATHer가 우아하게 열화한다"는 아직
  베팅. ⑥ 결과가 약하면 프레이밍 재조정 필요 (EWELD 실제 분포-이동
  윈도우가 백업 증거).
- λ 재측정(③)에서 판정이 바뀌면 FEATHer LTSF 160런(①에서 재충전된 것)의
  lambda_spec과 불일치 가능 → 그 경우 해당 데이터셋만 한 번 더 purge+재런
  (작음). 바뀌지 않으면 그대로 진행.

## 아카이브 (완료된 옛 단계)

- ~~OFAT 512 + combo 32~~ ✅ 2026-07-06 완료, LTSF 8행 동결 (Weather 행만
  P1-2로 재동결 대기)
- ~~표준 제조 아키텍처 확정~~ ✅ base d8/k7/p12/B3 (d16/p6는 sub-1K 초과로
  기각)
- ~~코드 감사 P0~P3~~ ✅ 2026-07-14c/d 전부 수정·검증·푸시 (`59c85da`)
- ~~GasTurbine/SML/TEP/PM~~ 제거됨; PdM(CMAPSS/CMAPSS3/PMSM)은 논문 밖 셸프
- 퍼시스턴스 베이스라인: `results/persistence_baseline.csv` 계산 완료
  (단, mfg 데이터 4종이 갭-정책으로 재생성됨 → ⑧ 전에 재계산 권장:
  `python tools/paper/persistence_baseline.py`)
