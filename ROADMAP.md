# FEATHer 실험 로드맵 (JMS) — 2026-07-14e 갱신

> 코드 감사(P0~P3) **전부 종결**. 사용자 결정(07-14e): λ-confound를
> 리트머스로 판정하지 않고 **FEATHer OFAT를 처음부터 전면 재실행** —
> 모든 FEATHer HP가 "교정된 spectral loss + 표준 Weather split" 아래에서
> 탐색되도록 하여 provenance caveat 자체를 소멸시킨다.
> 세부 배경: `CLAUDE.md` 상태 블록 + `manuscript/notes/code_audit_2026-07-14.md`.
> 사람 루프: **summary 전달 → 붙여넣기(로컬) → 커밋 → 서버 pull**.
> 푸시는 ijaehun(personal)만.

## 체인 한눈에 (07-14e 확정)

```
① LTSF 스윕 (FEATHer 제외, --exclude FEATHer)     ┐
② FEATHer OFAT 전면 재실행 512런 → validate 32    ├─ 동시 실행 (CSV 분리)
④a 제조 lr 서치 (FEATHer 제외)                     ┘
                     ▼  ② 완료
  사람 루프 1: hp --validate → --summary → LTSF 8행 재동결
              + canonical mfg 아키텍처 재확인 + λ 최종 판정 → 커밋 → pull
                     ▼
③ FEATHer LTSF 160런 (새 HP)   ∥   ④b FEATHer 제조 lr
                     ▼  ④ 완료
  사람 루프 2: lr --summary → lr 72칸 MERGE (기존 period 엔트리에, 키 중복 금지) → 커밋 → pull
                     ▼
⑤ 제조 본 스윕 1,440 (--save_model)   ∥   ⑦ ablation (겹쳐 가능)
                     ▼
⑥ Robustness 24,000행 (추론)  +  corruption-ablation/Pareto 코드
                     ▼
⑧ 표/그림/엣지  →  ⑨ 원고 완성
[상시 병행] ⑨ Intro/Related 초안 — 서버와 무관, 지금 제일 병목
```

## 서버 재발사 — ✅ 완료 (2026-07-14e, 터미널 3개)

퍼지(FEATHer fcst/lr 행 + hp_search 통째) 후 3개 동시 발사됨:

```bash
python run_forecast.py --group ltsf --exp_tag main --save_model --exclude FEATHer --ngpu 2
python run_hp_search.py --ngpu 2
python run_lr_search.py --exclude FEATHer --ngpu 2
```

발사 직후 사니티 체크 (숫자 이상하면 퍼지가 안 먹은 것):

```bash
python run_forecast.py --check --group ltsf --exp_tag main   # Weather 셀 missing, FEATHer는 목록에 없어야 정상
python run_hp_search.py --check                              # Missing = 512 (전체)
python run_lr_search.py --check                              # 4개 재생성 데이터셋 셀 missing
```

## 진행 보드

| # | 실험 | 커맨드 | 규모 | 상태 |
|---|---|---|---|---|
| ① | LTSF 베이스라인 (FEATHer 제외) | `run_forecast.py --group ltsf --exp_tag main --save_model --exclude FEATHer --ngpu 2` | 11모델 잔여 + Weather 재충전 | 🔄 진행 중 (07-14e 발사, 터미널 1) |
| ② | FEATHer OFAT 전면 재실행 | `run_hp_search.py --ngpu 2` → `--validate` | 512 + 32 | 🔄 진행 중 (07-14e 발사, 터미널 2) |
| ④a | 제조 lr (FEATHer 제외) | `run_lr_search.py --exclude FEATHer --ngpu 2` | 11모델 몫 (재생성 4데이터셋 재충전 포함) | 🔄 진행 중 (07-14e 발사, 터미널 3) |
| ③ | FEATHer LTSF 본실험 | `run_forecast.py --model FEATHer --group ltsf --exp_tag main --save_model --ngpu 2` | 160 | ⏳ 사람 루프 1 후 |
| ④b | FEATHer 제조 lr | `run_lr_search.py --model FEATHer --ngpu 2` | 5lr×9데이터셋×2H×2시드=180 | ⏳ 사람 루프 1 후 (canonical 아키텍처 재확인 뒤) |
| ⑤ | 제조 본 스윕 | `run_forecast.py --group mfg --exp_tag main --save_model --ngpu 2` | 1,440 | ⏳ 사람 루프 2 후 |
| ⑥ | Robustness | `run_robustness.py --train_exp_tag main --exp_tag robust --ngpu 2` | 24,000행 | ⏳ ⑤ + ①의 ckpt |
| ⑦ | Ablation | `run_forecast.py --ablation_axis all --data <mfg> --exp_tag ablation --ngpu 2` | 스코프 ⑤때 확정 | ⏳ ⑤와 병행 가능 |

페이스 확인:
```bash
python run_forecast.py --check --group ltsf --exp_tag main
python run_hp_search.py --check
python run_lr_search.py --check
python tools/audit/check_progress.py --exp_tag main
```

## 사람 루프 상세

### 루프 1 — ② OFAT 완료 시

```bash
python run_hp_search.py --validate --ngpu 2          # 추천 조합 검증 (~32런)
PYTHONUTF8=1 python run_hp_search.py --summary       # 최종 판정 출력 전달
```

Claude가 하는 일 (로컬):
1. **LTSF 8행 재동결** → `_DATASET_OVERRIDES` 교체 (직전 동결분과 diff 기록)
2. **canonical mfg 아키텍처 재확인** — 횡단 집계 + sub-1K 제약. 종전 결론
   d8/k7/p12/B3가 유지되는지; 바뀌면 ④b·⑤의 FEATHer 몫 전제가 바뀌므로
   여기서 확정하고 넘어감
3. **λ 최종 판정 기록** — 교정 loss 아래 λ축이 진짜 flat인지;
   `memory/project_lambda_flat_confound.md` 갱신 (confound 해소로 종결)
4. 커밋 → 서버 pull → ③·④b 발사

`--summary`의 `(incomplete)`/`(seeds n/2)` 플래그가 붙은 셀은 동결 전에
마저 채울 것.

### 루프 2 — ④a+④b 완료 시

```bash
python run_lr_search.py --summary
```
→ lr 72칸(12모델×6데이터셋)을 `_DATASET_OVERRIDES`의 **기존 period 엔트리에
MERGE (키 중복 금지!)** → 커밋 → pull → ⑤ 발사.

## ⑥ Robustness — 논문 #1 기둥 (추론만)

체크포인트 조건: ⑤(mfg 3종) + ①·③(ETTh1/Weather)이 `--save_model`로 완주.
ROBUSTNESS_DATASETS = [Steel, WindSCADA, EWELD, ETTh1, Weather].
감사 P0 버그(override 미전달/시딩/resume) 전부 수정·검증 — 발사 블로커 없음.
per-condition resume이라 중간에 죽어도 재실행만 하면 됨.

## 아직 없는 코드 (Claude, 서버 도는 동안 로컬 선행)

- [ ] **corruption 하 component ablation**: gate-off/band-off 변형을
      robustness 스윕에 통과시키는 배선 ("robustness가 FFT 게이트 덕분"의
      직접 증거)
- [ ] **Pareto plot 생성기**: MSE-vs-params 프런티어 (R3 "why 1K" 답변)

## ⑧ 표/그림 생성 체인 (전부 로컬)

`results/*.csv` 로컬 복사 후: `main_table.py`(퍼시스턴스 열 자동; EWELD는
seasonal-naive ~0.55 병기) → `wilcoxon.py --reference FEATHer` →
`robust_summary.py` → `gate_logging.py` → `ablation_table.py`(데이터셋별
분리 출력) → `deployment/cortex_m3/run.py`.
주의: 감사 이전 CSV의 RMSE 열은 편향값 → 표에서 sqrt(MSE) 유도.
퍼시스턴스 베이스라인은 **데이터 재생성 4종 때문에 재계산 필요**:
`python tools/paper/persistence_baseline.py`.

## ⑨ 원고 (상시 병행 — 현재 최대 병목)

- [ ] **A+C Intro/Motivation 초안** (`manuscript/drafts/`) — air-gap/OT
      isolation 인용; "진짜 로컬" 설정으로 스코프
- [ ] Related/Datasets 제조 리프레임 (워딩 가드:
      `manuscript/notes/dataset_precedent_audit.md`)
- [ ] Method: half-RevIN(means만 detach) 서술은 코드 그대로 정확히
- [ ] 갭-필 정책 서술: "≤1h 선형보간, 그 이상 hold-last(미래 미참조)" +
      BlastFurnace winsorize train-영역 (prep_manufacturing.py 주석 참고)
- 제출 포맷: Word (Elsevier YPYW)

## 리스크 메모

- **robustness 스윕 = 논지 #1 리스크**: "num_bands clean-flat"은 확인된
  사실이지만 "corruption에서 우아한 열화"는 아직 베팅. ⑥ 결과가 약하면
  프레이밍 재조정 (EWELD 실제 분포-이동 윈도우가 백업 증거).
- ~~λ-confound~~ → **OFAT 전면 재실행으로 구조적으로 해소** (07-14e 결정).
  재실행 후 새 8행이 종전과 크게 다르면 그 자체가 "버그 영향이 실재했다"는
  기록이 되므로 diff를 남길 것.
- canonical mfg 아키텍처가 재집계에서 바뀔 가능성: 낮음(sub-1K 제약이
  지배적)이지만, 바뀌면 ④b 이후 전부 새 canonical로 진행하면 됨 —
  ④b를 루프 1 뒤로 배치한 이유.

## 아카이브 (완료된 옛 단계)

- ~~코드 감사 P0~P3~~ ✅ 2026-07-14c/d 전부 수정·검증·푸시 (`59c85da`)
- ~~구(舊) OFAT 512 + combo 32 (07-06 동결)~~ → 07-14e 결정으로 **전면
  재실행 대상** (교정 loss + 표준 Weather split 아래에서 재탐색)
- ~~GasTurbine/SML/TEP/PM~~ 제거됨; PdM(CMAPSS/CMAPSS3/PMSM)은 논문 밖 셸프
- 갭-정책 데이터 재생성 ✅ (MetroPT3 16.6%/N2O 2.7%/WindSCADA 1.1%/
  BlastFurnace 0.9% 셀 변경; 서버 전송 + lr purge는 SERVER PROCEDURE 참고)
