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

## ⚑ MASK-FIX (2026-07-15) — 결측 = 관측-only 채점 (구현·검증 완료, 로컬)

결측 데이터 처리 정론화. **모든 갭을 hold-last(인과적, 보간 제거)로 채우고,
companion `mask.csv`로 imputed 셀을 표시 → 로더가 정답(data_y)만 NaN 처리 →
loss/metric이 관측값만 채점**(traffic/의료/센서 표준). 입력(data_x)은 채운
값 유지. 12모델 동일 적용이라 비교 공정.

- 코드: `prep_manufacturing.py`(mask.csv 생성+hold-last) · `data_loader.py`
  (Dataset_Custom가 정답 NaN) · `data_factory.py`(mask 로드) ·
  `run_forecast.py`(`_masked_loss` 관측-only) · `metrics.py`(MSE/MAE nan-aware).
  robustness worker는 공유 nan-aware metric 자동 상속(무수정).
- 검증: 배선 스모크 ALL PASS + 학습 스모크 Steel 0.5648 / MetroPT3(17.7%
  마스킹) 0.7533 유한. 마스크 없는 데이터셋은 nanmean==mean → 바이트 동일.
- 실측 결측률/마스크: MetroPT3 17.67% · N2O 5.79% · WindSCADA 1.20% ·
  **BlastFurnace 1칸**(첫캐스트-이전 Si; winsorize·캐스트간 홀드는 정당→제외)
  · EWELD 0(진짜 clean) · Steel/LTSF/PdM 0.
- **재실행 대상 = MetroPT3/N2O/WindSCADA 3개**(data.csv가 interp→hold로
  바뀜 + 마스킹). EWELD 무변화·BF 1칸(노이즈이하)=재실행 불필요.
- **서버 전송**: 재생성 data.csv 3개(WindSCADA/N2O/MetroPT3) + mask.csv 4개
  (+BlastFurnace) → 그다음 그 3개 lr purge+refill (아래 ④a에 반영).
- 로컬 백업: `data/{MetroPT3,N2O,WindSCADA}/data.csv.premask` (전환 확정까지).

## ⚑ JMS 재포지셔닝 — 클래식 베이스라인 & 배포 (2026-07-15)

제조 리뷰어 신뢰 앵커. 세부: `memory/project_jms_baseline_repositioning.md`.

**✅ 구현 완료 (미커밋/로컬, 서버 미활성화):**
- **LSTM / GRU / TCN** — `baselines/{LSTM,GRU,TCN}/` nn.Module, `list_models()`
  15개. 기존 worker로 학습(lr/main/robustness 자동). arch 고정 + lr 서치.
- **XGBoost** — 별도 `run_classical.py` (채널독립 horizon-feature 직접다스텝,
  마스킹-aware, GPU, 고정config+early-stop). `fcst_results.csv`에 model="XGBoost".
- 스모크 전부 유한 (Steel·MetroPT3 마스킹 포함). 파라미터 76k~290k노드=실무층.

**⏳ 해야할 것 (서버·원고):**
- [ ] **커밋 + push** (다음 커밋에 MASK-FIX 5파일 + 클래식 베이스라인 묶음).
- [ ] **서버 활성화 (현 체인 ①②④ 정리된 뒤 의도적으로)**: push 후
      `list_models()`=15라 lr/main/robustness가 3모델↑ (resume이 채움) +
      XGBoost는 `python run_classical.py --group mfg --exp_tag main`(GPU,
      120셀). ⚠ 지금 켜면 현 스윕과 겹치니 **② OFAT / ④ lr 끝난 뒤**.
- [ ] **lr 서치에 LSTM/GRU/TCN 포함** → `--summary`로 mfg lr 머지(기존 절차).
- [ ] **ARIMA / LightGBM** (별도 확장): ARIMA=느림·long-H 약함→mfg 짧은 H만
      고려; LightGBM=pip 설치 필요(xgboost가 트리계열 이미 커버) → 후순위.
- [ ] **배포 실험 = JMS #1 구멍**: `deployment/cortex_m3/` Layer-2 QEMU
      skeleton → 실측(FLOPs/메모리/레이턴시/사이클). 별도 세션.
- [ ] **SWIFT related-work 문단** (원고 시): 고정 웨이블릿+공유linear vs 우리
      적응형 다중대역+FFT게이트+주기헤드; robustness/제조/실배포 부재. 인용만.
- [ ] **2층 프레이밍 표**: 실무층(LSTM/GRU/TCN/XGBoost, 정확·비배포) vs
      엣지층(FEATHer/SparseTSF/FITS, sub-1K·배포). main_table 배치.

## 서버 실행 — 🔁 07-18b 2-서버 스윕-분할

07-14e 3-동시(`--ngpu 2`) → 07-18 순차 단일-GPU(느려서) → **07-18b 전용 GPU
1개짜리 신 서버 확보 → 스윕별 CSV 분리를 이용해 두 박스에 병렬 분배**(사용자
결정). ①②④a는 쓰는 CSV가 달라(`fcst_results`/`hp_search`/`lr_search`) 동시 안전.

**신 서버 (전용, GPU 1개, 일반 모드) ← ② FEATHer OFAT 이사:**
구 서버에서 ②는 **멈춰있었음** → `results/hp_search.csv`를 신 서버 `results/`로
복사하면 resume이 남은 것만 채움. 단독 GPU라 **큐/스레드 옵션 전부 제거**:
```bash
# (구 서버) hp_search.csv 신 서버로 복사 — 멈춰있으니 정지 불필요
scp results/hp_search.csv <신서버user>@<신서버host>:<repo>/results/hp_search.csv
# (신 서버) fresh clone이면: git clone + pull(880c83e까지) + conda activate feather
python run_hp_search.py --check     # Missing<512 = 이사 성공(진행분 인식)
python run_hp_search.py             # 일반 모드 — --ngpu/--jobs_per_gpu/FORCE_OMP/CUDA_VISIBLE_DEVICES 다 뺌
```

**구 서버 (공유, GPU 1) = ① LTSF 계속:**
```bash
FORCE_OMP=5 CUDA_VISIBLE_DEVICES=1 python run_forecast.py --group ltsf --exp_tag main --save_model --exclude FEATHer --ngpu 1 --num_seeds 3 --jobs_per_gpu 2
# ①/② 중 먼저 끝난 쪽에 ④a (lr_search.csv — hp와 안 겹침):
FORCE_OMP=5 CUDA_VISIBLE_DEVICES=1 python run_lr_search.py --exclude FEATHer --ngpu 1 --jobs_per_gpu 2
```

**⚠ `hp_search.csv`는 한 번에 한 박스에서만**(단일 CSV → 두 박스 동시 = 남은 행
중복 + 병합 충돌). 최종엔 각 CSV를 한 results/로 모으고 `check_duplicates.py` 검증.
(옛 3-동시 `--ngpu 2` / 07-18 순차 단일 커맨드는 히스토리; 위가 현재.)

발사 직후 사니티 체크 (숫자 이상하면 퍼지가 안 먹은 것):

```bash
python run_forecast.py --check --group ltsf --exp_tag main   # Weather 셀 missing, FEATHer는 목록에 없어야 정상
python run_hp_search.py --check                              # Missing = 512 (전체)
python run_lr_search.py --check                              # 4개 재생성 데이터셋 셀 missing
```

## 진행 보드

| # | 실험 | 커맨드 | 규모 | 상태 |
|---|---|---|---|---|
| ① | LTSF 베이스라인 (FEATHer 제외) | `run_forecast.py --group ltsf … --ngpu 1 --jobs_per_gpu 2` (구 서버) | 11모델 잔여 + Weather 재충전 | 🔄 ~절반 (4모델 완주, 5 미착수; 07-19 중간점검 `manuscript/notes/ltsf_midsweep_check_2026-07-19.md` — 값 문헌범위 OK, DLinear/ETTh2 시드불안정=특성) |
| ② | FEATHer OFAT 전면 재실행 | `run_hp_search.py` (신 서버, 일반 모드) → `--validate` | 512 + 32 | 🔄 신 서버로 이사 중 (hp_search.csv 복사 → resume) |
| ④a | 제조 lr (FEATHer 제외) | `run_lr_search.py --exclude FEATHer …` | 11모델 몫 (재생성 4데이터셋 재충전 포함) | ⏳ ①/② 중 먼저 비는 박스에 |
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

## ⚑ 07-19 검토 발견 — ✅ 07-20 전부 수정 완료

서브에이전트 검토 A/B 결과 = `manuscript/notes/review_findings_2026-07-19.md`.
활성 오염은 없었고, 타이밍 딸린 보류 수정 10건을 07-20에 전부 적용:
- **A `1e078d7`**(#1 제로카운트·#2 pred-NaN 가드·#9 resume 파손가드·#10 override
  블록리스트) → 서버
- **B `c9ddd94`**(#3 전역 관측-셀 MSE/MAE 추정량 통일) → 서버
- **C·D `bbe6067`**(#4 RMSE=sqrt(MSE)·#5 `--seeds` 필터·#6 robust 공통-셀) → 로컬 도구
- **E 로컬 미커밋**(#7 exit1·#8 XGBoost 진짜 val split) → 클래식 묶음과 함께 대기
검증: Steel 0.4884 / N2O 0.7333 byte-identical → **기존 행 재실행 불필요**,
현재 돌던 ①·② 무영향. 서버는 다음 pull에 반영. 남은 것은 논문 서술 봉합뿐.

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
