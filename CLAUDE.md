# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEATHer (Fourier-Efficient Adaptive Temporal Hierarchy Forecaster) is an
ultra-lightweight model for long-term time-series forecasting under sub-1K
parameter / edge-MCU constraints. It combines multi-scale frequency
decomposition with a shared temporal kernel and a period-aware sparse
forecasting head.

## Current status (2026-07-14e)

**△ 2026-07-16 업데이트 (요약 — 상세는 아래 블록/Decision history):**
- **베이스라인 5→3 시드 (교수 결정)**: `--exclude FEATHer` 스윕을 `--num_seeds 3`
  으로 낮춰 제출 우선, **리비전에서 `--num_seeds 5`로 보강**(3시드=[2025,2026,
  2027]⊂5시드라 resume이 남은 2개만 채움; base_seed/exp_tag 불변 필수). **① LTSF는
  3시드로 재발사됨.** ⑤ 제조 본 스윕은 FEATHer 포함 통째로 `--num_seeds 3`(메인
  테이블 12모델 시드 일관성). ④a lr은 원래 2시드 고정이라 무관. **코드 수정
  없음**(CLI 플래그; 다운스트림 도구 전부 `nunique` 동적 집계, wilcoxon은 셀 넘어
  풀링이라 3시드도 유의성 충분). ② OFAT 선택은 3시드로 하지 말 것(안정성).
- **원고 텍스트 파트 재작성 (서버 무관)**: `manuscript/drafts/*_jms.md` 5개
  (abstract/intro/related/datasets/methodology+theory)를 mfg-6 + A+C 고속-"다리"
  프레이밍(동기만; 우리는 slow-process 평가) + **robustness 보수 서술** + SWIFT
  차별화 + half-RevIN + 이론 트림(R8 placeholder Theorem 3/4/5 제거)으로 갱신.
  옛 `0X_*_draft.md`는 IoT-J 아카이브. 상세: `memory/project_manuscript_jms_drafts.md`.
  남은 텍스트: Conclusion 리프레임 + Appendix 증명 2개 + Algorithm 1 검토.
- **고속 프레이밍 판정**: HR-SHM(고속 센싱→다스텝 예측, decimation이 이벤트를
  aliasing) = 실재 문헌이나 우리 데이터는 slow-process → **동기 인용만**, 우리
  평가 regime이라 주장 금지("horizon=step count" 다리로 봉합). EWELD = 금속산업
  전력**부하**(≠"극한기상 부하") + 극한기상 이벤트 창.

**⚡ 지금 서버 상태 (07-18b 2-서버 스윕-분할 — 체인의 단일 소스는 `ROADMAP.md`):**
**전용 GPU 1개짜리 신 서버 확보 → 스윕별 CSV 분리 이용해 병렬 분배**(사용자
결정). ①②④a는 CSV가 달라(`fcst_results`/`hp_search`/`lr_search`) 동시 안전.
- **구 서버 (Linux, 공유 128GB, GPU1)** = **① LTSF** (`FORCE_OMP=5
  CUDA_VISIBLE_DEVICES=1 python run_forecast.py --group ltsf … --ngpu 1
  --jobs_per_gpu 2 --exclude FEATHer --num_seeds 3`). `fcst_results.csv`.
- **신 서버 (Windows, 전용 32GB, GPU 1개, env `lee`, `C:\Users\user\Desktop\
  FEATHer`)** = **② FEATHer OFAT + ④a 제조 lr 둘 다** (CSV 다름 → 동시 안전).
  구 서버서 멈춰있던 hp를 `results/hp_search.csv` 복사해 이사(resume). 단일 GPU
  라 큐 옵션 최소: **② OFAT = `python run_hp_search.py`(jobs 1 유지 — elec/traffic
  워커가 RAM 무거움)**, **④a lr = `python run_lr_search.py --exclude FEATHer
  --jobs_per_gpu 3`(mfg는 RAM 가벼워 병렬 3배; GPU 10%·RAM ~8GB 여유라 이게 속도
  레버)**. RAM은 ② OFAT가 Traffic(862ch) 잡을 때만 조심(`free`). ④a엔 mfg
  data/ + `lr_search.csv` 전송 필요(뱅크 Steel/EWELD/BF resume).
- **⚠ 신 서버 `bash`는 bash 4+ 아님** — `setup_baselines.sh`의 `declare -A`가
  조용히 실패("division by 0" → clone 0개인데 "All present" 가짜 성공). **11개
  upstream repo는 cmd에서 수동 `git clone --depth 1 <url> baselines/<N>/<N>-main`**
  (URL = `setup_baselines.sh`의 REPOS). reformer_pytorch는 이미 설치됨. 안 하면
  베이스라인이 upstream import에서 죽음(예: DiPE_Linear `No module 'timeprophet'`).
- **⚠ 규칙: 같은 CSV는 한 박스에서만.** `lr_search.csv`=신 서버만(구 서버 lr 금지),
  `hp_search.csv`=신 서버만, `fcst_results.csv`=구 서버만. 최종엔 각 CSV를 한
  results/로 모으고 `check_duplicates.py` 검증. `FORCE_OMP` env = 커밋 `880c83e`
  (worker당 OMP/MKL 스레드 고정 → 박스 공유 시만 필요).
- **⚠ 07-19 `d56aa05`(MASK-NAN-GRAD) 푸시됨 → 양 서버 pull 필수, 특히 신
  서버는 pull 전 ④a 실행 금지**(마스크 3종이 가짜 0.0으로 또 기록됨).
  pull 후 클린 lr_search.csv(844행) 배치 → `run_lr_search.py --exclude FEATHer`
  재발사 → resume이 마스크 3종+미착수 모델 재충전.
**다음 사람 루프(루프 1)**: ②·④a 완료 → `--validate`(32) → `PYTHONUTF8=1
--summary` 출력 전달받아 LTSF 8행 재동결 + 07-06 diff 기록 + canonical mfg
아키텍처(d8/k7/p12/B3) 재확인 + λ 최종 판정 → 커밋 → pull → ③ FEATHer LTSF
160 + ④b FEATHer lr 발사.

**Positioning (settled):** Fresh **JMS** submission (TPAMI reject → IoT-J →
JMS manufacturing pivot). Main table = **6 manufacturing datasets** (Steel ·
WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD), LTSF-8 demoted to a
generalization section, short-horizon PdM (CMAPSS/CMAPSS3/PMSM) **NOT in the
paper** (lr-banked shelf, `--group cmapss`). FEATHer's 453p > SparseTSF 41p is
justified by **ROBUSTNESS, not accuracy** (OFAT num_bands FLAT on clean).
Framing guard: NOT "more params = more robust" but "41p linear lacks headroom;
FEATHer's inductive bias buys graceful degradation at sub-1K". R3's
"long-term-on-edge is contrived" attack resolved via **A+C framing**: (A)
industrial edge NEEDS on-device long-horizon (air-gap/OT isolation, per-sensor
scale, outage autonomy — our datasets are the evidence), (C) edge = param
budget, horizon = task, orthogonal axes; "why 1K" → accuracy-per-param Pareto
frontier. 12-model baseline lineup unchanged.

**⚑ CODE AUDIT (2026-07-14c) — full-repo review done, findings in
`manuscript/notes/code_audit_2026-07-14.md`. ✅ THE CRITICAL BATCH IS FIXED
(same session, all validated end-to-end):**
- ✅ **P0-1 robustness override plumbing**: `run_robustness.py` dispatch now
  mirrors the training-time `get_method_defaults`+`get_dataset_overrides`
  merge → `--model_overrides`; the worker gained the flag +
  `_apply_overrides()`; `gate_logging.py` applies the same merge itself.
  Verified live: trained ETTh1 d16/k3/p8/B4 ckpt loads + clean MSE
  reproduces training-time test MSE exactly.
- ✅ **P0-2 corruption seeding**: salted `hash((ft,sev))` →
  `utils/noise.corruption_seed()` (deterministic index-based). Verified:
  re-run reproduces corrupt-MSE to 6 decimals across processes.
- ✅ **P0-3 robustness resume**: `load_robust_set` counts DISTINCT
  (fault,severity) pairs; worker skips conditions already in the CSV
  (per-condition resume — no duplicate rows on retry). Verified via
  truncate-and-refill test.
- ✅ **P1-1 spectral loss band order** fixed (`bands[S-1-s]` — POINT owns the
  highest slice, LOW the lowest). Synthetic check: correct placement → loss
  0.0, swapped → 2.0. Frozen LTSF OFAT rows kept (λ axis was flat).
- ✅ **P1-2 Weather split** → standard 0.7/0.1/0.2 (legacy hard-coded branch
  + dead PM/PEMS_BAY/METR branches deleted). ALL pre-fix Weather rows are
  invalid → purge + re-run (see server procedure below).
- ✅ **P1-3 RMSE** = sqrt(global MSE) in both workers (+ NaN-aware CORR/R2
  accumulation). Old rows: derive sqrt(MSE) at table time.
- ✅ **P2-1 worker exit code**: training worker now exits 1 if any seed run
  failed (kills the reformer-style silent-skip class). **P2-2 CSV header
  race**: atomic `open(mode="x")` create-with-header in both workers.
- ✅ **P2-3..8 + P3 batch (2026-07-14d) — AUDIT FULLY CLOSED** (23-check
  smoke suite passed + all 4 orchestrators `--check` exit 0; details in the
  audit doc UPDATE 2 block):
  - P2-3 check_duplicates schema-detects the robust CSV (key extends with
    train_exp_tag/fault_type/severity — `--fix` now safe on it). P2-4
    explicit FEATHer CLI arch flags beat `_DATASET_OVERRIDES` (flags default
    None; sweep behavior byte-identical). P2-5 best_state=None raises (no
    row/ckpt; exits via P2-1 path). P2-6 mCE drops no-ref + model-incomplete
    cells for ALL models (warns; emits n_cells). P2-7 ablation_table = one
    table per dataset (no raw-MSE pooling). P2-8 lr/hp `--summary` flag seed
    under-coverage; hp combo "complete" requires full seeds.
  - P3: prep `_fill_gaps` gap policy (linear interp ≤ ~1 h, hold-last
    beyond — no lookahead/linear bridging of outages; audited max-gap
    asserts); BlastFurnace winsorize bounds from the TRAIN region (also
    finally clips the late Tp4 ~3311 sentinel: hi 3313→855); EWELD D=14
    assert; metrics.py np.divide out= init; severity tables +
    corruption_seed → torch-free `utils/severity.py` (noise.py re-exports,
    robust_summary shares the object, utils/__init__ now lazy PEP 562);
    FEATHer half-RevIN (means detached, stdev not) KEPT + documented as
    deliberate (changing it would invalidate all frozen OFAT rows); stale
    docstrings/comments swept.
  - ⚠ **DATA REGENERATED** under the new gap policy — cells changed:
    MetroPT3 16.6% / N2O 2.7% / WindSCADA 1.1% / BlastFurnace 0.9% /
    EWELD·Steel 0%. Pre-fix copies kept at `data/<name>/data.csv.preP3bak`.
    → transfer the 4 changed data.csv to the server and purge/refill their
    lr_search rows (added to SERVER PROCEDURE below).

**▶ SERVER PROCEDURE after pulling the fix commit** (order matters):
```bash
git pull   # (at a natural pause; running workers finish on old code harmlessly)
# invalidated rows — resume refills them:
python tools/audit/purge_rows.py --csv results/fcst_results.csv --data Weather --apply
python tools/audit/purge_rows.py --csv results/hp_search.csv    --data Weather --apply
python tools/audit/purge_rows.py --csv results/fcst_results.csv --model FEATHer --apply
python tools/audit/purge_rows.py --csv results/lr_search.csv    --model FEATHer --apply
# (07-14e 결정 — λ 슬라이스 부분퍼지를 대체) FEATHer OFAT 전면 재실행:
# hp_search를 통째로 비우고 run_hp_search.py가 512런 전부 재탐색
# (교정된 spectral loss + 표준 Weather split 아래에서):
python tools/audit/purge_rows.py --csv results/hp_search.csv --model FEATHer --apply
# (P3 data regen, 2026-07-14d) transfer the 4 regenerated data.csv files
# from local (WindSCADA / BlastFurnace / N2O / MetroPT3 — Steel/EWELD
# unchanged), THEN purge their lr rows so lr is re-selected on the new data
# (~960 of 2,160 lr runs; MetroPT3 changed 16.6% — too big to caveat away):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,BlastFurnace,N2O,MetroPT3 --apply
# then just re-run the same sweep commands — resume fills the gaps.
# Weather + lambda OFAT slices (hp_search) refill via run_hp_search.py;
# re-freeze the Weather row in _DATASET_OVERRIDES from --summary afterwards.
#
# (MASK-FIX, 2026-07-15) observed-only scoring — transfer + re-refill:
#   1. transfer the 3 regenerated data.csv (WindSCADA / N2O / MetroPT3 — now
#      hold-last, no interp) AND the 4 companion mask.csv (WindSCADA / N2O /
#      MetroPT3 / BlastFurnace) to data/<name>/.  (EWELD unchanged: 0 imputed.)
#   2. purge + refill lr for the 3 whose data changed (BF's 1-cell mask is
#      below noise -> its rows stay valid; EWELD unchanged):
python tools/audit/purge_rows.py --csv results/lr_search.csv --data WindSCADA,N2O,MetroPT3 --apply
#   (this SUPERSEDES the 07-14d BlastFurnace lr purge line above — BF no longer
#    needs a refill under MASK-FIX; the other 3 do, for both the P3 regen and
#    the interp->hold+mask change.)
```

**Evidence state (verified from hp_search.csv)**: only **num_bands is
confirmed FLAT on clean** LTSF-8 (B2/B3/B4 mean MSE 0.4505/0.4418/0.4457).
Gate on/off clean effect UNTESTED (needs ablation sweep); ALL robustness
UNTESTED (sweep unrun). The robustness thesis = 1 confirmed fact + an unproven
bet → the robustness sweep is the #1 risk.
⚠ **λ-flat confound → 해소 방식 격상 (user 결정 2026-07-14e)**: λ 슬라이스만
재는 대신 **FEATHer OFAT 512런 전면 재실행** — 모든 FEATHer HP가 교정된
loss + 표준 Weather split 아래에서 탐색되어 provenance caveat 자체가 소멸.
07-06 동결분과의 diff를 기록할 것 (크게 다르면 그 자체가 버그 영향의 증거).
canonical mfg 아키텍처(d8/k7/p12/B3)도 재집계로 재확인 (sub-1K 제약이
지배적이라 유지 예상). Memory: project_lambda_flat_confound.md.
All audit-fix sites are tagged in-code — `grep -rn "AUDIT-FIX"` lists every
fixed location with its finding ID.

**Server state**: 07-18b **2-서버 스윕-분할** — 구 서버(Linux, 공유, GPU1)=
**① LTSF**(`fcst_results.csv`), 신 서버(Windows, 전용 GPU1, env `lee`)=
**② FEATHer OFAT**(`hp_search.csv`) + **④a 제조 lr**(`lr_search.csv`) 둘 다
(CSV 달라 동시 안전; 위 ⚡ 블록 + ROADMAP.md 보드).
**⚠ 07-20 신 서버 GPU 사고+수리**: env `lee`에 torch가 `2.8.0+cpu`(CPU 전용
빌드)로 깔려 있어 `cuda.is_available()=False` → ②·④a가 **전부 CPU로** 돌아
GPU 0%. `pip install torch==2.5.1 --index-url .../cu121`로 교체(`2.5.1+cu121
True 1` 확인). 로컬 `feather`와 동일 버전(병합 정합성). CPU로 돈 OFAT/lr 행은
**선택용 스윕이라 폐기 불요**(최종 숫자는 ③/⑤ GPU 재실행). 발사 확인 = 워커
`Using device: cuda:0` + `nvidia-smi`에 python 프로세스(util 낮은 건 초소형
모델이라 정상). 세부 = `memory/reference_new_server_env_lee.md`.
**07-20 Ctrl+C 패치 `37e4758`**(personal push): `utils/dispatch_queue.py` —
단일 Ctrl+C가 스윕 트리 전체 종료(stop 이벤트 + in-flight 자식 핸들 추적 →
`subprocess.run`→`Popen`, 워커가 새 잡 안 집고 돌던 자식 terminate, 고아 방지).
**양 서버 pull 필요.** (기존엔 Ctrl+C가 죽인 자식을 "실패 잡"으로 보고 다음 잡을
띄워 매 런마다 Ctrl+C 해야 했음.)

**Local git**: main fully PUSHED to personal/ijaehun through 37e4758
(07-20 Ctrl+C 패치). Push policy: **ijaehun (personal) ONLY**, not WDSLab,
unless asked. 미커밋 로컬 = 클래식 베이스라인(LSTM/GRU/TCN + run_classical.py +
`baselines/__init__.py` M) + 검토수정 E — 현 체인(①②④) 끝난 뒤 별도 커밋(섞으면
list_models=15로 스윕 자동 확장).

**▶ NEXT**
1. ✅ ~~Apply audit fixes~~ — ENTIRE audit closed 2026-07-14d (P0/P1 in
   cfd90c5, P2-3..8+P3 in the close-out commit). Nothing left in the audit
   doc.
2. **Server (07-18 순차 단일-GPU 모드 — ROADMAP.md가 단일 소스)**: 3-동시
   → GPU 1 순차로 전환. 지금 **① LTSF 실행 중** → 끝나면 **② OFAT → ④a lr**
   하나씩. 남은 체인: **②·④a 완료** → `--validate` → `--summary` 전달 →
   (Claude) 8행 재동결 + 07-06 diff + canonical 재확인 + λ 판정 → 커밋 →
   pull → ③ FEATHer LTSF 160 + ④b FEATHer lr → lr `--summary` MERGE → mfg
   본 스윕 1,440 (`--save_model`) → robustness → `robust_summary.py` +
   `gate_logging.py`.
3. **A+C Intro/Motivation draft** (`manuscript/drafts/`) — needs air-gap /
   OT-isolation citations; scope to genuinely-local settings.
4. **Still-missing analysis code**: component ablation UNDER corruption
   (gate-off/band-off variants through the robustness sweep) + Pareto plot
   generator (MSE-vs-params frontier).
4b. **JMS 클래식 베이스라인 (구현됨, 미커밋/미활성화)**: LSTM/GRU/TCN
   (`baselines/`, list_models 15) + XGBoost (`run_classical.py`). 커밋 후
   **현 체인 끝난 뒤** 서버 활성화(15모델 스윕 + `run_classical.py --group
   mfg`). ARIMA/LightGBM·배포실측(#1구멍)·SWIFT문단은 후속. 상세 =
   ROADMAP.md "JMS 재포지셔닝" + `memory/project_jms_baseline_repositioning.md`.
5. ✅ ~~Push to ijaehun~~ — done 2026-07-14d (through 59c85da).
6. Raw-data cleanup under `data/`: DECIDED skip (raw = regen safety net; drop
   PMSM raw first if disk pressure).

### Decision history (compact — full blocks in `manuscript/notes/CLAUDE_STATUS_ARCHIVE_2026-07-14.md`)

- **2026-07-21 (데이터셋 6종 재확인 + 베이스라인 제외사유 통일 + CycleNet 판정
  — 전부 텍스트/결정, 코드·데이터·서버 변경 0)**: (1) **mfg-6 유지, MetroPT3
  KEEP** — 드롭 후보로 4턴 숙고했으나 대체 후보 전부 패(PMSM=2Hz persistence-
  trivial / TEP=시뮬 / flotation=공정중복·이미 07-06 패스 / ETT=generalization+
  robustness 삼중사용 / Korean10=감사리스크+전력클러스터중복). MetroPT3=코드
  no-op(드롭된 적 없음)+회전기계 자산클래스가 채우는 축. **선정기준 프레이밍=
  "비트리비얼(persistence-감사)+다양성+실측, 결측=프로토콜-handled"**(⚠ 무결성
  threshold·도메인순도 금지 — MetroPT3/WindSCADA가 아웃라이어라 자기모순).
  [[project-mfg6-metropt3-decision-0721]] + 데이터셋 **기각 로그**
  [[reference-dataset-reject-log]](데이터셋 논의 재개 시 먼저 던질 것). (2)
  **베이스라인 제외사유 self-consistency 픽스**(CLAUDE.md "Excluded from
  benchmark" 반영): S-Mamba "no MCU port"→**CUDA커널 재현불가**, TimeCMA
  "5 orders"→**사전학습=다른 자원클래스**, SWIFT "18k"→**official code 미성숙+
  FITS tie**(DLinear 18k가 표에 있어 sub-1K 논리 즉사), 방어문단 재작성, seq_len=
  96 "LTSF convention"→**배포프로토콜+MCU 입력버퍼 바이트예산**(lookback만 통일
  명시). **size는 절대 제외기준 아님 — 41p–1.6M 스팬이 목적**. (3) **CycleNet/
  RLinear=cite-not-baseline**(중간-tier·미요청·게이팅; "period 모델이라"는 폐기
  — SparseTSF도 period; 원칙상 포함가능이라 요구/1차스윕강하면 즉시 escalation).
  (4) **robustness 방어실험 재설계**: ablation-under-corruption=**1차 등-용량
  confound-closer**(gate/band-off ≈453p) / 2×2 용량스왑=보강(SparseTSF period-
  노브↑·FEATHer num_bands↓, clean정확도 유지 보고로 detune반론 봉합) / 이중프로토콜
  (메인 96 + 부록 literature lookback, **SparseTSF@720 필수**) / int8=limitations
  한 문장(리비전) / **전부 1차 스윕 효과확인에 게이팅**. [[project-baseline-lineup]]
  [[project-robustness-justification]] 07-21 블록. **라인업 12·①②④ 스윕 무영향**;
  체인 후 클래식 4종(LSTM/GRU/TCN+XGBoost)만 실제 추가. CLAUDE.md 텍스트분 미커밋
  (다음 원고/문서 커밋에 동봉).
- **2026-07-20c (① LTSF 스윕 sanity-audit — 전부 clean)**: 돌던 LTSF CSV 훑음.
  TPAMI Table 3(ETTh1/ETTh2) 재현 MSE·MAE 대부분 ±0.01(FITS/TQNet 동일값,
  PatchTST 현재가 더 좋음=개선, DLinear/ETTh2=시드불안정 특성), Weather 유효
  (07-14+ 재실행), RMSE 07-15+만 정상(옛 행 sqrt(MSE) 유도), HP소싱 표준
  (7 per-dataset 공식 / 4 단일 default = 어쩔 수 없고 정답), split 표준
  (ETT 12/4/4, 나머지 0.7/0.1/0.2), 공정성 OK. **실험 설계 문제없음, 남은 건
  덜 돈 것뿐.** 세부 = `memory/project_ltsf_baseline_verification_0720.md`.
  **decimation-vs-native 실험 = 안 함**(하중주장 아님+slow-process라 역풍
  리스크 → 리비전 카드; 인트로 anti-decimation은 인용+톤다운으로 봉합).
- **2026-07-20b (신 서버 GPU 수리 + Ctrl+C 패치)**: 신 서버 env `lee` torch가
  `2.8.0+cpu`라 ②·④a가 CPU로만 돌던 것 발견 → `torch==2.5.1+cu121` 재설치로
  수리(`2.5.1+cu121 True 1`). CPU 행은 선택용이라 유지. `dispatch_queue.py`
  단일-Ctrl+C 종료 패치(`37e4758`, personal push, 양 서버 pull). 신 서버 재실행
  = 터미널A `run_hp_search.py --jobs_per_gpu 3` / 터미널B `run_lr_search.py
  --exclude FEATHer --jobs_per_gpu 4`(RAM 여유로 3/4 워커; ② Traffic 겹칠 때
  RAM 주시). 세부 = `memory/reference_new_server_env_lee.md`.
- **2026-07-20 (코드/프로토콜 검토 수정 A~E 완료)**: 서브에이전트 검토 A/B
  (`manuscript/notes/review_findings_2026-07-19.md`) 발견 10건 전부 수정 —
  **A `1e078d7`**(#1 test/robustness 제로카운트→실패·#2 pred-NaN 가드·#9
  robustness resume 파손가드·#10 override 프로토콜축 블록리스트) + **B
  `c9ddd94`**(#3 전역 관측-셀 MSE/MAE 추정량 통일: 배치-bs가중→nansum/finite-
  count, persistence·classical과 일치, 무마스크 byte-identical) + **C·D
  `bbe6067`**(#4 main_table RMSE=√MSE 행별 유도·#5 `--seeds` 필터+시드수 불일치
  경고·#6 robust_summary dw/μ_rel/AUC 공통-셀 가드) 서버 푸시. **E는 로컬
  미커밋**(#7 run_classical 실패시 exit1·#8 XGBoost early-stop을 진짜 val
  split으로 — 클래식 묶음과 함께 대기). 검증: Steel 0.4884/N2O 0.7333
  byte-identical → **기존 행 재실행 불필요, ①·② 무영향**. 남은 것=논문 서술
  봉합 3건(lr-config 명시/FFT-period 스크립트/wilcoxon 상대화).
- **2026-07-19 (MASK-NAN-GRAD, `d56aa05` 푸시됨)**: MASK-FIX 후속 치명 버그
  발견·수정 — `_masked_loss`가 full-tensor로 `(output-target)` 먼저 계산 →
  NaN 타깃 셀이 autograd 그래프 진입 → backward `0*NaN=NaN` gradient →
  **마스크 3종(N2O/WindSCADA/MetroPT3)에서 첫 스텝 가중치 전체 NaN 발산** →
  train/val/test 집계 `sum/max(1,cnt)=0`이 **가짜 0.0 기록**(P2-5 가드도
  0.0=유한이라 통과). 수정 (1) 인덱싱 선행 `output[mask]-target[mask]`,
  (2) val_cnt==0→NaN(P2-5가 시끄럽게 실패). 3중 스모크 검증(N2O 0.0→0.7333 /
  MetroPT3 0.6914 / Steel 0.4884 무회귀). **양 서버 pull 후 ④a 재개 필수**.
  전송용 lr_search.csv는 마스크3종+GasTurbine 286행 purge 완료(1130→844).
  같은 날: ① 중간점검(886행 클린, 값 문헌범위, DLinear/ETTh2 시드불안정=특성)
  + 옛 TPAMI FEATHer 수치 vs 새 베이스라인 비교(ETT 1위권 유지, DiPE 우세,
  Electricity만 약점) → `manuscript/notes/ltsf_midsweep_check_2026-07-19.md`.
- **2026-07-16**: (1) **베이스라인 5→3 시드**(교수 결정: 제출 우선 + 리비전
  보강) — `--num_seeds 3`, 코드 수정 없음, ① LTSF 3시드 재발사, ⑤는 통째로 3시드,
  lr은 2시드 유지; 리비전 top-up = `--num_seeds 5`(resume이 2028/2029만 채움).
  (2) **원고 텍스트 재작성** — `*_jms.md` 5개(abstract/intro/related/datasets/
  methodology) mfg-6 + A+C 고속-다리(동기만) + robustness 보수 + SWIFT + half-RevIN
  + 이론 트림. (3) **고속 프레이밍**: HR-SHM은 실재 문헌이나 우리 데이터=slow-
  process → 동기 인용만. EWELD=금속산업 전력부하(+극한기상 창). Memory:
  project_manuscript_jms_drafts.md.
- **2026-07-15 (MASK-FIX)**: 결측 처리 정론화 — 모든 갭 **hold-last**(보간
  제거, 인과적) + companion `mask.csv`로 imputed 셀 표시 → 로더가 정답만
  NaN → **loss/metric 관측-only 채점**(traffic/의료/센서 표준; 12모델 동일→
  공정). 코드 5파일(prep/data_loader/data_factory/run_forecast/metrics;
  robustness는 공유 nan-aware metric 자동 상속). 검증: 배선 스모크 ALL PASS
  + 학습 Steel 0.5648/MetroPT3(17.7%마스킹) 0.7533. 실측 결측률: MetroPT3
  17.67%/N2O 5.79%/WindSCADA 1.20%/BlastFurnace 1칸/EWELD 0. **재실행 3개**
  (MetroPT3/N2O/WindSCADA — data가 interp→hold). BF 1칸·EWELD 0 = 불필요.
  선형보간 lookahead 공격면 소멸. 세부: ROADMAP.md "MASK-FIX" 블록.
- **2026-07-15 (JMS 재포지셔닝 + 클래식 베이스라인 구현)**: JMS 리뷰어=제조
  연구자 → 실무 신뢰 앵커로 **클래식 베이스라인** 추가 + **"실무층(정확·비배포)
  vs 엣지층(sub-1K·배포)" 2층 프레이밍** + **배포 실험=#1 구멍**(QEMU Layer-2
  skeleton → 실측) + ETT/Weather 강등(이미 결정). **SWIFT(2501.16178)=인용만**
  (18.1k=sub-1K 아님·FITS 타이·robustness 0·ICLR2025 withdrawn). **구현 완료
  (미커밋/로컬)**: LSTM/GRU/TCN(nn.Module, `baselines/{LSTM,GRU,TCN}/`,
  list_models 15, 기존 worker로 학습, arch 고정+lr서치) + XGBoost(별도
  `run_classical.py`, 채널독립 horizon-feature 직접다스텝, 마스킹-aware,
  GPU, 고정config+early-stop, lr탐색X). 스모크 전부 유한. **⚠ 서버 활성화
  보류**(push하면 스윕 3모델↑; 현 체인 끝난 뒤 의도적으로). ARIMA/LightGBM
  미결. Memory: project_jms_baseline_repositioning.md.
- **2026-07-14d/e**: 감사 P2-3..8+P3 전부 수정(23-check 검증, `59c85da`) →
  감사 종결; prep 갭 정책으로 데이터 4종 재생성(MetroPT3 16.6% 등) +
  BlastFurnace winsorize train-영역; **07-14e 사용자 결정: λ-슬라이스
  재측정 대신 FEATHer OFAT 512 전면 재실행**(07-06 동결 8행 대체 예정,
  diff 기록); LTSF/lr은 `--exclude FEATHer`로 재발사, FEATHer 몫(③④b)은
  재동결 후; TSLib 대조로 eval 프로토콜 정석 확인; push through 30f037d.
- **2026-07-14b**: `robust_summary.py` +4 metrics (μ_rel / AUC / mCE-vs-
  SparseTSF / degradation curve) + `gate_logging.py`, both synthetic-validated;
  R3 resolved (A+C, no short-term in paper); ROBUSTNESS_DATASETS
  GasTurbine→EWELD = [Steel, WindSCADA, EWELD, ETTh1, Weather].
- **2026-07-14**: mfg lineup LOCKED at 6 (EWELD in; GasTurbine fully removed —
  broken, no revival); freq-tied period HPs data-driven in
  `_DATASET_OVERRIDES`; SCINet added-then-reverted; TEP/ziya07/AI4I/GasSensor
  vetted+rejected; PyTSF-MfG = cite for mfg-scarcity gap.
- **2026-07-13c**: short-horizon PdM DROPPED from paper (kept as shelf);
  mfg 3→6 (+MetroPT3 +N2O +BlastFurnace-stress); Tier-2 loader prune
  (data_dict = 18 keys); push policy → ijaehun only.
- **2026-07-13/b**: strategy pivot — robustness (not accuracy) justifies
  params, protocol+citations in
  `manuscript/notes/robustness_justification_evidence.md`; server bugs fixed
  (reformer_pytorch import-time dep; --ngpu `torch.cuda.set_device`) —
  4 models were silently missing from CSVs, resume fills;
  `check_duplicates.py` added (fcst 769 rows, 0 dups).
- **2026-07-06/b**: OFAT 512 + combo 32 done → FEATHer LTSF 8 rows FROZEN into
  `_DATASET_OVERRIDES` (raw-verified); dataset precedent audit — all 6
  confirmed, "first to cast as edge LTSF" holds; wording guards in
  `manuscript/notes/dataset_precedent_audit.md`.
- **2026-07-03**: OFAT verdict — canonical mfg arch stays d8/k7/p12/B3
  (d16/p6 win val but break sub-1K); `--ngpu` dynamic queue + `--jobs_per_gpu`
  added; ⚠ FEATHer params scale with pred_len → sub-1K claim is H=96-scoped.
- **2026-07-02/b**: PMSM dropped from main (2Hz trivial persistence), rebuilt
  for PdM; PHM-2018 vetted+rejected; protocol sequencing locked (arch from
  LTSF OFAT, lr-only on mfg, symmetric).
- **2026-06-25/26**: JMS manufacturing pivot (professor's call, from IoT-J);
  TEP dropped, C-MAPSS added as separate short-horizon section (later
  shelved); Word submission format.
- **2026-06-11..18**: eval protocol rewritten (best-val-epoch, test-once);
  per-dataset FEATHer tuning replaces single-config narrative; baseline LTSF
  sweep launched; tex symbol/citation fixes.

## Environment

- **Conda env**: `feather` at `C:\Users\lee\.conda\envs\feather\` — Activate
  with `conda activate feather` before any commands. Has `torch 2.5.1+cu121`,
  `darts 0.37.1`, `reformer_pytorch 1.4.4` (for iTransformer's
  SelfAttention_Family).
- `mamba-ssm` is intentionally **not** installed (CUDA-toolchain build,
  out-of-scope for our edge-deployable benchmark).

## Repository structure

```
FEATHer/
├── run_forecast.py                 user-facing orchestrator (--check, resume, dispatch)
├── run_robustness.py               robustness orchestrator (loads --save_model checkpoints)
├── run_hp_search.py                FEATHer OFAT HP search (LTSF generalization configs + R2 #23 sensitivity)
├── run_lr_search.py                per-(method,dataset) lr search for the MFG main table (Option B, all 12 models)
├── setup_baselines.sh              git-clone baseline upstream repos (idempotent) + pip deps
├── scripts/
│   └── benchmarks/
│       ├── run_forecast.py         worker (one model × dataset × pred_len × N seeds)
│       └── run_robustness.py       robustness worker (Gauss/miss/impulse/quant sweep)
├── baselines/                      top-level model directory
│   ├── _import_helper.py           sys.modules isolation for upstream imports
│   ├── __init__.py                 registry (12 main + 30 ablation) + _METHOD_DEFAULTS + _DATASET_OVERRIDES
│   ├── FEATHer/                    our model (FEATHer.py + ablation/{multiscale,dtk,gating,head,complexity}.py)
│   └── <11 baseline dirs>/wrapper.py + <Name>-main/ upstream clones (gitignored)
├── utils/
│   ├── seed.py                     set_seed() + parse_seed_list()
│   ├── data_factory.py             data_provider + data_select (lazy darts import; data_dict = 18 keys)
│   ├── data_loader.py              Dataset classes (ETT_hour/minute, Custom, CMAPSS)
│   ├── dispatch_queue.py           --ngpu dynamic job queue (one worker subprocess per GPU)
│   ├── losses.py                   spectral_separation_loss_scales (FEATHer-specific)
│   ├── metrics.py                  metric() (MSE/MAE/RMSE/CORR/R2)
│   ├── noise.py                    4-axis corruption (gauss/miss/impulse/quant)
│   ├── severity.py                 torch-free severity grids + corruption_seed (single source)
│   └── timefeatures.py
├── tools/
│   ├── prep_manufacturing.py       reproducible raw→data.csv for the 6 MFG datasets (tracked; data/ gitignored)
│   ├── audit/check_progress.py     read results CSV, print coverage + mean±std preview
│   ├── audit/check_duplicates.py   resume-key dup audit + --fix (schema-aware: robust CSV key auto-extends)
│   └── paper/                      Phase 5 table/figure generators
│       ├── main_table.py           5-seed mean±std table + persistence column (md / latex / csv)
│       ├── wilcoxon.py             pairwise signed-rank vs FEATHer
│       ├── ablation_table.py       5 axes × 4 horizons aggregator (clean accuracy only)
│       ├── robust_summary.py       μ_rel / AUC / mCE-vs-SparseTSF / degradation curves + heatmaps
│       ├── gate_logging.py         FFT gate-under-corruption logging (per-band weights, CSV+PNG)
│       └── persistence_baseline.py copy-last baseline under the exact worker protocol
├── deployment/cortex_m3/           Layer-1 edge-cost estimator + qemu/ Layer-2 skeleton (WSL)
├── results/                        fcst_results.csv · robust_results.csv · edge_estimates.csv · checkpoints/<exp_tag>/
├── data/                           raw + processed data (gitignored — transfer to server manually)
└── manuscript/                     reviews, drafts, notes (gitignored; tex_workspace/ + drafts/*.md selectively tracked)
```

## Common commands

```bash
# Always activate the env first
conda activate feather

# === One-time: clone the 11 baseline upstream repos (+ reformer_pytorch dep) ===
bash setup_baselines.sh           # clone all
bash setup_baselines.sh DLinear   # clone one

# === Manufacturing main table (default group; 6 datasets) ===
python run_forecast.py --check --exp_tag main             # 1,440 mfg runs (12×6×4×5)
python run_forecast.py --exp_tag main --save_model        # group=mfg by default; --save_model makes robustness ckpts
python run_forecast.py --group ltsf --exp_tag main --save_model   # LTSF-8 generalization (1,920)
python run_forecast.py --group cmapss --exp_tag main --save_model # PdM shelf (540; NOT in paper)
python run_forecast.py --group all  --exp_tag main --save_model   # mfg + ltsf = 3,360
python run_forecast.py --model FEATHer --save_model                # one model across group
python run_forecast.py --data ETTh1 --pred_len 96 --save_model     # one (data, horizon)
python run_forecast.py --exclude TimesNet,MDMLP_EIA                # skip heavy models
python run_forecast.py --num_seeds 1 --num_epochs 2 --exp_tag smoke

# Multi-GPU: all four orchestrators take --ngpu N (dynamic queue, one worker
# subprocess per GPU; utils/dispatch_queue.py; no DDP) and --jobs_per_gpu M
# (ngpu×M workers, auto OMP/MKL pinning). ngpu=1 default = sequential.
python run_forecast.py --exp_tag main --save_model --ngpu 2 --jobs_per_gpu 3

# === lr search for the mfg main table (Option B; runs BEFORE the main sweep) ===
# 12 models × 5 lr × 9 datasets (6 mfg + 3 PdM shelf) × 2 horizons × 2 seeds = 2,160.
# Bypasses _DATASET_OVERRIDES (default arch) — period overrides only affect the main sweep.
python run_lr_search.py --check
python run_lr_search.py --ngpu 2            # resumable
python run_lr_search.py --summary           # best lr per (method,dataset) → MERGE into _DATASET_OVERRIDES
                                            # (merge into EXISTING period entries — no duplicate keys!)

# === FEATHer OFAT HP search (LTSF generalization only; DONE, rows frozen) ===
python run_hp_search.py --check / --summary / --validate
# Windows gotcha: run --summary with PYTHONUTF8=1 PYTHONIOENCODING=utf-8 (em-dash vs cp949)

# === Robustness sweep (after mfg-main checkpoints exist) ===
# (audit fully closed 2026-07-14d — only blocker left is the mfg-main
# checkpoints themselves)
python run_robustness.py --check            # 1,200 ckpt × 20 cond = 24,000 rows
python run_robustness.py --train_exp_tag main --exp_tag robust
python run_robustness.py --model FEATHer --train_exp_tag main

# === Ablation sweep (5 axes × 30 FEATHer variants; scope extends to mfg) ===
python run_forecast.py --ablation_axis all --data ETTh1 --exp_tag ablation
# axes: ms (15) / gate (4) / dtk (4) / head (4) / complexity (3)

# === Progress audit ===
python tools/audit/check_progress.py --exp_tag main
python tools/audit/check_duplicates.py      # schema-aware (robust CSV key auto-extends)

# === Paper artifact generators (Phase 5) ===
python tools/paper/main_table.py --metric MSE --format latex > main_mse.tex
python tools/paper/wilcoxon.py --reference FEATHer
python tools/paper/ablation_table.py --axis ms --format latex
python tools/paper/robust_summary.py --output_dir manuscript/figures/
python tools/paper/gate_logging.py --data Steel --pred_len 96 --ckpt <path>

# === Data regen (data/ is gitignored) ===
python tools/prep_manufacturing.py            # all 6 mfg recipes
python tools/prep_manufacturing.py EWELD      # one (BlastFurnace needs openpyxl)

# === Cortex-M3 edge cost estimator / QEMU (Layer 1 / Layer 2, WSL) ===
python -m deployment.cortex_m3.run --check
bash deployment/cortex_m3/qemu/run_qemu.sh FEATHer ETTh1 96 64

# === Manuscript build (IoT-J tex now legacy; JMS = Word via drafts/*.md) ===
cd manuscript/tex_workspace
pdflatex feather_iotj && bibtex feather_iotj && pdflatex feather_iotj && pdflatex feather_iotj
```

## Model registry (12 entries)

```python
from baselines import get_model, list_models, get_method_defaults, get_dataset_overrides
print(list_models())
# ['DLinear', 'DiPE_Linear', 'FEATHer', 'FITS', 'LMS_AutoTSF', 'MDMLP_EIA',
#  'PatchTST', 'SparseTSF', 'TQNet', 'TimeMixer', 'TimesNet', 'iTransformer']
```

| Model | Venue | Year | Params (ETTh1 paper config, D=7) |
|---|---|---|---|
| DiPE_Linear | DASFAA | 2026 | 267 |
| SparseTSF | ICML Oral | 2024 | 41 |
| FEATHer (ours) | (JMS) | 2026 | 453 |
| FITS | ICLR Spotlight | 2024 | 1,200 |
| DLinear | AAAI | 2023 | 18,624 |
| PatchTST | ICLR | 2023 | 35,171 |
| TimeMixer | ICLR | 2024 | 75,497 |
| TQNet | ICML | 2025 | 86,192 |
| LMS_AutoTSF | arXiv | 2024 | 181,318 |
| TimesNet | ICLR | 2023 | 605,479 |
| iTransformer | ICLR | 2024 | 841,568 |
| MDMLP_EIA | AAAI | 2026 | 1,632,832 |

(SCINet was briefly added 2026-07-13 then REMOVED — reviewer only named it as
prior multi-scale work, not a baseline request; Related-Work differentiation
instead.) → 5 orders of magnitude (267 → 1.6M). Two 2026 SOTA papers
(DiPE-Linear, MDMLP-EIA) address R4's "outdated baselines" critique.

Each baseline lives at `baselines/<Name>/`. `wrapper.py` exposes
`build(args, n_features, seq_len, pred_len)` that:
1. Uses `_import_helper.isolated_import(repo_root, "models.Foo")` to load
   the upstream `Model` class with full `sys.modules` isolation.
2. Constructs an `argparse.Namespace` with paper-default hyperparameters.
3. Wraps non-standard forward signatures into uniform `forward(x) -> (B, pred_len, D)`.

### Excluded from benchmark (cite in Related Work only)

- **S-Mamba** — selective-scan **custom CUDA kernels** → not fairly
  reproducible under our unified from-scratch protocol (AMP off, identical
  training loop); `mamba-ssm` intentionally not installed. *(NOT "no MCU
  port" — the ceiling models, e.g. iTransformer 841k / MDMLP 1.6M, don't fit
  an MCU either; MCU-portability can't be the exclusion criterion.)*
- **TimeCMA** (AAAI 2025) — frozen **pretrained GPT-2 features = a different
  resource class** (not trainable from scratch). *(NOT a size cutoff — we KEEP
  MDMLP 1.6M as a ceiling; size is never the exclusion criterion, the 41p–1.6M
  span IS the point.)*
- **SWIFT** (arXiv 2501.16178) — **no mature official code** → not fairly
  reproducible under our protocol; performance **ties FITS** (already in the
  table → FITS represents that point). ICLR2025-withdrawn = secondary note
  only. *(NOT "18k not sub-1K" — DLinear 18,624p is a benchmarked baseline.)*

Reviewer-defense paragraph (paper draft):
> "Model size is not our exclusion criterion — our benchmark deliberately
> spans 41 to 1.6M parameters to map the capacity–accuracy frontier. We
> exclude only architectures we cannot fairly reproduce under a unified
> from-scratch protocol — Mamba-family models (S-Mamba) requiring custom
> CUDA kernels, and methods without mature official code (SWIFT) — or that
> belong to a different resource class, namely LLM-augmented / pretrained
> forecasters (e.g., TimeCMA) depending on frozen external features rather
> than being trained from scratch. These are cited in Related Work but lie
> outside our reproducibility-controlled benchmark."

## Training protocol — TFB-style per-(method, dataset) paper HP

Resolved 2026-06-02 via TFB-style per-(method, dataset) overrides. See
`memory/project_open_lr_policy.md` for the full rationale.

### ✅ Unified across all (method, dataset) cells

| Axis | Value | Reason |
|---|---|---|
| `seq_len` | 96 | Deployment protocol: unified across all models; justified by MCU input-buffer byte budget (720×D×4B vs 96×D×4B). Other HP from each paper's recipe — lookback is the one deliberate override (state this to avoid the "published recipe vs 96 override" self-contradiction; SparseTSF@720 goes in an appendix rank-stability check). |
| `batch_size` | 32 | Memory protocol axis |
| `num_epochs` | 50 | Epoch-budget protocol axis |
| `patience` | 10 | Early stop on val loss; selection = best-val epoch, test evaluated once on that model |
| Optimizer | Adam | Standard |
| Scheduler | CosineAnnealingLR | Standard |
| AMP | **OFF** | Fairness + FITS complex grads + TimesNet cuFFT half-precision |
| Seeds | 2025–2027 (3 seeds; ⤴5 in revision) | 07-16 triage: `--num_seeds 3` submit-first, top-up to 5 (`--num_seeds 5`) in revision — 3⊂5, resume fills the 2; base_seed/exp_tag fixed. lr search stays `LR_SEEDS=[2025,2026]`. |
| `deterministic` | `cudnn.deterministic=True` | Reproducibility |
| Data split | identical | All baselines see the same train/val/test |

### ⚠️ Per-(method, dataset) overrides

| Axis | Source |
|---|---|
| `lr` | baselines: each paper's per-dataset script; mfg: `run_lr_search.py` (merge pending) |
| `loss` | each paper's default (MSE for baselines, L1 for FEATHer) |
| Architecture HPs | paper Table or official script; FEATHer LTSF-8 rows from OFAT (frozen) |
| Freq-tied (`cycle`, `period_len`, FEATHer `period`) | data-driven per mfg dataset (FFT-detected), same rule for all 3 period-models |

Routing: `_METHOD_DEFAULTS` + `_DATASET_OVERRIDES` in `baselines/__init__.py`;
orchestrator merges defaults + overrides + CLI, forwards `lr`/`loss` as worker
flags and the rest as `--model_overrides "k=v;k2=v2"`; worker's
`_apply_overrides()` sets each key on `args` before model construction.
FEATHer is tuned per-dataset like every baseline (symmetric fair-comparison
framing); canonical mfg arch = base d8/k7/p12/B3.

## Architecture (FEATHer model)

`baselines/FEATHer/FEATHer.py`, four components:

1. **Multi-band Decomposition** — depthwise 1D convs separate input into
   frequency bands (POINT, HIGH?, MID, LOW). `num_bands ∈ {2,3,4}`.
2. **DenseTemporalKernel (DTK)** — project to latent → depthwise causal conv
   → project back. Shared across all bands.
3. **FFTFrequencyGate** — FFT magnitude → Conv1d → softmax weights, one per
   band, adaptive per sample. Casts to float32 (cuFFT half-precision).
4. **SparsePeriodKernel (SPK)** — period-aware head: reshape by period phases,
   shared linear projection, reconstruct.

Loss = main (L1/MSE) + `lambda_spec` × spectral-separation loss (FEATHer-only;
⚠ band-order inversion bug — audit P1-1 — fix pending).

Key FEATHer args: `--d_state 8` (DTK latent), `--kernel_size 7`, `--period 12`
(must divide seq_len AND pred_len), `--num_bands 3`, `--lambda_spec 0.01`.

## Results CSV schema

Single source of truth: `results/fcst_results.csv`. Append-only.

Columns (resume key in bold):
**`exp_tag`**, **`model`**, **`data`**, **`pred_len`**, **`seq_len`**,
**`seed`**, `MSE`, `MAE`, `RMSE`, `CORR`, `R2`, `val_loss`, `best_epoch`,
`num_params`, `timestamp`

Test metrics computed once on the best-val-epoch model. `exp_tag` partitions
experiments (`main`, `robust`, `ablation_dtk`, `smoke`, `lr_*`, `hp_*`).
Use `--results_csv PATH` to redirect verification runs.

## Datasets

Loaded via `utils/data_factory.py` (lazy darts import; `data_dict` = 18 keys
after the 2026-07-13c Tier-2 prune).

| Dataset | Source | Horizon set |
|---|---|---|
| **Steel · WindSCADA · BlastFurnace · N2O · MetroPT3 · EWELD** (MFG **main table**; BlastFurnace D=28 = stress tier) | local CSV (`tools/prep_manufacturing.py`) | [96, 192, 336, 720] |
| CMAPSS / CMAPSS3 / PMSM (PdM **shelf**, not in paper; unit-aware `Dataset_CMAPSS`) | local CSV | [24, 48, 96] |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange/Electricity/Traffic (generalization) | darts | [96, 192, 336, 720] |
| nrel (kept per user) | local CSV | [96, 192, 336, 720] |

Robustness set: `run_robustness.py:ROBUSTNESS_DATASETS` =
**[Steel, WindSCADA, EWELD, ETTh1, Weather]** (EWELD's 5,741 extreme-weather
event windows = the only non-synthetic distribution-shift test).
EWELD gotcha: strong daily cycle → MUST report a seasonal-naive baseline
(copy-last-day ~0.55). Each `data/<name>/` holds raw + cleaned `data.csv`;
regenerate with `python tools/prep_manufacturing.py [<name> ...]`
(BlastFurnace needs `openpyxl`). `data/` is gitignored — transfer manually.

## Known issues / gotchas

- **⚑ AUDIT (2026-07-14) — FULLY CLOSED** (P0/P1/P2-1,2 in cfd90c5;
  P2-3..8 + P3 in the 07-14d close-out). Residual DATA facts to remember:
  pre-fix Weather rows and stored-RMSE values in old CSVs are stale (purge /
  derive sqrt(MSE) respectively); lr rows for WindSCADA/BlastFurnace/N2O/
  MetroPT3 predate the gap-policy data regen (purge+refill per SERVER
  PROCEDURE); `data/<name>/data.csv.preP3bak` = pre-fix data copies (delete
  once the server transition is confirmed).
- **`--ngpu N` + upstream hardcoded `.cuda()` (fixed 2026-07-13)**: workers
  call `torch.cuda.set_device(args.gpu)` at startup. Do NOT sed the upstream
  `baselines/*/*-main/` files (gitignored, re-cloned).
- **`reformer_pytorch` is an import-time dep of 4 baselines** (iTransformer,
  LMS_AutoTSF, TimeMixer, TimesNet). Missing → those 4 die at import, write
  no CSV row, and look "not yet run" (failed workers currently exit 0 —
  audit P2-1). `setup_baselines.sh` installs `reformer_pytorch==1.4.4`.
- **AMP is forcibly disabled** in the worker. Re-enabling breaks FITS
  (complex grads) and TimesNet (half-precision cuFFT).
- **`_import_helper.isolated_import`** must be used for every upstream
  baseline import (sys.modules collisions otherwise).
- **TQNet** receives `cycle_index = torch.zeros(B)` (our DataLoader has no
  per-sample cycle position) — consistent across all batches.
- **R2 metric** can explode to extreme negatives on near-constant channels
  (SST≈0); MSE/MAE/RMSE unaffected; R2/CORR supplementary only.
- **FEATHer params scale as D²** (`in_proj`): sub-1K holds for D≤14 only.
  Weather 1.4K / Electricity 115K / Traffic 776K / BlastFurnace(D=28) 1,986
  are framed as stress tests.
- **FEATHer params also scale with pred_len** (SPK backbone): quoted counts
  are H=96 values; at H=720 D≥11 exceeds 1K. Sub-1K claim must be scoped to
  H=96 in the manuscript (flagged 2026-07-03, unresolved).

## Workflow phases

- **Phase 1–3** ✅ infra (multi-seed, 12-model registry, 4-axis noise).
- **Phase 4** (running) main sweeps on server: mfg 1,440 + LTSF 1,920.
- **Phase 4b** (blocked on P0 fixes + mfg checkpoints) robustness sweep:
  12 models × 5 datasets × 4 horizons × 5 seeds × 20 conditions = 24,000 rows,
  inference-only from `--save_model` checkpoints.
- **Phase 5** paper artifacts under `tools/paper/` (+ still to write:
  ablation-under-corruption aggregation, Pareto plot).

## Paper context

- Target venue: **Journal of Manufacturing Systems (JMS)**, Elsevier/SME, Q1.
  Story: edge PdM/process-monitoring forecasting on machine-side MCUs;
  FEATHer = sub-1K Cortex-M3-deployable forecaster, robust to shop-floor
  sensor faults, validated across 6 industrial domains. Contributions:
  (1) sub-1K multiscale-frequency forecaster, (2) first edge/sub-1K mfg LTSF
  benchmark (6 domains), (3) sensor-fault robustness, (4) Cortex-M3
  deployment evidence.
- **Submission format: Word** (Elsevier "Your Paper Your Way"; no rigid
  template; Highlights 3–5 bullets ≤85 chars; Vancouver refs; CRediT +
  declarations). Draft in markdown under `manuscript/drafts/` → paste into
  Word. Order: Intro (A+C framing) → Datasets → Related → Method polish →
  results when sweeps land.
- Reuse map from `feather_iotj.tex`: Methodology + Theoretical Analysis
  ~as-is (trim theory); Intro/Related/Datasets need the manufacturing
  reframe; Results/Ablation/Robustness wait for sweep numbers.
- Wording guards: CMAPSS = "first to cast as a fixed-horizon multivariate
  sensor forecasting BENCHMARK" (not "first to forecast"); PMSM cite AIP
  Advances 16(4):045101; Steel all-channels departure stated explicitly;
  mfg-LTSF precedent = "first to cast as edge PdM LTSF"
  (`manuscript/notes/dataset_precedent_audit.md`).
- Reviewer-by-reviewer notes in memory; reject letter at
  `manuscript/notes/reject_mail.md`; robustness protocol citations at
  `manuscript/notes/robustness_justification_evidence.md`.
