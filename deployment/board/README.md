# deployment/board — 실보드(NUCLEO-F103RB) float32 포팅 트랙

계획·결정·로드맵의 단일 소스는 `../EDGE_EXPERIMENT.md`. 이 디렉토리는
그 코드 산출물이다. (기존 `../cortex_m3/qemu/`는 int8/CMSIS-NN
방향의 Layer-2 스켈레톤 — 07-18 float32 결정으로 이 트랙과 별개.)

## 구성

```
export_golden_<m>.py     PyTorch 모델 → golden/ 헤더 (가중치 · 스테이지 골든 벡터 · DFT 트위들)
                         m = feather2 · feather2ci · dlinear · stsf · stsf_mlp · mixlinear · dipe · fits
                         · cyclenet · cmos · timebase · phaseformer · samformer
                         (10-07 의 4개는 --config/--data 로 후보 설정을 받음 = protocol.resolve 와 같은 해석;
                          구조값(seg·num_map·period·latent …)은 가중치 헤더에 찍혀 C 가 그대로 씀)
                         (export_golden.py = 08-24 의 938p FEATHer, 기록용)
export_batch.py          100윈도우 배치 MSE 기준값 → golden/batch_ref.h
verify_deployed_accuracy.py  학습 체크포인트 → 실제 테스트 창 200개 PyTorch 예측 (make realcheck 입력)
measure_footprint.py     ARM 크로스컴파일 → results/board_footprint.csv (모델 × D × H, 보드 불필요)
src/fea_kernels.{h,c}    공용 float32 프리미티브 + DFT/iDFT
src/<m>_infer.{h,c}      모델별 forward (보드 빌드 = 캡처 없음, 테스트 빌드 = -D<PFX>_CAPTURE_STAGES)
test/test_*.c            모델별 대조 (혼합 허용오차 atol=1e-5 + rtol=1e-4)
golden/                  생성물 (gitignored)
Makefile                 x86 빌드 (-O2 -ffp-contract=off -fno-fast-math)
```

## 실행

```bash
# 골든 생성 (Windows, conda feather env) - 기본은 H=96, D=7
python deployment/board/export_golden_<m>.py [--pred_len H] [--d_model D]
# 빌드 + 대조 (WSL)
make -C deployment/board test            # 79 검사
make -C deployment/board realcheck       # 먼저 export_golden_feather2.py --ckpt <MetroPT3 체크포인트>
# 칩 최대판 탐색의 구조값 골든 (10-14): FEATHer num_bands·point_kernel·use_norm·gate none·mlp_rank·
# head_seg·agg_shared (mix/CI), TimeBase use_period_norm 0, DiPE individual_*·use_freq_w/use_time_w 0
python deployment/board/check_dials.py   # 케이스마다 export → WSL make → 실행, 끝나면 golden/ 복원
# 발자국 실측
python deployment/board/measure_footprint.py --dims 1,4,6,7,8,14,16,21,28,32 --horizons 96,192,336,720
```

**10-07 추가 포트 메모.** 넓은 PhaseFormer 설정(latent 32·128, 헤드 2·4)과 CMoS L336 은 랜덤 골든에서
허용오차를 조금 넘는다 — float64 기준으로 C 오차가 PyTorch float32 오차와 같은 크기(1e-4~1e-3)임을 확인
(`export_golden_phaseformer.py --f64`), 즉 float32 덧셈 순서 한계. 기본 골든(make test)은 모두 통과.
CMoS 원본은 채널 1개에서 오류(squeeze) → D = 1 골든은 2채널 모델의 0번 채널(채널 독립이라 정확히 같음).

## 공통 C 구현 규칙 (규칙 ①, 2026-09-14)

포트는 전부 우리가 짰다. 포트마다 손본 정도가 다르면 배포표가 모델의 성질이 아니라
우리의 노력을 재게 되므로, **모든 포트(우리 모델 포함)에 아래 네 가지를 똑같이 적용한다.**
원칙은 한 문장이다: *PyTorch 모델의 계산은 그대로 수행하고, 메모리는 계산 순서를
정리하는 방식으로만 줄인다.* 코드 주석의 "equal-effort rule" 은 이 절을 가리킨다.

1. **센서 하나씩.** 센서끼리 계산이 섞이지 않는 모델은 센서를 하나씩 처리한다.
   섞는 모델은 섞는 데 필요한 만큼만 전 센서를 들고 있는다. 경량 계열에서 섞는 모델은
   FEATHer-mix 하나이고(08-27 행동 측정: 한 센서 파형을 바꿨을 때 다른 센서 출력이 변하는
   것은 mix 뿐), 그 대가는 잔차 MLP 가 요구하는 hidden state (L×D) 하나다.
2. **출력은 바로 내보내고, 쓰지 않을 출력은 계산하지 않는다.** 추가 계산 없이 끊을 수
   있는 단위로 호출자에게 넘긴다 — DLinear 1스텝 · SparseTSF / FEATHer 주기 · MixLinear
   위상 · FITS / DiPE 는 슬라이스로 남는 H 샘플만.
3. **동시에 쓰지 않는 작업 공간은 같이 쓴다.**
4. **재계산 금지.** 메모리를 줄이려고 같은 계산을 다시 하지 않는다. 메모리↔연산 교환은
   어느 모델에나 똑같이 가능하므로 비교 대상에서 뺀다 (원고 한계점에 명시).

**측정 조건 (모든 모델 동일).** SRAM = `.data+.bss` + 입력 창 L×D×4B + 스택 528B.
flash = `.text+.rodata` + 공용 커널 + 펌웨어 바닥 10.4KB. 호출자의 출력 버퍼는 어떤
모델에도 계상하지 않는다. 호라이즌에 따라 커지는 임시 배열은 스택이 아니라 static 에
둬서 계상되게 한다.

**검증.**
- 규칙 적용 전후의 보드 빌드 출력이 바이트 단위로 같다 (H 96/192/336/720 × D 7/28).
- 골든은 **모든 가중치를 랜덤으로** 채워 export 한다 (복소수 허수부 포함). 초기값 가중치
  (DiPE 의 1/0 항등 초기화, FITS·MixLinear 의 허수부 0)는 계산 경로를 검사하지 못한다 —
  DiPE 의 패딩 고정값 버그가 H=720 골든을 그대로 통과했던 이유다.
- `make test` 79 검사, `make realcheck` (학습 체크포인트 · 실제 테스트 창 200개).

**이 규칙으로 고친 것 (2026-09-14).** SparseTSF · SparseTSF-MLP · DLinear · MixLinear
앞단(센서 전체 버퍼) · FEATHer-mix (정규화 입력 사본, 출력 블록) · FITS / DiPE / MixLinear
출력단(쓰지 않는 출력 계산과 H 길이 버퍼) · FEATHer-CI 사본 2개. 그리고 DiPE 패딩을
H 에서 계산하도록 수정 (이전 값 1 은 H=96 에서만 맞았다).

## 기록: 1~2주차 (2026-07-21)

- **1주차 통과**: canonical mfg config(d8/k7/p12/B3, D=7, L=H=96,
  **453 params**)로 forward 전 스테이지 20개 PASS.
  §6 함정 4종 모두 골든 벡터로 커버: interp align_corners(fL) /
  DTK 인과 트림(oP·oM·oL) / eps-in-var(stdev) / rfft96(rfft_re·im).
- rfft 근사치의 max_rel이 0 근처 빈에서 ~2.2까지 튀지만 max_abs
  5.7e-6 → **혼합 허용오차 결정(#4)이 정확히 필요했던 지점** (순수
  상대오차 기준이었으면 가짜 FAIL).
- 트위들 테이블은 Python double로 계산해 float32로 반올림한 flash 상수
  (768B) → x86/ARM에서 동일 상수 보장.
- 2주차 (07-21f): 아레나 재작성 + 100윈도우 배치 MSE 5모델 일치
  (`export_batch.py` + `test/test_batch.c`, 10-14 에 `_archive/board_legacy/` 로 보관).
