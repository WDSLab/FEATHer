# deployment/board — 실보드(NUCLEO-F103RB) float32 포팅 트랙

계획·결정·로드맵의 단일 소스는 `../EDGE_EXPERIMENT.md`. 이 디렉토리는
그 1~4주차 코드 산출물이다. (기존 `../cortex_m3/qemu/`는 int8/CMSIS-NN
방향의 Layer-2 스켈레톤 — 07-18 float32 결정으로 이 트랙과 별개.)

## 구성

```
export_golden.py       PyTorch FEATHer → golden/ 헤더 (레이어별 골든 벡터)
export_golden_stsf.py  SparseTSF (실제 upstream, 41p) → golden/ 헤더
export_golden_dipe.py  DiPE-Linear (실제 upstream, 339p·복소194) → golden/
export_golden_fits.py  FITS (실제 upstream, 1200 복소p) → golden/
src/fea_kernels.{h,c}  공용 float32 프리미티브 + generic DFT/iDFT
src/feather_infer.{h,c} FEATHer forward (스테이지 캡처 포함)
src/stsf_infer.{h,c}   SparseTSF forward (MetroPT3 셀 p24)
src/dipe_infer.{h,c}   DiPE-Linear forward (96/193점 DFT 경로)
src/fits_infer.{h,c}   FITS forward (rfft96→복소 Linear→irfft192)
src/pers_infer.h       persistence copy-last (0p, 표의 바닥 행)
test/test_*.c          모델별 대조 (혼합 허용오차 atol=1e-5 + rtol=1e-4)
golden/                생성물 (gitignored): weights / vectors / DFT 트위들
Makefile               x86 빌드 (-O2 -ffp-contract=off -fno-fast-math)
```

## 실행

```bash
# 1. 골든 벡터 생성 (Windows, conda feather env)
python deployment/board/export_golden.py
# 2. 빌드 + 대조 (WSL 또는 아무 gcc)
make -C deployment/board test
```

## 상태 (2026-07-21)

- **1주차 통과**: canonical mfg config(d8/k7/p12/B3, D=7, L=H=96,
  **453 params** — 논문 수치와 일치)로 forward 전 스테이지 20개 PASS.
  §6 함정 4종 모두 골든 벡터로 커버: interp align_corners(fL) /
  DTK 인과 트림(oP·oM·oL) / eps-in-var(stdev) / rfft96(rfft_re·im).
- rfft 근사치의 max_rel이 0 근처 빈에서 ~2.2까지 튀지만 max_abs
  5.7e-6 → **혼합 허용오차 결정(#4)이 정확히 필요했던 지점** (순수
  상대오차 기준이었으면 가짜 FAIL).
- 트위들 테이블은 Python double로 계산해 float32로 반올림한 flash 상수
  (768B) → x86/ARM에서 동일 상수 보장.

## 남은 것 (로드맵 주차)

- ✅ ~~2주차~~ (07-21f): 아레나 재작성(정적 11,216B + 입력 2,688B =
  **13.6KB ≤ 16KB**; `feather_forward_ip()` = 보드 진입점, 캡처는
  `-DFEA_CAPTURE_STAGES` 스위치) + 100윈도우 배치 MSE 5모델 일치
  (|diff| ≤ 3e-8; `export_batch.py` + `test/test_batch.c`).
- 가중치 export를 학습 ckpt(`--save_model`)에서 뽑는 옵션 추가
  (⑤ 재동결 후; FEATHer는 MetroPT3 실배포 config p24로 재생성).
- **3주차**: STM32CubeIDE 프로젝트(M3 타깃, 동일 플래그) + map 검증.
- **4주차**: DWT 사이클 / 스택 페인팅 / IDD 전류 실측 (§7 프로토콜).
