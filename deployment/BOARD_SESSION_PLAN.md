# 보드 측정 세션 계획 (7모델 전수 재측정)

작성 2026-08-24. 절차(손으로 어떻게)는 `board/BOARD_RUNBOOK.md`,
연구 맥락·과거 로그는 `EDGE_EXPERIMENT.md`. 이 문서는 **이번 세션에 뭘 재고
어디에 적을지**만 다룬다.

## 왜 전수 재측정인가

07-27 측정 이후 FEATHer가 두 번 바뀌었고(잔차 MLP 07-31, d16/B4 08-11),
라인업에 두 모델이 늘었다(MixLinear 포팅만 완료, SparseTSF-MLP 신규). 옛 값과
새 값을 섞으면 같은 표 안에 서로 다른 조건이 들어간다. 모델당 5분이면 되므로
**한 세션에서 같은 빌드 설정으로 7개를 다 잰다.**

부수 효과가 하나 더 있다: `tools/paper/measure_complexity.py`의 연산 회계가
지금 4개 실측점으로 r²=0.90을 얻고 있는데, 7점이 되면 그 검증이 눈에 띄게
단단해진다.

## 사전 조건 (전부 완료됨)

x86 골든 8개 전부 PASS — `wsl bash deployment/board/run_tests.sh`

| 모델 | params (D=7,H=96) | C 포팅 | x86 골든 |
|---|---:|---|---|
| persistence | 0 | pers_infer.h | PASS |
| SparseTSF-Linear | 41 | stsf_infer.c | PASS |
| MixLinear | 49 | mixlinear_infer.c | PASS |
| DiPE-Linear | 339 | dipe_infer.c | PASS |
| **FEATHer-N (ours)** | **938** | feather_infer.c | PASS (23단계) |
| SparseTSF-MLP | 1,181 | stsf_mlp_infer.c | PASS |
| FITS | 1,200 | fits_infer.c | PASS |

## 파일 이관 (CubeIDE 프로젝트로 복사)

대상 = `C:\Users\lee\STM32Cube_projects\blink_test`

`deployment/board/src/*.c` → `Core\Src\`
`deployment/board/src/*.h` + `deployment/board/golden/*.h` → `Core\Inc\`

이번에 **반드시 새로 복사해야 하는 것** (나머지는 변경 없음):
- `feather_infer.c/.h`, `fea_kernels.c/.h` — HIGH 밴드·GELU 추가로 바뀜
- `feather_weights.h`, `feather_vectors.h` — 938p로 재생성
- `stsf_mlp_infer.c/.h`, `stsf_mlp_weights.h`, `stsf_mlp_vectors.h` — 신규

`main.c`에 스위치가 없는 모델은 기존 `RUN_STSF` 분기를 복사해 추가
(RUNBOOK §2-1). 필요한 것: `RUN_MIXLINEAR`, `RUN_STSF_MLP`.

## 측정 항목 (모델당)

### A. 일반 모드 — `POWER_*` 둘 다 주석 처리
Build → Run → Tera Term(COM3, 115200)

| 읽을 값 | 어디서 | 표의 어느 칸 |
|---|---|---|
| golden PASS / bad=0 | UART | 정확도 일치 증빙(표에는 안 들어감) |
| `latency_us` | UART | T2 board latency |
| `stack=` | UART | peak SRAM |
| flash `text+data` | CubeIDE Console 빌드 출력 | T2 flash (실측) |

⚠ flash는 **빌드만 하면 나온다** — Run 안 해도 됨. 모델 바꿔 빌드할 때마다
Console 마지막 줄의 `text data bss dec` 를 적어둘 것. 지금 표의 flash 열은
params×4B 추정치라, 실측으로 바꾸면 그대로 더 강해진다.

### B. 전력 모드 — JP6 점퍼 + 멀티미터 (RUNBOOK §3)
- `POWER_ACTIVE` → 활성 전류(mA)
- `POWER_IDLE` → 대기 전류(mA), 모델 무관하므로 **1회만**

에너지/추론 = 3.3V × 활성전류(A) × latency(s)

07-27에 5모델 전수 29–31mA로 "모델 무관"이 나왔다. 새 3모델로 재확인되면
7모델 근거가 되어 배터리 수명 계산의 전제가 단단해진다.

⚠ 끝나면 **JP6 캡 반드시 원위치** — 안 꽂으면 다음에 MCU 전원이 안 들어감.

## 기록할 표 (그대로 채워서 주시면 됩니다)

```
모델              latency_us   stack(B)   flash text+data   활성 mA   golden
persistence
SparseTSF-Linear
MixLinear
DiPE-Linear
FEATHer-N
SparseTSF-MLP
FITS
대기(idle)           —           —              —
```

## 함정 (RUNBOOK §5 요약)

- 모델 `#define`은 **한 번에 하나만** — 여러 개면 RAM 20KB 초과.
  FEATHer가 16.6KB를 쓰므로 이번엔 여유가 더 줄었다.
- **Debug(🐞) 말고 Run(▶)**
- 최적화 **-O2** 확인 (`-O0`이면 훨씬 느리게 나옴)
- 플래그 바꿨는데 반영 안 되면 `Project → Clean`

## 이번 세션에서 안 하는 것

- **호라이즌별 측정**(H=192/336/720): 값어치는 크다(FEATHer는 H에 따라 거의
  안 늘고 SparseTSF-MLP는 3.8배 는다 — SPK의 존재 이유를 실측으로 보이는
  실험). 다만 모델×호라이즌마다 골든 재생성이 필요해 준비가 선행되어야 한다.
  **다음 세션.**
- **연속 운전 데모**: 펌웨어 하네스 확장 필요. 논문 필수는 아님.

## 측정 후 (내가 할 것)

1. `analysis/mfg/T2_cost.xlsx` 재생성 — latency/flash/energy 실측으로 교체
2. `measure_complexity.py --validate` 재실행 — 검증점 4 → 7
3. `EDGE_EXPERIMENT.md`에 세션 로그 추가
4. 배터리 수명 projection 계산
