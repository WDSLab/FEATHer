# 보드 측정 실전 매뉴얼 (NUCLEO-F103RB)

임베디드 처음이어도 **이 문서만 보고** 측정을 시작·수행·종료할 수 있게 정리한
운영 절차서. (연구 맥락·결과·결정은 `../EDGE_EXPERIMENT.md`가 단일 소스. 이 문서는
"손으로 어떻게 하는가"만 다룸.)

_최초 작성 2026-07-27 (보드 도착·5모델 실측 완료한 세션)._

---

## 0. 어디에 뭐가 있나 (경로)

| 항목 | 위치 |
|---|---|
| **CubeIDE 프로젝트** | `C:\Users\lee\STM32Cube_projects\blink_test` (⚠ repo 밖, Dropbox 아님) |
| 편집하는 소스 | 위 폴더 `Core\Src\main.c` |
| 모델 C코드 원본(백업) | `deployment/board/src/` (repo 안) |
| 골든/가중치 헤더 원본 | `deployment/board/golden/` (repo 안) |
| 결과·로그·결정 | `deployment/EDGE_EXPERIMENT.md` (repo 안) |
| 시리얼 터미널 | **Tera Term** (PC에 설치됨) |
| 멀티미터 | **ZOYI ZT-102** (전력 측정용) |
| 보드 시리얼 포트 | **COM3**, **115200** baud (바뀔 수 있음 → 장치관리자 확인) |

**툴체인 주의**: STM32CubeIDE **2.2.0은 CubeMX가 분리**돼 있음. 새 프로젝트를
만들 일이 있으면 **독립 실행형 STM32CubeMX(숫자 없는 것 = HAL1; "MX2" 아님)** 로
Board Selector → NUCLEO-F103RB 생성해야 함. (이번 `blink_test`는 이미 만들어져
있으니 새로 만들 필요 없음.)

---

## 1. 다시 시작하기 (세션 재개 절차)

1. **USB 미니B 케이블로 보드 연결** → 잠시 뒤 탐색기에 `NODE_F103RB` 드라이브가
   뜨면 인식 OK. (안 뜨면: 데이터 케이블인지 확인 / 다른 USB 포트)
   - 이미 구워둔 펌웨어가 있으면 **연결하자마자 자동 실행**됨 (초록 LD2 깜빡).
2. **STM32CubeIDE 실행** → 워크스페이스는 기본값 Launch.
   - 프로젝트가 안 보이면: `File → Open Projects from File System...` →
     `C:\Users\lee\STM32Cube_projects\blink_test` 지정 → Finish.
3. **Tera Term 실행** (결과를 눈으로 볼 통로):
   - `Serial` 선택 → 포트 **COM3**(장치관리자 → 포트(COM & LPT) →
     `STLink Virtual COM Port`에서 번호 확인) → OK
   - `Setup → Serial port...` → **Speed = 115200** → OK
   - 보드가 돌고 있으면 여기 출력이 뜸. 안 뜨면 보드 검은색 **리셋 버튼(B2)** 한 번.

준비 끝. 이제 §2(측정)로.

---

## 2. 측정하는 법 (핵심 워크플로)

측정 대상·모드는 **`Core\Src\main.c` 맨 위 `USER CODE BEGIN PD` 블록의 `#define`
스위치**로 고른다. 고르고 → **Build(망치) → Run(▶)** → Tera Term/멀티미터로 읽는다.

### 2-1. 모델 고르기 (한 번에 하나만!)
```c
/* #define RUN_STSF */     // SparseTSF
/* #define RUN_DIPE */     // DiPE-Linear
/* #define RUN_FITS */     // FITS
/* #define RUN_PERS */     // persistence
#define RUN_FEATHER        // FEATHer (ours)  <- 활성화된 건 이 줄 하나뿐
```
- **반드시 하나만** 활성화(`#define`), 나머지는 `/* */`로 주석.
- ⚠ **RAM 20KB 제약**: 여러 모델을 동시에 넣으면 초과. 그래서 "한 빌드 = 한 모델".
  링커(`--gc-sections`)가 **호출하는 모델만** 링크하므로, 소스 파일은 다 놔둬도
  됨(안 부르는 모델은 자동으로 빠짐). **빌드 제외(Exclude from Build) 토글 필요
  없음** — `#define`만 바꾸면 됨.
  - (과거에 feather_infer.c를 빌드 제외한 적 있는데, 지금 방식(한 모델씩)에선
    불필요. feather_infer.c는 **포함(un-exclude) 상태로 둘 것**.)

### 2-2. 무엇을 잴지 (모드 스위치, 바로 아래 줄)
```c
/* #define POWER_ACTIVE */  // 켜면: 추론 무한반복(전류 안정) — 전력 측정용
/* #define POWER_IDLE */    // 켜면: CPU 재움(WFI) — 대기 전류 측정용
```
- **둘 다 주석(꺼짐)** = **일반 모드**: golden 검증(PASS/FAIL) + 레이턴시(cycles/µs)
  + stack 출력, 초록 LED 깜빡. → **레이턴시·정확도 잴 때 이 모드.**
- **POWER_ACTIVE만 켬** = 추론만 무한반복(출력·LED 없음, "hello"만 한 줄) →
  **멀티미터로 활성 전류 잴 때.**
- **POWER_IDLE만 켬** = CPU sleep → **멀티미터로 대기 전류 잴 때.**

### 2-3. 빌드 & 실행
1. **Build(망치)** → Console에 `Build Finished. 0 errors` 확인.
   - `undefined reference to feather_forward` 뜨면 → feather_infer.c가 빌드 제외됨.
     `Core/Src/feather_infer.c` 우클릭 → `Resource Configurations → Exclude from
     Build...` → **체크 해제**.
   - `region RAM overflowed` 뜨면 → `#define`이 2개 이상 켜졌는지 확인(하나만).
   - 플래그 바꿨는데 반영 안 되면 → `Project → Clean...` 후 다시 Build.
2. **Run(▶)** → Console에 `Download verified successfully` = 플래시 성공.
   - ⚠ **Debug(벌레🐞) 말고 Run(▶ 초록 재생)** 을 쓸 것. (Debug는 시작점에서
     멈춰 있어서 `F8`(Resume)을 눌러야 돎.)

### 2-4. 결과 읽기
- **일반 모드**: Tera Term에 예)
  `FEATHer PASS bad=0 maxabs(x1e6)=0 cycles=25174074 latency_us=349639 stack=504 B`
  - `PASS`+`bad=0` = PyTorch와 동일 함수(정확도 OK). `latency_us` = 추론 시간.
  - 값이 매 줄 (거의) 동일 = 결정적(정상). 빌드마다 ~0.2% 차이는 flash 정렬 탓(정상).
- **전력 모드**: 멀티미터의 mA 값을 읽음(§3 참고).

### 2-5. 모델 바꿔 반복
`#define`만 다른 모델로 바꾸고 → Build → Run → 읽기. 끝.

---

## 3. 전력 측정 (하드웨어 연결 필요)

**원리**: 보드의 `JP6`(=`IDD`) 점퍼는 MCU 전원 경로. 캡을 빼고 그 자리에
멀티미터를 **직렬**로 넣으면 MCU가 먹는 전류(mA)를 잰다.

**안전**: MCU 전력은 ~0.1W라 터지거나 타지 않음. 최악 = 멀티미터 퓨즈 나감/보드
안 켜짐(다시 연결). USB 뽑고 작업하면 특히 안전.

### 절차
1. `main.c`에서 `POWER_ACTIVE`(활성) 또는 `POWER_IDLE`(대기) 켜고, 모델 하나 선택
   → **Build → Run** (캡 꽂힌 상태로 먼저 플래시).
2. **멀티미터 세팅**: 다이얼 **`mA`** / 화면 **`DC`**(AC면 SELECT로 토글) /
   **적색 리드 = `mAμA` 잭**(10A 아님) / **흑색 = `COM`**.
3. **USB 뽑기** → **JP6 캡 빼기**(잃어버리지 말 것) → JP6 두 핀에 멀티미터를
   직렬 연결:
   - 점퍼와이어(암-수) 또는 종이클립으로 각 핀에 물림 → 악어클립으로 리드 연결.
   - 핀1 ↔ 적색, 핀2 ↔ 흑색 (극성 반대면 값에 −만 붙음, 무해).
   - ⚠ 두 클립이 서로 닿으면 안 됨(닿으면 전류가 새서 0 근처 = 위험X, 다시).
4. **USB 다시 꽂기** → 화면에 안정된 mA = 그 모델의 활성(또는 대기) 전류.
5. **에너지 계산**: `에너지/추론 = 3.3V × 활성전류(A) × 레이턴시(s)`.
   - 활성 전류는 모델 무관 ~30mA로 확인됨 → 실질적으로 레이턴시가 에너지를 결정.

### 전력 측정 끝나면 (중요!)
- **USB 뽑기 → 멀티미터/클립 제거 → JP6 캡 다시 꽂기 → USB 재연결.**
- ⚠ **캡을 다시 안 꽂으면 다음에 켤 때 MCU에 전원이 안 들어감**(ST-LINK만 켜짐).

---

## 4. 다 하고 나서 (종료 절차)

아무것도 안 잃어버림 — **펌웨어는 flash에 영구 저장**되어 USB 다시 꽂으면 자동 실행.

1. (전력 쟀으면) **JP6 캡 원위치** 먼저.
2. **Tera Term 닫기** (창 X).
3. **CubeIDE**: 실행 세션 살아있으면 **빨간 네모(■ Terminate)** → CubeIDE 닫기
   (저장 물으면 Yes). 안 눌러도 닫을 때 정리됨.
4. **USB 뽑기** — 아무 때나 안전, "안전 제거" 불필요. (모든 LED 꺼짐 = 정상)

---

## 5. 알아두면 좋은 함정/팁 (이번 세션에 겪은 것)

- **LED**: 빨강 `LD3` = 전원(항상 켜짐, 정상). 초록 `LD2` = 우리 코드가 제어하는
  것(이게 깜빡여야 정상). 일반 모드에서 LD2가 매 사이클 5번 빠르게 깜빡이게 해둠.
- **printf에 실수(float)가 안 찍힘**: newlib-nano 기본이라 float 출력 안 됨. 그래서
  오차를 정수(×1e6)로 출력함. 굳이 float 찍고 싶으면 프로젝트 설정에서
  "Use float with printf" 켜야 함(지금은 불필요).
- **최적화 플래그**: 논문 측정은 `-O2 -ffp-contract=off -fno-fast-math`. 프로젝트
  Properties → C/C++ Build → Settings → MCU GCC Compiler → Optimization(=`Optimize
  more (-O2)`) / Miscellaneous → Other flags(fp 플래그)에 설정돼 있음. `-O0`(기본)이면
  훨씬 느리게 나옴 — 반드시 -O2로.
- **글자 깨짐(□□□)**: Tera Term baud가 115200 아님.
- **Tera Term에 아무것도 없음**: 포트 틀림 / 보드 리셋(B2) 눌러보기 / 일반 모드인지
  확인(POWER 모드는 출력 없음).

---

## 6. 현재까지 결과(2026-07-27) & 남은 일

**실측 완료** (canonical p12/453, -O2 soft-float, golden 전부 bit-exact PASS):

| 모델 | latency | 총 flash | 활성전류 | 에너지/추론 |
|---|---|---|---|---|
| persistence | 0.08 ms | 9.6 KB | ~30 mA | 0.008 mJ |
| SparseTSF | 47 ms | 12.4 KB | 30.5 mA | 4.8 mJ |
| **FEATHer** | **349 ms** | **17.6 KB** | 30 mA | **~35 mJ** |
| FITS | 796 ms | 24.0 KB | ~30 mA | 80 mJ |
| DiPE | 1312 ms | 17.0 KB | ~30 mA | 132 mJ |
| 대기(idle) | — | — | 11.4 mA | — |

**남은 일**:
1. **FEATHer 최종 숫자 = period=24(489p)로 재측정** — 서버 config(⑤/그리드서치)
   확정 후. `deployment/board/export_golden.py`로 골든/가중치를 p24로 재생성 →
   `Core/Inc`의 `feather_weights.h`/`feather_vectors.h` 교체 → RUN_FEATHER로 재측정.
   (지금 값은 p12/453 기준.)
2. 논문 §7/§8 배포 섹션 표·문단 작성.
3. (선택) 점퍼와이어(암-수) 도착하면 전력 접촉 안정화 재확인.
