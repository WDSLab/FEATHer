# 엣지 디바이스 구현실험 (Layer 3 — 실보드 실측)

**이 문서가 이 트랙의 단일 소스.** 서버 실험 체인(ROADMAP.md)·CLAUDE.md와 독립.
**손으로 보드 측정하는 실전 절차(재시작/측정/종료)는 `deployment/board/BOARD_RUNBOOK.md`.**
목적: 논문 기여 4번("Cortex-M3 배포 증거")을 추정치가 아닌 **실측 숫자**로 채운다
(#1 구멍). 서버 체인과 완전 병행 가능 — 레이턴시/SRAM/flash는 가중치 값과
무관하므로 canonical 체크포인트(⑤ 이후)를 기다릴 필요 없음.

_상태: 2026-07-27 — **★ 보드 도착. 개통 + FEATHer 실보드 실측 성공**(3주차 ✅ +
4주차 레이턴시/SRAM ✅). **5모델 전부 실측 완료** (golden 전부 bit-exact PASS):
persistence 0.08ms / SparseTSF 47ms / **FEATHer(p12) 349ms** / FITS 796ms /
DiPE 1312ms @72MHz -O2 soft-float. 파라미터순≠레이턴시순(DiPE 339p가 최장).
프로젝트 = `C:\Users\lee\STM32Cube_projects\blink_test`. **남음 = 전력 측정(DMM
도착✅+JP6) + .map서 모델별 static SRAM/flash 3분할 + p24 FEATHer 최종 재측정.**
상세는 문서 맨 아래 07-27 로그. (이하 07-22 스냅샷) —
1~2주차(x86 파트) 완료. 보드(F103RB)
+ 미니B 케이블 확보; **멀티미터는 UT33C+ 단종 취소 → ZOYI ZT-102로 대체 구매
완료**(쿠팡, 바나나-악어클립 리드 포함 — 07-22 로그 참조). F411RE 미구매.
5모델(FEATHer/SparseTSF/DiPE/FITS/persistence) C 포팅 + 골든 + 100윈도우
배치 전부 PASS, FEATHer 아레나 13.6KB ≤ 16KB. 코드 = `deployment/board/`
(README 참조). 보드 도착 시 §5 체크리스트 → 3주차. 진행 로그는 문서 맨 아래._

---

## 1. 확정 결정 (2026-07-18 전문가 감수 채택)

| # | 결정 | 이유 |
|---|---|---|
| 1 | **float32 soft-float가 메인 서사. int8 양자화 안 함** (리비전 카드로만) | M3 soft-float로도 추론 ~0.2초 = 1분 샘플링 예산의 ~1/300. 양자화하면 robustness 스윕 재실행까지 끌려 들어가 6주 붕괴. float 유지 → 보드 모델 = 논문 모델(같은 함수) → 서버 robustness 결과가 그대로 성립 |
| 2 | **FFT는 정확한 96점 변환** (naive DFT 또는 mixed-radix). **128 제로패딩 절대 금지** | 패딩 시 빈 49→65인데 `AdaptiveAvgPool1d(1)` 탓에 에러 없이 조용히 다른 게이트 값 → 논문 모델과 다른 함수를 실측하게 되는 폭탄. 96점 DFT는 트위들 ~768B, soft-float ~0.1초로 충분 |
| 3 | **SparseTSF 동반 실측 필수** (+여유 시 DiPE-Linear, persistence; FITS는 추정치 각주) — *07-21e 갱신: 5모델 전부 포팅 완료돼 FITS도 실측 옵션 행으로 승격 가능; DLinear↑는 의도적 미포팅 유지(2층 프레이밍)* | "41p 대비 453p의 가치=robustness" 논리가 배포 표에서 끊기지 않게. 선형 모델이라 공용 프리미티브 재사용 시 각 1–2일 |
| 4 | 포팅 성공 기준 = **혼합 허용오차**(요소별 atol=1e-5 + rtol=1e-4) + 테스트 100윈도우 MSE 소수 4자리 일치 | "상대오차<1e-5" 단독 기준은 0 근처 출력에서 반드시 깨짐 |

**안 올리는 모델도 논문 메시지**: DLinear(18K)↑ 전부 미배포 = 2층 프레이밍
(실무층 vs 엣지층)의 증거. iTransformer(84만)는 메모리부터 불가.

**보드에서 하는 것/안 하는 것**: 예측 실행 O(그게 측정 대상) / 테스트셋 정확도
채점 X(골든 벡터 10–100윈도우로 "PC와 같은 함수" 입증, 정확도 표는 PyTorch 숫자)
/ 학습 X(가중치는 flash 상수로 소성).

## 2. 하드웨어

| 보드 | 칩 | 역할 |
|---|---|---|
| **NUCLEO-F103RB** | STM32F103RB — Cortex-M3, **FPU 없음**, 72MHz, 20KB SRAM / 128KB flash | **주력 실측** (논문 서사 그 자체). DWT CYCCNT·온보드 ST-LINK·IDD 점퍼(JP6) 보유 |
| **NUCLEO-F411RE** | STM32F411RE — Cortex-M4F(FPU), 100MHz, 128KB SRAM / 512KB flash | 포팅 검증 사다리 (품절 시 F401RE 대체 OK) — **07-21 미구매 결정: 3주차에 막히면 그때 추가 주문** (soft-float도 IEEE-754 준수 → x86 골든 벡터를 레이어별로 잘게 쪼개면 대부분 커버) |

- 배포 대상 데이터셋: **MetroPT3** (D=7, 1분 샘플링 → 레이턴시 예산 널널; 453p 인용과 정합)
- 구매 시: "재고보유" 리스팅으로(주문수입=10–15일), **USB 미니B 케이블**(데이터용) 동봉 안 되니 별도, 멀티미터(mA DC)는 4주차 전까지만 있으면 됨
- Blue Pill 금지(클론 칩 → 논문 실측 신뢰성 훼손)

## 3. 왜 하는가 (요약)

1. 기여 4번을 주장→증거로 (기여 4개짜리 논문 유지)
2. JMS 독자층(제조 현장)에는 "2만원 칩에서 X ms, Y mJ"가 신뢰 신호
3. 경량 LTSF 경쟁 논문(SparseTSF/FITS/SWIFT) 중 실칩 실측 없음 → 차별점
4. 공격 방어: "라즈베리파이 쓰면 되잖아" → 전력 실측으로 답 / "41p가 더 싸잖아" → 동반 실측 표가 robustness 논지로 가는 다리
5. Layer-1 추정기 교정("Layer 3가 Layer 1을 검증"으로 프레이밍)

## 4. 6주 로드맵

| 주차 | 작업 | 통과 기준 | 상태 |
|---|---|---|---|
| 1 | 공용 C 프리미티브(depthwise conv1d 인과/same, linear, instance-norm, **96점 DFT**, 3원소 softmax, 선형보간 `align_corners=False`) + **x86 골든 벡터 대조** | 혼합 허용오차 통과; §6 함정 4종 단위 테스트 | ✅ 07-21b (20스테이지 PASS) |
| 2 | FEATHer forward 조립(버퍼 스케줄 문서화, 피크 ≤16KB 사전 설계 — 20KB의 70–80% 참) + 가중치 export(로컬 스모크 ckpt로 충분) | x86에서 100윈도우 MSE 소수 4자리 일치 | ✅ 07-21f (아레나 13.6KB; MSE |diff|≤3e-8) |
| 3 | M3 크로스 빌드(`-O2 -ffp-contract=off -fno-fast-math`; F411RE 미구매라 x86 골든 레이어별 격리가 사다리 역할 대체) | M3 링크 성공 + map으로 SRAM 예산 확인 | ✅ 07-27 (CubeIDE 프로젝트 빌드+플래시+golden PASS) |
| 4 | **M3 실측**: DWT 사이클, 스택 페인팅 peak SRAM, size/map 3분할, IDD+DMM → 추론당 mJ + duty-cycle 에너지 | 사이클 분산 ≈ 0(결정성); Layer-1 추정 대비 오차율 기록 | 🔶 07-27 부분: FEATHer 레이턴시(25.1M cyc≈349ms)·peak SRAM(528B)·결정성(±1cyc) 완료 / **전력(mJ)+size·map 3분할 남음** |
| 5 | SparseTSF + DiPE-Linear + FITS + persistence 동일 파이프라인 | 모델별 동일 표 | ✅ 07-27 보드 실측 완료 (5모델 레이턴시+golden PASS; 표는 07-27 로그) |
| 6 | 표/그림 + 논문 배포 섹션 + 재현 스크립트 커밋 | 표 초안 완성 | — |

## 5. 보드 도착 시나리오 (D-Day 체크리스트)

각 단계는 직전 단계 성공이 전제. 막히면 단계 이름으로 보고.

- **도착 전**: ✅ 전부 완료 (1~2주차 + STM32CubeIDE + Tera Term 설치, 07-21)
- **D-0 개통 (F103RB 단독 플랜, ~1시간)**:
  1. USB 연결 → 탐색기에 "NODE_F103RB" 드라이브 뜸 (Nucleo는 .bin 드래그=플래싱; 안 뜨면 충전전용 케이블 의심)
  2. LED 블링크 예제 빌드+플래시 → LD2 깜빡임 = 빌드→플래시→실행 파이프라인 검증
  3. printf → ST-LINK 가상 COM(115200) → Tera Term에 "hello" = 모든 측정 결과의 통로
- **D+1 자(측정 도구)부터 검증**:
  4. DWT CYCCNT 스모크: 결과를 아는 루프 측정 → 두 번 재서 **정확히 같은 값**(다르면 IRQ 안 죽인 것)
  5. 스택 페인팅 스모크(0xAA 패턴)
- **D+2~**: 골든 벡터 펌웨어(입력·기대출력 flash 상수, UART로 PASS/FAIL; 올리기 전 map으로 SRAM 확인) → §4의 4주차 실측. **수치 불일치로 막히면 그때 F411RE 추가 주문**(재고 리스팅 며칠)이 격리 사다리.

## 6. 골든 벡터 함정 4종 (전부 x86에서 잡을 것)

1. LOW 밴드 `F.interpolate(mode="linear", align_corners=False)` — 보간 규약 `src=(dst+0.5)·scale−0.5`
2. DTK `padding=k-1` 후 `[:L]` 트림 (인과 conv)
3. half-RevIN `sqrt(var + 1e-5)` — **eps가 std가 아니라 var에** 붙은 뒤 sqrt. `detach()`는 학습 전용이라 추론 수식엔 무흔적 = 배포 입장에선 표준 인스턴스 정규화(구조 변경 불가 제약의 배포 비용 0)
4. rfft(96) 재현 — 구현 후 PyTorch `torch.fft.rfft` 대비 요소별 단위 테스트

기타: spectral loss는 학습 전용 → 펌웨어 불필요. softmax는 3원소뿐(비쟁점).

## 7. 측정 프로토콜 (논문 표 각주에 들어갈 것)

- 고정 클럭(72MHz) + flash wait-state(2WS)/prefetch 설정 명기, 모든 IRQ off, systick off
- gcc 버전 + 플래그(`-O2 -ffp-contract=off -fno-fast-math`) 명기, 두 보드 동일
- 원시 사이클 수를 표에 병기(클럭 무관 재현성), 1000회 반복 — **분산≈0이 정상**(M3 무캐시 결정성; 분산 크면 측정 버그) → 결정성 자체를 셀링 포인트로
- 입력 윈도우 ~10개 flash 상수로(UART 수신 시간 배제), 측정 범위 = RevIN 정규화+역변환+FFT **포함 end-to-end**
- 전력: IDD 점퍼(JP6) + DMM 평균 전류 × 3.3V × 레이턴시 = 추론당 mJ; duty-cycle(슬립 포함) 에너지도 계산 → "왜 이 칩" 방어
- Layer-1 추정기: `cycles_per_mac_fp32=9.0`은 낙관적(실제 soft-float 30–100cyc/MAC) → 괴리를 "실측이 추정기를 교정"으로 서술 + F103RB(72MHz/20KB/128KB/2WS) 프로파일을 `op_costs.py`에 추가(현재 LM3S6965 기준)

## 8. 논문 표기 규칙

- "**453 learnable parameters**; total flash image XX KB" — sub-1K는 학습 파라미터로 한정
- flash 3분할 표: learnable(1,812B fp32) / const tables(트위들 ~768B) / code(soft-float 런타임 포함). FITS류도 같은 테이블 비용 → 3분할에서 불리하지 않음; SparseTSF 테이블 0은 숨기지 말고 표에 노출
- H=96·D≤14 스코프 문장을 배포 섹션에도 재기재
- 예상 리뷰어 공격 3: contrived deployment(→duty-cycle 에너지로 정량 방어) / 논문 모델↔배포 모델 불일치(→float 배포+골든 벡터로 봉쇄) / sub-1K vs 이미지 크기(→3분할+스코핑)

---

## 진행 로그 (최신이 위)

- **2026-07-27**: **★ 보드 도착 → 개통 + FEATHer 실보드 실측 성공 (3주차 완료
  + 4주차 레이턴시/SRAM 완료).** 프로젝트 = `C:\Users\lee\STM32Cube_projects\
  blink_test` (repo 밖, IDE 워크스페이스).
  - **D-0 개통**: `NODE_F103RB` 인식 → LED 블링크(빌드→플래시→실행 파이프라인) →
    printf→USART2(115200)→Tera Term "hello" 통로 확보. (COM3)
  - **툴체인 함정**: STM32CubeIDE **2.2.0은 CubeMX 통합 제거**(2.0.0부터) →
    새 프로젝트는 **독립 실행형 STM32CubeMX(숫자 없는 것, HAL1; MX2는 STM32C5+
    전용이라 F1 미지원)** 로 Board Selector→NUCLEO-F103RB 생성, Toolchain=
    STM32CubeIDE. 클럭 72MHz(HSE8×PLL9)/`FLASH_LATENCY_2` 자동 생성 확인(=§7 요구).
  - **D+1 측정도구 검증**: DWT `CYCCNT`(IRQ off 시 지터 0=결정적) + 스택 페인팅
    (0xAA, `_end`~SP 채우고 잔존 0xAA 스캔). 둘 다 결정성 확인.
  - **D+2 포팅**: `fea_kernels.{c,h}`+`feather_infer.{c,h}`+골든 헤더 3개(weights/
    vectors/twiddle)를 `Core/{Src,Inc}`에 복사. 보드는 `-DFEA_CAPTURE_STAGES`
    **끄고** compact arena 경로(≈13.9KB)만 사용 → 최종 출력 `y`만 `G_Y`와 혼합
    허용오차 비교(스테이지별은 메모리 초과라 x86서 이미 완료). 진입점=
    `feather_forward_ip()` 직접 호출(mutable 입력 복사 후), 래퍼 xbuf는 gc-sections로 제거.
  - **실측 (canonical p12/453p, `-O2 -ffp-contract=off -fno-fast-math`)**:
    **golden PASS, bad=0, bit-exact**(max_abs < 5e-7) / **25,126,919 cycles ≈
    349 ms @72MHz** / **±1 cycle 결정적** / **peak stack 528 B**. duty cycle =
    0.58%(60s 샘플링) → §1 "예산의 1/300" 논리 실측으로 성립(추정 0.2s, 실측 0.35s).
  - **발견 (논문용)**: -O0→-O2가 **14%만** 단축(29.3M→25.1M) = FPU 없는 M3에서
    모든 float op이 **libgcc soft-float 호출**(`__aeabi_fmul` 등)로 처리돼 실행
    시간을 지배 → 우리 -O2는 정수·루프 오버헤드만 깎음. **"연산량=레이턴시"**
    실증 → §7 `cycles_per_mac` 추정기 교정 서사와 직결.
  - **printf 함정 회피**: newlib-nano 기본은 float 미출력 → 오차를 정수(×1e6)로
    출력해 `-u _printf_float` 설정 불필요하게 함.
  - ⚠ **최종 논문 숫자는 MetroPT3 배포 셀 period=24(489p)로 골든 재생성 후**
    (가중치 헤더만 교체; ⑤ 재동결/그리드서치 확정 뒤). 이번엔 **포팅·측정
    파이프라인이 실보드에서 작동함을 실증**한 것(제일 어려운 관문 통과).
  - **★ 5모델 실보드 레이턴시 표 완성 (F103RB @72MHz, -O2 soft-float, golden
    전부 bit-exact PASS bad=0):**

    | 모델 | params | cycles | latency | peak stack |
    |---|---|---|---|---|
    | persistence | 0 | 5,573 | 0.08 ms | ~136 B |
    | SparseTSF | 41 | 3,412,788 | 47 ms | (printf 지배) |
    | **FEATHer (ours, p12)** | **453** | 25,126,919 | **349 ms** | 528 B |
    | FITS | 1,200 | 57,328,304 | 796 ms | (printf 지배) |
    | DiPE | 339 | 94,499,910 | 1,312 ms | (printf 지배) |

    **핵심 발견: 파라미터 순서 ≠ 레이턴시 순서** — DiPE(339p)가 최장(193pt DFT),
    FITS(1200p)도 FEATHer보다 느림 → "sub-1K라도 설계가 레이턴시를 결정"
    논거. 전부 ±수 cycle 결정적. **RAM 함정**: companion `*_st_*`는 이름만
    스테이지지 실제 작업버퍼라 `#ifdef` 제거 불가 → 4개 동시는 20KB 초과,
    **모델당 1펌웨어**(main.c `#define RUN_STSF/DIPE/FITS/PERS` 스위치, gc-sections가
    호출 모델만 링크)로 측정. FEATHer는 반대로 companion 제외하고 측정.
  - **★ 5모델 메모리 풋프린트 (`size` on -O2 .o, 07-27):**

    | 모델 | learnable params | const tables | code(모델+공유커널) | working RAM(.bss) |
    |---|---|---|---|---|
    | persistence | 0 | 0 | ~64 B | ~0 |
    | SparseTSF | 164 B (41p) | 0 | 3,532 B | 5,788 B |
    | **FEATHer** | **1,812 B (453p)** | **768 B** | 6,240 B | 13,904 B |
    | DiPE | 1,356 B | 2,312 B | 7,352 B | 10,440 B |
    | FITS | 9,600 B (1,200 복소p) | 1,536 B | 14,796 B | 7,360 B |

    공유 커널 `fea_kernels` 2,980B는 4모델 공유; soft-float 런타임/HAL/startup
    고정 베이스라인(~10-15KB)은 제외(증분 비교값). **§8 수치 실측 검증**:
    FEATHer learnable **정확히 1,812B(=453×4)** + 트위들 **정확히 768B** → 문서
    추정과 일치. **FITS 9,600B = sub-1K 아님(배제 논리 강화)**; SparseTSF 테이블
    0은 노출. FEATHer working RAM 13,904B(arena 11,216 + 입력복사 2,688)가 최대이나
    20KB 내. ⚠ companion `.bss`엔 실사용 `*_st_*` 포함(순수-inspection 섞였으면
    소폭 과대 가능 → 최종 표 전 순 arena 재확인).
  - ⚠ **peak stack 값 주의**: 현재 출력의 stack은 **printf가 지배**(~504–528B)해
    모델별 순 추론 스택이 아님. 논문용 정확한 SRAM 풋프린트 = 각 모델 static
    .bss(작업버퍼) + 순수 추론 스택 → **`.map`/`size`에서 재추출 필요**(TODO).
  - **★ 전력 실측 (JP6/IDD 직렬, ZOYI ZT-102 DC mA; 악어클립 직결 — 점퍼와이어
    미보유라 임시):** 활성 **30.5 mA**(101 mW @3.3V), 대기(WFI+SysTick) **11.4 mA**
    (37.6 mW). STM32F103 datasheet run@72MHz 전형(25-37mA)과 정합 → 측정 유효.
    **추론당 에너지 = 3.3V × 30.5mA × latency**(활성전류는 CPU+flash 지배라 모델
    무관 가정, SparseTSF서 실측 확인; FEATHer 실측 확인 TODO):

    | 모델 | latency | 에너지/추론 |
    |---|---|---|
    | persistence | 0.08 ms | ~0.008 mJ |
    | SparseTSF | 47 ms | 4.8 mJ |
    | **FEATHer** | 349 ms | **~35 mJ** |
    | FITS | 796 ms | ~80 mJ |
    | DiPE | 1312 ms | ~132 mJ |

    **핵심 서사**: 60s duty-cycle 기준 추론 35mJ는 그 1분 대기 에너지(37.6mW×60≈
    2.25J)의 **~1.5%** → "예측 연산은 사실상 공짜, 대기 전력이 지배". "왜 MCU냐"
    답 = STOP 모드(µA대)로 배터리 수개월 자율. ⚠ 11.4mA=WFI+SysTick "대기 상한",
    실배포 STOP은 µA대(리비전 한 문장). 펌웨어: main.c `#define POWER_ACTIVE`
    (연속추론=안정전류) / `POWER_IDLE`(WFI) 스위치 + `RUN_FEATHER`(companion 제외
    대신 feather 포함).
  - **✅ 활성전류 5모델 전수 실측 = 전부 29-31mA** (SparseTSF 30.5 / FEATHer 30 /
    DiPE 28-31 / FITS 29-31 / persistence 30-31) → "활성전류 모델 무관(CPU@72MHz+
    flash 지배, 연산종류 무관)" **가정 아니라 전수 실측**(리뷰어 방어). 에너지=
    3.3V×30mA×latency: pers 0.008 / STSF 4.8 / **FEATHer 35** / FITS 80 / DiPE 132 mJ.
  - **★ 총 flash 이미지 실측 (모델별 재링크 → `size` .elf; text+data, HAL+soft-float
    런타임+newlib+startup 전부 포함):** persistence **9.6KB** / SparseTSF **12.4KB** /
    DiPE **17.0KB** / **FEATHer 17.6KB** / FITS **24.0KB** (전부 128KB의 8-19%).
    고정 베이스라인 ≈9.6KB(=persistence); 증분 STSF+2.8 / DiPE+7.4 / FEATHer+8 /
    FITS+14.8KB. **§8 문장**: "FEATHer 453 learnable params(**1,812B**) + twiddle
    **768B**; total flash image **~17.6KB**". FITS 24KB(params 9,600B=FEATHer 5배)=
    sub-1K 아님·배포부담 큼 논거. ⚠ printf/UART/측정 HAL 포함된 디버그빌드라 **상한**
    (실배포 스트립 시 축소).
  - **남음**: (c) FEATHer 최종 숫자는 p24(489p) 골든 재생성 후 재측정(서버 config
    확정 시; 가중치 헤더만 교체), (d) 점퍼와이어 F-M 도착 후 접촉 안정화 재측정
    (현재 악어클립 직결은 값 소폭 흔들림), (e) §7/§8 프로토콜대로 논문 배포 섹션
    표·문단 작성. **latency·memory·flash·power 실측은 이번 07-27 세션에 전부 완료.**
- **2026-07-22**: **멀티미터 교체 (UT33C+ 단종).** 디바이스마트에서 UT33C+
  단종 통보 → 해당 품목만 자동 취소(익영업일 17시 원결제 환불), **보드는
  정상 발송 예정**(하루 이틀 내 송장 안 뜨면 070-7019-8887 내선 1 확인).
  대체 검토: UT136B+/UT33D+/UT131 계열 전부 디바이스마트 품절 → **쿠팡에서
  ZOYI ZT-102 구매**(6000카운트 오토레인지, DC 전류 60mA/600mA/6A/10A
  ±(1.2%+3), 이중 퓨즈). 우리 측정점(~25mA)은 **60mA 레인지 0.01mA 분해능**
  — UT33C+(200mA 레인지 0.1mA)보다 분해능 우위, 논문 표기 "ZOYI ZT-102
  (6000-count) handheld DMM". **측정 연결재도 함께 구매**: 4mm 바나나플러그↔
  악어클립 리드 세트(Cleqee 5P 또는 JM-012 적/흑). 4주차 IDD 측정 연결 =
  JP6 캡 제거 → 핀1↔적색리드↔DMM mA 잭 / 핀2↔흑색리드↔COM 잭 (핀에 암수
  점퍼와이어 꽂고 악어클립으로 물기; 점퍼와이어는 연구실 보유분). 구매 완결
  — 남은 것 = 배송 대기뿐.
- **2026-07-21f**: **2주차 완료 (보드 도착 전 x86 파트 전부 종료).**
  (1) **버퍼 스케줄**: FEATHer forward를 아레나 구조로 재작성 — DTK 인과
  conv 링버퍼 인플레이스, 게이트 선계산 후 밴드 즉시 누적, LAT 버퍼 4중
  재사용(게이트 스펙트럼/풀링/agg-conv) → **정적 아레나 11,216B + 입력
  2,688B = 13.6KB** (목표 ≤16KB, SRAM 20KB의 68%). 스테이지 캡처는
  `-DFEA_CAPTURE_STAGES` 스위치로 동일 코드 경로 유지 → 골든 20스테이지
  재통과. (2) **100윈도우 배치 MSE**(결정 #4 후반): 5모델 전부 C vs
  PyTorch MSE 일치 |diff| 1e-8~1e-11 (요구는 소수 4자리, 실측은 7자리+).
  남은 것 = 보드 도착 후 3주차(CubeIDE/M3 빌드)부터.
- **2026-07-21e**: **FITS까지 포팅 — 배포 표 후보 5모델 전부 x86 ALL
  PASS** (사용자 결정: "들어갈만한 건 다 만들어두기" → FITS 각주를 실측
  옵션 행으로 승격 가능). FITS = 1,200 복소 params(레지스트리 일치),
  rfft96→복소 Linear(24→48)→irfft(192) 경로, 192점 트위들 추가.
  DLinear(18K)↑는 **의도적으로 포팅 안 함 유지**(2층 프레이밍의 증거 —
  올리면 스토리 자해). 라인업 확정: FEATHer 453 / SparseTSF 41 /
  DiPE 339(복소194) / FITS 1200(복소) / persistence 0.
- **2026-07-21d**: **동반 실측 라인업 완성 — DiPE-Linear + persistence도
  x86 ALL PASS.** 배포 표 4모델(FEATHer 453p / SparseTSF 41p / DiPE 339p /
  persistence 0p) C 포팅 + 골든 검증 전부 완료. DiPE는 193점 DFT(복소
  가중+바이어스) 경로 포함, generic DFT/iDFT 프리미티브를 `fea_kernels`에
  추가. ⚠ DiPE 파라미터 실측 = **339** (복소 194개 포함; flash 기준 533
  floats) — 레지스트리 표의 "267"과 다름, 논문 표 각주에서 카운팅 기준
  (복소=2 floats) 명시 필요. FITS는 결정대로 포팅 안 함(추정치 각주).
- **2026-07-21c**: **SparseTSF 동반 포팅도 x86 ALL PASS** (결정 #3을 5주차
  에서 당김 — 보드 불필요 파트). 실제 upstream 모델 + MetroPT3 배포 셀
  (`sparsetsf_period_len=24`) 기준 **정확히 41 params**. 프리미티브를
  `src/fea_kernels.c`로 분리(모델별 펌웨어 이미지가 자기 가중치만 링크).
  ⚠ 발견: `_DATASET_OVERRIDES`상 FEATHer MetroPT3 배포 셀은 **period=24**
  (=489 params) — 현 골든은 canonical p12/453으로 프리미티브 검증, **최종
  보드 빌드 전 p24로 상수 2줄 바꿔 재생성 필요**(⑤ 재동결 후 확정).
- **2026-07-21b**: **1주차 핵심 완료 — x86 골든 벡터 ALL PASS.**
  `deployment/board/` 신설(README 참조): `export_golden.py`(PyTorch →
  레이어별 골든 벡터 + 가중치 + DFT 트위들 헤더) + `src/feather_infer.c`
  (float32 프리미티브 + full forward) + `test/test_golden.c`(혼합
  허용오차 하니스). canonical config 453 params 일치 확인, forward
  스테이지 20개 전부 PASS, §6 함정 4종 전부 골든으로 커버. PC 준비도
  완료(Tera Term 5.6.2 설치, STM32CubeIDE 설치 완료). 남은 것 = 2주차:
  100윈도우 배치 MSE + 버퍼 스케줄 ≤16KB 설계(현재 분리 버퍼 ~48KB).
- **2026-07-21**: **주문 완료** — NUCLEO-F103RB(디바이스마트 2만원) + UNI-T
  UT33C+ 멀티미터(23,100원, UT120C 품절 대체; DC 200mA 레인지 확인) + USB
  미니B 데이터 케이블(쿠팡, 디바이스마트 케이블류 품절/수입 2주라 우회).
  **F411RE는 미구매 결정** — soft-float도 IEEE-754 준수라 사다리의 격리 가치가
  당초 평가보다 작음; 3주차에 원인 불명 불일치로 막히면 그때 추가 주문(재고
  리스팅 며칠 소요). 다음 = 1주차(x86 골든 벡터) 착수 + CubeIDE/Tera Term 설치.
- **2026-07-20**: 본 문서 생성(이 트랙의 단일 소스로 분리). 보드 주문 진행 중.
- **2026-07-18**: 전문가 감수 → 4대 결정 확정(float 메인/96점 DFT/SparseTSF 동반/혼합 허용오차). 6주 로드맵 수립.
