# 2026-08-31 아카이브 — 모델 확정 직전

여기 있는 표는 전부 **옛 FEATHer(in_proj 있는 ds16_k7_B4)** 기준이라 무효.
밴드축 실험으로 모델이 확정되면 전부 재생성된다.

또한 이 표들은 다음 두 가지가 **반영되기 전** 상태다:
- DLinear ETTh2 lr 발산 수정 (4H 평균 0.5440 -> 0.4468)
- SparseTSF-MLP 편입 (mfg 24칸, OVERALL 0.7271) — 배포집합 경쟁자인데 빠져 있었음

`diagnostics/*.py` (linear_ceiling, gate_init, window_stdev 등)는 계속 쓰는 도구라
아카이브하지 않았다.
