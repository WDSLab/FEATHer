# -*- coding: utf-8 -*-
"""선형 천장을 전 벤치마크 칸에 대해 구해 CSV 로 낸다.

    python analysis/diagnostics/linear_ceiling_sweep.py            # mfg+LTSF 56칸
    python analysis/diagnostics/linear_ceiling_sweep.py --validate # 기존 도구와 대조

`linear_ceiling.py` 와 **같은 정의**다: 과거 L 값에서 미래 H 값으로 가는 최적
선형 사상을 train 최소제곱으로 직접 구한 것으로, 어떤 선형 예측기도 넘을 수
없는 한계선. 채널은 배치로 접는다(채널독립 = DLinear/FITS/SparseTSF 계열의
구조). 벤치마크와 같은 공간에서 재므로 results/*_raw.csv 의 MSE 와 직접 비교
가능하고, 관측(NaN 아님) 셀만 채점한다(MASK-FIX 규약).

구현만 다르다: 원본은 전체 표본을 메모리에 올려 lstsq 를 부르는데, 채널을 접으면
Traffic(862ch)은 수천만 행이 되어 못 돌린다.  여기서는 정규방정식을 스트리밍으로
누적한다 - A^T A 는 (L+1)x(L+1), A^T Y 는 (L+1)xH 이므로 표본 수와 무관하게
메모리가 일정하다.  float64 로 누적하고, 같은 칸에서 원본과 값이 일치하는지
--validate 로 확인한다.
"""
import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils.data_factory import data_provider  # noqa: E402

MFG = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange",
        "Electricity", "Traffic"]
HS = [96, 192, 336, 720]


def _fold(batch, L, H):
    """(B, L, D) / (B, *, D) -> 채널을 배치로 접어 (B*D, L), (B*D, H)."""
    x = batch[0].float().numpy()
    y = batch[1].float().numpy()[:, -H:]
    B, _, D = x.shape
    return (x.transpose(0, 2, 1).reshape(B * D, L),
            y.transpose(0, 2, 1).reshape(B * D, H))


def fit(data, L, H, max_windows):
    """train 에서 정규방정식을 누적해 plain / center 두 사상을 푼다."""
    G = np.zeros((L + 1, L + 1))          # A^T A
    Cp = np.zeros((L + 1, H))             # A^T Y      (plain)
    Cc = np.zeros((L + 1, H))             # A^T (Y-m)  (center)
    Gc = np.zeros((L + 1, L + 1))
    seen = 0
    _, dl = data_provider("data/", data, "M", 512, L, 48, H, "train",
                          train_shuffle_flag=False)
    for batch in dl:
        X, Y = _fold(batch, L, H)
        ok = np.isfinite(Y).all(1) & np.isfinite(X).all(1)
        X, Y = X[ok], Y[ok]
        if not len(X):
            continue
        A = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
        G += A.T @ A
        Cp += A.T @ Y.astype(np.float64)
        m = X.mean(1, keepdims=True)
        Ac = np.concatenate([X - m, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
        Gc += Ac.T @ Ac
        Cc += Ac.T @ (Y - m).astype(np.float64)
        seen += batch[0].shape[0]
        if max_windows and seen >= max_windows:
            break
    Wp = np.linalg.lstsq(G, Cp, rcond=None)[0]
    Wc = np.linalg.lstsq(Gc, Cc, rcond=None)[0]
    return Wp, Wc, seen


def evaluate(data, L, H, Wp, Wc):
    """test 를 스트리밍하며 관측 셀만 채점한다."""
    acc = dict(plain=0.0, center=0.0, persistence=0.0, winmean=0.0)
    n = 0
    _, dl = data_provider("data/", data, "M", 512, L, 48, H, "test",
                          train_shuffle_flag=False)
    for batch in dl:
        X, Y = _fold(batch, L, H)
        obs = np.isfinite(Y)
        fin = np.isfinite(X).all(1)
        obs &= fin[:, None]
        if not obs.any():
            continue
        Xs = np.nan_to_num(X).astype(np.float64)
        A = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
        m = Xs.mean(1, keepdims=True)
        Ac = np.concatenate([Xs - m, np.ones((len(Xs), 1))], 1)
        preds = dict(plain=A @ Wp,
                     center=Ac @ Wc + m,
                     persistence=np.repeat(Xs[:, -1:], H, 1),
                     winmean=np.repeat(m, H, 1))
        for k, P in preds.items():
            acc[k] += ((P[obs] - Y[obs]) ** 2).sum()
        n += int(obs.sum())
    return {k: v / max(n, 1) for k, v in acc.items()}, n


def run_cell(data, H, L=96, max_windows=20000):
    Wp, Wc, seen = fit(data, L, H, max_windows)
    sc, n = evaluate(data, L, H, Wp, Wc)
    sc.update(data=data, pred_len=H, seq_len=L,
              train_windows=seen, test_cells=n,
              ceiling=min(sc["plain"], sc["center"]))
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="쉼표 목록 (기본: mfg+LTSF 전부)")
    ap.add_argument("--pred_len", default=",".join(str(h) for h in HS))
    ap.add_argument("--max_train", type=int, default=20000,
                    help="train 창 수 상한 (채널 접기 전 기준)")
    ap.add_argument("--out", default="results/linear_ceiling.csv")
    ap.add_argument("--validate", action="store_true",
                    help="기존 linear_ceiling.py 가 보고한 칸을 재현하는지 확인")
    a = ap.parse_args()

    if a.validate:
        # 2026-09-09 에 원본 linear_ceiling.py 를 직접 돌려 얻은 값.
        # ⚠ CLAUDE.md 에는 0.3996 ("문헌 DLinear 0.399 와 일치")로 적혀 있는데
        # 원본 도구는 오늘 0.3818(plain) / 0.3835(center) 를 낸다.  기록 쪽이
        # 문헌 숫자를 옮겨 적은 것으로 보인다 - 천장은 기록보다 낮다(= 선형이
        # 더 강하다).  기준값은 도구 출력으로 잡는다.
        REF = {("ETTh1", 96): 0.3818}
        print("칸               구현      기존 기록   차이")
        for (d, h), ref in REF.items():
            sc = run_cell(d, h)
            got = sc["ceiling"]
            print(f"{d} H={h:<4} {got:8.4f} {ref:10.4f} {abs(got-ref):8.4f}"
                  f"   {'OK' if abs(got-ref) < 0.005 else '불일치'}")
        return

    datasets = ([s.strip() for s in a.data.split(",")] if a.data else MFG + LTSF)
    hs = [int(s) for s in a.pred_len.split(",")]
    rows = []
    cols = ["data", "pred_len", "seq_len", "ceiling", "plain", "center",
            "persistence", "winmean", "train_windows", "test_cells"]
    out = os.path.join(_ROOT, a.out)
    for d in datasets:
        for h in hs:
            try:
                sc = run_cell(d, h, max_windows=a.max_train)
            except Exception as e:                      # noqa: BLE001
                print(f"{d:<13} H={h:<4} SKIP  {type(e).__name__}: {e}")
                continue
            rows.append(sc)
            print(f"{d:<13} H={h:<4} ceiling {sc['ceiling']:.4f} "
                  f"(plain {sc['plain']:.4f} / center {sc['center']:.4f})  "
                  f"pers {sc['persistence']:.4f}")
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    print(f"\n-> {out}  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
