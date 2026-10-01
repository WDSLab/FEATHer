"""선형 예측의 천장 vs FEATHer.

과거 seq_len 값에서 미래 pred_len 값으로 가는 최적 선형 사상을 train 최소제곱으로
직접 구한다. 이것은 어떤 선형 예측기(DLinear/FITS/SparseTSF/MixLinear/DiPE)도
넘을 수 없는 한계선이다. 벤치마크와 같은 공간(train Scaler 표준화, 창별 재정규화
없음)에서 재므로 results/*_raw.csv 의 MSE 와 직접 비교할 수 있다.

두 변형을 함께 보고한다:
  plain   과거 창을 그대로 특징으로 사용
  center  RevIN 처럼 창 평균을 빼고 예측 후 다시 더함 (모델들이 하는 것)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import argparse
import csv
import numpy as np
from utils.data_factory import data_provider

p = argparse.ArgumentParser()
p.add_argument("--data", default="ETTh1")
p.add_argument("--seq_len", type=int, default=96)
p.add_argument("--pred_len", type=int, default=96)
p.add_argument("--max_train", type=int, default=20000)
a = p.parse_args()
L, H = a.seq_len, a.pred_len


def collect(flag, cap):
    ds, dl = data_provider("data/", a.data, "M", 512, L, 48, H, flag,
                           train_shuffle_flag=False)
    xs, ys, n = [], [], 0
    for batch in dl:
        xs.append(batch[0].float().numpy())
        ys.append(batch[1].float().numpy()[:, -H:])
        n += xs[-1].shape[0]
        if cap and n >= cap:
            break
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    if cap:
        X, Y = X[:cap], Y[:cap]
    B, _, D = X.shape
    # 채널독립: 채널을 배치로 접음 (FEATHer LTSF 설정과 동일)
    return X.transpose(0, 2, 1).reshape(B * D, L), Y.transpose(0, 2, 1).reshape(B * D, H)


Xtr, Ytr = collect("train", a.max_train)
Xte, Yte = collect("test", None)
print("train %d / test %d samples (채널 폴딩 후)" % (len(Xtr), len(Xte)))

# NaN 마스킹된 타깃 셀은 제외 (관측-only 채점, MASK-FIX 규약과 동일)
obs = np.isfinite(Yte)


def score(P):
    return ((P[obs] - Yte[obs]) ** 2).mean()


rows = []

# --- plain 선형 천장 ---
A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
ok = np.isfinite(Ytr).all(1)
W, *_ = np.linalg.lstsq(A[ok], Ytr[ok], rcond=None)
rows.append(("선형 천장 (plain)", score(np.concatenate([Xte, np.ones((len(Xte), 1))], 1) @ W)))

# --- center 선형 천장 (RevIN 스타일) ---
mtr = Xtr.mean(1, keepdims=True)
mte = Xte.mean(1, keepdims=True)
A = np.concatenate([Xtr - mtr, np.ones((len(Xtr), 1))], 1)
W, *_ = np.linalg.lstsq(A[ok], (Ytr - mtr)[ok], rcond=None)
P = np.concatenate([Xte - mte, np.ones((len(Xte), 1))], 1) @ W + mte
rows.append(("선형 천장 (center=RevIN식)", score(P)))

# --- 참고 기준 ---
rows.append(("persistence (마지막값 반복)", score(np.repeat(Xte[:, -1:], H, 1))))
rows.append(("창 평균 반복", score(np.repeat(mte, H, 1))))

print()
print("%-32s %10s" % ("방법", "test MSE"))
print("-" * 44)
for nm, v in rows:
    print("%-32s %10.4f" % (nm, v))

# --- 실제 모델 성적 (CSV 에서) ---
print()
print("%-32s %10s" % ("실제 모델 (results/*_raw.csv, 3시드 평균)", "test MSE"))
print("-" * 44)
for csv_path in ("results/ltsf_raw.csv", "results/mfg_raw.csv"):
    if not os.path.exists(csv_path):
        continue
    acc = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["data"] != a.data or r["pred_len"] != str(a.pred_len):
                continue
            if r.get("exp_tag") not in ("main", "final"):
                continue
            try:
                acc.setdefault(r["model"], []).append(float(r["MSE"]))
            except ValueError:
                pass
    for m, v in sorted(acc.items(), key=lambda kv: np.mean(kv[1])):
        print("%-32s %10.4f  (n=%d)" % (m, np.mean(v), len(v)))
