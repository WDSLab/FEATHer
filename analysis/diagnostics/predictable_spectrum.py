"""미래에 대해 예측 가능한 정보는 스펙트럼의 어디에 있는가?

과거 창의 rFFT 계수(실수부/허수부)에서 미래 H스텝으로 가는 '최적 선형 사상'을
train 최소제곱으로 구하고 test 에서 채점한다. 사용 대역을 넓혀가며 재면 예측력이
어느 대역에서 나오는지가 보인다. 최소제곱은 그 특징 집합으로 도달 가능한 선형
예측의 천장이므로, 모델 구조와 무관하게 "그 정보가 애초에 있는가"를 말해준다.

두 가지 함정을 피한다 (2026-08-27, 초판에서 둘 다 밟았음):
  * 창 안이 거의 상수인 구간이 있어 (x-mu)/sd 가 발산한다 -> Steel 적합이
    MSE 2.3e12 로 터지고, 그 결과 특징을 늘려도 값이 변하지 않는 것처럼 보였다.
    표준편차가 작은 창은 제외한다.
  * R^2 의 기준선. '각 창의 미래 평균'을 기준으로 잡으면 미래를 미리 아는
    오라클과 비교하는 셈이라 좋은 예측도 음수가 된다(ETTh1 선형 천장 -2.6%).
    tools/paper/compute_r2.py 와 같은 전체 분산 기준을 쓴다(같은 예측 +23.7%).
검증: 시간축 원본 전체를 쓴 선형 천장이 ETTh1 에서 MSE 0.400 -> 문헌 DLinear
0.399 와 일치.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from utils.data_factory import data_provider

L, H = 96, 96
MAX_WIN = 4000
MIN_STD = 1e-2          # 이보다 평평한 창은 정규화가 무의미 -> 제외


def collect(name, flag, cap):
    ds, dl = data_provider("data/", name, "M", 512, L, 48, H, flag,
                           train_shuffle_flag=False)
    xs, ys, n = [], [], 0
    for batch in dl:
        xs.append(batch[0].float().numpy())
        ys.append(batch[1].float().numpy()[:, -H:])
        n += xs[-1].shape[0]
        if n >= cap:
            break
    X = np.concatenate(xs)[:cap]
    Y = np.concatenate(ys)[:cap]
    B, _, D = X.shape
    X = X.transpose(0, 2, 1).reshape(B * D, L)
    Y = Y.transpose(0, 2, 1).reshape(B * D, H)
    # CENTER ONLY — no division. Dividing by the lookback std blew Steel up even
    # after filtering (normalized-space MSE 79): a window that is flat in the
    # lookback but moves in the horizon gets Y/sd amplified by 100x, and a
    # handful of such rows dominate the least-squares fit. Centering removes the
    # window level (which is what makes the task window-relative) without any
    # division, and the R^2 below is scale-free anyway.
    mu = X.mean(1, keepdims=True)
    return X - mu, Y - mu, 1.0


def fit_eval(A_tr, Ytr, A_te, Yte):
    A_tr = np.concatenate([A_tr, np.ones((len(A_tr), 1))], 1)
    A_te = np.concatenate([A_te, np.ones((len(A_te), 1))], 1)
    W, *_ = np.linalg.lstsq(A_tr, Ytr, rcond=None)
    P = A_te @ W
    mse = ((P - Yte) ** 2).mean()
    r2 = 1 - ((P - Yte) ** 2).sum() / ((Yte - Yte.mean()) ** 2).sum()
    return mse, r2


for name in ("ETTh1", "Steel"):
    Xtr, Ytr, kt = collect(name, "train", MAX_WIN)
    Xte, Yte, ke = collect(name, "test", MAX_WIN)
    Ftr, Fte = np.fft.rfft(Xtr, axis=1), np.fft.rfft(Xte, axis=1)
    n_freq = Ftr.shape[1]

    print("\n=== %s   train %d / test %d  (평평한 창 %.1f%% 제외) ==="
          % (name, len(Xtr), len(Xte), 100 * (1 - ke)))
    print("%-32s %9s %9s %9s" % ("사용한 정보", "MSE", "R^2", "R^2 증가"))
    print("-" * 62)

    prev = None
    for m in (1, 2, 3, 5, 8, 12, 24, n_freq):
        mse, r2 = fit_eval(np.concatenate([Ftr[:, :m].real, Ftr[:, :m].imag], 1),
                           Ytr,
                           np.concatenate([Fte[:, :m].real, Fte[:, :m].imag], 1),
                           Yte)
        lbl = ("저주파 %2d개 (주기 >= %.0f스텝)" % (m, L / (m - 1))) if m > 1 \
            else "저주파  1개 (상수항)"
        inc = "" if prev is None else "%8.1f%%" % (100 * (r2 - prev))
        print("%-32s %9.4f %8.1f%% %9s" % (lbl, mse, 100 * r2, inc))
        prev = r2

    for lo in (12, 24):
        mse, r2 = fit_eval(np.concatenate([Ftr[:, lo:].real, Ftr[:, lo:].imag], 1),
                           Ytr,
                           np.concatenate([Fte[:, lo:].real, Fte[:, lo:].imag], 1),
                           Yte)
        print("%-32s %9.4f %8.1f%%" % ("고주파만 (bin %d~%d)" % (lo, n_freq - 1),
                                       mse, 100 * r2))

    mse, r2 = fit_eval(Xtr, Ytr, Xte, Yte)
    print("%-32s %9.4f %8.1f%%" % ("시간축 96개 전부 (선형 천장)", mse, 100 * r2))

    mse = ((np.repeat(Xte[:, -1:], H, 1) - Yte) ** 2).mean()
    r2 = 1 - ((np.repeat(Xte[:, -1:], H, 1) - Yte) ** 2).sum() \
        / ((Yte - Yte.mean()) ** 2).sum()
    print("%-32s %9.4f %8.1f%%" % ("(참고) persistence", mse, 100 * r2))
