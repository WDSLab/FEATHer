# -*- coding: utf-8 -*-
"""센서를 따로 볼 때와 같이 볼 때의 선형 천장을 나란히 구한다.

    python analysis/diagnostics/linear_ceiling_mix.py                 # mfg 6 x 4H
    python analysis/diagnostics/linear_ceiling_mix.py --data Steel --pred_len 96

왜 필요한가 (2026-09-10)
  linear_ceiling_sweep.py 의 천장은 채널을 배치로 접는다 - 센서마다 같은 96->H
  사상을 따로 적용한다.  그래서 어떤 모델이 그 천장을 넘어도 이유가 둘로 갈린다:
    (가) 선형으로는 원리상 못 잡는 비선형 구조가 있다
    (나) 센서끼리의 관계가 있다 - 이건 센서를 섞기만 하면 선형으로도 잡힌다
  우리 두 변종이 바로 이 둘을 가른다: FEATHer_CI(비선형 O, 섞기 X)는 mfg 경량
  9개 중 7등, FEATHer_mix(비선형 O, 섞기 O)는 2등.  즉 제조에서의 우위가 비선형
  때문인지 섞기 때문인지는 이 둘을 따로 재지 않고는 말할 수 없다.

두 천장
  CI   센서별 96 -> H     (특징 L+1,   채널을 배치로 접음)
  MIX  전 센서 L*D -> H*D (특징 L*D+1, 출력 채널마다 따로 풂)

공정성
  - 두 천장 모두 같은 규칙: ridge 강도를 val 로 고르고 test 는 한 번만 채점.
    MIX 는 특징이 D 배라 규제가 모자라면 과적합으로 불리해진다 - 첫 스윕은
    격자 끝값(1.0)이 뽑히며 CI 보다 50~108% 나빴다.  격자를 1e4 까지 넓히고
    끝값이 뽑힌 칸은 lam_edge 로 표시한다.
  - 결측은 두 천장 모두 (창, 센서) 단위로 버린다.  첫 판은 MIX 가 창 전체를
    버려서 결측이 흩어진 N2O 는 창이 하나도 남지 않았다(SKIP).
  - test 는 벤치마크와 같은 관측-only 채점 (MASK-FIX): 셀 단위로 NaN 제외.
  - lambda=0 CI 는 linear_ceiling_sweep.py 와 같은 정의다.  결측 없는 데이터
    (Steel)에서는 1e-9 로 재현되고, 결측 있는 데이터에서는 두 도구의 선택
    단계 결측 처리 차이로 최대 2.5% 어긋난다 (WindSCADA H720).  이 도구 안의
    CI vs MIX 비교는 같은 처리를 쓰므로 일관된다.

구현
  정규방정식 충분통계만 누적한다.  val MSE 는 통계로 닫힌 식으로 구하므로 val
  데이터를 저장하지 않는다.  MIX 는 모든 채널이 관측된 창을 공유 행렬 하나에
  모으고, 일부 채널만 관측된 창은 채널별 보정항으로 더한다 - 결측이 드문
  데이터는 사실상 한 번의 연립방정식으로 끝난다.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils.data_factory import data_provider  # noqa: E402

MFG = ["Steel", "MetroPT3", "WindSCADA", "N2O", "EWELD", "BlastFurnace"]
HS = [96, 192, 336, 720]
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]   # x mean(diag(G))
# MIX 는 0 이 아니라 CI 해답 쪽으로 당긴다 (아래 _mix_weights).  inf = CI 그 자체.
MIX_LAMBDAS = LAMBDAS + [float("inf")]


def _batches(data, L, H, flag):
    _, dl = data_provider("data/", data, "M", 512, L, 48, H, flag,
                          train_shuffle_flag=False)
    for b in dl:
        yield b[0].double().numpy(), b[1].double().numpy()[:, -H:]


def _bias(X):
    return np.concatenate([X, np.ones((len(X), 1))], 1)


def _solve(G, C, lam, s):
    if lam > 0:
        return np.linalg.solve(G + lam * s * np.eye(len(G)), C)
    return np.linalg.lstsq(G, C, rcond=None)[0]


# ------------------------------------------------------------------- CI
class CIStats:
    """채널을 배치로 접은 행 (창, 센서) 단위.  결측 행은 버린다."""

    def __init__(self):
        self.G = self.C = None
        self.syy, self.n = 0.0, 0

    def add(self, x, y):
        B, L, D = x.shape
        A = _bias(x.transpose(0, 2, 1).reshape(B * D, L))
        Y = y.transpose(0, 2, 1).reshape(B * D, -1)
        ok = np.isfinite(Y).all(1)
        A, Y = A[ok], Y[ok]
        if not len(A):
            return
        if self.G is None:
            self.G = np.zeros((A.shape[1],) * 2)
            self.C = np.zeros((A.shape[1], Y.shape[1]))
        self.G += A.T @ A
        self.C += A.T @ Y
        self.syy += float((Y * Y).sum())
        self.n += Y.size

    def sse(self, W):
        return self.syy - 2.0 * float((W * self.C).sum()) + float((W * (self.G @ W)).sum())


def fit_ci(tr, va):
    s = float(np.mean(np.diag(tr.G)))
    best, W0 = None, None
    for lam in LAMBDAS:
        W = _solve(tr.G, tr.C, lam, s)
        v = va.sse(W) / max(va.n, 1)
        if lam == 0.0:
            W0 = W
        if best is None or v < best[0]:
            best = (v, lam, W)
    return best, W0


# ------------------------------------------------------------------- MIX
class MixStats:
    """전 센서 특징 -> 출력 채널별 결측 처리.  (창, 센서) 단위로 버린다."""

    def __init__(self):
        self.G = self.C = None        # (F,F), (F,D,H): 모든 채널 관측된 창
        self.corr, self.ccorr = {}, {}  # d -> (F,F), (F,H): 일부만 관측된 창
        self.syy = self.n = None      # (D,)

    def add(self, x, y):
        B, L, D = x.shape
        A = _bias(x.reshape(B, L * D))
        Yc = y.transpose(0, 2, 1)                     # (B,D,H)
        H, F = Yc.shape[2], A.shape[1]
        if self.G is None:
            self.G = np.zeros((F, F))
            self.C = np.zeros((F, D, H))
            self.syy, self.n = np.zeros(D), np.zeros(D)
        obs = np.isfinite(Yc).all(2)                  # (B,D)
        full = obs.all(1)
        if full.any():
            Af, Yf = A[full], Yc[full]
            self.G += Af.T @ Af
            self.C += np.einsum("bf,bdh->fdh", Af, Yf)
            self.syy += (Yf ** 2).sum((0, 2))
            self.n += full.sum() * H
        part = (~full) & obs.any(1)
        if part.any():
            for d in range(D):
                rows = part & obs[:, d]
                if not rows.any():
                    continue
                Ad, Yd = A[rows], Yc[rows, d, :]
                if d not in self.corr:
                    self.corr[d] = np.zeros((F, F))
                    self.ccorr[d] = np.zeros((F, H))
                self.corr[d] += Ad.T @ Ad
                self.ccorr[d] += Ad.T @ Yd
                self.syy[d] += (Yd ** 2).sum()
                self.n[d] += rows.sum() * H

    def Gd(self, d):
        return self.G + self.corr[d] if d in self.corr else self.G

    def Cd(self, d):
        return self.C[:, d, :] + self.ccorr[d] if d in self.ccorr else self.C[:, d, :]

    def sse(self, W):                                 # W: (F,D,H)
        F, D, H = W.shape
        shared = [d for d in range(D) if d not in self.corr]
        tot = 0.0
        if shared:
            Wb = W[:, shared, :].reshape(F, -1)
            Cb = self.C[:, shared, :].reshape(F, -1)
            tot += float(self.syy[shared].sum()) - 2.0 * float((Wb * Cb).sum()) \
                + float((Wb * (self.G @ Wb)).sum())
        for d in range(D):
            if d in self.corr:
                Wd = W[:, d, :]
                tot += float(self.syy[d]) - 2.0 * float((Wd * self.Cd(d)).sum()) \
                    + float((Wd * (self.Gd(d) @ Wd)).sum())
        return tot


def _embed_ci(W_ci, L, D):
    """CI 가중치 (L+1, H) 를 MIX 공간 (F, D, H) 으로 옮긴다.

    MIX 특징은 x.reshape(B, L*D) 라 센서 d 의 시점 t 는 열 t*D + d, 마지막 열이
    bias.  출력 채널 d 는 자기 센서 열에만 CI 가중치를 두고 나머지는 0.
    MIX 가 CI 를 특수경우로 포함한다는 사실을 코드로 쓴 것이다."""
    F, H = L * D + 1, W_ci.shape[1]
    E = np.zeros((F, D, H))
    for d in range(D):
        E[np.arange(L) * D + d, d, :] = W_ci[:L, :]
        E[F - 1, d, :] = W_ci[L, :]
    return E


def _mix_weights(tr, lam, s, E):
    """min ||Y - A W||^2 + lam*s*||W - E||^2.

    0 이 아니라 CI 해답 E 쪽으로 당긴다.  첫 판은 0 쪽으로 당겨서 EWELD 에서
    검증셋조차 CI 보다 52% 나빴다 - 규제가 센서간 가중치와 함께 자기 센서
    과거값 신호까지 죽였기 때문이다.  lam=inf 면 정확히 CI, lam=0 이면 무규제 MIX.
    학습 데이터가 전혀 없는 채널(N2O 의 Inlet_Q 초기 구간)은 풀 수 없으므로 E."""
    if np.isinf(lam):
        return E.copy()
    F, D, H = tr.C.shape
    W = np.empty((F, D, H))
    shared = [d for d in range(D) if d not in tr.corr and tr.n[d] > 0]
    if shared:
        Cb = tr.C[:, shared, :].reshape(F, -1)
        Eb = E[:, shared, :].reshape(F, -1)
        W[:, shared, :] = (_solve(tr.G, Cb, 0.0, s) if lam == 0 else
                           np.linalg.solve(tr.G + lam * s * np.eye(F), Cb + lam * s * Eb)
                           ).reshape(F, len(shared), H)
    for d in range(D):
        if d in tr.corr and tr.n[d] > 0:
            Gd, Cd = tr.Gd(d), tr.Cd(d)
            W[:, d, :] = (_solve(Gd, Cd, 0.0, s) if lam == 0 else
                          np.linalg.solve(Gd + lam * s * np.eye(F), Cd + lam * s * E[:, d, :]))
        elif tr.n[d] == 0:
            W[:, d, :] = E[:, d, :]
    return W


def fit_mix(tr, va, E):
    D = tr.C.shape[1]
    # 공유 행렬이 비어도(N2O: 전 센서 동시관측 창 0개) 쓸 수 있는 척도
    s = float(np.mean([np.mean(np.diag(tr.Gd(d))) for d in range(D) if tr.n[d] > 0]))
    best, W0 = None, None
    for lam in MIX_LAMBDAS:
        W = _mix_weights(tr, lam, s, E)
        v = va.sse(W) / max(float(va.n.sum()), 1.0)
        if lam == 0.0:
            W0 = W
        if best is None or v < best[0]:
            best = (v, lam, W)
    return best, W0


# ------------------------------------------------------------------ cell
def run_cell(data, H, L=96, max_train=0):
    t0 = time.time()
    tr_ci, va_ci, tr_mx, va_mx = CIStats(), CIStats(), MixStats(), MixStats()
    seen = 0
    for x, y in _batches(data, L, H, "train"):
        tr_ci.add(x, y)
        tr_mx.add(x, y)
        seen += len(x)
        if max_train and seen >= max_train:
            break
    for x, y in _batches(data, L, H, "val"):
        va_ci.add(x, y)
        va_mx.add(x, y)

    (v_ci, lam_ci, W_ci), W_ci0 = fit_ci(tr_ci, va_ci)
    F, D, _ = tr_mx.C.shape
    E = _embed_ci(W_ci, L, D)
    (v_mx, lam_mx, W_mx), W_mx0 = fit_mix(tr_mx, va_mx, E)
    del E
    # (F,D,H) -> (F,H*D) in the loader's time-major flatten order (h*D + d)
    to_flat = lambda W: W.transpose(0, 2, 1).reshape(F, -1)   # noqa: E731
    Wm, Wm0 = to_flat(W_mx), to_flat(W_mx0)
    del W_mx, W_mx0

    sse = dict(ci=0.0, ci_lam0=0.0, mix=0.0, mix_lam0=0.0)
    cnt = dict(ci=0, mix=0)
    for x, y in _batches(data, L, H, "test"):
        B = len(x)
        # CI
        A = _bias(x.transpose(0, 2, 1).reshape(B * D, L))
        Y = y.transpose(0, 2, 1).reshape(B * D, H)
        obs = np.isfinite(Y)
        Yz = np.where(obs, Y, 0.0)
        for key, W in (("ci", W_ci), ("ci_lam0", W_ci0)):
            sse[key] += float((((A @ W - Yz) ** 2) * obs).sum())
        cnt["ci"] += int(obs.sum())
        # MIX
        A = _bias(x.reshape(B, L * D))
        Y = y.reshape(B, H * D)
        obs = np.isfinite(Y)
        Yz = np.where(obs, Y, 0.0)
        for key, W in (("mix", Wm), ("mix_lam0", Wm0)):
            sse[key] += float((((A @ W - Yz) ** 2) * obs).sum())
        cnt["mix"] += int(obs.sum())

    return dict(
        data=data, pred_len=H, D=D, train_windows=seen,
        ci=sse["ci"] / cnt["ci"], mix=sse["mix"] / cnt["mix"],
        ci_lam0=sse["ci_lam0"] / cnt["ci"], mix_lam0=sse["mix_lam0"] / cnt["mix"],
        lam_ci=lam_ci, lam_mix=lam_mx,
        lam_edge_ci=lam_ci == LAMBDAS[-1], lam_edge_mix=lam_mx == LAMBDAS[-1],
        mix_is_ci=bool(np.isinf(lam_mx)),     # 검증셋이 "센서간 정보 없음"을 골랐다
        val_ci=v_ci, val_mix=v_mx,
        partial_channels_mix=len(tr_mx.corr),
        secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=",".join(MFG))
    ap.add_argument("--pred_len", default=",".join(str(h) for h in HS))
    # 0 = 학습 데이터 전부.  예전 기본값 20000 은 창을 앞에서부터 잘라, N2O 에서는
    # 학습 데이터의 86% 를 버리고 센서가 아직 안 붙어 있던 초기 구간만 남겼다.
    # (--max_train 20000 이면 linear_ceiling_sweep.py 와 같은 조건 = 재현 검증용)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--out", default="results/analysis/linear_ceiling_mix.csv")
    a = ap.parse_args()

    cols = ["data", "pred_len", "D", "ci", "mix", "ci_lam0", "mix_lam0",
            "lam_ci", "lam_mix", "lam_edge_ci", "lam_edge_mix", "mix_is_ci",
            "val_ci", "val_mix",
            "partial_channels_mix", "train_windows", "secs"]
    out = os.path.join(_ROOT, a.out)
    rows = []
    for d in [s.strip() for s in a.data.split(",") if s.strip()]:
        for h in [int(s) for s in a.pred_len.split(",")]:
            try:
                r = run_cell(d, h, max_train=a.max_train)
            except Exception as e:                        # noqa: BLE001
                print(f"{d:<13} H={h:<4} SKIP {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(r)
            gain = (r["ci"] - r["mix"]) / r["ci"] * 100
            edge = " [lam 끝값]" if (r["lam_edge_ci"] or r["lam_edge_mix"]) else ""
            print(f"{d:<13} H={h:<4} CI {r['ci']:.4f}  MIX {r['mix']:.4f}  "
                  f"섞기이득 {gain:+6.2f}%   (lam {r['lam_ci']:g}/{r['lam_mix']:g}"
                  f"{edge}, CI lam0 {r['ci_lam0']:.4f}, {r['secs']}s)", flush=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    print(f"\n-> {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
