"""EWELD 불안정 가설 1차 관문: 평평한 창(작은 stdev)이 정말 많은가?

FEATHer 의 RevIN 은 half-RevIN 이다 — means 는 detach 하지만 stdev 는 gradient 가
통과한다(FEATHer.py, 2026-07-14 감사에서 의도적이라고 결정). 그러면 x_norm = (x-mu)/sd
의 sd 에 대한 미분이 1/sd^2 로 커지므로, 창이 평평할수록 gradient 가 폭발한다.

EWELD 는 공장 전력이라 야간·휴일에 평평한 구간이 많을 것으로 의심된다. 학습을 돌리기
전에 이것부터 재서, 가설이 성립할 여지가 있는지 확인한다(없으면 가설 폐기).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
from utils.data_factory import data_provider

L, H = 96, 96
DATASETS = ["Steel", "EWELD", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3", "ETTh1"]

print("창별 표준편차 분포 (train, 채널 폴딩 후). 데이터는 train Scaler 로 표준화돼 있으므로")
print("전형적인 창은 sd ~ 1 근처. sd 가 작을수록 1/sd gradient 가 커진다.")
print()
print("%-14s %8s %8s %8s %8s %9s %9s %9s"
      % ("data", "창 수", "sd 중앙", "sd 5%", "sd 1%", "sd<0.1", "sd<0.03", "sd<0.01"))
print("-" * 84)

for name in DATASETS:
    try:
        ds, dl = data_provider("data/", name, "M", 512, L, 48, H, "train",
                               train_shuffle_flag=False)
    except Exception as exc:
        print("%-14s [실패] %s" % (name, str(exc)[:40]))
        continue
    sds = []
    n = 0
    for b in dl:
        x = b[0].float().numpy()                       # (B, L, D)
        s = x.std(axis=1)                              # (B, D)
        sds.append(s.reshape(-1))
        n += x.shape[0]
        if n >= 4000:
            break
    s = np.concatenate(sds)
    s = s[np.isfinite(s)]
    print("%-14s %8d %8.3f %8.3f %8.4f %8.1f%% %8.1f%% %8.2f%%"
          % (name, len(s), np.median(s), np.percentile(s, 5), np.percentile(s, 1),
             100 * (s < 0.1).mean(), 100 * (s < 0.03).mean(), 100 * (s < 0.01).mean()))

print()
print("해석: sd=0.01 인 창에서 1/sd = 100 배. 그런 창이 배치(512)에 하나만 섞여도")
print("      그 스텝의 gradient 를 지배할 수 있다.")
