"""EWELD 불안정 가설 2: 초기 게이트가 밴드를 죽이는가?

실패 시드(2025, 2027)와 성공 시드(2026)는 1에폭째부터 갈리고, train loss 도 같이
나쁘다(0.293 vs 0.278) — 과적합이 아니라 최적화 실패이며 초기화에서 갈린다는 뜻.

canonical 의 "global" 게이트는 Conv1d 가 랜덤 초기화라 학습 시작 전부터 밴드 가중치가
크게 치우칠 수 있다(관측 예: [0.5968, 0.0003, 0.2617, 0.1413] — 2번 밴드가 사실상 OFF).
softmax 가 포화된 밴드는 gradient 를 거의 못 받으므로 죽은 가지가 된다.

대조: gate_mode="banded" 는 zero-init 이라 정확히 균등(1/S)에서 출발한다. 그리고
실측에서 EWELD 성공률이 33% -> 83% 로 올랐다.

여기서는 시드별 초기 게이트를 실제 EWELD 창으로 재서, 실패 시드가 더 치우쳐 있는지 본다.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import torch
from baselines.FEATHer.FEATHer import FEATHer
from utils.data_factory import data_provider
from utils.seed import set_seed

L, H = 96, 96
DATA = os.environ.get("GATE_INIT_DATA", "EWELD")

ds, dl = data_provider("data/", DATA, "M", 512, L, 48, H, "train",
                       train_shuffle_flag=False)
xb = next(iter(dl))[0].float()
D = xb.shape[2]
xn = (xb - xb.mean(1, keepdim=True)) / torch.sqrt(
    xb.var(1, keepdim=True, unbiased=False) + 1e-5)

print("%s  D=%d   초기 게이트 (학습 전, 실제 창 512개 평균)" % (DATA, D))
print("%-6s %-9s %-34s %8s %9s" % ("seed", "gate_mode", "평균 밴드 가중치", "최소", "최대/최소"))
print("-" * 76)
for gm in ("global", "banded"):
    for seed in (2025, 2026, 2027, 2028, 2029):
        set_seed(seed, deterministic=True)
        m = FEATHer(seq_len=L, pred_len=H, d_model=D, d_state=16, kernel_size=7,
                    num_bands=4, period=24, use_residual_mlp=True, mlp_rank=16,
                    channel_mode="mix", gate_mode=gm).eval()
        with torch.no_grad():
            g = m.freq_gate(xn).mean(0)
        lo = g.min().item()
        print("%-6d %-9s %-34s %8.4f %9.1f"
              % (seed, gm, str([round(v, 4) for v in g.tolist()]), lo,
                 g.max().item() / max(lo, 1e-9)))
print()
print("실패 시드 = 2025, 2027 / 성공 시드 = 2026  (EWELD, canonical)")
