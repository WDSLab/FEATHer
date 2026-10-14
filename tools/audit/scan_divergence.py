"""발산 의심 칸 스캔.

증상: best_epoch 가 0~2 = 첫 에폭 근처가 가장 좋았고 이후 계속 나빠짐 = 학습률 과다.
예: DLinear ETTh2 는 lr 0.05(공표값, seq_len=336 기준)로 발산해 MSE 0.483, lr 0.005 면 0.296.

원인 구조: configs/<모델>.yaml 의 lr 은 각 논문의 공식 스크립트에서
가져온 값인데, 그 값들은 그 논문의 룩백(대개 336/720)에서 튜닝됐다. 우리는
seq_len=96 을 강제하므로 같은 lr 이 과할 수 있다. 긴 룩백 네이티브 모델
(DLinear/FITS/SparseTSF/MixLinear/DiPE/CycleNet)이 특히 위험하다.

    python tools/audit/scan_divergence.py
    python tools/audit/scan_divergence.py --csv results/_archive/pre_raw_20261007/mfg_raw.csv --max_epoch 3
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/_archive/pre_raw_20261007/ltsf_raw.csv,results/_archive/pre_raw_20261007/mfg_raw.csv")
    p.add_argument("--max_epoch", type=int, default=2,
                   help="best_epoch <= this counts as a divergence suspect.")
    p.add_argument("--exp_tag", default="main,final")
    a = p.parse_args()
    tags = set(a.exp_tag.split(","))

    cells = defaultdict(list)
    for path in a.csv.split(","):
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("exp_tag") not in tags:
                    continue
                try:
                    cells[(r["data"], r["model"], int(r["pred_len"]))].append(
                        (int(float(r["best_epoch"])), float(r["MSE"])))
                except (ValueError, KeyError):
                    pass

    bad = []
    for (ds, model, h), rows in cells.items():
        eps = [e for e, _ in rows]
        if not eps:
            continue
        frac = sum(1 for e in eps if e <= a.max_epoch) / len(eps)
        if frac >= 0.5:                       # 과반 시드가 발산 지문
            bad.append((ds, model, h, sum(eps) / len(eps),
                        sum(m for _, m in rows) / len(rows), len(rows), frac))

    if not bad:
        print("발산 의심 칸 없음 (best_epoch <= %d 가 과반인 칸)" % a.max_epoch)
        return

    print("발산 의심 칸: %d개   (best_epoch <= %d 인 시드가 과반)" % (len(bad), a.max_epoch))
    print("%-14s %-14s %5s %10s %9s %6s" % ("data", "model", "H", "평균 best_ep", "MSE", "seeds"))
    print("-" * 66)
    per_model = defaultdict(int)
    for ds, model, h, ep, mse, n, frac in sorted(bad, key=lambda t: (t[1], t[0], t[2])):
        print("%-14s %-14s %5d %10.1f %9.4f %6d" % (ds, model, h, ep, mse, n))
        per_model[model] += 1
    print()
    print("모델별 의심 칸 수:")
    for m, c in sorted(per_model.items(), key=lambda kv: -kv[1]):
        print("   %-14s %d" % (m, c))


if __name__ == "__main__":
    main()
