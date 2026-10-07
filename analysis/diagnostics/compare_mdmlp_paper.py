"""MDMLP-EIA (AAAI'26) 논문 Table 과 우리 재현값 대조.

논문 표의 숫자는 사용자가 제공한 것을 그대로 옮김. 우리 값은 results/_archive/pre_raw_20261007/ltsf_raw.csv
(통일 프로토콜, 3시드 평균).

목적: 우리 재현이 어긋난 것인지, 아니면 그 표의 특정 베이스라인이 깨진 것인지 구분.
"""
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

# 논문 Table: dataset -> horizon -> {method: MSE}
# 열 순서: MDMLP-EIA(Ours) xPatch Amplifier TimeMixer CycleNet SOFTS
#          iTransformer FilterNet FITS SparseTSF FreTS DLinear
COLS = ["MDMLP_EIA", "xPatch", "Amplifier", "TimeMixer", "CycleNet", "SOFTS",
        "iTransformer", "FilterNet", "FITS", "SparseTSF", "FreTS", "DLinear"]

PAPER = {
    ("ETTh1", 96):  [0.365, 0.374, 0.377, 0.372, 0.375, 0.376, 0.392, 0.390, 0.731, 0.409, 0.452, 0.376],
    ("ETTh1", 192): [0.404, 0.419, 0.413, 0.413, 0.425, 0.418, 0.443, 0.421, 0.806, 0.433, 0.436, 0.412],
    ("ETTh1", 336): [0.431, 0.439, 0.437, 0.448, 0.467, 0.437, 0.458, 0.441, 0.894, 0.452, 0.482, 0.447],
    ("ETTh1", 720): [0.441, 0.451, 0.458, 0.459, 0.542, 0.439, 0.496, 0.458, 1.105, 0.457, 0.551, 0.465],
    ("ETTh2", 96):  [0.265, 0.280, 0.279, 0.273, 0.280, 0.290, 0.291, 0.296, 0.277, 0.303, 0.937, 0.301],
    ("ETTh2", 192): [0.330, 0.344, 0.333, 0.331, 0.352, 0.374, 0.362, 0.349, 0.337, 0.366, 1.005, 0.430],
    ("ETTh2", 336): [0.358, 0.375, 0.376, 0.348, 0.381, 0.408, 0.390, 0.390, 0.357, 0.393, 1.005, 0.525],
    ("ETTh2", 720): [0.388, 0.392, 0.387, 0.426, 0.419, 0.421, 0.414, 0.426, 0.383, 0.413, 0.700, 0.696],
    ("ETTm1", 96):  [0.288, 0.288, 0.292, 0.300, 0.289, 0.284, 0.301, 0.303, 0.296, 0.299, 0.307, 0.295],
    ("ETTm2", 96):  [0.159, 0.163, 0.166, 0.163, 0.159, 0.165, 0.171, 0.165, 0.214, 0.168, 0.179, 0.165],
    ("Weather", 96): [0.141, 0.147, 0.146, 0.143, 0.144, 0.147, 0.157, 0.154, 0.147, 0.153, 0.151, 0.172],
    ("Exchange", 96): [0.082, 0.083, 0.084, 0.083, 0.107, 0.084, 0.086, 0.087, 0.147, 0.138, 0.082, 0.096],
    ("Electricity", 96): [0.133, 0.127, 0.135, 0.131, 0.126, 0.856, 0.131, 0.140, 0.186, 0.145, 0.138, 0.135],
    ("Traffic", 96): [0.362, 0.361, 0.378, 0.370, 0.377, 0.393, 0.360, 0.369, 0.421, 0.406, 0.395, 0.398],
}

# 우리가 함께 돌리는 모델만 대조 가능
SHARED = ["MDMLP_EIA", "TimeMixer", "CycleNet", "iTransformer", "FITS",
          "SparseTSF", "DLinear"]

ours = {}
for path in ("results/_archive/pre_raw_20261007/ltsf_raw.csv",):
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("exp_tag") not in ("main", "final"):
                continue
            try:
                key = (r["data"], int(r["pred_len"]), r["model"])
                ours.setdefault(key, []).append(float(r["MSE"]))
            except (ValueError, KeyError):
                pass

print("논문값 vs 우리 재현값 (MSE, 우리=3시드 평균).  Δ = (우리-논문)/논문")
print("=" * 78)
print("%-12s %4s %-14s %8s %8s %9s" % ("dataset", "H", "model", "논문", "우리", "Δ"))
print("-" * 78)
big = []
for (ds, h), vals in sorted(PAPER.items()):
    for name, pv in zip(COLS, vals):
        if name not in SHARED:
            continue
        got = ours.get((ds, h, name))
        if not got:
            continue
        ov = sum(got) / len(got)
        d = 100 * (ov - pv) / pv
        flag = ""
        if abs(d) >= 20:
            flag = "  <== 20%+ 불일치"
            big.append((ds, h, name, pv, ov, d))
        print("%-12s %4d %-14s %8.3f %8.3f %+8.1f%%%s" % (ds, h, name, pv, ov, d, flag))
    print("-" * 78)

print()
print("20%% 이상 어긋난 칸: %d개" % len(big))
for ds, h, name, pv, ov, d in big:
    print("   %-12s H=%-4d %-14s 논문 %.3f  우리 %.3f  (%+.0f%%)" % (ds, h, name, pv, ov, d))
