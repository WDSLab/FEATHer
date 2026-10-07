# -*- coding: utf-8 -*-
"""Would the candidate's design choices survive if they were made on VAL, not TEST?
(2026-09-22)  The dev comparisons that shaped the candidate were read off test MSE
(lr alone was val-selected).  Recompute every decision with the validation loss and
report whether the choice flips.  No training.

    python tools/dev/val_recheck.py
"""
import numpy as np
import pandas as pd

d = pd.read_csv("results/dev_mix0914.csv", on_bad_lines="skip")
d = d[d.pred_len == 96].copy()
d["arm"] = d.exp_tag.str.replace(r"^mix0914_", "", regex=True).str.replace(r"_[0-9.e-]+$", "", regex=True)
DS = ["Steel", "EWELD", "WindSCADA", "BlastFurnace", "N2O", "MetroPT3"]


def pick(arm, ds):
    g = d[(d.arm == arm) & (d.data == ds)]
    if g.empty:
        return None
    tag = g.groupby("exp_tag").val_loss.median().idxmin()      # lr chosen on val, as in the dev loop
    return g[g.exp_tag == tag].set_index("seed")[["val_loss", "MSE"]]


def compare(arms, ds):
    rows = [pick(a, ds) for a in arms]
    if any(r is None for r in rows):
        return None
    common = sorted(set.intersection(*[set(r.index) for r in rows]))
    return {a: r.loc[common].mean() for a, r in zip(arms, rows)}, len(common)


def pct(x, ref):
    return 100 * (x / ref - 1)


print("=== pairwise: % of arm vs reference, VAL loss | TEST MSE (neg = arm better) ===")
PAIRS = [
    ("D1 mix location: input vs hidden (R16)", "mx_base", "mx_in1"),
    ("D2a rank: R4 vs R16 (input)", "mx_in1", "mx_in1_r4"),
    ("D2b rank: R2 vs R4", "mx_in1_r4", "mx_in1_r2"),
    ("D3 gate: per-sensor vs sensor-mean", "mx_in1_r4", "mx_in1_r4_gsensor"),
    ("D4a bounds: octave vs linear", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_oct"),
    ("D4b bounds: tercile vs linear", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_ter"),
    ("D4c bounds: tercile/sensor vs linear", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_ters"),
    ("D5 static ratio vs FFT gate", "mx_in1_r4_gsensor", "mx_in1_r4_static"),
    ("D6 raw-input mixer vs candidate", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_raw"),
    ("D7 zero-sum bands vs candidate", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_zs"),
    ("D8 G2 fft vs candidate", "mx_in1_r4_gsensor", "g2_fft"),
    ("D9 G4 fft vs candidate", "mx_in1_r4_gsensor", "g4_fft"),
    ("D10 A listen vs candidate", "mx_in1_r4_gsensor", "mxA"),
    ("D11 B lag vs candidate", "mx_in1_r4_gsensor", "mxB"),
    ("D12 C zero-init vs candidate", "mx_in1_r4_gsensor", "mxC"),
    ("D13 ABC vs candidate", "mx_in1_r4_gsensor", "mxABC"),
    ("ref: mixer off vs candidate", "mx_in1_r4_gsensor", "mx_in1_r4_gsensor_ci"),
]
out = []
for name, ref, arm in PAIRS:
    line_v, line_t = [], []
    for ds in DS:
        c = compare([ref, arm], ds)
        if c is None:
            line_v.append("    --"); line_t.append("    --"); continue
        m, n = c
        v, t = pct(m[arm].val_loss, m[ref].val_loss), pct(m[arm].MSE, m[ref].MSE)
        line_v.append(f"{v:+6.2f}"); line_t.append(f"{t:+6.2f}")
        out.append(dict(decision=name, data=ds, seeds=n, val_pct=v, test_pct=t))
    print(f"\n{name}   [{arm} vs {ref}]")
    print("   " + "  ".join(f"{ds[:9]:>9s}" for ds in DS))
    print("   val  " + "   ".join(line_v))
    print("   test " + "   ".join(line_t))

# D2 as it was actually decided: share of the mixing gain kept, ci -> base
print("\n=== D2 as decided: mixing gain kept = (ci - arm) / (ci - hidden R16) ===")
for arm in ["mx_in1", "mx_in1_r4", "mx_in1_r2"]:
    kv, kt = [], []
    for ds in DS:
        c = compare(["mx_ci", "mx_base", arm], ds)
        if c is None:
            continue
        m, _ = c
        for key, acc in (("val_loss", kv), ("MSE", kt)):
            gain = m["mx_ci"][key] - m["mx_base"][key]
            if gain > 0:
                acc.append((m["mx_ci"][key] - m[arm][key]) / gain)
    print(f"{arm:10s} kept: val mean {100*np.mean(kv):5.1f}% (n={len(kv)}) | test mean {100*np.mean(kt):5.1f}% (n={len(kt)})")
pd.DataFrame(out).to_csv("results/_val_recheck.csv", index=False)
print("\nwrote results/_val_recheck.csv")
