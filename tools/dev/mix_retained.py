"""Mixer cost-reduction screen: how much of the channel-mixing gain does each arm keep?

    python tools/dev/mix_retained.py --csv results/dev_mix0914.csv --prefix mix0914

For every dataset, each arm's lr is picked on the MEDIAN VAL loss over seeds
(never test), exactly like dev_summary.py.  Then

    retained% = (MSE_ci - MSE_arm) / (MSE_ci - MSE_base) * 100

where base = the shipping mix model (mx_base) and ci = no mixing (mx_ci).
100% = keeps the whole mixing gain, 0% = no better than not mixing, <0 = worse
than not mixing.  On a dataset where mixing buys almost nothing (small
ci - base gap) the ratio is noise, so the gap is printed next to it and a
dataset is flagged when the gap is below the seed noise of the base arm.

Failed optimisations are counted with the same rule as dev_summary.py (val
loss more than --bad_margin above the best val seen on that dataset) and the
headline is the median, so one bad seed cannot fake an architecture effect.
"""
import argparse
import re
import statistics as st

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/dev_mix0914.csv")
    p.add_argument("--prefix", default="mix0914")
    p.add_argument("--base", default="mx_base")
    p.add_argument("--ci", default="mx_ci")
    p.add_argument("--metric", default="MSE", choices=["MSE", "MAE"])
    p.add_argument("--bad_margin", type=float, default=0.08)
    a = p.parse_args()

    d = pd.read_csv(a.csv).dropna(subset=[a.metric, "val_loss"])
    pat = re.compile(r"^%s_(.+)_([0-9][0-9.eE+-]*)$" % re.escape(a.prefix))
    parsed = d.exp_tag.str.extract(pat)
    d = d.assign(arm=parsed[0], lr=parsed[1]).dropna(subset=["arm"])
    if d.empty:
        raise SystemExit("no rows for prefix %r in %s" % (a.prefix, a.csv))

    summary = []
    for data, g in d.groupby("data"):
        bad_cut = g.val_loss.min() * (1 + a.bad_margin)
        best = {}
        for (arm, lr), c in g.groupby(["arm", "lr"]):
            vmed = c.val_loss.median()
            if arm not in best or vmed < best[arm]["vmed"]:
                vals = c[a.metric].tolist()
                best[arm] = dict(vmed=vmed, lr=lr, med=st.median(vals),
                                 std=(st.stdev(vals) if len(vals) > 1 else 0.0),
                                 n=len(vals), bad=int((c.val_loss > bad_cut).sum()),
                                 params=int(c.num_params.iloc[0]))
        if a.base not in best or a.ci not in best:
            print("%s: base or ci arm missing, skipped" % data)
            continue
        b, c = best[a.base], best[a.ci]
        gap = c["med"] - b["med"]
        noisy = abs(gap) < b["std"]
        print("\n=== %s   ci %.4f  base %.4f  mixing gain %.2f%%%s" % (
            data, c["med"], b["med"], 100 * gap / c["med"],
            "   (gap < base seed std: ratios are noise)" if noisy else ""))
        print("%-10s %6s %3s %4s %9s %7s %9s %10s %7s" % (
            "arm", "lr", "n", "bad", a.metric, "std%", "vs base%", "retained%", "params"))
        for arm, r in sorted(best.items(), key=lambda kv: kv[1]["med"]):
            ret = 100 * (c["med"] - r["med"]) / gap if gap else float("nan")
            print("%-10s %6s %3d %4s %9.4f %7.2f %+9.2f %10.0f %7d" % (
                arm, r["lr"], r["n"], ("%d!" % r["bad"]) if r["bad"] else "-",
                r["med"], 100 * r["std"] / r["med"], 100 * (r["med"] - b["med"]) / b["med"],
                ret, r["params"]))
            summary.append(dict(data=data, arm=arm, retained=ret, noisy=noisy,
                                vs_base=100 * (r["med"] - b["med"]) / b["med"]))

    s = pd.DataFrame(summary)
    if s.empty:
        return
    print("\n=== retained% per arm (datasets where the mixing gain clears the base seed noise) ===")
    clean = s[~s.noisy]
    if clean.empty:
        print("(no dataset has a mixing gain above noise)")
    else:
        print(clean.pivot(index="arm", columns="data", values="retained").round(0).to_string())
    print("\n=== MSE vs base (%) on every dataset ===")
    print(s.pivot(index="arm", columns="data", values="vs_base").round(2).to_string())


if __name__ == "__main__":
    main()
