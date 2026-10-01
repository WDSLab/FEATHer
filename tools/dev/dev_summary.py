"""Summarise a fast dev-loop CSV: per-arm mean MSE, seed spread, delta vs a reference.

The dev loop (see tools/dev/README.md) trades the shipping protocol's
batch_size=32 for batch_size=512 + a scaled lr, which is ~15x cheaper per run.
That is only legitimate for RANKING arms, so every number here is reported with
its seed std and every delta with the noise floor it has to clear.

Rows are grouped by exp_tag of the form ``dev_<arm>_<lr>``; for each arm the lr
with the best mean VAL loss is selected (val-only selection, never test).

    python tools/dev/dev_summary.py --csv results/dev_loop.csv --ref canon
    python tools/dev/dev_summary.py --prefix devcal --ref canon
"""
import argparse
import csv
import math
import re
from collections import defaultdict


def _mean_std(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _median(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/dev_loop.csv")
    p.add_argument("--prefix", default="dev",
                   help="Only exp_tags starting with this are considered.")
    p.add_argument("--ref", default="canon",
                   help="Arm name used as the 0%% reference.")
    p.add_argument("--data", default=None, help="Filter to one dataset.")
    p.add_argument("--pred_len", default=None, help="Filter to one horizon.")
    p.add_argument("--metric", default="MSE", choices=["MSE", "MAE"])
    p.add_argument("--bad_margin", type=float, default=0.08,
                   help="A run is counted as a failed optimisation if its val "
                        "loss exceeds the best val seen anywhere in the table "
                        "by more than this fraction.")
    args = p.parse_args()

    # exp_tag -> (arm, lr). The lr is the LAST underscore-separated token; the
    # arm is everything before it, so arm names may contain underscores. The lr
    # token is matched loosely because a shell may rewrite "8e-3" to "0.008"
    # before it reaches the tag (PowerShell parses a comma-separated list of
    # scientific literals as an array of doubles).
    head = args.prefix + "_"
    lr_re = re.compile(r"^[0-9][0-9.eE+-]*$")

    def split_tag(tag):
        if not tag.startswith(head):
            return None
        rest = tag[len(head):]
        if "_" not in rest:
            return None
        arm, lr = rest.rsplit("_", 1)
        if not arm or not lr_re.match(lr):
            return None
        return arm, lr

    # (arm, lr) -> {"test": [...], "val": [...], "params": int}
    cells = defaultdict(lambda: {"test": [], "val": [], "params": None})
    with open(args.csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = split_tag(row["exp_tag"])
            if not m:
                continue
            if args.data and row["data"] != args.data:
                continue
            if args.pred_len and row["pred_len"] != str(args.pred_len):
                continue
            key = m
            try:
                cells[key]["test"].append(float(row[args.metric]))
                cells[key]["val"].append(float(row["val_loss"]))
            except (ValueError, KeyError):
                continue
            cells[key]["params"] = row.get("num_params")

    if not cells:
        raise SystemExit("no rows matched prefix %r in %s" % (args.prefix, args.csv))

    # A dev run either converges (Steel H96: val ~0.43) or falls into a clearly
    # separated worse optimum (val ~0.49) that early-stops around epoch 12-67.
    # It is seed-dependent and NOT undertraining (4x the epochs does not move
    # it). Averaging over it turns one bad draw into a fake architecture
    # penalty, so failures are counted and the headline number is the MEDIAN.
    all_val = [v for c in cells.values() for v in c["val"]]
    good_floor = min(all_val) if all_val else float("nan")
    bad_cut = good_floor * (1.0 + args.bad_margin)

    # Per arm: pick the lr with the best median val loss.
    best = {}
    for (arm, lr), c in cells.items():
        vmed = _median(c["val"])
        if arm not in best or vmed < best[arm][0]:
            tmed = _median(c["test"])
            tm, ts = _mean_std(c["test"])
            n_bad = sum(1 for v in c["val"] if v > bad_cut)
            best[arm] = (vmed, lr, tmed, ts, len(c["test"]), c["params"], n_bad, tm)

    ref = best.get(args.ref)
    print("bad-basin cutoff: val > %.4f (best val %.4f + %.0f%%)"
          % (bad_cut, good_floor, 100 * args.bad_margin))
    print("%-20s %6s %4s %4s %8s %7s %9s %9s" %
          ("arm", "lr", "n", "bad", args.metric, "std%", "delta%", "params"))
    print("-" * 76)
    for arm, (vm, lr, tmed, ts, n, params, n_bad, tm) in sorted(
            best.items(), key=lambda kv: kv[1][2]):
        stdpct = 100 * ts / tm if tm else float("nan")
        if ref and ref[2]:
            dstr = "%+8.2f" % (100 * (tmed - ref[2]) / ref[2])
        else:
            dstr = "%8s" % "-"
        pstr = params or "-"
        if ref and params and ref[5]:
            try:
                pstr = "%s (%+.0f%%)" % (params, 100 * (int(params) - int(ref[5])) / int(ref[5]))
            except (TypeError, ValueError):
                pass
        print("%-20s %6s %4d %4s %8.4f %7.2f %s %9s" %
              (arm, lr, n, ("%d!" % n_bad) if n_bad else "-", tmed, stdpct,
               dstr, pstr))
    if ref:
        print("\nreference = %r (median %s = %.4f)." % (args.ref, args.metric, ref[2]))
        print("'bad' = runs that landed in the worse optimum; an arm with bad>0 "
              "needs more seeds before its delta means anything.")
        print("%s is the MEDIAN over seeds; std%% is over all seeds including "
              "bad ones, so a large std%% with bad>0 is an optimisation "
              "artifact, not a result." % args.metric)


if __name__ == "__main__":
    main()
