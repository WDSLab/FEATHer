"""Fill values of the size candidates (protocol.SIZE_CANDIDATES) for the 1K tier -> docs/size_candidates.json.

    python tools/results/size_candidates.py            # report
    python tools/results/size_candidates.py --write    # write (commit the file)

For each candidate (fill key, fixed keys) and dataset: set the fixed keys, then grow the fill key step by step and keep
the last value whose model stays within 1,000 stored numbers at all four H.  If even the smallest value is over the
budget (DiPE), the smallest one is kept and marked over_budget.  The chip-max tier needs real builds (not done here).
"""
import copy
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))]
import protocol as P   # noqa: E402
import baselines as B   # noqa: E402

_argv, sys.argv = sys.argv, ["train.py", "--model", "DLinear", "--data", "Steel", "--pred_len", "96"]
import train as T   # noqa: E402
DEFAULT_ARGS = T.parse_args()                        # worker defaults
sys.argv = _argv

BUDGET = 1000
CHANNELS = {"Steel": 6, "WindSCADA": 14, "BlastFurnace": 28, "N2O": 14, "MetroPT3": 7, "EWELD": 14}
# values tried for the fill key, from the smallest model to the largest
SEGMENT_STEPS = [96, 48, 24, 16, 12, 8, 6, 4, 3, 2, 1]
COUNT_STEPS = range(1, 4001)


def stored_numbers(model, data, H, knobs):
    """Numbers the chip stores for this model (complex weights count twice); None if it cannot be built."""
    base = P.mfg_main_config_id(model)
    _, _, settings = P.resolve(model, "mfg", data, H, base)
    args = copy.deepcopy(DEFAULT_ARGS)
    args.model = P.registry_name(model, base)
    for k, v in {**settings, **knobs}.items():
        setattr(args, k, v)
    try:
        net = B.get_model(args.model, args, n_features=CHANNELS[data], seq_len=96, pred_len=H)
    except Exception:
        return None
    n = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
    if model == "CMoS":   # CMoS builds one spare mapping that forward never uses
        seg = args.cmos_seg_size
        n -= (96 // seg) * (H // seg) + H // seg
    return n


def largest_over_h(model, data, knobs):
    sizes = [stored_numbers(model, data, H, knobs) for H in P.HS]
    return None if None in sizes else max(sizes)


def fill(model, data, key, fixed):
    """(value, stored) of the largest model within the budget, or the smallest model if none fits."""
    steps = SEGMENT_STEPS if ("seg" in key or "period" in key) else COUNT_STEPS
    best, smallest = None, None
    for v in steps:
        n = largest_over_h(model, data, {**fixed, key: v})
        if n is None:
            continue                                   # e.g. a segment that does not divide H
        smallest = smallest or (v, n)
        if n > BUDGET:
            break
        best = (v, n)
    return best, smallest


def main(write):
    out = {}
    for model, cands in P.SIZE_CANDIDATES.items():
        out[model] = []
        print(f"\n== 1k {model}")
        for key, fixed in cands:
            name = "+".join([*(f"{k}={v}" for k, v in fixed.items()), f"fill:{key}"])
            cand = {"name": name, "knobs": {}, "stored": {}, "over_budget": False}
            for data in CHANNELS:
                best, smallest = fill(model, data, key, fixed)
                if best is None:
                    best, cand["over_budget"] = smallest, True
                cand["knobs"][data] = {**fixed, key: best[0]}
                cand["stored"][data] = best[1]
            out[model].append(cand)
            print(f"   {name}" + ("   OVER BUDGET (reference, unranked)" if cand["over_budget"] else ""))
            for data in CHANNELS:
                print(f"      {data:12s} {cand['knobs'][data]}  stored {cand['stored'][data]}")
    if write:
        old = json.load(open(P.SIZE_CAND_FILE, encoding="utf-8")) if os.path.exists(P.SIZE_CAND_FILE) else {}
        old["1k"] = out
        json.dump(old, open(P.SIZE_CAND_FILE, "w", encoding="utf-8"), indent=1)
        print("\nwritten", P.SIZE_CAND_FILE)


if __name__ == "__main__":
    main("--write" in sys.argv)
