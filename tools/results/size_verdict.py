"""Pick one search trial per tier, model and dataset for the size-matched comparison (configs/experiments/size.yaml).

    python tools/results/size_verdict.py            # report
    python tools/results/size_verdict.py --write    # -> docs/chosen_size.json + results/analysis/size_selection_audit.csv

Validation only, seed 2025, H96 + H720: per dataset the trials are ranked by validation loss at each H and the best mean
rank over the two H wins (ties: trial order).  A trial with a non-finite value on that
dataset is not eligible (only divergence counts as failure, protocol §5.6).  Test errors are never read.  A (model,
dataset) pair is written only when all its trials are finished.  lr_edge = the chosen lr lies in the lowest or highest
decade of the common range (a model whose choice sits there on half of the datasets or more gets the range extended on
that side, same number of extra trials).  The audit file keeps every trial.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402

RES = os.path.join(P.ROOT, "results")
AUDIT = os.path.join(RES, "analysis", "size_selection_audit.csv")
VAL = [f"val_H{h}" for h in P.SIZE_SELECT_HS]


def selection_rows():
    frames = [pd.read_csv(os.path.join(RES, "raw_mfg.csv"))]
    for f in glob.glob(os.path.join(RES, "_incoming", "*.csv")):
        d = pd.read_csv(f, on_bad_lines="skip")
        if "exp_tag" in d and len(d):
            d["role"], d["config_id"] = zip(*[P.split_exp_tag(t) for t in d.exp_tag])
            frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a = a[(a.role == "sizesel") & (a.seed == P.SELECT_SEED) & a.pred_len.isin(P.SIZE_SELECT_HS)]
    a = a[a.hparams.astype(str).str.contains('"budget":')]   # runs with the current search budget only
    return a.drop_duplicates(P.KEY, keep="last")


def judge(rows, tier, model, data):
    """Table of the trials of one (tier, model, dataset), best first."""
    lo, hi = (math.log10(x) for x in P.size_lr_range(tier, model))
    table = []
    for t in P.size_trials(tier, model):
        cid = P.size_config_id(tier, t["id"], t["knobs"][data])
        r = rows[(rows.model == model) & (rows.data == data) & (rows.config_id == cid)].set_index("pred_len")
        val = [r.val_loss.get(h, np.nan) for h in P.SIZE_SELECT_HS]
        lr = t["knobs"][data]["lr"]
        table.append(dict(tier=tier, model=model, data=data, trial=t["id"], lr=lr,
                          lr_edge=not lo + 1 <= math.log10(lr) <= hi - 1, stored=t["stored"][data], cells=len(r),
                          eligible=len(r) == len(VAL) and bool(np.isfinite(val).all()), **dict(zip(VAL, val))))
    t = pd.DataFrame(table)
    t["mean_val_rank"] = np.nan
    ok = t[t.eligible]
    t.loc[ok.index, "mean_val_rank"] = ok[VAL].rank().mean(axis=1)
    return t.sort_values("mean_val_rank", kind="stable")


def main(write):
    rows = selection_rows()
    chosen, audit = {}, []
    for tier in P.SIZE_TIERS:
        for model in P.SIZE_MODELS:
            if not P.size_trials(tier, model):
                continue
            for data in P.MFG:
                t = judge(rows, tier, model, data)
                done = bool((t.cells == len(VAL)).all())
                best = t.iloc[0]
                picked = done and np.isfinite(best.mean_val_rank)
                t["pick"] = picked & (t.index == best.name)
                audit.append(t)
                print(f"== {tier} {model} {data}: pick {best.trial} lr {best.lr:g} stored {best.stored}"
                      + ("  [lr at range edge]" if best.lr_edge else "")
                      + ("" if done else f"  (unfinished: {int(t.cells.sum())} / {len(t) * len(VAL)} cells)"))
                if picked:
                    chosen[f"{tier}|{model}|{data}"] = best.trial
    if write:
        json.dump(chosen, open(P.CHOSEN_SIZE_FILE, "w", encoding="utf-8"), indent=1, sort_keys=True)
        if audit:
            P.write_csv_safely(pd.concat(audit, ignore_index=True), AUDIT)
        print(f"\nwritten {P.CHOSEN_SIZE_FILE} ({len(chosen)} choices) and {AUDIT}")


if __name__ == "__main__":
    main("--write" in sys.argv)
