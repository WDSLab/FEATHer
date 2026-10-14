"""Pick (size candidate, lr) per tier and model for the size-matched comparison (EXPERIMENT_PROTOCOL.md §10).

    python tools/results/size_verdict.py            # report
    python tools/results/size_verdict.py --write    # -> docs/chosen_size.json + results/analysis/size_selection_audit.csv

A trial = one candidate (docs/size_candidates.json) with one lr multiplier.  Validation only, seed 2025, H96 + H720:
per dataset the trials are ranked by the mean validation loss, and the best mean rank over the 6 datasets wins
(ties: the order in which trials are listed).  A trial is not eligible if a value is not finite or if its best epoch
is <= 2 in more than half of its cells (same health check as select_verdict.py).  Test errors are never read.
Only models whose trials are all finished are written.  The audit file keeps every trial (the record behind the pick).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
AUDIT = os.path.join(RES, "analysis", "size_selection_audit.csv")


def selection_rows():
    frames = [pd.read_csv(os.path.join(RES, "raw_mfg.csv"))]
    for f in glob.glob(os.path.join(RES, "_incoming", "*.csv")):
        d = pd.read_csv(f, on_bad_lines="skip")
        if "exp_tag" in d and len(d):
            d["role"], d["config_id"] = zip(*[P.split_exp_tag(t) for t in d.exp_tag])
            frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a = a[(a.role == "sizesel") & (a.seed == P.SELECT_SEED) & a.pred_len.isin(P.SIZE_SELECT_HS)]
    return a.drop_duplicates(P.KEY, keep="last")


def judge(rows, tier, model):
    """Table of trials (candidate, lr, cells, eligible, mean_val_rank) for one model, best first."""
    cands = P.size_candidates(tier, model)
    mults = P.SIZE_LR_BY_N[len(cands)]
    table = []
    for c in cands:
        for mult in mults:
            ids = [P.size_config_id(tier, model, mult, c["knobs"][d]) for d in P.MFG]
            r = rows[(rows.model == model) & rows.config_id.isin(ids)]
            healthy = np.isfinite(r.val_loss).all() and (r.best_epoch <= 2).mean() <= 0.5
            table.append(dict(tier=tier, model=model, candidate=c["name"], lr_mult=mult, cells=len(r),
                              eligible=len(r) == len(P.MFG) * len(P.SIZE_SELECT_HS) and healthy,
                              over_budget=c["over_budget"], val=r.groupby("data").val_loss.mean()))
    t = pd.DataFrame(table)
    t["mean_val_rank"] = np.nan
    ok = t[t.eligible]
    if len(ok):
        ranks = pd.DataFrame(list(ok.val), index=ok.index).rank().mean(axis=1)   # rank per dataset, then mean
        t.loc[ok.index, "mean_val_rank"] = ranks
    t["lr_edge"] = t.lr_mult.isin([min(mults), max(mults)])
    return t.drop(columns="val").sort_values("mean_val_rank", kind="stable")


def main(write):
    rows = selection_rows()
    chosen, audit = {}, []
    for tier in P.SIZE_TIERS:
        for model in P.SIZE_CANDIDATES:
            if not P.size_candidates(tier, model):
                continue
            t = judge(rows, tier, model)
            done = bool((t.cells == len(P.MFG) * len(P.SIZE_SELECT_HS)).all())
            best = t.iloc[0]
            t["pick"] = done & (t.index == best.name)
            audit.append(t)
            print(f"\n== {tier} {model}: cells {t.cells.sum()} / {len(t) * len(P.MFG) * len(P.SIZE_SELECT_HS)}"
                  + ("" if done else "  (unfinished)") + ("  [over budget: reference]" if best.over_budget else ""))
            print(t[["candidate", "lr_mult", "cells", "eligible", "mean_val_rank", "pick"]].to_string(index=False))
            if done:
                chosen[f"{tier}|{model}"] = {"cand": best.candidate, "mult": float(best.lr_mult)}
    if write:
        json.dump(chosen, open(P.CHOSEN_SIZE_FILE, "w", encoding="utf-8"), indent=1, sort_keys=True)
        if audit:
            P.write_csv_safely(pd.concat(audit, ignore_index=True), AUDIT)
        print(f"\nwritten {P.CHOSEN_SIZE_FILE} ({len(chosen)} choices) and {AUDIT}")


if __name__ == "__main__":
    main("--write" in sys.argv)
