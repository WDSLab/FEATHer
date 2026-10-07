"""Choose each baseline's industrial setting from the selection runs (EXPERIMENT_PROTOCOL.md, rule 3).

    python tools/results/select_verdict.py            # report only
    python tools/results/select_verdict.py --write    # also write results/chosen_mfg.json

Same procedure that fixed FEATHer's configuration: per dataset, rank the candidates by validation loss
(mean over the 4 horizons, seed 2025, L 96); the candidate with the best mean rank over the 6 datasets
wins.  Test errors are never read.  Ties: fewer parameters, then the ETTh1 candidate.
Health check (rule 6): a candidate whose best epoch is <= 2 in more than half of its 24 cells, or
that has a missing / non-finite cell, is not eligible.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402


def selection_rows():
    frames = []
    p = os.path.join(RES, "select_mfg.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
    r = pd.read_csv(os.path.join(RES, "raw_mfg.csv"))
    frames.append(r[(r.role == "main") & r.config_id.str.startswith("cand:") & (r.seed == P.SELECT_SEED)
                    & (r.seq_len == 96)])
    a = pd.concat(frames, ignore_index=True)
    return a.sort_values("timestamp").drop_duplicates(["model", "config_id", "data", "pred_len"], keep="last")


def main(write):
    a = selection_rows()
    chosen = {}
    for m, cands in P.CANDIDATES.items():
        if len(cands) < 2:
            continue
        ids = ["cand:" + c for c in cands]
        s = a[(a.model == m) & a.config_id.isin(ids)]
        cnt = s.groupby("config_id").size().reindex(ids, fill_value=0)
        bad_ep = s.assign(low=s.best_epoch <= 2).groupby("config_id").low.mean().reindex(ids, fill_value=1.0)
        fin = s.assign(f=np.isfinite(s.val_loss)).groupby("config_id").f.all().reindex(ids, fill_value=False)
        ok = [c for c in ids if cnt[c] == 24 and fin[c] and bad_ep[c] <= 0.5]
        print(f"\n== {m}: {len(ids)} candidates, complete+healthy {len(ok)}")
        if len(ok) < len(ids):
            for c in ids:
                if c not in ok:
                    print(f"   not eligible {c}: cells {cnt[c]}/24, finite {bool(fin[c])}, best_epoch<=2 share {bad_ep[c]:.2f}")
        if not ok:
            continue
        v = s[s.config_id.isin(ok)].groupby(["data", "config_id"]).val_loss.mean().unstack()
        rank = v.rank(axis=1, method="average").mean().sort_values()
        par = s.groupby("config_id").num_params.max()
        order = sorted(ok, key=lambda c: (round(rank[c], 6), par[c], c != "cand:ETTh1"))
        for c in order:
            print(f"   {c:18s} mean val rank {rank[c]:.2f}   max params {int(par[c]):>7,}")
        complete = len(ok) == len(ids)
        chosen[m] = order[0]
        print(f"   -> {order[0]}" + ("" if complete else "   (PRELIMINARY: not every candidate finished)"))
    if write:
        json.dump(chosen, open(P.CHOSEN_FILE, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print(f"\nwritten {P.CHOSEN_FILE}")


if __name__ == "__main__":
    main("--write" in sys.argv)
