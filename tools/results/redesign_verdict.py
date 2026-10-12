"""Stage verdict for docs/prereg_redesign_1012.md - VALIDATION loss only (test columns are never read).

    python tools/results/redesign_verdict.py --stage 1
    python tools/results/redesign_verdict.py --stage 2 --winner rd:mean

Per dataset the candidates are ranked by validation loss (mean over the 4 horizons, seed 2025, L 96); the best
mean rank over the 6 datasets wins.  Ties: fewer parameters, then the pre-registered order A > B > D > E > C.
Health check as select_verdict.py: non-finite, or best epoch (0-based) <= 2 in more than half the cells.
Reads raw_mfg.csv plus not-yet-merged shards in results/_incoming/.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402

ORDER = ["frozen", "rd:mean", "rd:learnlevel", "rd:mixstats", "rd:norevin"]
COLS = ["config_id", "data", "pred_len", "val_loss", "best_epoch", "num_params", "timestamp"]


def rows():
    r = pd.read_csv(os.path.join(ROOT, "results", "raw_mfg.csv"))
    a = r[(r.model == "FEATHer_mix") & (r.seq_len == 96) & (r.seed == P.SELECT_SEED)
          & (((r.role == "main") & (r.config_id == "frozen")) | (r.role == "redesign"))][COLS]
    frames = [a]
    for p in glob.glob(os.path.join(ROOT, "results", "_incoming", "*_redesign.csv")):
        d = pd.read_csv(p, on_bad_lines="skip")
        if len(d):
            d["config_id"] = d.exp_tag.astype(str).str.split("|", n=1).str[1]
            frames.append(d[(d.model == "FEATHer_mix") & (d.seed == P.SELECT_SEED) & (d.seq_len == 96)][COLS])
    a = pd.concat(frames, ignore_index=True)
    return a.sort_values("timestamp").drop_duplicates(["config_id", "data", "pred_len"], keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--winner", default=None, help="stage 2: the stage-1 winner config_id")
    a = ap.parse_args()
    if a.stage == 1:
        ids = ["frozen"] + P.REDESIGN_STAGE1
    else:
        w = a.winner
        base = "rd:frozen" if w == "frozen" else w
        ids = [w, f"{base}@lr0.0003", f"{base}@lr0.003"]
    s = rows()
    s = s[s.config_id.isin(ids)]
    n = len(P.MFG) * len(P.HS)
    cnt = s.groupby("config_id").size().reindex(ids, fill_value=0)
    low = s.assign(low=s.best_epoch <= 2).groupby("config_id").low.mean().reindex(ids, fill_value=1.0)
    fin = s.assign(f=np.isfinite(s.val_loss)).groupby("config_id").f.all().reindex(ids, fill_value=False)
    for c in ids:
        print(f"  {c:<26} cells {cnt[c]:>2}/{n}  best_epoch<=2 share {low[c]:.2f}  finite {bool(fin[c])}")
    if not (cnt == n).all():
        print("unfinished - no verdict yet")
        return
    ok = [c for c in ids if fin[c] and low[c] <= 0.5]
    v = s[s.config_id.isin(ok)].groupby(["data", "config_id"]).val_loss.mean().unstack()[ok]
    rank = v.rank(axis=1, method="average")
    pd.set_option("display.width", 200)
    print("\nvalidation loss (H mean):\n" + v.round(4).to_string())
    print("\nrank per dataset:\n" + rank.to_string())
    mr = rank.mean()
    npar = s.groupby("config_id").num_params.mean()
    pos = lambda c: ORDER.index(c.split("@")[0]) if c.split("@")[0] in ORDER else (
        ORDER.index("frozen") if c.startswith("rd:frozen") else 99)
    best = sorted(ok, key=lambda c: (mr[c], npar[c], pos(c)))
    print("\nmean rank: " + "  ".join(f"{c} {mr[c]:.2f}" for c in best))
    print(f"\nWINNER (stage {a.stage}): {best[0]}")


if __name__ == "__main__":
    main()
