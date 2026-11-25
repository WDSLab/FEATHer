"""Merge run shards from results/_incoming/ into the managed result files.

    python tools/results/merge_incoming.py            # merge (shards stay; re-merging is harmless)
    python tools/results/merge_incoming.py --archive  # merge, then move the shards to results/_archive/incoming_merged/

Shards are already in the final layout (protocol.RESULT_FIELDS; an older-layout shard is converted by
protocol.read_results).  Routing: <mfg|ltsf>_<stage>.csv by the dataset group and the stage.  Duplicate
runs (protocol.KEY) keep the newest row.  Copy shards from other machines into results/_incoming/ first (any file name).
"""
import glob
import os
import shutil
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402


def main(archive):
    shards = sorted(glob.glob(os.path.join(RES, "_incoming", "*.csv")))
    if not shards:
        print("no shards in results/_incoming/")
        return
    new = []
    for p in shards:
        d = P.read_results(p)
        if d.empty:
            continue
        # a row cut off by a writer that was still running has NaN key fields -> drop it (the cell stays
        # missing and the next run.py fills it); failed runs (MSE NaN, best_epoch -1) are complete rows
        cut = d[P.KEY + ["time"]].isna().any(axis=1)
        if cut.any():
            print(f"  {os.path.basename(p)}: {int(cut.sum())} incomplete row(s) dropped")
        new.append(d[~cut])
        print(f"  {os.path.basename(p)}: {int((~cut).sum())} rows")
    if not new:
        return
    new = pd.concat(new, ignore_index=True)
    group = new.data.map({**{x: "mfg" for x in P.MFG}, **{x: "ltsf" for x in P.LTSF}})
    if group.isna().any():
        print(f"  [WARN] {int(group.isna().sum())} row(s) of datasets outside the protocol are not merged:",
              sorted(new[group.isna()].data.unique()))
    # guard: one setting per cell -> the parameter count must not differ between seeds of a cell
    g = new.groupby([k for k in P.KEY if k != "seed"]).params.nunique()
    if (g > 1).any():
        print("  [WARN] cells whose seeds report different params (mixed settings?):")
        print(g[g > 1].to_string())
    for (grp, stage), add in new[group.notna()].groupby([group[group.notna()], "stage"]):
        name = f"{grp}_{stage}.csv"
        path = os.path.join(RES, name)
        old = P.read_results(path) if os.path.exists(path) else pd.DataFrame(columns=P.RESULT_FIELDS)
        allr = pd.concat([old, add], ignore_index=True)
        n0 = len(old)
        allr = allr.sort_values("time", kind="stable").drop_duplicates(P.KEY, keep="last")
        P.write_csv_safely(allr[P.RESULT_FIELDS].sort_values(P.KEY, kind="stable"), path)
        print(f"{name}: {n0} -> {len(allr)} rows")
    if archive:
        dst = os.path.join(RES, "_archive", "incoming_merged")
        os.makedirs(dst, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for p in shards:
            shutil.move(p, os.path.join(dst, f"{stamp}_{os.path.basename(p)}"))
        print(f"moved {len(shards)} shard(s) to _archive/incoming_merged/")


if __name__ == "__main__":
    main("--archive" in sys.argv)
