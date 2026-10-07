"""Merge run shards from results/_incoming/ into the managed result files (2026-10-07).

    python tools/results/merge_incoming.py            # merge (shards stay; re-merging is harmless)
    python tools/results/merge_incoming.py --archive  # merge, then move the shards to _archive/incoming_merged/

Routing: exp_tag "<role>|<config_id>" (written by run.py) -> role select -> select_{mfg,ltsf}.csv,
otherwise industrial datasets -> raw_mfg.csv, LTSF datasets -> raw_ltsf.csv.  Duplicate cells
(role, config_id, model, data, pred_len, seq_len, seed) keep the newest row.  Copy a shard from the
server into results/_incoming/ first (any file name).
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

KEY = P.KEY


def main(archive):
    shards = sorted(glob.glob(os.path.join(RES, "_incoming", "*.csv")))
    if not shards:
        print("no shards in results/_incoming/")
        return
    new = []
    for p in shards:
        d = pd.read_csv(p, on_bad_lines="skip")
        if d.empty:
            continue
        rc = [P.split_exp_tag(t) for t in d.exp_tag]
        d["role"], d["config_id"], d["source"] = [r for r, _ in rc], [c for _, c in rc], os.path.basename(p)
        bad = d.role.isna()
        if bad.any():
            print(f"  {os.path.basename(p)}: {int(bad.sum())} row(s) without '<role>|<config_id>' exp_tag skipped")
        # a row cut off by a writer that was still running has NaN key fields -> drop it (the cell stays
        # missing and the next run.py fills it); failed runs (MSE NaN, best_epoch -1) are complete rows
        cut = d[~bad][KEY + ["timestamp"]].isna().any(axis=1)
        if cut.any():
            print(f"  {os.path.basename(p)}: {int(cut.sum())} incomplete row(s) dropped")
        d = d[~bad][~cut.values]
        new.append(d)
        print(f"  {os.path.basename(p)}: {len(d)} rows")
    if not new:
        return
    new = pd.concat(new, ignore_index=True)
    stray = ~new.data.isin(P.MFG + P.LTSF)
    if stray.any():
        print(f"  [WARN] {int(stray.sum())} row(s) of datasets outside the protocol are not merged:",
              sorted(new[stray].data.unique()))
    # guard: one setting per cell -> the parameter count must not differ between seeds of a cell
    g = new.groupby(["role", "config_id", "model", "data", "pred_len", "seq_len"]).num_params.nunique()
    if (g > 1).any():
        print("  [WARN] cells whose seeds report different num_params (mixed settings?):")
        print(g[g > 1].to_string())
    targets = {
        "select_mfg.csv": new[(new.role == "select") & new.data.isin(P.MFG)],
        "select_ltsf.csv": new[(new.role == "select") & new.data.isin(P.LTSF)],
        "raw_mfg.csv": new[(new.role != "select") & new.data.isin(P.MFG)],
        "raw_ltsf.csv": new[(new.role != "select") & new.data.isin(P.LTSF)],
    }
    for name, add in targets.items():
        if add.empty:
            continue
        path = os.path.join(RES, name)
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=add.columns)
        allr = pd.concat([old, add], ignore_index=True)
        n0 = len(old)
        allr = allr.sort_values("timestamp").drop_duplicates(KEY, keep="last")
        cols = ["role", "config_id"] + [c for c in allr.columns if c not in ("role", "config_id", "source")] + ["source"]
        P.write_csv_safely(allr[cols].sort_values(KEY), path)
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
