"""Merge run shards from results/_incoming/ into the managed result files (2026-10-07).

    python tools/results/merge_incoming.py            # merge (shards stay; re-merging is harmless)
    python tools/results/merge_incoming.py --archive  # merge, then move the shards to _archive/incoming_merged/

Routing: exp_tag "<role>|<config_id>" (written by run_plan.py) -> role select -> select_{mfg,ltsf}.csv,
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

KEY = ["role", "config_id", "model", "data", "pred_len", "seq_len", "seed"]


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
        sp = d.exp_tag.astype(str).str.split("|", n=1, expand=True)
        if sp.shape[1] != 2 or sp[1].isna().any():
            print(f"  skip {os.path.basename(p)}: exp_tag is not '<role>|<config_id>'")
            continue
        d["role"], d["config_id"], d["source"] = sp[0], sp[1], os.path.basename(p)
        new.append(d)
        print(f"  {os.path.basename(p)}: {len(d)} rows")
    if not new:
        return
    new = pd.concat(new, ignore_index=True)
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
        allr[cols].sort_values(KEY).to_csv(path, index=False)
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
