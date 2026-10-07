"""Choose baseline settings from the selection runs (docs/EXPERIMENT_PROTOCOL.md §5.3-5.4, rule 6).

    python tools/results/select_verdict.py                  # industrial: report only
    python tools/results/select_verdict.py --write          # + results/chosen_mfg.json
    python tools/results/select_verdict.py --group ltsf     # LTSF cells without an official setting
    python tools/results/select_verdict.py --group ltsf --write   # + results/chosen_ltsf.json

Industrial: per dataset, rank a model's candidates by validation loss (mean over the 4 horizons, seed 2025,
L 96); the best mean rank over the 6 datasets wins and is used everywhere.  LTSF (no-official cells): one
dataset, so the candidates are ranked per horizon and the best mean rank over the 4 horizons wins.
Test errors are never read.  Ties: fewer parameters, then the ETTh1 candidate.

Health check: a candidate whose best epoch (0-based) is <= 2 in more than half of its cells, or that has a
non-finite value, is not eligible.  --write refuses while any candidate of a model is unfinished (a model's
choice is only written when all its candidates are complete), and copies the chosen candidate's seed-2025
selection rows into raw_*.csv as role "main" so the main stage does not rerun them.
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

KEY = ["role", "config_id", "model", "data", "pred_len", "seq_len", "seed"]


def selection_rows(group):
    frames = []
    p = os.path.join(RES, f"select_{group}.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
    r = pd.read_csv(os.path.join(RES, f"raw_{group}.csv"))
    frames.append(r[(r.role == "main") & r.config_id.str.startswith("cand:") & (r.seed == P.SELECT_SEED)
                    & (r.seq_len == 96)].assign(role="select"))
    a = pd.concat(frames, ignore_index=True)
    return a.sort_values("timestamp").drop_duplicates(["model", "config_id", "data", "pred_len"], keep="last")


def choose(s, ids, n_cells, unit):
    """s = selection rows of one model (and one dataset for LTSF). Returns (winner or None, complete, lines)."""
    lines = []
    cnt = s.groupby("config_id").size().reindex(ids, fill_value=0)
    low = s.assign(low=s.best_epoch <= 2).groupby("config_id").low.mean().reindex(ids, fill_value=1.0)
    fin = s.assign(f=np.isfinite(s.val_loss)).groupby("config_id").f.all().reindex(ids, fill_value=False)
    complete = bool((cnt == n_cells).all())
    ok = [c for c in ids if cnt[c] == n_cells and fin[c] and low[c] <= 0.5]
    for c in ids:
        if c not in ok:
            why = "unfinished" if cnt[c] < n_cells else ("non-finite" if not fin[c] else "health check")
            lines.append(f"   not eligible {c}: {why} (cells {cnt[c]}/{n_cells}, best_epoch<=2 share {low[c]:.2f})")
    if not ok:
        return None, complete, lines
    v = s[s.config_id.isin(ok)].groupby([unit, "config_id"]).val_loss.mean().unstack()
    rank = v.rank(axis=1, method="average").mean()
    par = s.groupby("config_id").num_params.max()
    order = sorted(ok, key=lambda c: (round(rank[c], 6), par[c], c != "cand:ETTh1"))
    for c in order:
        lines.append(f"   {c:18s} mean val rank {rank[c]:.2f}   max params {int(par[c]):>9,}")
    return order[0], complete, lines


def main(group, write):
    a = selection_rows(group)
    chosen, copy = {}, []
    if group == "mfg":
        units = [(m, None, ["cand:" + c for c in cs]) for m, cs in P.CANDIDATES.items() if len(cs) > 1]
    else:
        units = [(m, d, P.ltsf_candidates(m, d)) for m in P.LIGHT for d in P.LTSF if len(P.ltsf_candidates(m, d)) > 1]
    for m, d, ids in units:
        s = a[(a.model == m) & a.config_id.isin(ids)]
        if d is not None:
            s = s[s.data == d]
        n = 24 if group == "mfg" else 4
        win, complete, lines = choose(s, ids, n, "data" if group == "mfg" else "pred_len")
        tag = m if d is None else f"{m}|{d}"
        print(f"\n== {tag}: {len(ids)} candidates" + ("" if complete else "  (INCOMPLETE - not written)"))
        print("\n".join(lines))
        if win is None:
            continue
        print(f"   -> {win}")
        if complete:
            chosen[tag] = win
            rows = s[s.config_id == win]
            copy.append(rows)
    if not write:
        print("\n(report only; --write records the complete choices)")
        return
    path = P.CHOSEN_FILE if group == "mfg" else P.CHOSEN_LTSF_FILE
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    old.update(chosen)
    json.dump(old, open(path, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(f"\nwritten {path}: {len(chosen)} choice(s)")
    if copy:
        add = pd.concat(copy, ignore_index=True).assign(role="main")
        rp = os.path.join(RES, f"raw_{group}.csv")
        r = pd.read_csv(rp)
        allr = pd.concat([r, add[r.columns.intersection(add.columns)]], ignore_index=True)
        allr = allr.sort_values("timestamp").drop_duplicates(KEY, keep="last").sort_values(KEY)
        allr.to_csv(rp, index=False)
        print(f"raw_{group}.csv: {len(r)} -> {len(allr)} rows (chosen seed-{P.SELECT_SEED} selection rows reused as main)")


if __name__ == "__main__":
    g = "ltsf" if "--group" in sys.argv and sys.argv[sys.argv.index("--group") + 1] == "ltsf" else "mfg"
    main(g, "--write" in sys.argv)
