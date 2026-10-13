"""lr choice of the size-matched comparison (2026-10-13, docs/EXPERIMENT_PROTOCOL.md §10) - validation only.

    python tools/results/size_verdict.py            # report
    python tools/results/size_verdict.py --write    # -> docs/chosen_size.json (commit it)

Per (tier, model): the lr multipliers (protocol.SIZE_LR_MULT) are ranked per dataset by the validation loss
(mean over the 4 H, seed 2025, role sizesel); the best mean rank over the 6 datasets wins - the same rule as the
industrial setting choice (select_verdict.py).  Health check as there: a multiplier whose best epoch (0-based) is
<= 2 in more than half of its cells, or with a non-finite value, is not eligible.  Ties: multiplier 1, then the
smaller one.  Test errors are never read.  --write records only (tier, model) units whose cells are all complete.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def rows():
    frames = [pd.read_csv(os.path.join(RES, "raw_mfg.csv"))]
    inc = os.path.join(RES, "_incoming")
    for f in os.listdir(inc) if os.path.isdir(inc) else []:
        if f.endswith(".csv"):
            d = pd.read_csv(os.path.join(inc, f), on_bad_lines="skip")
            if "exp_tag" in d and len(d):
                rc = [P.split_exp_tag(t) for t in d.exp_tag]
                d["role"], d["config_id"] = [r for r, _ in rc], [c for _, c in rc]
                frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a = a[(a.role == "sizesel") & (a.seed == P.SELECT_SEED) & (a.seq_len == 96)]
    return a.drop_duplicates(P.KEY, keep="last")


def main(write):
    a = rows()
    chosen = {}
    for tier in P.SIZE_TIERS:
        for m in P.SIZE_DIALS:
            want = {d: {mult: P.size_config_id(tier, m, d, mult) for mult in P.SIZE_LR_MULT} for d in P.MFG}
            if any(c is None for v in want.values() for c in v.values()):
                print(f"== {tier} {m}: dial values incomplete - skipped")
                continue
            s = a[a.model == m].copy()
            s["mult"] = [P._sz(c)[2] if str(c).startswith("sz") else np.nan for c in s.config_id]
            s = s[[want.get(d, {}).get(mu) == c for d, mu, c in zip(s.data, s.mult, s.config_id)]]
            cnt = s.groupby("mult").size().reindex(P.SIZE_LR_MULT, fill_value=0)
            complete = bool((cnt == 24).all())
            low = s.assign(l=s.best_epoch <= 2).groupby("mult").l.mean().reindex(P.SIZE_LR_MULT, fill_value=1.0)
            fin = s.assign(f=np.isfinite(s.val_loss)).groupby("mult").f.all().reindex(P.SIZE_LR_MULT, fill_value=False)
            ok = [mu for mu in P.SIZE_LR_MULT if cnt[mu] == 24 and fin[mu] and low[mu] <= 0.5]
            print(f"== {tier} {m}: cells {dict(cnt)}" + ("" if complete else "  (INCOMPLETE - not written)"))
            for mu in P.SIZE_LR_MULT:
                if mu not in ok:
                    why = "unfinished" if cnt[mu] < 24 else ("non-finite" if not fin[mu] else "health check")
                    print(f"   not eligible x{mu:g}: {why} (best_epoch<=2 share {low[mu]:.2f})")
            if not ok:
                continue
            v = s[s.mult.isin(ok)].groupby(["data", "mult"]).val_loss.mean().unstack()
            rank = v.rank(axis=1, method="average").mean()
            order = sorted(ok, key=lambda mu: (round(rank[mu], 6), mu != 1, mu))
            for mu in order:
                print(f"   x{mu:<4g} mean val rank {rank[mu]:.2f}")
            print(f"   -> x{order[0]:g}")
            if complete:
                chosen[f"{tier}|{m}"] = order[0]
    if write:
        old = json.load(open(P.CHOSEN_SIZE_FILE, encoding="utf-8")) if os.path.exists(P.CHOSEN_SIZE_FILE) else {}
        old.update(chosen)
        json.dump(old, open(P.CHOSEN_SIZE_FILE, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print(f"\nwritten {P.CHOSEN_SIZE_FILE}: {len(chosen)} choice(s)")
    else:
        print("\n(report only; --write records the complete choices)")


if __name__ == "__main__":
    main("--write" in sys.argv)
