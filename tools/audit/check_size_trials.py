"""Sanity checks of the drawn size-matched trials (docs/size_trials.json) before they are committed / run.

    python tools/audit/check_size_trials.py [--tier 1k|chip]

Per model: trial count and rejections, every key inside its search space, stored count within the tier cap on every
dataset, no duplicate trial, how many distinct architectures (ignoring lr), the drawn distribution of each list key.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
import protocol as P  # noqa: E402

tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else "1k"
J = json.load(open(P.SIZE_TRIALS_FILE, encoding="utf-8"))
meta_all = J.get("_meta" if tier == "1k" else f"_meta_{tier}", {})
lo, hi = P.SIZE_COMMON["lr"]
for model in P.SIZE_MODELS:
    trials, meta = J.get(tier, {}).get(model), meta_all.get(model, {})
    if trials is None:
        print(f"\n== {model}: not drawn")
        continue
    if not trials:
        print(f"\n== {model}: excluded ({meta.get('rejected')})")
        continue
    cap = meta.get("cap") or {d: 1000 for d in trials[0]["knobs"]}
    space = {k: c for k, c in P.SIZE_SPACE[model].items() if c != "unused"}
    issues, sigs, archs = [], set(), collections.Counter()
    for t in trials:
        for d, k in t["knobs"].items():
            n = t["stored"][d]
            if n is None or n > cap[d]:
                issues.append(f"{t['id']} {d} stored {n} > cap {cap[d]}")
            if not lo <= k["lr"] <= hi:
                issues.append(f"{t['id']} lr {k['lr']}")
            if set(space) - set(k):
                issues.append(f"{t['id']} {d} missing {set(space) - set(k)}")
            for key, kind in space.items():
                allowed = kind if isinstance(kind, list) else P.SIZE_COMMON.get(kind)
                if allowed is not None and k.get(key) not in allowed:
                    issues.append(f"{t['id']} {d} {key}={k.get(key)}")
                if kind == "size" and (not isinstance(k.get(key), int) or k[key] < 1):
                    issues.append(f"{t['id']} {d} {key}={k.get(key)}")
        sig = json.dumps(t["knobs"], sort_keys=True)
        issues += [f"{t['id']} duplicate"] if sig in sigs else []
        sigs.add(sig)
        archs[json.dumps({d: {a: b for a, b in k.items() if a != "lr"} for d, k in t["knobs"].items()}, sort_keys=True)] += 1
    steel = [t["stored"]["Steel"] for t in trials]
    print(f"\n== {model}: {len(trials)} trials, rejected {meta.get('rejected')}, Steel stored {min(steel)}-{max(steel)}")
    print(f"   distinct architectures {len(archs)} / {len(trials)}")
    for key, kind in space.items():
        if kind != "size":
            print(f"   {key}: {dict(collections.Counter(str(t['knobs']['Steel'].get(key)) for t in trials))}")
    print("   ISSUES: " + ("; ".join(issues[:12]) + (f" ... (+{len(issues) - 12})" if len(issues) > 12 else "")
                         if issues else "none"))
