"""Appendix table: every hyperparameter of the size-matched baselines, how it is searched (configs/experiments/size.yaml),
its author value and whether it changes the stored count -> results/analysis/size_hparams.csv.

    python tools/results/size_hparams.py [--write]
"""
import os
import sys

import pandas as pd

sys.argv, _argv = sys.argv[:1], sys.argv
import size_trials as S   # noqa: E402  (sets up the paths and the worker defaults)
import protocol as P   # noqa: E402
sys.argv = _argv

OUT = os.path.join(S.ROOT, "results", "analysis", "size_hparams.csv")
TRIES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def tries(value, kind):
    """Values a key is probed at: its search list, else a few values around the author value."""
    v = S.values(kind) if kind != "unused" else None
    if v is not None:
        return v
    if isinstance(value, bool):
        return [False, True]
    if isinstance(value, int):
        return sorted({value, *TRIES})
    if isinstance(value, float):
        return [value / 2, value, value * 2]
    return [value]


def main(write):
    rows = []
    for model, keys in P.SIZE_SPACE.items():
        print(f"\n== {model}")
        for key, kind in keys.items():
            kind_s = ("list " if isinstance(kind, list) else "") + str(kind)
            counts, author = {}, set()
            for data in S.CHANNELS:
                base = {k: S.author_value(model, data, k) for k in keys}   # unused keys too (their author value)
                author.add(str(base[key]))
                for v in tries(base[key], kind):
                    n = S.stored(model, data, {**base, key: v})
                    if n is not None:
                        counts.setdefault(v, set()).add(n)
            sizes = sorted({n for ns in counts.values() for n in ns})
            rows.append(dict(model=model, hparam=key, searched_as=kind_s, author="/".join(sorted(author)),
                             changes_size=len({frozenset(ns) for ns in counts.values()}) > 1,
                             stored_range=f"{sizes[0]}-{sizes[-1]}" if sizes else "-"))
            r = rows[-1]
            print(f"   {key:28s} {kind_s:22s} author {r['author']:10s} changes size: {str(r['changes_size']):5s}"
                  f"  stored {r['stored_range']}")
    if write:
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print("\nwritten", OUT)


if __name__ == "__main__":
    main("--write" in sys.argv)
