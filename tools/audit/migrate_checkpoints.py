# -*- coding: utf-8 -*-
"""One-time migration: flat checkpoints -> per-model subfolders (2026-07-31).

Old layout:  results/checkpoints/<exp_tag>/<model>_<data>_H<h>_L<L>_s<seed>.pth
New layout:  results/checkpoints/<exp_tag>/<model>/<data>_H<h>_L<L>_s<seed>.pth

Model names contain underscores (DiPE_Linear, LMS_AutoTSF, MDMLP_EIA), so the
filename is split by matching the LONGEST known model prefix, not the first '_'.
Files that match no known model are left untouched and reported.

Dry-run by default; pass --apply to actually move. Idempotent: files already in
a per-model subfolder are ignored.

Usage:
    python tools/audit/migrate_checkpoints.py --exp_tag main            # dry-run
    python tools/audit/migrate_checkpoints.py --exp_tag main --apply
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines import list_models  # noqa: E402

# Known models to match filename prefixes against — active registry plus the
# archived classics, in case their checkpoints exist from an earlier sweep.
KNOWN = sorted(set(list_models()) | {"LSTM", "GRU", "TCN"}, key=len, reverse=True)


def main():
    p = argparse.ArgumentParser(description="Flat -> per-model checkpoint migration")
    p.add_argument("--exp_tag", default="main")
    p.add_argument("--root", default="results/checkpoints")
    p.add_argument("--apply", action="store_true", help="actually move (default: dry-run)")
    args = p.parse_args()

    base = os.path.join(args.root, args.exp_tag)
    if not os.path.isdir(base):
        print(f"[missing] {base} does not exist")
        return

    entries = os.listdir(base)  # snapshot before we create subfolders
    moved = skipped_dir = unmatched = 0
    for fn in entries:
        src = os.path.join(base, fn)
        if not (fn.endswith(".pth") and os.path.isfile(src)):
            skipped_dir += 1
            continue
        model = next((m for m in KNOWN if fn.startswith(m + "_")), None)
        if model is None:
            print(f"  [UNMATCHED] {fn}")
            unmatched += 1
            continue
        rest = fn[len(model) + 1:]                      # strip "<model>_"
        dst_dir = os.path.join(base, model)
        dst = os.path.join(dst_dir, rest)
        if args.apply:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
        moved += 1

    action = "moved" if args.apply else "would move"
    print("-" * 60)
    print(f"  {action}: {moved}  |  unmatched: {unmatched}  |  "
          f"non-file/dir skipped: {skipped_dir}")
    if not args.apply and moved:
        print("  (dry-run — re-run with --apply to execute)")
    if unmatched:
        print("  [WARN] unmatched files left in place — check the model names above")


if __name__ == "__main__":
    main()
