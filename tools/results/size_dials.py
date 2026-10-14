"""Dial values of the size-matched comparison from the chip builds (docs/EXPERIMENT_PROTOCOL.md §10).

    python tools/results/size_dials.py            # report
    python tools/results/size_dials.py --write    # -> docs/size_dials.json (commit it)
    python tools/results/size_dials.py --write --tier 1k   # write one tier only (the other keeps its file value)

Sources (real firmware builds, deployment/board/chip_verdict.py one_cell):
  results/board/chip_verdict_budget1k_20261013.csv  1K tier (value = largest with stored numbers <= 1,000 at all H)
  results/board/chip_verdict_sizes_20261013.csv     purpose "1k" (FITS) and purpose "max" (chip-max search)
A value counts only when its four H were built on the model's CURRENT base setting (docs/chosen_mfg.json) and all
four fit 128 KB flash / 20 KB SRAM.  Chip-max = the largest such model: the smallest segment for CMoS, the largest
value for every other dial.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P   # noqa: E402

# the single size hyperparameter per model and the file of its chip-build values (input for the chip-max candidates)
SIZE_DIALS = {"CMoS": "cmos_seg_size", "SparseTSF_MLP": "sparsetsf_d_model", "TimeBase": "timebase_basis_num",
              "PhaseFormer": "phaseformer_latent_dim", "DiPE_Linear": "dipe_num_experts", "FITS": "fits_cut_freq",
              "FEATHer_mix": "head_seg"}
SIZE_DIALS_FILE = os.path.join(os.path.dirname(P.SIZE_CAND_FILE), "size_dials.json")
BOARD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "board")
B2R = {"cmos": "CMoS", "stsf_mlp": "SparseTSF_MLP", "timebase": "TimeBase", "phaseformer": "PhaseFormer",
       "dipe": "DiPE_Linear", "fits": "FITS", "feather_mix": "FEATHer_mix"}
SMALLER_IS_BIGGER = {"CMoS", "FEATHer_mix"}   # segment length: a smaller segment = a larger model


def load():
    a = pd.read_csv(os.path.join(BOARD, "chip_verdict_budget1k_20261013.csv")).assign(purpose="1k")
    p = os.path.join(BOARD, "chip_verdict_sizes_20261013.csv")
    b = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=a.columns)
    x = pd.concat([a, b], ignore_index=True)
    x = x[x.model.isin(B2R)].assign(reg=lambda d: d.model.map(B2R))
    base = {r: (P.mfg_main_config_id(r) or "").replace("cand:", "") for r in B2R.values()}
    x = x[[str(c).split("+")[0] == base[r] for c, r in zip(x.config, x.reg)]]      # current base setting only
    return x.sort_values("timestamp").drop_duplicates(["purpose", "model", "data", "H", "value"], keep="last")


def main(write, only=None):
    x = load()
    out = {t: {} for t in P.SIZE_TIERS}
    for (purpose, reg, d), g in x.groupby(["purpose", "reg", "data"]):
        ok = g.groupby("value").apply(lambda s: set(s.H[s.verdict == "FIT"]) >= set(P.HS))
        good = sorted(v for v, f in ok.items() if f)
        if not good:
            print(f"  [none] {purpose} {reg} {d}: no value fits all four H")
            continue
        tier = "1k" if purpose == "1k" else "max"
        if tier == "1k" and len(good) > 1:
            print(f"  [warn] 1k {reg} {d}: several values built {good}; using the largest model")
        v = (good[0] if reg in SMALLER_IS_BIGGER else good[-1])
        out[tier].setdefault(reg, {})[d] = int(v)
    for t in P.SIZE_TIERS:
        print(f"\n== {t}")
        for reg in SIZE_DIALS:
            row = out[t].get(reg, {})
            print(f"  {reg:14s} {SIZE_DIALS[reg]:24s} " + "  ".join(f"{d}={row.get(d, '-')}" for d in P.MFG))
    missing = [(t, r, d) for t in P.SIZE_TIERS for r in SIZE_DIALS for d in P.MFG if d not in out[t].get(r, {})]
    print(f"\nmissing {len(missing)} of {len(P.SIZE_TIERS) * len(SIZE_DIALS) * len(P.MFG)}")
    if write:
        if only:
            old = json.load(open(SIZE_DIALS_FILE, encoding="utf-8")) if os.path.exists(SIZE_DIALS_FILE) else {}
            out = {t: (out[t] if t == only else old.get(t, {})) for t in P.SIZE_TIERS}
        json.dump(out, open(SIZE_DIALS_FILE, "w", encoding="utf-8"), indent=1, sort_keys=True)
        print("written", SIZE_DIALS_FILE, "(cells still missing stay absent; run.py skips them)")


if __name__ == "__main__":
    main("--write" in sys.argv, sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else None)
