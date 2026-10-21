"""Draw the random-search trials of the size-matched comparison -> docs/size_trials.json (commit before any run).

    python tools/results/size_trials.py [--tier 1k|chip] [--models CMoS,FITS]   # report
    python tools/results/size_trials.py --write [--tier ...] [--models ...]     # write (other models are kept)

Rule (configs/experiments/size.yaml), per model, 1K tier: 40 trials, all drawn with draw_seed (no author or frozen
setting as a starting trial, so no model starts from settings fitted elsewhere or to these data): lr log-uniform, the
common / model lists, a target size uniform in (0, budget], one size
key (a key that changes the stored count) picked at random; the other size keys are drawn (counts log-uniform from 1 to
the largest value that can fit, never past the key's ceiling) and the picked key takes, per dataset, the value with the
largest stored count <= target.  A draw that cannot be built and run, does not fit on all six datasets, or repeats an
earlier trial is drawn again (counted in the JSON).  "Stored" = the numbers the chip keeps (complex x2, CMoS spare
mapping excluded) at H720 during the search; every accepted trial is checked at all four H.
Chip tier: the same rule with a per-dataset cap instead of 1,000: cap = (128 KB - the model's non-weight flash) / 4,
the non-weight flash measured by one real firmware build of the author setting at H720 (flash proxy); the target is
the same fraction of each dataset's cap; every accepted trial is then built for real at H720 on every dataset
(deployment/board/chip_verdict.one_cell: flash <= 128 KB and SRAM <= 20 KB) and redrawn if it does not fit.
"""
import ast
import json
import math
import os
import random
import re
import sys
import warnings
from argparse import Namespace
from functools import lru_cache

import torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.dirname(os.path.abspath(__file__))]
import protocol as P   # noqa: E402
import baselines as B   # noqa: E402

_argv, sys.argv = sys.argv, ["train.py", "--model", "DLinear", "--data", "Steel", "--pred_len", "96"]
import train as T   # noqa: E402
DEFAULT_ARGS = vars(T.parse_args())                  # worker defaults
sys.argv = _argv

BUDGET = 1000
TIER = "1k"
CAP = {}                                             # dataset -> cap in stored numbers (1k: BUDGET; chip: flash proxy)
CHANNELS = {"Steel": 6, "WindSCADA": 14, "BlastFurnace": 28, "N2O": 14, "MetroPT3": 7, "EWELD": 14}
SIZE_CLASSES = ("size", "segment", "period")
COUNT_MAX = 4096                                     # chip tier: 1 << 16, searched by bisection
MAX_FAILS, PLATEAU = 32, 64                          # stop a count scan: values in a row that do not build / do not change
MAX_DRAWS = 400                                      # draws per trial before giving up
DEPENDENT = {"mixlinear_lpf", "cmos_conv_stride"}    # counts whose ceiling depends on another key (period / kernel)


@lru_cache(maxsize=None)
def _stored(model, data, H, knobs):
    base = P.mfg_main_config_id(model)
    _, _, settings = P.resolve(model, "mfg", data, H, base)
    args = Namespace(**{**DEFAULT_ARGS, "model": P.registry_name(model, base), **settings, **dict(knobs)})
    C = CHANNELS[data]
    try:
        net = B.get_model(args.model, args, n_features=C, seq_len=96, pred_len=H)
        with torch.no_grad():                          # a model that builds but cannot run (e.g. kernel > L) is invalid
            y = net(torch.randn(2, 96, C))             # (values are not checked: DiPE is not finite before training)
    except Exception:
        return None
    if tuple(y.shape) != (2, H, C):
        return None
    n = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
    if model == "CMoS":                                # one spare mapping that forward never uses
        seg = args.cmos_seg_size
        n -= (96 // seg) * (H // seg) + H // seg
    return n


def stored(model, data, knobs, all_h=False):
    """Stored count at H720, the largest horizon (the search); all_h=True: the largest over the four H, None if any
    horizon cannot be built or run (the final check of every accepted trial).  lr is not a model key."""
    k = tuple(sorted((a, b) for a, b in knobs.items() if a != "lr"))
    sizes = [_stored(model, data, H, k) for H in (P.HS if all_h else [max(P.HS)])]
    return None if None in sizes else max(sizes)


# ---- chip tier: real firmware builds (deployment/board, one at a time) ---------------------------------------
BOARD = {"CMoS": "cmos", "SparseTSF_MLP": "stsf_mlp", "TimeBase": "timebase", "PhaseFormer": "phaseformer",
         "DiPE_Linear": "dipe", "FITS": "fits", "SparseTSF": "stsf", "MixLinear": "mixlinear",
         "FEATHer_mix": "feather_mix", "FEATHer_CI": "feather_ci", "FEATHer_CI_noMLP": "feather_ci"}
# training-only keys: no effect on the firmware (left out of the build so equal firmware is built once)
TRAIN_ONLY = {"lr", "cmos_dropout", "cmos_use_pi", "cmos_period", "phaseformer_attn_dropout", "timebase_orthogonal_weight",
              "timebase_use_orthogonal", "dipe_loss_alpha", "lambda_spec", "head_mode"}
CHIP_OFF = [0]
_BUILDS = {}


def chip_build(model, data, knobs, H=720):
    """chip_verdict.one_cell row for a trial's knobs on one dataset (cached per firmware)."""
    sys.path.insert(0, os.path.join(ROOT, "deployment", "board"))
    import chip_verdict as CV
    k = {a: b for a, b in sorted(knobs.items()) if a not in TRAIN_ONLY and not a.endswith("@u")}
    key = (model, data, H, json.dumps(k, sort_keys=True))
    if key not in _BUILDS:
        bm = BOARD[model]
        if bm in ("stsf", "feather_mix", "feather_ci"):          # these ports take the knobs as geometry
            geo = {("period" if a == "sparsetsf_period_len" else a): (int(b) if isinstance(b, bool) else b)
                   for a, b in k.items()}
            _BUILDS[key] = CV.one_cell(bm, data, H, extra_geo=geo)
        else:                                                    # config models: "<author src>+key=value..."
            src = P.mfg_main_config_id(model).replace("cand:", "")
            _BUILDS[key] = CV.one_cell(bm, data, H, config=src + "".join(f"+{a}={b}" for a, b in k.items()))
    return _BUILDS[key]


def chip_caps(model):
    """dataset -> cap in stored numbers: (128 KB - non-weight flash) / 4, the non-weight flash from one real build of
    the author setting at H720 (flash = code + constants + 4 bytes per stored number)."""
    sys.path.insert(0, os.path.join(ROOT, "deployment", "board"))
    import board_prep as bp
    out = {}
    for d in CHANNELS:
        knobs = author_knobs(model, d)
        r = chip_build(model, d, knobs)
        n = stored(model, d, knobs)
        if r.get("flash_B") is None or n is None:
            raise SystemExit(f"{model} {d}: author setting does not build ({r.get('why')})")
        out[d] = (bp.FLASH_BYTES - (r["flash_B"] - 4 * n)) // 4
        print(f"   chip cap {model} {d}: flash {r['flash_B']} B, stored {n} -> cap {out[d]} stored numbers", flush=True)
    return out


def author_value(model, data, key):
    """Value in the model's industrial setting (H 96), else the worker default, else the wrapper's getattr default."""
    lr, _, settings = P.resolve(model, "mfg", data, 96, P.mfg_main_config_id(model))
    if key == "lr":
        return lr
    v = settings.get(key, DEFAULT_ARGS.get(key))
    if v is None:
        folder = "SparseTSF" if model == "SparseTSF_MLP" else P.registry_name(model)
        src = open(os.path.join(ROOT, "baselines", folder, "wrapper.py"), encoding="utf-8").read()
        m = re.search(rf'getattr\(args, "{key}", ([^)]+)\)', src)
        v = ast.literal_eval(m.group(1).strip()) if m else None
    return v


def space(model):
    """{key: kind} of the searched keys (unused ones dropped)."""
    return {k: c for k, c in P.SIZE_SPACE[model].items() if c != "unused"}


def author_knobs(model, data):
    return {k: author_value(model, data, k) for k in space(model)}


def values(kind):
    """List a non-count key is drawn from; None for a count (1 .. ceiling)."""
    return kind if isinstance(kind, list) else P.SIZE_COMMON.get(kind)


def ceiling(key, knobs):
    """Largest meaningful value of a count key (larger values are clamped by the model or give the same model)."""
    if key == "fits_cut_freq":
        return 96 // 2                                   # protocol rule 7: cut frequency <= L / 2
    if key == "mixlinear_lpf" and "mixlinear_period_len" in knobs:
        return math.ceil(96 / knobs["mixlinear_period_len"])   # the wrapper clamps lpf to the segment count
    if key in ("cmos_kernel_size", "point_kernel"):
        return 96
    if key == "cmos_conv_stride" and "cmos_kernel_size" in knobs:
        return 96 - knobs["cmos_kernel_size"] + 1        # larger strides all give one conv output
    return COUNT_MAX


def scan(model, data, key, kind, knobs, stop=None):
    """{value: stored} of `key`, the other keys at `knobs`, from the smallest model to the largest.  A count scan stops
    once it has grown past the budget, after MAX_FAILS values in a row that cannot build, or after PLATEAU equal ones."""
    v = values(kind)
    count = v is None
    out, fails, same, prev, lo = {}, 0, 0, None, None
    for x in (range(1, ceiling(key, knobs) + 1) if count else sorted(v, reverse=True)):
        n = stored(model, data, {**knobs, key: x})
        if n is None:
            fails += 1
            if count and fails >= MAX_FAILS:
                break
            continue
        fails, out[x] = 0, n
        same = same + 1 if n == prev else 0
        prev, lo = n, n if lo is None else min(lo, n)
        if count and ((n > (stop or CAP[data]) and n > lo and len(out) > 2) or same >= PLATEAU):
            break
    return out


def bisect(model, data, key, knobs, cap):
    """(value, stored) of the largest count <= cap, for a count whose size grows with the value (chip tier); an
    invalid value (e.g. an even kernel) is replaced by the one below it."""
    def at(v):
        for x in (v, v - 1):
            n = stored(model, data, {**knobs, key: x}) if x >= 1 else None
            if n is not None:
                return x, n
        return None
    best, top = at(1), ceiling(key, knobs)
    if best is None or best[1] > cap:
        return None
    lo, hi = 1, 2
    while hi <= top and (r := at(hi)) and r[1] <= cap:
        best, lo, hi = r, hi, hi * 2
    hi = min(hi, top + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = at(mid)
        if r and r[1] <= cap:
            best, lo = r, mid
        else:
            hi = mid
    return best


def fill(model, data, key, kind, knobs, cap=None):
    """(value, stored) of the largest model <= cap (ties: the smaller value), or None."""
    cap = CAP[data] if cap is None else cap
    if TIER == "chip" and kind == "size" and key not in DEPENDENT:
        return bisect(model, data, key, knobs, cap)
    s = {v: n for v, n in scan(model, data, key, kind, knobs).items() if n <= cap}
    if not s:
        return None
    v = max(s, key=lambda x: (s[x], -x))
    return v, s[v]


def size_keys(model):
    """Size-class keys whose value changes the stored count (author setting, any dataset), yaml order."""
    return [k for k, kind in space(model).items() if kind in SIZE_CLASSES and any(
        len(set(scan(model, d, k, kind, author_knobs(model, d), stop=BUDGET).values())) > 1 for d in CHANNELS)]


def smallest(model, data, key, kind):
    """The key's value with the fewest stored numbers (author setting elsewhere)."""
    s = scan(model, data, key, kind, author_knobs(model, data))
    return min(s, key=lambda v: (s[v], v)) if s else None


def count_cap(model, key):
    """Largest count of `key` that can fit on all six datasets: every other searched key (size, switch, list) in turn
    at the value that gives the smallest model while `key` still changes the size (e.g. the MLP stays on for its
    width)."""
    sp, cap = space(model), COUNT_MAX
    for d in CHANNELS:
        base = author_knobs(model, d)
        for k in sp:
            if k == key:
                continue
            best = None
            cand = ([1] if TIER == "chip" and sp[k] == "size" and k not in DEPENDENT       # chip: a count's smallest model is 1
                    else scan(model, d, k, sp[k], {**base, key: 1}))
            for v in cand:
                b = {**base, k: v}
                n1, n2 = stored(model, d, {**b, key: 1}), stored(model, d, {**b, key: 2})
                if n1 is not None and n1 != n2 and (best is None or n1 < best[1]):
                    best = (v, n1)
            if best:
                base[k] = best[0]
        if TIER == "chip":
            f = bisect(model, d, key, base, CAP[d])
            cap = min(cap, f[0] if f else 1)
            continue
        fits = [v for v, n in scan(model, d, key, "size", base).items() if n <= CAP[d]]
        cap = min(cap, max(fits) if fits else 1)
    return cap


def draw_value(rng, kind, cap=None):
    if kind == "size":
        return int(round(math.exp(rng.uniform(0, math.log(cap)))))
    return rng.choice(values(kind))


def draw_trial(rng, model, keys, caps):
    """One random draw -> trial dict, or None if it cannot fit on all datasets."""
    sp = space(model)
    lo, hi = (math.log10(x) for x in P.size_lr_range(TIER, model))
    drawn = {"lr": float(f"{10 ** rng.uniform(lo, hi):.2g}")}
    target = rng.uniform(0, BUDGET)                    # chip: the same fraction of each dataset's cap
    fill_key = rng.choice(keys) if keys else None
    frac = {}                                          # dependent counts: a log-position u in [0, 1) of 1 .. ceiling,
    for k, kind in sp.items():                         # resolved once the key they depend on is known (maybe filled)
        if k in DEPENDENT and k != fill_key:
            frac[k] = rng.random()
        elif k != fill_key:
            drawn[k] = draw_value(rng, kind, caps.get(k))
    t = {"fill": fill_key, "target": round(target), "drawn": {**drawn, **{f"{k}@u": u for k, u in frac.items()}},
         "knobs": {}, "stored": {}}
    tgt = {d: target if CAP[d] == BUDGET else target / BUDGET * CAP[d] for d in CHANNELS}
    for d in CHANNELS:
        knobs = {**drawn, **{k: 1 for k in frac}}
        if fill_key:
            f = fill(model, d, fill_key, sp[fill_key], knobs, cap=tgt[d])
            if f is None:
                return None
            knobs[fill_key] = f[0]
        for k, u in frac.items():
            knobs[k] = max(1, int(round(ceiling(k, knobs) ** u)))
        n = stored(model, d, knobs)
        if n is None or n > tgt[d]:
            return None
        t["knobs"][d], t["stored"][d] = knobs, n
    for d in CHANNELS:                                 # accepted at H720: every horizon must build, run and fit
        n = stored(model, d, t["knobs"][d], all_h=True)
        if n is None or n > tgt[d]:
            return None
        t["stored"][d] = n
    if TIER == "chip":                                 # the real firmware at H720 on every dataset
        for d in CHANNELS:
            r = chip_build(model, d, t["knobs"][d])
            if r.get("verdict") != "FIT":
                CHIP_OFF[0] += 1
                return None
            t.setdefault("chip", {})[d] = {"flash_B": r["flash_B"], "ram_B": r.get("ram_stream_B") or r["ram_B"]}
    return t


def build(model):
    global CAP
    sp = space(model)
    CAP = {d: BUDGET for d in CHANNELS}
    keys = size_keys(model)
    if TIER == "chip":
        CAP = chip_caps(model)
    caps = {k: count_cap(model, k) for k, kind in sp.items() if kind == "size" and k not in DEPENDENT}
    # edge rule (size.yaml lr_widened): the committed first trials stay, `trials` more come from the widened lr range
    wide = f"{TIER}|{model}" in P.SIZE_LR_WIDENED
    trials = P.size_trials(TIER, model)[:P.SIZE_TRIALS] if wide else []
    if wide and len(trials) != P.SIZE_TRIALS:
        raise RuntimeError(f"{TIER} {model}: lr_widened needs the first {P.SIZE_TRIALS} trials drawn and committed")
    seen = {json.dumps(t["knobs"], sort_keys=True) for t in trials}
    rng = random.Random(f"{P.SIZE_DRAW_SEED}|{model}" + ("" if TIER == "1k" else f"|{TIER}") + ("|lr-wide" if wide else ""))
    rejected = {"no_fit": 0, "repeat": 0}
    CHIP_OFF[0] = 0
    while len(trials) < P.SIZE_TRIALS * (2 if wide else 1):
        for _ in range(MAX_DRAWS):
            t = draw_trial(rng, model, keys, caps)
            if t is None:
                rejected["no_fit"] += 1
                continue
            sig = json.dumps(t["knobs"], sort_keys=True)
            if sig in seen:
                rejected["repeat"] += 1
                continue
            seen.add(sig)
            t["id"] = f"t{len(trials) + 1:02d}"
            trials.append(t)
            break
        else:
            if not trials:                             # no setting fits the tier (DiPE in 1K): left out, nothing runs
                print(f"   [excluded] {model}: no setting fits the tier on all six datasets")
                return keys, caps, [], {**rejected, "excluded": "no setting fits the tier"}
            print(f"   [stop] {model}: no fitting draw in {MAX_DRAWS} tries after {len(trials)} trials")
            break
    if TIER == "chip":
        rejected["chip_off"] = CHIP_OFF[0]                       # drawn, fit the proxy, failed the real build
    return keys, caps, trials, rejected


def main(write, models, tier_name="1k"):
    global TIER, COUNT_MAX
    TIER = tier_name
    if TIER == "chip":
        COUNT_MAX = 1 << 16
    path = P.SIZE_TRIALS_FILE
    for model in models:
        keys, caps, trials, rejected = build(model)
        print(f"\n== {TIER} {model}: size keys {keys or '-'}  caps {caps}  trials {len(trials)}  rejected {rejected}")
        for t in trials:
            print(f"   {t['id']} fill={t['fill']}  Steel {t['knobs']['Steel']}  stored "
                  + "/".join(str(t["stored"][d]) for d in CHANNELS), flush=True)
        if write:                                                # written per model: a long chip run keeps its progress
            out = P._load(path)
            out.setdefault(TIER, {})[model] = trials
            meta = {"size_keys": keys, "count_caps": caps, "rejected": rejected}
            if TIER != "1k":
                meta["cap"] = CAP
            metas = out.setdefault("_meta" if TIER == "1k" else f"_meta_{TIER}", {})
            if f"{TIER}|{model}" in P.SIZE_LR_WIDENED:               # keep the first draw's record, add the widened one
                meta = {**metas[model], "lr_widened": P.size_lr_range(TIER, model), "rejected_lr_widened": rejected}
            metas[model] = meta
            json.dump(out, open(path, "w", encoding="utf-8"), indent=1, sort_keys=True)
            print("written", path, flush=True)


if __name__ == "__main__":
    pick = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--models=")), None)
    if "--models" in sys.argv:
        pick = sys.argv[sys.argv.index("--models") + 1]
    tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else "1k"
    main("--write" in sys.argv, pick.split(",") if pick else P.SIZE_MODELS, tier)
