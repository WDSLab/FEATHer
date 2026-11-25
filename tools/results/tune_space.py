"""Search space of the tuning (configs/experiments/size.yaml) as an Optuna define-by-run sampler.

One Optuna study = one (tier, model, dataset, H).  `Space(model, group, data, H, cap)` is built once per study (it
finds the size keys and their upper bounds); `space.sample(trial)` draws one setting:

    lr        log-uniform on common.lr, rounded to 3 significant digits
    target    uniform in (0, cap]                     cap = the tier's limit in stored numbers (1K: 1,000)
    fill      one size key whose value changes the stored count; it takes the value with the largest stored count
              <= target (ties: the smaller value), so the trials spread over model sizes
    size      the other size keys: integer, log-uniform from 1 to the largest value that can fit (`{size: N}`: <= N)
    list      the common lists (switch / dropout / mix / period / segment) or the model's own list

A key that has no effect given an earlier key is not drawn and keeps the model's base value (CONDITIONS).  A draw
that cannot be built and run, or is larger than its target, raises Reject: the trial is not trained and not counted.
The runner tells a rejected trial to Optuna as PRUNED (user attribute "reject"), so TPE learns to avoid the region:
told as FAIL, TPE ignores it and keeps drawing settings that do not fit (in a check without training, 600 draws gave
10-18 trained trials as FAIL, 40 trained in 47-97 draws as PRUNED).
"Stored" = the numbers the chip keeps (complex x2, the CMoS spare mapping excluded) at this H.
"""
import math
import os
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
import train as T   # noqa: E402

DEFAULT_ARGS = vars(T.parse_args(["--model", "FEATHer", "--data", "Steel", "--pred_len", "96"]))   # worker defaults
L = 96
CHANNELS = {"Steel": 6, "WindSCADA": 14, "BlastFurnace": 28, "N2O": 14, "MetroPT3": 7, "EWELD": 14,
            "ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21, "Electricity": 321,
            "Traffic": 862}
SIZE_CLASSES = ("size", "segment", "period")          # kinds whose value can change the stored count
DEPENDENT = {"mixlinear_lpf", "cmos_conv_stride"}    # counts whose upper bound depends on another key
COUNT_MAX = 1 << 16                                  # search limit of a count (far above any tier)
# key -> (parent key, values of the parent for which the key has an effect)
CONDITIONS = {
    "phaseformer_revin_affine": ("phaseformer_use_revin", {True}),     # affine of a RevIN that is off
    "phaseformer_pos_dropout": ("phaseformer_pos_embed", {True}),      # dropout on an embedding that is off
    "mlp_rank": ("use_residual_mlp", {True}),                          # width of an MLP that is off (FEATHer_CI)
    "gate_mode": ("num_bands", {2, 3}),                                # one band: the gate weight is always 1
    "lambda_spec": ("num_bands", {2, 3}),                              # one band: nothing to separate (loss 0)
    "period": ("head_mode", {"spk"}),                                  # the linear head has no period / segment /
    "head_seg": ("head_mode", {"spk"}),                                # smoothing kernel
    "agg_shared": ("head_mode", {"spk"}),
    "sparsetsf_d_model": ("sparsetsf_model_type", {"mlp"}),            # hidden width of the linear head (none)
}
# Not listed (left as run, so the records match): timebase_orthogonal_weight with use_orthogonal off and
# cmos_dropout (never used upstream) are drawn although they have no effect - the same model can appear under two
# settings strings.


class Reject(Exception):
    """The drawn setting is not trained: `why` = cannot_build / no_fit."""
    def __init__(self, why):
        super().__init__(why)
        self.why = why


@lru_cache(maxsize=None)
def _stored(model, group, data, H, knobs):
    _, _, settings = P.resolve(model, group, data, H, P.mfg_main_config_id(model))
    args = Namespace(**{**DEFAULT_ARGS, "model": P.registry_name(model), **settings, **dict(knobs)})
    C = CHANNELS[data]
    try:
        net = B.get_model(args.model, args, n_features=C, seq_len=L, pred_len=H)
        with torch.no_grad():                          # a model that builds but cannot run (e.g. kernel > L) is invalid
            y = net(torch.randn(2, L, C))              # (values are not checked: some models are not finite before training)
    except Exception:
        return None
    if tuple(y.shape) != (2, H, C):
        return None
    n = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
    if model == "CMoS":                                # one spare mapping that forward never uses
        seg = args.cmos_seg_size
        n -= (L // seg) * (H // seg) + H // seg
    return n


def kind(spec):
    """("count", max) for a size key, ("list", values) otherwise."""
    if spec == "size":
        return "count", COUNT_MAX
    if isinstance(spec, dict):
        return "count", int(spec["size"])
    return "list", (spec if isinstance(spec, list) else P.SIZE_COMMON[spec])


def keys_of(model):
    """{key: yaml spec} of the drawn keys, in yaml order with every CONDITIONS parent before its child."""
    sp = {k: s for k, s in P.SIZE_SPACE[model].items() if s != "unused"}
    parents = [k for k in sp if any(c in sp and p == k for c, (p, _) in CONDITIONS.items())]
    return {k: sp[k] for k in parents + [k for k in sp if k not in parents]}


class Space:
    def __init__(self, model, group, data, H, cap):
        self.model, self.group, self.data, self.H, self.cap = model, group, data, H, cap
        self.keys = keys_of(model)
        _, _, base = P.resolve(model, group, data, H, P.mfg_main_config_id(model))
        # value of a key that is not drawn (a key with no value anywhere keeps the wrapper's own default)
        self.base = {k: v for k in self.keys if (v := base.get(k, DEFAULT_ARGS.get(k))) is not None}
        self.size_keys = [k for k in self.keys if self._changes_size(k)]
        self.upper = {k: self._upper(k) for k, s in self.keys.items()
                      if kind(s)[0] == "count" and k not in DEPENDENT}

    # ---- stored count -----------------------------------------------------------------------------------
    def stored(self, knobs):
        k = tuple(sorted((a, b) for a, b in knobs.items() if a != "lr"))
        return _stored(self.model, self.group, self.data, self.H, k)

    def ceiling(self, key, knobs):
        """Largest meaningful value of a count (larger values are clamped by the model or give the same model)."""
        top = kind(self.keys[key])[1]
        if key == "fits_cut_freq":
            top = min(top, L // 2)                       # cut frequency <= L / 2
        elif key == "mixlinear_lpf" and "mixlinear_period_len" in knobs:
            top = min(top, math.ceil(L / knobs["mixlinear_period_len"]))   # the wrapper clamps lpf to the segments
        elif key in ("cmos_kernel_size", "point_kernel"):
            top = min(top, L)
        elif key == "cmos_conv_stride" and "cmos_kernel_size" in knobs:
            top = min(top, L - knobs["cmos_kernel_size"] + 1)   # larger strides all give one conv output
        return top

    def _values(self, key, knobs):
        """{value: stored} of a list key, the other keys at knobs (values that cannot be built left out)."""
        out = {}
        for v in kind(self.keys[key])[1]:
            n = self.stored({**knobs, key: v})
            if n is not None:
                out[v] = n
        return out

    def _changes_size(self, key):
        s = self.keys[key]
        if s not in SIZE_CLASSES and not isinstance(s, dict):
            return False
        knobs = dict(self.base)
        cond = CONDITIONS.get(key)
        if cond and cond[0] in self.keys and knobs.get(cond[0]) not in cond[1]:
            knobs[cond[0]] = sorted(cond[1])[0]         # the key in effect (SparseTSF base = linear: its MLP width)
        if kind(s)[0] == "list":
            return len(set(self._values(key, knobs).values())) > 1
        ns = {self.stored({**knobs, key: v}) for v in (1, 2, 3)}
        return len(ns - {None}) > 1

    def largest(self, key, knobs, cap):
        """(value, stored) of the largest count of `key` with stored <= cap, the other keys at knobs, or None.
        Bisection on a count whose size grows with the value; an invalid value (e.g. an even kernel) is replaced by
        the one below it."""
        def at(v):
            for x in (v, v - 1):
                n = self.stored({**knobs, key: x}) if x >= 1 else None
                if n is not None:
                    return x, n
            return None
        best, top = at(1), self.ceiling(key, knobs)
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

    def _upper(self, key):
        """Upper bound of a drawn count: the largest value that fits the cap with every other key at the value that
        gives the smallest model while `key` still changes the size (e.g. the MLP stays on for its width)."""
        knobs = dict(self.base)
        for k, s in self.keys.items():
            if k == key:
                continue
            if kind(s)[0] == "count":
                knobs[k] = 1
                continue
            best = None
            for v in kind(s)[1]:
                b = {**knobs, k: v}
                n1, n2 = self.stored({**b, key: 1}), self.stored({**b, key: 2})
                if n1 is not None and n1 != n2 and (best is None or n1 < best[1]):
                    best = (v, n1)
            if best:
                knobs[k] = best[0]
        r = self.largest(key, knobs, self.cap)
        return r[0] if r else 1

    def fill(self, key, knobs, target):
        """(value, stored) of the largest model <= target through `key` (ties: the smaller value), or None."""
        s = self.keys[key]
        if kind(s)[0] == "count" and key not in DEPENDENT:
            return self.largest(key, knobs, target)
        if kind(s)[0] == "count":                       # a dependent count: its bound is small (<= L), every value
            cand = {v: self.stored({**knobs, key: v}) for v in range(1, self.ceiling(key, knobs) + 1)}
            cand = {v: n for v, n in cand.items() if n is not None}
        else:
            cand = self._values(key, knobs)
        vals = {v: n for v, n in cand.items() if n <= target}
        if not vals:
            return None
        v = max(vals, key=lambda x: (vals[x], -x))
        return v, vals[v]

    # ---- one draw ---------------------------------------------------------------------------------------
    def sample(self, trial):
        """Draw one setting on an optuna.Trial -> (knobs, stored); raises Reject.  Drawn values are the trial's
        params; the settings actually used (fill value, rounded lr) are its user attributes."""
        lo, hi = P.SIZE_COMMON["lr"]
        knobs = {"lr": float(f"{trial.suggest_float('lr', lo, hi, log=True):.3g}")}
        target = trial.suggest_float("target", 0.0, float(self.cap))
        fill = trial.suggest_categorical("fill", self.size_keys) if len(self.size_keys) > 1 else (
            self.size_keys[0] if self.size_keys else None)
        frac, skipped = {}, set()
        for k, s in self.keys.items():
            cond = CONDITIONS.get(k)
            if cond and cond[0] in knobs and knobs[cond[0]] not in cond[1]:
                if k in self.base:                      # no effect here: the base value, not drawn
                    knobs[k] = self.base[k]
                skipped.add(k)
                continue
            if k == fill:
                continue
            if k in DEPENDENT:
                frac[k] = trial.suggest_float(f"{k}@u", 0.0, 1.0)
            elif kind(s)[0] == "count":
                knobs[k] = trial.suggest_int(k, 1, max(1, self.upper[k]), log=True)
            else:
                knobs[k] = trial.suggest_categorical(k, kind(s)[1])
        for k in frac:
            knobs[k] = 1                                # resolved after the fill (their bound depends on it)
        filled = bool(fill) and fill not in skipped     # a fill key without effect here: nothing to fill
        if filled:
            f = self.fill(fill, knobs, target)
            if f is None:
                raise Reject("no_fit")
            knobs[fill] = f[0]
        for k, u in frac.items():
            knobs[k] = max(1, int(round(self.ceiling(k, knobs) ** u)))
        n = self.stored(knobs)
        if n is None:
            raise Reject("cannot_build")
        if n > (target if filled else self.cap):
            raise Reject("no_fit")
        trial.set_user_attr("knobs", knobs)
        trial.set_user_attr("stored", n)
        return knobs, n
