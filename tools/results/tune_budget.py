"""Tier limit of one tuning study (configs/experiments/size.yaml tiers): is a drawn setting small enough?

    1k          stored numbers <= 1,000 at the run's H (tune_space draws within it; nothing to build)
    chip, ltsf  the F103 firmware fits 128 KB flash / 20 KB SRAM at the run's H with the dataset's channel count
                Every trained trial fits, so the pick is simply the best
                validation loss (no fallback after tuning):
                  - the draw is first limited by a flash proxy, cap = (128 KB - non-weight flash) / 4 stored numbers,
                    the non-weight flash from one real build of the model's base setting at this H (free: no build)
                  - every draw that passes the proxy is then built for real, every model (the proxy sees flash only:
                    e.g. MetroPT3 H96: PhaseFormer 1 of 6 over SRAM, SparseTSF_MLP 20,248 B of 20,480 B)

Real builds = deployment/board/chip_verdict.one_cell (a scratch copy of the CubeIDE project, the CubeIDE GCC 14.3).
Keys that only change training (TRAIN_ONLY) are left out of the build, so settings with the same firmware are built
once (cached per process).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT, "deployment", "board", "tools")]
import board_prep as bp   # noqa: E402
import protocol as P   # noqa: E402
import tune_space as S   # noqa: E402

CHIP_TIERS = ("chip", "ltsf")
ONE_K = 1000
BOARD = {"CMoS": "cmos", "SparseTSF_MLP": "stsf_mlp", "TimeBase": "timebase", "PhaseFormer": "phaseformer",
         "FITS": "fits", "SparseTSF": "stsf", "MixLinear": "mixlinear",
         P.FEATHER_CORE: "feather_ci", P.FEATHER_CI: "feather_ci"}
# keys that do not change the firmware (training only)
TRAIN_ONLY = {"lr", "cmos_dropout", "cmos_use_pi", "cmos_period", "phaseformer_attn_dropout", "phaseformer_pos_dropout",
              "phaseformer_use_huber", "timebase_orthogonal_weight", "lambda_spec"}
_BUILDS = {}


def firmware(model, knobs):
    """(board model, config, geo) of a setting - how board_prep builds it (tuning, board runs, C checks, replay):
    SparseTSF / FEATHer take the keys as geometry (on top of the dataset's period, chip_verdict.geo_of); the config
    models take "<author src>+key=value...".  Keys that only change training are left out."""
    k = {a: b for a, b in sorted(knobs.items()) if a not in TRAIN_ONLY}
    if k.pop("head_mode", "spk") != "spk":                       # FEATHer_CI on LTSF: the C port has the SPK head only
        raise ValueError("no C port of FEATHer's linear head")
    if model == "SparseTSF" and k.pop("sparsetsf_model_type", "linear") == "mlp":
        model = "SparseTSF_MLP"                                  # the MLP head = the stsf_mlp C port
    elif model == "SparseTSF":
        k.pop("sparsetsf_d_model", None)                         # linear head: no hidden width
    bm = BOARD[model]
    if bm in ("stsf", "feather_ci"):
        geo = {("period" if a == "sparsetsf_period_len" else a): (int(b) if isinstance(b, bool) else b)
               for a, b in k.items()}
        return bm, None, geo
    return bm, P.mfg_main_config_id(model).replace("cand:", "") + "".join(f"+{a}={b}" for a, b in k.items()), None


def real_build(model, data, H, knobs):
    """chip_verdict.one_cell row of a setting (cached per firmware): verdict FIT / OFF / ?, flash_B, ram_B, ..."""
    import chip_verdict as CV
    bm, config, geo = firmware(model, knobs)
    key = (model, data, H, config, json.dumps(geo, sort_keys=True))
    if key not in _BUILDS:
        _BUILDS[key] = CV.one_cell(bm, data, H, config=config, extra_geo=geo)
    return _BUILDS[key]


class Budget:
    """The limit of one study.  `cap` = the most stored numbers a draw may have (tune_space.Space(cap=...));
    `check(knobs)` raises tune_space.Reject("chip_off") when a real build does not fit."""

    def __init__(self, tier, model, group, data, H):
        self.tier, self.model, self.data, self.H = tier, model, data, H
        self.real = tier in CHIP_TIERS
        self.base_build = None
        if tier not in CHIP_TIERS:
            self.cap = ONE_K
            return
        sp = S.Space(model, group, data, H, cap=1)               # only for the base setting and its stored count
        r = real_build(model, data, H, sp.base)
        n = sp.stored(sp.base)
        if r.get("flash_B") is None or n is None:
            raise RuntimeError(f"{model} {data} H{H}: base setting does not build ({r.get('why')})")
        self.base_build = {"flash_B": r["flash_B"], "stored": n}
        self.cap = (bp.FLASH_BYTES - (r["flash_B"] - 4 * n)) // 4

    def check(self, knobs):
        """Before training a draw that passed the proxy: the real build on the chip tiers (row); None on 1K."""
        if not self.real:
            return None
        if knobs.get("head_mode", "spk") != "spk":
            raise S.Reject("no_c_port")                          # no firmware to fit (FEATHer_CI linear head)
        r = real_build(self.model, self.data, self.H, knobs)
        if r.get("build") == "EXPORT_FAIL":                      # the exporter refused the geometry
            raise S.Reject("cannot_build")
        if r.get("verdict") == "?":
            # a compile / link failure is the machine (WSL, compiler, project copy), not the setting: stop the study
            # (a re-run resumes) instead of counting it as a reject
            raise RuntimeError(f"{self.model} {self.data} H{self.H}: firmware build failed: {r.get('why')}")
        if r.get("verdict") != "FIT":                            # flash and SRAM
            raise S.Reject("chip_off")
        return r
