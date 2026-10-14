# -*- coding: utf-8 -*-
"""Prepare the CubeIDE project for ONE board measurement.

For each measurement cell this script:

  1. exports the golden vectors + weights of the model under test at (H, D)
     (random weights - the golden only has to prove the function; latency does
     not depend on the weight values);
  2. copies the ports, the kernels, every golden header and the harness into
     the CubeIDE project;
  3. writes Core/Inc/board_cfg.h (RUN_* switch, geometry, power mode) - the port
     headers pick it up through a __has_include hook, so nothing in the repo
     changes and the x86 tests never see it;
  4. restores this repo's golden/ directory, so `make test` is unaffected.

Then in CubeIDE:  Project -> Clean,  Build (hammer),  Run (green play),
and read the line in Tera Term.

    python deployment/board/board_prep.py --plan                 # the measurement list
    python deployment/board/board_prep.py --model feather_mix --H 96 --D 7
    python deployment/board/board_prep.py --model feather_mix --H 96 --D 7 --power active
    python deployment/board/board_prep.py --verify_plan          # dry-build every cell (WSL, no board)

--verify / --verify_plan build the complete firmware in a SCRATCH copy of the
project with the CubeIDE compiler flags (arm-none-eabi-gcc in WSL) and report
flash / RAM against the part - so a cell that would not compile or not fit is
caught before anyone sits at the board.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GOLDEN = HERE / "golden"
SRC = HERE / "src"
DEFAULT_PROJECT = Path(r"C:\Users\lee\STM32Cube_projects\blink_test")

FLASH_BYTES = 128 * 1024
RAM_BYTES = 20 * 1024

# model -> (RUN switch, printed name, export stem, extra export args, full-buffer API)
MODELS = {
    "feather_mix": ("RUN_FEATHER_MIX", "FEATHer-mix", "feather2",   ["--mlp_rank", "4"], False),
    "feather_ci":  ("RUN_FEATHER_CI",  "FEATHer-CI",  "feather2ci", ["--mlp_rank", "4"], False),
    # FEATHer-mix without bands + FFT gate (control arm FEATHer_frontnone): the chip
    # cost of the band/gate front end; same recompute schedule
    "feather_fn":  ("RUN_FEATHER_FN",  "FEATHer-frontnone", "feather2fn", ["--mlp_rank", "4"], False),
    "dlinear":     ("RUN_DLINEAR",     "DLinear",     "dlinear",    [], False),
    "cyclenet":    ("RUN_CYCLENET",    "CycleNet",    "cyclenet",   [], False),
    "stsf":        ("RUN_STSF",        "SparseTSF",   "stsf",       [], True),
    "stsf_mlp":    ("RUN_STSF_MLP",    "STSF-MLP",    "stsf_mlp",   [], True),
    "mixlinear":   ("RUN_MIXLINEAR",   "MixLinear",   "mixlinear",  [], True),
    "dipe":        ("RUN_DIPE",        "DiPE",        "dipe",       [], True),
    "fits":        ("RUN_FITS",        "FITS",        "fits",       [], True),
    "pers":        ("RUN_PERS",        "persist",     "stsf",       [], True),
    # Structure depends on the chosen author setting (see config_args): the export
    # resolves it the way the experiments do.
    "cmos":        ("RUN_CMOS",        "CMoS",        "cmos",        [], True),
    "timebase":    ("RUN_TIMEBASE",    "TimeBase",    "timebase",    [], True),
    "phaseformer": ("RUN_PHASEFORMER", "PhaseFormer", "phaseformer", [], True),
    "samformer":   ("RUN_SAMFORMER",   "SAMformer",   "samformer",   [], True),
}

# board model -> registry model, for the models whose export takes --config (the
# chosen setting can change the structure: SparseTSF-MLP width, DiPE experts / RevIN,
# MixLinear lpf, FITS cut / per-channel).
CONFIG_MODELS = {"cmos": "CMoS", "timebase": "TimeBase", "phaseformer": "PhaseFormer",
                 "samformer": "SAMformer", "stsf_mlp": "SparseTSF_MLP", "dipe": "DiPE_Linear",
                 "mixlinear": "MixLinear", "fits": "FITS"}

# geometry macro prefix per port (feather handled separately)
PREFIX = {"dlinear": "DL", "cyclenet": "CN", "stsf": "STSF", "stsf_mlp": "SMLP", "dipe": "DIPE",
          "fits": "FITS", "mixlinear": "ML", "cmos": "CMOS", "timebase": "TB",
          "phaseformer": "PF", "samformer": "SAM"}

# A/B cells: the same three models with the direct 96-point DFT (-DFEA_DFT_DIRECT),
# so the paper can show how much of each latency is the transform implementation.
AB_PLAN = [("feather_mix", 96, 7), ("dipe", 96, 7), ("fits", 96, 7)]

# Extra cells go AFTER the A/B cells so the plan numbers of earlier cells
# (board_run.py resumes by number) do not move.  CycleNet H = 720 does not fit the flash.
EXTRA_PLAN = [("cyclenet", 96, 7)]
# At the chosen industrial setting (docs/chosen_mfg.json; measure after
# select_verdict.py --write).  SAMformer mixes channels, so it cannot be scaled
# from D = 1 - and only H = 96 fits.
EXTRA_PLAN += ([(m, 96, 7) for m in ("cmos", "timebase", "phaseformer", "samformer")]
               + [(m, 96, 1) for m in ("cmos", "timebase", "phaseformer")]
               + [(m, 720, 1) for m in ("cmos", "timebase", "phaseformer")])
# band + gate cost: FEATHer-mix without them, same cells as FEATHer-mix
EXTRA_PLAN += [("feather_fn", 96, 7), ("feather_fn", 720, 7)]

# The measurement list.  Channel-independent ports that only expose a full
# (H x D) output buffer run at D = 1 when H = 720 (a 720 x 7 buffer alone is
# 20 KB); their D = 7 latency is D x the D = 1 figure, which the H = 96 pair
# (D = 1 and D = 7) checks is linear.
PLAN = (
    [(m, 96, 7) for m in ("pers", "stsf", "stsf_mlp", "mixlinear", "dipe", "fits",
                          "dlinear", "feather_ci", "feather_mix")]
    + [(m, 96, 1) for m in ("stsf", "stsf_mlp", "mixlinear", "dipe", "fits")]
    + [(m, 720, 1) for m in ("stsf", "stsf_mlp", "mixlinear", "dipe", "fits")]
    + [("feather_ci", 720, 7), ("feather_mix", 720, 7),
       ("feather_mix", 96, 28), ("feather_mix", 720, 28)]
)


def cfg_header(model: str, H: int, D: int, power: str, direct_dft: bool = False,
               stream: bool = False, config: str | None = None, period: int | None = None,
               stream_ch: bool = False, geo: dict | None = None) -> str:
    run, name, _, _, _ = MODELS[model]
    if direct_dft:
        name += " (direct DFT)"
    if config:
        # candidate sweep: consecutive cells share model / H / D, so the printed name must
        # carry the setting, or the reader matches the previous firmware's lines
        name += f"[{config}]"
    lines = ["/* Generated by deployment/board/board_prep.py - do not edit.",
             f" * {name}  H={H}  D={D}  power={power} */",
             "#ifndef BOARD_CFG_H", "#define BOARD_CFG_H",
             f"#define {run}", f'#define BOARD_NAME "{name}"',
             f"#define BOARD_H {H}", f"#define BOARD_D {D}"]
    # period: ports whose period is a compile-time macro rather than a generated
    # geometry header (SparseTSF-MLP / MixLinear carry it in their geom header)
    if period and model in ("feather_mix", "feather_fn", "feather_ci"):
        lines.append(f"#define FEA2_PERIOD {period}")
    elif period and model == "stsf":
        lines.append(f"#define STSF_P {period}")
    if model in ("feather_mix", "feather_fn"):
        lines += [f"#define FEA2_H {H}", f"#define FEA2_D {D}", "#define FEA2_MLP_R 4",
                  "#define BOARD_SKIP_CI"]
        # size dial (head_seg / agg_shared); absent = the frozen model
        if (geo or {}).get("seg"):
            lines.append(f"#define FEA2_SEG {geo['seg']}")
        if (geo or {}).get("agg_shared"):
            lines.append("#define FEA2_AGG_SHARED 1")
        if model == "feather_fn":
            lines.append("#define FEA2_FRONT_NONE")
    elif model == "feather_ci":
        lines += [f"#define FEA2_H {H}", "#define FEA2_D 1", f"#define FEA2CI_D {D}",
                  "#define FEA2_MLP_R 4"]
    else:
        lines.append("#define BOARD_SKIP_CI")
        if model in PREFIX and model != "mixlinear":   # MixLinear's H comes from mixlinear_geom.h
            lines += [f"#define {PREFIX[model]}_H {H}", f"#define {PREFIX[model]}_D {D}"]
        elif model == "mixlinear":
            lines.append(f"#define ML_D {D}")
    if stream_ch:
        lines.append("#define BOARD_STREAM_CH")     # one channel at a time, H-float output
    if stream:
        lines.append("#define STREAM_MODE")        # main.c.stream (replay.py), trained weights
    if direct_dft:
        lines.append("#define FEA_DFT_DIRECT")      # A/B: direct 96-point DFT
    if power == "active":
        lines.append("#define POWER_ACTIVE")
    elif power == "idle":
        lines.append("#define POWER_IDLE")
    elif power == "max":
        lines.append("#define POWER_MAX")          # measurement sanity check only
    lines.append("#endif /* BOARD_CFG_H */")
    return "\n".join(lines) + "\n"


def check_cell(model: str, H: int, D: int) -> None:
    if model not in MODELS:
        raise SystemExit(f"unknown model {model}; one of {', '.join(MODELS)}")
    full = MODELS[model][4]
    if full and H * D * 4 > 4096:
        raise SystemExit(f"{model} only has a full-buffer API: a {H}x{D} output buffer "
                         f"is {H * D * 4 / 1024:.1f} KB.  Measure at --D 1 and scale by D.")
    if model == "dlinear" and H != 96:
        raise SystemExit("DLinear does not fit the flash at H >= 192 (156-556 KB of weights)")
    if model == "samformer" and H != 96:
        raise SystemExit("SAMformer does not fit the flash at H >= 192 (133-333 KB)")
    if model == "cyclenet" and H >= 336:
        raise SystemExit("CycleNet does not fit the flash at H >= 336 (139-315 KB of weights)")


def config_args(model: str, config: str | None = None) -> list[str]:
    """--config for CONFIG_MODELS = `config` if given (candidate sweep, board_run.py
    --candidates), else the chosen industrial setting (docs/chosen_mfg.json).  Before the
    selection is written, the export's default is used and a warning printed: such a
    measurement is a dry run, not a number for the paper.  SAMformer is exempt - its
    candidates differ in lr / SAM rho only, so every one builds the same firmware."""
    if model not in CONFIG_MODELS:
        return []
    if config:
        return ["--config", config]
    if model == "samformer":
        return []
    sys.path.insert(0, str(ROOT))
    from tools.results.protocol import mfg_main_config_id
    cid = mfg_main_config_id(CONFIG_MODELS[model])
    if cid is None:
        print(f"WARNING: no chosen setting for {CONFIG_MODELS[model]} yet - using the export default "
              "(dry run only; re-measure after select_verdict.py --write)", file=sys.stderr)
        return []
    return ["--config", cid.split(":", 1)[1]]


def geo_args(model: str, geo: dict | None) -> list[str]:
    """The dataset's period / CycleNet cycle (chip_verdict.py) for the exports that do
    not resolve the dataset themselves.  geo = None keeps every export's default."""
    if not geo:
        return []
    if model == "feather_mix":
        # size dial passes through to the export (absent = the frozen model)
        return ((["--period", str(geo["period"])] if geo.get("period") else [])
                + (["--head_seg", str(geo["seg"])] if geo.get("seg") else [])
                + (["--agg_shared", "1"] if geo.get("agg_shared") else []))
    if model in ("feather_ci", "stsf") and geo.get("period"):
        return ["--period", str(geo["period"])]
    if model == "cyclenet" and geo.get("cycle"):
        return ["--cycle", str(geo["cycle"])]
    return []


def export(model: str, H: int, D: int, ckpt: str | None = None, config: str | None = None,
           data: str | None = None, geo: dict | None = None) -> None:
    _, _, stem, extra, _ = MODELS[model]
    # data (chip_verdict.py): the industrial dataset whose period rule the config
    # models resolve with; None = each export's default (MetroPT3)
    cmd = [sys.executable, str(HERE / f"export_golden_{stem}.py"),
           "--pred_len", str(H), "--d_model", str(D)] + extra + config_args(model, config) \
        + (["--data", data] if data and model in CONFIG_MODELS else []) \
        + geo_args(model, geo) \
        + (["--ckpt", str(ckpt)] if ckpt else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    if r.returncode != 0:
        raise SystemExit("export failed:\n" + (r.stdout + r.stderr)[-2000:])
    if model == "feather_fn":
        # board_cfg.h sets FEA2_H / FEA2_D / FEA2_MLP_R for the whole image, and every
        # Core/Src file is compiled - so the frozen ports' feather2 golden must match
        # them too, or feather2_infer.c fails its static asserts.  Not linked in the
        # measured image (gc-sections); only has to compile.
        export("feather_mix", H, D)


# full-buffer ports with a <pfx>_forward_ch API: vectors header and output symbol
STREAM_CH = {"cmos": ("cmos_vectors.h", "CG_Y"), "dipe": ("dipe_vectors.h", "DG_Y"),
             "fits": ("fits_vectors.h", "FG_Y"), "stsf_mlp": ("stsf_mlp_vectors.h", "SM_Y"),
             "mixlinear": ("mixlinear_vectors.h", "MG_Y"), "timebase": ("timebase_vectors.h", "TG_Y"),
             "phaseformer": ("phaseformer_vectors.h", "PG_Y"), "stsf": ("stsf_vectors.h", "SG_Y")}


def write_gold_ch(model: str, H: int, D: int, out: Path) -> None:
    """board_gold_ch.h for BOARD_STREAM_CH: channel 0 of the golden output plus each channel's
    sum and absolute sum (the full H x D golden does not fit the flash at D = 28, H = 720)."""
    import re
    vec, sym = STREAM_CH[model]
    txt = (GOLDEN / vec).read_text(encoding="utf-8")
    m = re.search(r"static const float " + sym + r"\[(\d+)\] = \{(.*?)\};", txt, re.S)
    vals = [float(v.rstrip("f")) for v in m.group(2).replace(",", " ").split()]
    assert len(vals) == H * D, (len(vals), H, D)
    y = [vals[t * D:(t + 1) * D] for t in range(H)]                  # time-major
    fmt = lambda xs: ", ".join(f"{v:.9e}f" for v in xs)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"/* Generated by board_prep.py from {vec} ({sym}) - do not edit. */\n")
        f.write("#ifndef BOARD_GOLD_CH_H\n#define BOARD_GOLD_CH_H\n")
        f.write(f"static const float GOLD_Y_CH0[{H}] = {{ {fmt([y[t][0] for t in range(H)])} }};\n")
        f.write(f"static const float GOLD_SUM[{D}] = {{ {fmt([sum(y[t][d] for t in range(H)) for d in range(D)])} }};\n")
        f.write(f"static const float GOLD_ABSSUM[{D}] = {{ {fmt([sum(abs(y[t][d]) for t in range(H)) for d in range(D)])} }};\n")
        f.write("#endif\n")


def install(project: Path, model: str, H: int, D: int, power: str,
            direct_dft: bool = False, stream_ckpt: str | None = None,
            config: str | None = None, data: str | None = None,
            geo: dict | None = None, stream_ch: bool = False) -> None:
    """Export, copy into `project`, write board_cfg.h, restore golden/.

    stream_ckpt: install the STREAMING REPLAY firmware (main.c.stream,
    driven by replay.py) with the TRAINED weights of that checkpoint instead of the
    timing harness with random golden weights.  FEATHer-mix only."""
    if stream_ckpt and model != "feather_mix":
        raise SystemExit("--stream_ckpt is implemented for feather_mix only")
    src_dir, inc_dir = project / "Core" / "Src", project / "Core" / "Inc"
    for d in (src_dir, inc_dir):
        if not d.is_dir():
            raise SystemExit(f"not a CubeIDE project (missing {d})")
    with tempfile.TemporaryDirectory() as tmp:
        bak = Path(tmp) / "golden"
        shutil.copytree(GOLDEN, bak)
        try:
            export(model, H, D, stream_ckpt, config, data, geo)
            for f in SRC.glob("*.c"):
                shutil.copy2(f, src_dir / f.name)
            for f in list(SRC.glob("*.h")) + list(GOLDEN.glob("*.h")):
                shutil.copy2(f, inc_dir / f.name)
            if stream_ch:
                write_gold_ch(model, H, D, inc_dir / "board_gold_ch.h")
        finally:
            # restore file by file - never remove the directory itself (Dropbox
            # holds a handle on it, and a failed rmdir leaves it half-emptied)
            for f in bak.iterdir():
                shutil.copy2(f, GOLDEN / f.name)
            for f in GOLDEN.iterdir():
                if not (bak / f.name).exists():
                    f.unlink()
    for old in ("feather2in_infer.c", "feather_infer.c"):    # retired ports left in the CubeIDE project
        if (src_dir / old).exists():
            (src_dir / old).unlink()
    with (inc_dir / "board_cfg.h").open("w", encoding="utf-8", newline="\n") as f:
        f.write(cfg_header(model, H, D, power, direct_dft, stream=bool(stream_ckpt),
                           config=config, period=(geo or {}).get("period"), stream_ch=stream_ch,
                           geo=geo))
    main_c = src_dir / "main.c"
    keep = src_dir / "main.c.bak_pre_board_prep"
    if main_c.exists() and not keep.exists():
        shutil.copy2(main_c, keep)                  # the original harness, once
    shutil.copy2(HERE / ("main.c.stream" if stream_ckpt else "main.c.board"), main_c)


# ------------------------------------------------------------------ dry build
CFLAGS = ("-mcpu=cortex-m3 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F103xB -c "
          "-ICore/Inc -IDrivers/STM32F1xx_HAL_Driver/Inc "
          "-IDrivers/STM32F1xx_HAL_Driver/Inc/Legacy "
          "-IDrivers/CMSIS/Device/ST/STM32F1xx/Include -IDrivers/CMSIS/Include "
          "-O2 -ffunction-sections -fdata-sections -Wall --specs=nano.specs "
          "-mfloat-abi=soft -mthumb")
LDFLAGS = ("-mcpu=cortex-m3 -T STM32F103RBTX_FLASH.ld --specs=nosys.specs "
           "-Wl,--gc-sections -static --specs=nano.specs -mfloat-abi=soft -mthumb "
           "-Wl,--start-group -lc -lm -Wl,--end-group")


def wsl(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    inner = "cd \"$(wslpath '%s')\" && %s" % (cwd, cmd)
    return subprocess.run(["wsl.exe", "-e", "bash", "-lc", inner],
                          capture_output=True, text=True)


def dry_build(project: Path, model: str, H: int, D: int, direct_dft: bool = False,
              stream_ckpt: str | None = None, data: str | None = None,
              keep_elf: Path | None = None, relax_ld: bool = False,
              geo: dict | None = None, config: str | None = None, stream_ch: bool = False) -> dict:
    """Build the whole firmware for one cell in a scratch copy; never touches `project`.
    stream_ch: the BOARD_STREAM_CH harness (per-value emit, no output buffer)."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "proj"
        shutil.copytree(project, scratch, ignore=shutil.ignore_patterns("Debug", "Release"))
        install(scratch, model, H, D, "none", direct_dft, stream_ckpt, config=config, data=data, geo=geo,
                stream_ch=stream_ch)
        if relax_ld:
            # chip_verdict.py: link the scratch image against oversized regions so an
            # over-budget cell still reports both sizes; the caller compares them with
            # the real 128 K / 20 K (the scratch firmware is never flashed)
            ld = scratch / "STM32F103RBTX_FLASH.ld"
            ld.write_text(ld.read_text().replace("LENGTH = 20K", "LENGTH = 1024K")
                          .replace("LENGTH = 128K", "LENGTH = 65536K"))
        script = (
            "set -e; mkdir -p obj; "
            "for f in Core/Src/*.c Drivers/STM32F1xx_HAL_Driver/Src/*.c; do "
            f"  arm-none-eabi-gcc $f {CFLAGS} -o obj/$(basename $f .c).o 2> obj/$(basename $f .c).err "
            "  || { echo COMPILE_FAIL $f; cat obj/$(basename $f .c).err | grep -m5 error; exit 3; }; "
            "done; "
            "arm-none-eabi-gcc -mcpu=cortex-m3 -c -x assembler-with-cpp --specs=nano.specs "
            "-mfloat-abi=soft -mthumb Core/Startup/startup_stm32f103rbtx.s -o obj/startup.o; "
            f"arm-none-eabi-gcc -o fw.elf obj/*.o {LDFLAGS} 2> link.err "
            "|| { echo LINK_FAIL; grep -m5 -i 'overflow\\|error' link.err; exit 4; }; "
            "arm-none-eabi-size -A fw.elf")
        r = wsl(script, scratch)
        out = r.stdout + r.stderr
        if keep_elf is not None and (scratch / "fw.elf").exists():
            shutil.copy2(scratch / "fw.elf", keep_elf)
        if r.returncode != 0:
            return dict(ok=False, why=" ".join(out.split())[-300:])
        sec = {}
        for line in r.stdout.splitlines():
            p = line.split()
            if len(p) >= 2 and p[0].startswith(".") and p[1].isdigit():
                sec[p[0]] = int(p[1])
        flash = sec.get(".isr_vector", 0) + sec.get(".text", 0) + sec.get(".rodata", 0) \
            + sec.get(".ARM", 0) + sec.get(".init_array", 0) + sec.get(".fini_array", 0) \
            + sec.get(".data", 0)
        ram = sec.get(".data", 0) + sec.get(".bss", 0) + sec.get("._user_heap_stack", 0)
        return dict(ok=flash <= FLASH_BYTES and ram <= RAM_BYTES, flash=flash, ram=ram,
                    why="" if flash <= FLASH_BYTES and ram <= RAM_BYTES else "over budget")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--D", type=int, default=7)
    ap.add_argument("--power", choices=["none", "active", "idle", "max"], default="none")
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--direct_dft", action="store_true",
                    help="A/B: build with the direct (pre-2026-09-29) 96-point DFT")
    ap.add_argument("--stream_ckpt", default=None,
                    help="install the streaming replay firmware with this trained checkpoint "
                         "(feather_mix; drive it with replay.py)")
    ap.add_argument("--plan", action="store_true", help="print the measurement list")
    ap.add_argument("--verify", action="store_true",
                    help="dry-build this cell in a scratch copy instead of installing")
    ap.add_argument("--verify_plan", action="store_true",
                    help="dry-build every cell of the plan (no board needed)")
    a = ap.parse_args()

    if a.plan:
        for i, (m, H, D) in enumerate(PLAN, 1):
            print(f"{i:2d}. --model {m:<11} --H {H:<3} --D {D}")
        for j, (m, H, D) in enumerate(AB_PLAN, len(PLAN) + 1):
            print(f"{j:2d}. --model {m:<11} --H {H:<3} --D {D} --direct_dft   (A/B)")
        return
    if a.verify_plan:
        bad = 0
        cells = [(m, H, D, False) for m, H, D in PLAN] + [(m, H, D, True) for m, H, D in AB_PLAN]
        for i, (m, H, D, direct) in enumerate(cells, 1):
            check_cell(m, H, D)
            r = dry_build(a.project, m, H, D, direct)
            bad += not r["ok"]
            size = (f"flash {r['flash'] / 1024:6.1f} KB  RAM {r['ram'] / 1024:5.2f} KB"
                    if "flash" in r else "")
            print(f"{i:2d}. {m:<11} H={H:<3} D={D:<2}{' direct' if direct else '       '}  "
                  f"{'OK  ' if r['ok'] else 'FAIL'}  {size}  {r['why']}", flush=True)
        print(f"\n{len(cells) - bad}/{len(cells)} cells build and fit.")
        sys.exit(1 if bad else 0)
    if not a.model:
        ap.error("--model is required (or --plan / --verify_plan)")
    check_cell(a.model, a.H, a.D)
    if a.verify:
        r = dry_build(a.project, a.model, a.H, a.D, a.direct_dft, a.stream_ckpt)
        print(r)
        sys.exit(0 if r["ok"] else 1)
    install(a.project, a.model, a.H, a.D, a.power, a.direct_dft, a.stream_ckpt)
    print(f"prepared {MODELS[a.model][1]}  H={a.H}  D={a.D}  power={a.power}"
          f"{'  DIRECT DFT' if a.direct_dft else ''}\n"
          f"  -> {a.project}\n"
          "In CubeIDE: Project -> Clean, Build (hammer), Run (green play), read Tera Term.")


if __name__ == "__main__":
    main()
