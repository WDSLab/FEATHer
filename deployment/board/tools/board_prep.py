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

board_run.py does this per cell and flashes the board; chip_verdict.py / the tuning (tools/results/tune_budget.py)
call dry_build: the complete firmware in a SCRATCH copy of the project with the CubeIDE compiler flags
(arm-none-eabi-gcc in WSL), reporting flash / RAM - the project itself is never modified.

    python deployment/board/tools/board_prep.py --model feather_ci --H 96 --D 7 --verify          # dry build only
    python deployment/board/tools/board_prep.py --model feather_ci --H 96 --D 7 --power active    # into the project
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BOARD = Path(__file__).resolve().parents[1]          # deployment/board
ROOT = BOARD.parents[1]
GOLDEN = BOARD / "golden"
# golden/ is one shared folder: export -> copy -> restore runs one process at a time (parallel tuning workers).
# The lock file lives in the system temp folder (not in the repo), one per checkout.
GOLDEN_LOCK = Path(tempfile.gettempdir()) / f"feather_golden_{hashlib.md5(str(GOLDEN).encode()).hexdigest()[:8]}.lock"
SRC = BOARD / "src"
# the CubeIDE project (outside the repo); every build copies the ports and golden headers in fresh
DEFAULT_PROJECT = Path.home() / "STM32Cube_projects" / "blink_test"

FLASH_BYTES = 128 * 1024
RAM_BYTES = 20 * 1024

# model -> (RUN switch, printed name, export stem, extra export args, full-buffer API)
MODELS = {
    "feather_ci":  ("RUN_FEATHER_CI",  "FEATHer-CI",  "feather", ["--mlp_rank", "4"], False),
    "stsf":        ("RUN_STSF",        "SparseTSF",   "stsf",       [], True),
    "stsf_mlp":    ("RUN_STSF_MLP",    "STSF-MLP",    "stsf_mlp",   [], True),
    "mixlinear":   ("RUN_MIXLINEAR",   "MixLinear",   "mixlinear",  [], True),
    "fits":        ("RUN_FITS",        "FITS",        "fits",       [], True),
    # Structure depends on the chosen author setting (see config_args): the export
    # resolves it the way the experiments do.
    "cmos":        ("RUN_CMOS",        "CMoS",        "cmos",        [], True),
    "timebase":    ("RUN_TIMEBASE",    "TimeBase",    "timebase",    [], True),
    "phaseformer": ("RUN_PHASEFORMER", "PhaseFormer", "phaseformer", [], True),
}

# board model -> registry model, for the models whose export takes --config (the
# chosen setting can change the structure: SparseTSF-MLP width,
# MixLinear lpf, FITS cut / per-channel).
CONFIG_MODELS = {"cmos": "CMoS", "timebase": "TimeBase", "phaseformer": "PhaseFormer",
                 "stsf_mlp": "SparseTSF_MLP",
                 "mixlinear": "MixLinear", "fits": "FITS"}

# ports on pocketfft's transforms (src/mcu_pocket.c + plans from export/pocket_plan.py)
POCKET_PORTS = {"feather", "fits", "mixlinear"}

# geometry macro prefix per port (feather handled separately)
PREFIX = {"stsf": "STSF", "stsf_mlp": "SMLP", "fits": "FITS", "mixlinear": "ML",
          "cmos": "CMOS", "timebase": "TB", "phaseformer": "PF"}

# FEATHer size / structure keys a chip-tier trial can put in `geo` (feather_ci);
# absent = the frozen value.  "head_seg" is accepted for "seg" (the FEATHer.py name).
#   key           export flag        C macro (only when not the default)
#   seg           --head_seg         FEA2_SEG
#   agg_shared    --agg_shared 1     FEA2_AGG_SHARED 1
#   mlp_rank      --mlp_rank         FEA2_MLP_R (always written; board default 4)
#   point_kernel  --point_kernel     FEA2_KP            (odd)
#   num_bands     --num_bands        FEA2_NB            (1 / 2 / 3)
#   use_norm      --use_norm 0       FEA2_USE_NORM 0
#   gate_mode     --gate_mode none   FEA2_GATE_NONE 1
FEA2_DEFAULT = dict(seg=0, agg_shared=False, mlp_rank=4, point_kernel=3, num_bands=3,
                    use_norm=True, gate_mode="banded", use_residual_mlp=True)


def _truthy(v) -> bool:
    return str(v).lower() not in ("0", "false", "none", "")


def fea2_knobs(geo: dict | None) -> dict:
    """the FEATHer keys of `geo`, completed with the frozen values"""
    g = dict(geo or {})
    if "head_seg" in g:
        g.setdefault("seg", g.pop("head_seg"))
    k = {**FEA2_DEFAULT, **{x: g[x] for x in FEA2_DEFAULT if g.get(x) is not None}}
    return dict(seg=int(k["seg"] or 0), agg_shared=_truthy(k["agg_shared"]), mlp_rank=int(k["mlp_rank"]),
                point_kernel=int(k["point_kernel"]), num_bands=int(k["num_bands"]),
                use_norm=_truthy(k["use_norm"]), gate_mode=str(k["gate_mode"]),
                use_residual_mlp=_truthy(k["use_residual_mlp"]))


def fea2_export_args(geo: dict | None) -> list[str]:
    k = fea2_knobs(geo)
    return ((["--head_seg", str(k["seg"])] if k["seg"] else [])
            + (["--agg_shared", "1"] if k["agg_shared"] else [])
            + ["--mlp_rank", str(k["mlp_rank"])]
            + (["--point_kernel", str(k["point_kernel"])] if k["point_kernel"] != 3 else [])
            + (["--num_bands", str(k["num_bands"])] if k["num_bands"] != 3 else [])
            + (["--use_norm", "0"] if not k["use_norm"] else [])
            + (["--gate_mode", "none"] if k["gate_mode"] == "none" else [])
            + (["--use_residual_mlp", "0"] if not k["use_residual_mlp"] else []))


def fea2_macros(geo: dict | None) -> list[str]:
    """#define lines for the non-default FEATHer keys (FEA2_MLP_R is written by the caller)"""
    k = fea2_knobs(geo)
    return (([f"#define FEA2_SEG {k['seg']}"] if k["seg"] else [])
            + (["#define FEA2_AGG_SHARED 1"] if k["agg_shared"] else [])
            + ([f"#define FEA2_KP {k['point_kernel']}"] if k["point_kernel"] != 3 else [])
            + ([f"#define FEA2_NB {k['num_bands']}"] if k["num_bands"] != 3 else [])
            + (["#define FEA2_USE_NORM 0"] if not k["use_norm"] else [])
            + (["#define FEA2_GATE_NONE 1"] if k["gate_mode"] == "none" else [])
            + (["#define FEA2_USE_MLP 0"] if not k["use_residual_mlp"] else []))


def cfg_header(model: str, H: int, D: int, power: str, replay: bool = False, config: str | None = None,
               period: int | None = None, stream_ch: bool = False, geo: dict | None = None) -> str:
    run, name, _, _, _ = MODELS[model]
    if config:
        # consecutive cells share model / H / D, so the printed name must carry the setting,
        # or the reader matches the previous firmware's lines
        name += f"[{config}]"
    lines = ["/* Generated by deployment/board/tools/board_prep.py - do not edit.",
             f" * {name}  H={H}  D={D}  power={power} */",
             "#ifndef BOARD_CFG_H", "#define BOARD_CFG_H",
             f"#define {run}", f'#define BOARD_NAME "{name}"',
             f"#define BOARD_H {H}", f"#define BOARD_D {D}"]
    # period: ports whose period is a compile-time macro rather than a generated
    # geometry header (SparseTSF-MLP / MixLinear carry it in their geom header)
    if period and model == "feather_ci":
        lines.append(f"#define FEA2_PERIOD {period}")
    elif period and model == "stsf":
        lines.append(f"#define STSF_P {period}")
    if model == "feather_ci":
        # size / structure keys (FEA2_DEFAULT); absent = the frozen model
        lines += [f"#define FEA2_H {H}", "#define FEA2_D 1", f"#define FEA2CI_D {D}",
                  f"#define FEA2_MLP_R {fea2_knobs(geo)['mlp_rank']}"] + fea2_macros(geo)
    else:
        lines.append("#define BOARD_SKIP_CI")
        if model in PREFIX and model != "mixlinear":   # MixLinear's H comes from mixlinear_geom.h
            lines += [f"#define {PREFIX[model]}_H {H}", f"#define {PREFIX[model]}_D {D}"]
        elif model == "mixlinear":
            lines.append(f"#define ML_D {D}")
    if stream_ch:
        lines.append("#define BOARD_STREAM_CH")     # one channel at a time, H-float output
    if replay:
        lines.append("#define REPLAY_MODE")        # main.c.replay (replay.py), trained weights
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


def config_args(model: str, config: str | None = None) -> list[str]:
    """--config for CONFIG_MODELS = `config` if given ("<author src>+key=value..." of a tuned setting), else the
    model's base setting (protocol.mfg_main_config_id); without one the export's default is used and a warning
    printed: such a build is a dry run, not a number for the paper."""
    if model not in CONFIG_MODELS:
        return []
    if config:
        return ["--config", config]
    sys.path.insert(0, str(ROOT))
    from tools.results.protocol import mfg_main_config_id
    cid = mfg_main_config_id(CONFIG_MODELS[model])
    if cid is None:
        print(f"WARNING: no base setting for {CONFIG_MODELS[model]} - using the export default (dry run only)",
              file=sys.stderr)
        return []
    return ["--config", cid.split(":", 1)[1]]


def geo_args(model: str, geo: dict | None) -> list[str]:
    """The dataset's period (chip_verdict.py) for the exports that do
    not resolve the dataset themselves.  geo = None keeps every export's default."""
    if not geo:
        return []
    if model == "feather_ci":
        # size / structure keys pass through to the export (absent = the frozen model)
        return (["--period", str(geo["period"])] if geo.get("period") else []) + fea2_export_args(geo)
    if model == "stsf" and geo.get("period"):
        return ["--period", str(geo["period"])]
    return []


def export(model: str, H: int, D: int, ckpt: str | None = None, config: str | None = None,
           data: str | None = None, geo: dict | None = None) -> None:
    _, _, stem, extra, _ = MODELS[model]
    # data (chip_verdict.py): the industrial dataset whose period rule the config
    # models resolve with; None = each export's default (MetroPT3)
    cmd = [sys.executable, str(BOARD / "export" / f"export_golden_{stem}.py"),
           "--pred_len", str(H), "--d_model", str(D)] + extra + config_args(model, config) \
        + (["--data", data] if data and model in CONFIG_MODELS else []) \
        + geo_args(model, geo) \
        + (["--ckpt", str(ckpt)] if ckpt else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    if r.returncode != 0:
        raise SystemExit("export failed:\n" + (r.stdout + r.stderr)[-2000:])


# full-buffer ports with a <pfx>_forward_ch API: vectors header and output symbol
STREAM_CH = {"cmos": ("cmos_vectors.h", "CG_Y"),
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


@contextlib.contextmanager
def golden_lock():
    """Hold golden/ for this process: other processes wait here (checks every 0.2 s).  The OS frees the lock if the
    holder dies, so a crashed worker never blocks the others."""
    with open(GOLDEN_LOCK, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.2)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def install(project: Path, model: str, H: int, D: int, power: str, replay_ckpt: str | None = None,
            config: str | None = None, data: str | None = None,
            geo: dict | None = None, stream_ch: bool = False) -> None:
    """Export, copy into `project`, write board_cfg.h, restore golden/.

    replay_ckpt: install the REPLAY firmware (main.c.replay,
    driven by replay.py) with the TRAINED weights of that checkpoint instead of the
    timing harness with random golden weights."""
    if replay_ckpt and model not in ("feather_ci", *STREAM_CH):
        raise SystemExit(f"--replay_ckpt: {model} has no path in main.c.replay")
    src_dir, inc_dir = project / "Core" / "Src", project / "Core" / "Inc"
    for d in (src_dir, inc_dir):
        if not d.is_dir():
            raise SystemExit(f"not a CubeIDE project (missing {d})")
    with golden_lock(), tempfile.TemporaryDirectory() as tmp:
        GOLDEN.mkdir(exist_ok=True)                 # generated, not in git: a fresh checkout has none
        bak = Path(tmp) / "golden"
        shutil.copytree(GOLDEN, bak)
        try:
            export(model, H, D, replay_ckpt, config, data, geo)
            # only this model's port (+ the shared kernels): another port's .c would need that model's golden
            # headers, which a fresh checkout does not have; ports copied in earlier are removed
            stem = MODELS[model][2]
            need = {"mcu_kernels.c", "mcu_pt.c", f"{stem}_infer.c"} | ({"feather_core.c"} if stem == "feather" else set()) \
                | ({"mcu_pocket.c"} if stem in POCKET_PORTS else set())
            for f in SRC.glob("*.c"):
                if f.name in need:
                    shutil.copy2(f, src_dir / f.name)
                elif (src_dir / f.name).exists():
                    (src_dir / f.name).unlink()
            for f in list(SRC.glob("*.h")) + list(GOLDEN.glob("*.h")):
                shutil.copy2(f, inc_dir / f.name)
            sys.path.insert(0, str(BOARD / "export"))
            from pocket_plan import write_rfft96
            write_rfft96(inc_dir)
            if stream_ch:
                write_gold_ch(model, H, D, inc_dir / "board_gold_ch.h")
        finally:
            # restore file by file - never remove the directory itself (a sync client
            # may hold a handle on it, and a failed rmdir leaves it half-emptied)
            for f in bak.iterdir():
                shutil.copy2(f, GOLDEN / f.name)
            for f in GOLDEN.iterdir():
                if not (bak / f.name).exists():
                    f.unlink()
    for old in ("feather2in_infer.c", "feather2_infer.c", "feather2ci_infer.c", "dipe_infer.c", "fea_fftn.c",
                "fea_kernels.c", "fea_pocket.c", "fea_pt.c"):    # retired ports left in the CubeIDE project
        if (src_dir / old).exists():
            (src_dir / old).unlink()
    with (inc_dir / "board_cfg.h").open("w", encoding="utf-8", newline="\n") as f:
        f.write(cfg_header(model, H, D, power, replay=bool(replay_ckpt),
                           config=config, period=(geo or {}).get("period"), stream_ch=stream_ch,
                           geo=geo))
    main_c = src_dir / "main.c"
    keep = src_dir / "main.c.bak_pre_board_prep"
    if main_c.exists() and not keep.exists():
        shutil.copy2(main_c, keep)                  # the original harness, once
    shutil.copy2(BOARD / "firmware" / ("main.c.replay" if replay_ckpt else "main.c.board"), main_c)


# ------------------------------------------------------------------ dry build
# One ARM toolchain for the fit verdict and the board: the GNU tools of STM32CubeIDE 2.2.0 (GCC 14.3.1), the
# same Windows binaries on every machine (or a copy at the second path).  The build is a Windows batch
# file run by cmd.exe (also from WSL: gcc.exe started with a Linux path cannot find its own specs), so
# the scratch folder is on a Windows drive (build_tmp).  Flags = the CubeIDE Debug configuration, as board_run.py
# builds the measured firmware.
ARM_BIN = (r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-"
           r"stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin", r"C:\ST\gnu-tools-for-stm32-14.3.rel1\tools\bin")
# Plain Linux (no cmd.exe): the linux64 build of the same ST plugin (1.0.100.202602081740,
# CubeIDE update site 2.2.0); flash / RAM / verdict = the Windows builds byte for byte.
# Arm's own 14.3.rel1 does NOT match (+84 B flash in every cell) - do not use it.
ARM_BIN_LINUX = Path(os.environ.get("ARM_GCC_BIN", Path.home() / "toolchains" / "st14" / "tools" / "bin"))
NATIVE_LINUX = os.name != "nt" and shutil.which("cmd.exe") is None
CFLAGS = ("-mcpu=cortex-m3 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F103xB -c "
          "-ICore/Inc -IDrivers/STM32F1xx_HAL_Driver/Inc "
          "-IDrivers/STM32F1xx_HAL_Driver/Inc/Legacy "
          "-IDrivers/CMSIS/Device/ST/STM32F1xx/Include -IDrivers/CMSIS/Include "
          "-O2 -ffunction-sections -fdata-sections -Wall --specs=nano.specs "
          "-mfloat-abi=soft -mthumb")
LDFLAGS = ("-mcpu=cortex-m3 -T STM32F103RBTX_FLASH.ld --specs=nosys.specs "
           "-Wl,--gc-sections -static --specs=nano.specs -mfloat-abi=soft -mthumb "
           "-Wl,--start-group -lc -lm -Wl,--end-group")


def wsl(cmd: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run a shell command with the arm / x86 toolchain: through WSL on Windows, directly on Linux."""
    if os.name != "nt":
        return subprocess.run(["bash", "-lc", cmd], cwd=str(cwd), capture_output=True, text=True)
    inner = "cd \"$(wslpath '%s')\" && %s" % (cwd, cmd)
    return subprocess.run(["wsl.exe", "-e", "bash", "-lc", inner],
                          capture_output=True, text=True)


_BUILD_TMP = []


def build_tmp() -> str | None:
    """Scratch parent for dry builds: the Windows temp folder (the default on Windows; under /mnt/c in WSL, where the
    Windows .exe toolchain can read and write it)."""
    if os.name == "nt" or NATIVE_LINUX:
        return None
    if not _BUILD_TMP:
        win = subprocess.run(["cmd.exe", "/c", "echo %TEMP%"], capture_output=True, text=True,
                             cwd="/mnt/c").stdout.strip()
        _BUILD_TMP.append(subprocess.run(["wslpath", "-u", win], capture_output=True, text=True).stdout.strip())
    return _BUILD_TMP[0]


def dry_build(project: Path, model: str, H: int, D: int,
              replay_ckpt: str | None = None, data: str | None = None,
              relax_ld: bool = False,
              geo: dict | None = None, config: str | None = None, stream_ch: bool = False) -> dict:
    """Build the whole firmware for one cell in a scratch copy; never touches `project`.
    stream_ch: the BOARD_STREAM_CH harness (per-value emit, no output buffer)."""
    with tempfile.TemporaryDirectory(dir=build_tmp()) as tmp:
        scratch = Path(tmp) / "proj"
        shutil.copytree(project, scratch, ignore=shutil.ignore_patterns("Debug", "Release"))
        install(scratch, model, H, D, "none", replay_ckpt, config=config, data=data, geo=geo,
                stream_ch=stream_ch)
        if relax_ld:
            # chip_verdict.py: link the scratch image against oversized regions so an
            # over-budget cell still reports both sizes; the caller compares them with
            # the real 128 K / 20 K (the scratch firmware is never flashed)
            ld = scratch / "STM32F103RBTX_FLASH.ld"
            ld.write_text(ld.read_text().replace("LENGTH = 20K", "LENGTH = 65536K")
                          .replace("LENGTH = 128K", "LENGTH = 65536K"))
        if NATIVE_LINUX:
            r = _build_linux(scratch)
        else:
            r = _build_windows(scratch)
        out = (r.stdout or "") + (r.stderr or "")
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


def _build_linux(scratch: Path) -> subprocess.CompletedProcess:
    """The same compile / link / size steps as the Windows batch file, as a shell script."""
    gcc, size = ARM_BIN_LINUX / "arm-none-eabi-gcc", ARM_BIN_LINUX / "arm-none-eabi-size"
    sh = ["set -e", f'[ -x "{gcc}" ] || {{ echo NO_TOOLCHAIN gcc 14.3; exit 5; }}', "mkdir obj",
          "for f in Core/Src/*.c Drivers/STM32F1xx_HAL_Driver/Src/*.c; do",
          f'  n=$(basename "$f" .c); "{gcc}" "$f" {CFLAGS} -o obj/$n.o 2> obj/$n.err || '
          '{ echo COMPILE_FAIL $f; grep -i error obj/$n.err; exit 3; }', "done",
          f'"{gcc}" -mcpu=cortex-m3 -g3 -DDEBUG -c -x assembler-with-cpp --specs=nano.specs '
          "-mfloat-abi=soft -mthumb Core/Startup/startup_stm32f103rbtx.s -o obj/startup.o",
          "ls obj/*.o | LC_ALL=C sort -f > objs.rsp",      # the order cmd lists them in
          f'"{gcc}" -o fw.elf @objs.rsp {LDFLAGS} 2> link.err || {{ echo LINK_FAIL; cat link.err; exit 4; }}',
          f'"{size}" -A fw.elf']
    (scratch / "build.sh").write_text("\n".join(sh) + "\n", encoding="ascii")
    return subprocess.run(["bash", "build.sh"], cwd=str(scratch), capture_output=True, text=True, errors="replace")


def _build_windows(scratch: Path) -> subprocess.CompletedProcess:
    bat = ["@echo off", "set B="]
    bat += [f'if not defined B if exist "{d}\\arm-none-eabi-gcc.exe" set "B={d}"' for d in ARM_BIN]
    bat += ['if not defined B (echo NO_TOOLCHAIN gcc 14.3 & exit /b 5)',
            "mkdir obj",
            "for %%f in (Core\\Src\\*.c Drivers\\STM32F1xx_HAL_Driver\\Src\\*.c) do (",
            f'  "%B%\\arm-none-eabi-gcc.exe" %%f {CFLAGS} -o obj\\%%~nf.o 2> obj\\%%~nf.err || '
            '(echo COMPILE_FAIL %%f & findstr /i error obj\\%%~nf.err & exit /b 3)', ")",
            '"%B%\\arm-none-eabi-gcc.exe" -mcpu=cortex-m3 -g3 -DDEBUG -c -x assembler-with-cpp --specs=nano.specs '
            "-mfloat-abi=soft -mthumb Core\\Startup\\startup_stm32f103rbtx.s -o obj\\startup.o || exit /b 3",
            "(for %%o in (obj\\*.o) do @echo obj/%%~nxo) > objs.rsp",      # '\' is an escape in a .rsp
            f'"%B%\\arm-none-eabi-gcc.exe" -o fw.elf @objs.rsp {LDFLAGS} 2> link.err || '
            '(echo LINK_FAIL & type link.err & exit /b 4)',
            '"%B%\\arm-none-eabi-size.exe" -A fw.elf']
    (scratch / "build.bat").write_text("\r\n".join(bat) + "\r\n", encoding="ascii")
    bat_path = str(scratch / "build.bat") if os.name == "nt" else subprocess.run(
        ["wslpath", "-w", str(scratch / "build.bat")], capture_output=True, text=True).stdout.strip()
    return subprocess.run(["cmd.exe" if os.name != "nt" else "cmd", "/c", bat_path], cwd=str(scratch),
                          capture_output=True, text=True, errors="replace")   # cmd speaks the system codepage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--D", type=int, default=7)
    ap.add_argument("--power", choices=["none", "active", "idle", "max"], default="none")
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--replay_ckpt", default=None,
                    help="install the streaming replay firmware with this trained checkpoint "
                         "(drive it with replay.py)")
    ap.add_argument("--verify", action="store_true",
                    help="dry-build this cell in a scratch copy instead of installing")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model is required")
    check_cell(a.model, a.H, a.D)
    if a.verify:
        r = dry_build(a.project, a.model, a.H, a.D, a.replay_ckpt)
        print(r)
        sys.exit(0 if r["ok"] else 1)
    install(a.project, a.model, a.H, a.D, a.power, a.replay_ckpt)
    print(f"prepared {MODELS[a.model][1]}  H={a.H}  D={a.D}  power={a.power}\n"
          f"  -> {a.project}\n"
          "In CubeIDE: Project -> Clean, Build (hammer), Run (green play), read Tera Term.")


if __name__ == "__main__":
    main()
