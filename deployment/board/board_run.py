# -*- coding: utf-8 -*-
"""Automated board measurement: build -> flash -> read -> CSV.

For every cell of board_prep.PLAN (+ the direct-DFT A/B cells) this script
  1. installs the cell into ONE scratch copy of the CubeIDE project
     (board_prep.install: golden export, sources, board_cfg.h, main.c.board);
  2. compiles the whole firmware with the compiler bundled in STM32CubeIDE and
     the flags of the project's own Debug build (so every cell - ours and the
     baselines' - is built by the same toolchain);
  3. flashes it with STM32_Programmer_CLI (write + verify + reset);
  4. reads the harness line from the ST-LINK virtual COM port until the same
     measurement has been printed twice (the harness is deterministic, so two
     identical lines confirm a clean run);
  5. appends one row to the CSV (resumable: finished cells are skipped).

Only latency, golden PASS/FAIL and peak stack come from here.  Current / energy
need the multimeter in JP6 and stay manual (BOARD_RUNBOOK.md section 3).

    python deployment/board/board_run.py                 # all cells, resumable
    python deployment/board/board_run.py --only 9,21     # plan numbers (see --plan)
    python deployment/board/board_run.py --plan
    python deployment/board/board_run.py --candidates --plan   # every candidate setting
    python deployment/board/board_run.py --candidates          # of the CAND_MODELS
    python deployment/board/board_run.py --stream              # STREAM_CELLS at real channel count
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import board_prep as bp

TOOLS = Path(r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins"
             r"\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin")
CLI = Path(r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins"
           r"\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin"
           r"\STM32_Programmer_CLI.exe")
GCC, SIZE = TOOLS / "arm-none-eabi-gcc.exe", TOOLS / "arm-none-eabi-size.exe"
OUT = bp.ROOT / "results" / "board" / "board_latency_20260929.csv"
# These models change structure with the author setting, which is chosen on validation.
# Every candidate is measured up front (at the MetroPT3 deployment cell, period 24), so the
# chosen one is read off this file without another board session.  Candidates with the same
# structure at a horizon build the same firmware: measured once, the others in `same_as`.
OUT_CAND = bp.ROOT / "results" / "board" / "board_latency_candidates_20261007.csv"
CAND_MODELS = {"cmos": "CMoS", "timebase": "TimeBase", "phaseformer": "PhaseFormer",
               "stsf_mlp": "SparseTSF_MLP", "dipe": "DiPE_Linear", "mixlinear": "MixLinear",
               "fits": "FITS"}


def _struct_key(m, reg, cfg, H):
    """What decides the firmware of one candidate at horizon H (MetroPT3 cell)."""
    import importlib
    if m in ("cmos", "timebase", "phaseformer"):
        return tuple(sorted(importlib.import_module(
            f"export_golden_{bp.MODELS[m][2]}").geometry(cfg, H, "MetroPT3").items()))
    from tools.results.protocol import resolve
    _, _, ov = resolve(reg, "mfg", "MetroPT3", H, "cand:" + cfg)
    if m == "stsf_mlp":
        return (ov.get("sparsetsf_d_model", 128),)
    if m == "dipe":
        return (ov.get("dipe_num_experts", 1), bool(ov.get("dipe_use_revin", True)))
    if m == "mixlinear":                       # lpf is clamped to the input segments (96 / 24);
        return (min(int(ov.get("mixlinear_lpf", 15)), 96 // 24),)   # alpha is a constant only
    if m == "fits":
        import argparse
        sys.path.insert(0, str(bp.ROOT))
        from baselines.FITS.wrapper import _cut_freq
        return (_cut_freq(argparse.Namespace(**ov), 96), bool(ov.get("fits_individual", False)))
    raise KeyError(m)
CAND_CELLS = [(96, 7), (96, 1), (720, 1)]
CAND_COLS = ["model", "config", "same_as", "H", "D", "status", "bad", "maxabs_e6", "cycles",
             "latency_us", "latency_ms", "stack_B", "flash_B", "ram_B", "gcc", "timestamp"]

# = the CubeIDE project's Debug configuration (Debug/Core/Src/subdir.mk, Debug/makefile)
CFLAGS = ["-mcpu=cortex-m3", "-std=gnu11", "-g3", "-DDEBUG", "-DUSE_HAL_DRIVER",
          "-DSTM32F103xB", "-c", "-ICore/Inc", "-IDrivers/STM32F1xx_HAL_Driver/Inc",
          "-IDrivers/STM32F1xx_HAL_Driver/Inc/Legacy",
          "-IDrivers/CMSIS/Device/ST/STM32F1xx/Include", "-IDrivers/CMSIS/Include",
          "-O2", "-ffunction-sections", "-fdata-sections", "-Wall", "--specs=nano.specs",
          "-mfloat-abi=soft", "-mthumb"]
ASFLAGS = ["-mcpu=cortex-m3", "-g3", "-DDEBUG", "-c", "-x", "assembler-with-cpp",
           "--specs=nano.specs", "-mfloat-abi=soft", "-mthumb"]
LDFLAGS = ["-mcpu=cortex-m3", "-T", "STM32F103RBTX_FLASH.ld", "--specs=nosys.specs",
           "-Wl,--gc-sections", "-static", "--specs=nano.specs", "-mfloat-abi=soft",
           "-mthumb", "-Wl,--start-group", "-lc", "-lm", "-Wl,--end-group"]

LINE = re.compile(r"^(?P<name>.+?)\s+H=(?P<H>\d+) D=(?P<D>\d+) (?P<status>PASS|FAIL) "
                  r"bad=(?P<bad>\d+) maxabs\(x1e6\)=(?P<maxabs>\d+) cycles=(?P<cycles>\d+) "
                  r"latency_us=(?P<lat>\d+) stack=(?P<stack>\d+) B")
COLS = ["idx", "model", "H", "D", "direct_dft", "status", "bad", "maxabs_e6", "cycles",
        "latency_us", "latency_ms", "stack_B", "flash_B", "ram_B", "gcc", "timestamp"]

PS_READ = r"""
$p = New-Object System.IO.Ports.SerialPort "{port}",115200,"None",8,"One"
$p.Open(); $buf = ""; $sw = [Diagnostics.Stopwatch]::StartNew()
# count only complete lines that START with this cell's name: a fragment left in
# the driver by the previous firmware cannot match
$rx = "(?m)^" + [regex]::Escape("{name}") + " +H={H} D={D} (PASS|FAIL) .*stack=\d+ B"
while ($sw.ElapsedMilliseconds -lt {max_ms}) {{
  $buf += $p.ReadExisting()
  if (([regex]::Matches($buf.Replace("`r", ""), $rx)).Count -ge {want}) {{ break }}
  Start-Sleep -Milliseconds 100
}}
$p.Close(); $buf
"""


def run(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed:\n{(r.stdout + r.stderr)[-1500:]}")
    return r.stdout


def build(proj: Path, hal_cache: Path) -> tuple[Path, int, int]:
    obj = proj / "obj"
    obj.mkdir(exist_ok=True)
    objs = []
    for c in sorted((proj / "Core" / "Src").glob("*.c")):
        o = obj / (c.stem + ".o")
        run([GCC, f"Core/Src/{c.name}"] + CFLAGS + ["-o", o], cwd=proj)
        objs.append(o)
    if not any(hal_cache.glob("*.o")):                       # HAL does not depend on the cell
        hal_cache.mkdir(exist_ok=True)
        for c in sorted((proj / "Drivers" / "STM32F1xx_HAL_Driver" / "Src").glob("*.c")):
            run([GCC, f"Drivers/STM32F1xx_HAL_Driver/Src/{c.name}"] + CFLAGS
                + ["-o", hal_cache / (c.stem + ".o")], cwd=proj)
        run([GCC] + ASFLAGS + ["Core/Startup/startup_stm32f103rbtx.s",
                               "-o", hal_cache / "startup.o"], cwd=proj)
    objs += sorted(hal_cache.glob("*.o"))
    elf = proj / "fw.elf"
    run([GCC, "-o", elf] + objs + LDFLAGS, cwd=proj)
    sec = {}
    for line in run([SIZE, "-A", elf]).splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].startswith(".") and p[1].isdigit():
            sec[p[0]] = int(p[1])
    flash = sum(sec.get(k, 0) for k in (".isr_vector", ".text", ".rodata", ".ARM",
                                        ".init_array", ".fini_array", ".data"))
    ram = sum(sec.get(k, 0) for k in (".data", ".bss", "._user_heap_stack"))
    return elf, flash, ram


def flash_board(elf: Path) -> None:
    try:
        out = run([CLI, "-c", "port=SWD", "-w", elf, "-v", "-rst"])
    except RuntimeError as e:
        if "No STM32 target found" not in str(e):
            raise
        # a core sleeping in WFI (power=idle firmware) cannot be attached:
        # connect while the reset line is held
        out = run([CLI, "-c", "port=SWD", "mode=UR", "-w", elf, "-v", "-rst"])
    if "verified successfully" not in out.lower() and "download verified" not in out.lower():
        raise RuntimeError("programmer did not report a verified download:\n" + out[-800:])


def read_lines(port: str, name: str, H: int, D: int, want: int, max_s: int) -> list[dict]:
    ps = PS_READ.format(port=port, max_ms=max_s * 1000, want=want, name=name, H=H, D=D)
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = []
    for line in r.stdout.replace("\r", "").split("\n"):
        m = LINE.match(line)
        if m and m["name"].strip() == name and int(m["H"]) == H and int(m["D"]) == D:
            rows.append(m.groupdict())
    return rows


def candidate_cells() -> list[tuple]:
    """(model, H, D, config, same_as) - one per distinct structure; same_as lists the other
    candidates that resolve to the same structure at that horizon."""
    sys.path.insert(0, str(bp.ROOT))
    from tools.results.protocol import CANDIDATES
    cells = []
    for m, reg in CAND_MODELS.items():
        for H, D in CAND_CELLS:
            groups = {}
            for cfg in CANDIDATES[reg]:
                groups.setdefault(_struct_key(m, reg, cfg, H), []).append(cfg)
            for cfgs in groups.values():
                cells.append((m, H, D, cfgs[0], ";".join(cfgs[1:])))
    return cells


def run_candidates(a) -> None:
    cells = candidate_cells()
    if a.plan:
        for i, (m, H, D, cfg, same) in enumerate(cells, 1):
            print(f"{i:2d}. {m:<11} H={H:<3} D={D}  {cfg}{'  (= ' + same + ')' if same else ''}")
        return
    pick = set(int(x) for x in a.only.split(",")) if a.only else None
    done = set()
    if a.out_cand.exists():
        with a.out_cand.open(encoding="utf-8") as f:
            done = {(r["model"], int(r["H"]), int(r["D"]), r["config"])
                    for r in csv.DictReader(f) if r["status"] in ("PASS", "NOFIT")}
    gcc_ver = run([GCC, "--version"]).splitlines()[0]

    def write(row):
        new = not a.out_cand.exists()
        with a.out_cand.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAND_COLS)
            if new:
                w.writeheader()
            w.writerow(row)

    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        hal = Path(tmp) / "hal"
        for i, (m, H, D, cfg, same) in enumerate(cells, 1):
            if (pick and i not in pick) or (m, H, D, cfg) in done:
                continue
            tag = f"{i:2d}. {m:<11} H={H:<3} D={D} {cfg:<11}"
            base = dict(model=m, config=cfg, same_as=same, H=H, D=D, gcc=gcc_ver,
                        timestamp=dt.datetime.now().isoformat(timespec="seconds"))
            try:
                bp.check_cell(m, H, D)
                bp.install(proj, m, H, D, "none", config=cfg)
                elf, fl, ram = build(proj, hal)
            except Exception as e:
                msg = str(e)
                if "overflow" in msg or "will not fit" in msg:
                    write({**base, "status": "NOFIT"})     # the part's flash / RAM is too small
                    print(f"{tag}  does not fit the chip", flush=True)
                else:
                    print(f"{tag}  ERROR  {msg.splitlines()[0][:120]}", flush=True)
                continue
            if fl > bp.FLASH_BYTES or ram > bp.RAM_BYTES:
                write({**base, "status": "NOFIT", "flash_B": fl, "ram_B": ram})
                print(f"{tag}  does not fit the chip  flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB",
                      flush=True)
                continue
            try:
                flash_board(elf)
                lines = read_lines(a.port, f"{bp.MODELS[m][1]}[{cfg}]", H, D, 2, a.max_s)
            except Exception as e:
                print(f"{tag}  ERROR  {str(e).splitlines()[0][:120]}", flush=True)
                continue
            if len(lines) < 2 or lines[0]["cycles"] != lines[1]["cycles"]:
                print(f"{tag}  UNSTABLE/NO OUTPUT  {lines}", flush=True)
                continue
            r = lines[0]
            write({**base, "status": r["status"], "bad": r["bad"], "maxabs_e6": r["maxabs"],
                   "cycles": r["cycles"], "latency_us": r["lat"],
                   "latency_ms": f"{int(r['lat']) / 1000:.2f}", "stack_B": r["stack"],
                   "flash_B": fl, "ram_B": ram})
            print(f"{tag}  {r['status']}  {int(r['lat']) / 1000:9.2f} ms  stack {r['stack']} B  "
                  f"flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB", flush=True)


# The tightest on-chip cell of every channel-streaming port at its chip-best setting
# (deployment/board/chip_verdict.py --candidates), run at the dataset's REAL channel count with
# <pfx>_forward_ch (one channel at a time, H-float output).  (board model, setting, dataset, H)
STREAM_CELLS = [("cmos", "Weather", "BlastFurnace", 720), ("dipe", "Electricity", "BlastFurnace", 720),
                ("fits", "ETTm1", "BlastFurnace", 336), ("fits", "ETTm1", "Steel", 720),
                ("stsf_mlp", "Traffic", "BlastFurnace", 720), ("mixlinear", "Traffic", "BlastFurnace", 720),
                ("timebase", "ETTm2", "BlastFurnace", 720), ("phaseformer", "ETTh2", "BlastFurnace", 720),
                ("phaseformer", "ETTh2", "BlastFurnace", 336), ("stsf", None, "BlastFurnace", 720)]
OUT_STREAM = bp.ROOT / "results" / "board" / "board_stream_20261012.csv"
STREAM_COLS = ["model", "config", "data", "H", "D", "status", "bad", "maxabs_e6", "cycles",
               "latency_us", "latency_ms", "stack_B", "flash_B", "ram_B", "gcc", "timestamp"]


def run_stream(a) -> None:
    import chip_verdict as cv
    gcc_ver = run([GCC, "--version"]).splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        hal = Path(tmp) / "hal"
        for i, (m, cfg, data, H) in enumerate(STREAM_CELLS, 1):
            if a.only and str(i) not in a.only.split(","):
                continue
            D = cv.DATA_D[data]
            tag = f"{i:2d}. {m:<11} {str(cfg):<11} {data:<12} D={D:<2} H={H:<3}"
            base = dict(model=m, config=cfg or "", data=data, H=H, D=D, gcc=gcc_ver,
                        timestamp=dt.datetime.now().isoformat(timespec="seconds"))
            try:
                bp.install(proj, m, H, D, "none", config=cfg, stream_ch=True,
                           data=data if m in bp.CONFIG_MODELS else None, geo=cv.geo_of(m, data, H))
                elf, fl, ram = build(proj, hal)
            except Exception as e:
                msg = str(e)
                st = "NOFIT" if ("overflow" in msg or "will not fit" in msg) else "ERROR"
                print(f"{tag}  {st}  {msg.splitlines()[-1][:140] if st == 'ERROR' else ''}", flush=True)
                if st == "NOFIT":
                    _append(OUT_STREAM, STREAM_COLS, {**base, "status": "NOFIT"})
                continue
            try:
                flash_board(elf)
                name = bp.MODELS[m][1] + (f"[{cfg}]" if cfg else "")
                lines = read_lines(a.port, name, H, D, 2, a.max_s)
            except Exception as e:
                print(f"{tag}  ERROR  {str(e).splitlines()[0][:120]}", flush=True)
                continue
            if not lines:
                print(f"{tag}  NO OUTPUT (flash {fl / 1024:.1f} KB, RAM {ram / 1024:.2f} KB)", flush=True)
                continue
            r = lines[0]
            _append(OUT_STREAM, STREAM_COLS, {**base, "status": r["status"], "bad": r["bad"],
                    "maxabs_e6": r["maxabs"], "cycles": r["cycles"], "latency_us": r["lat"],
                    "latency_ms": f"{int(r['lat']) / 1000:.2f}", "stack_B": r["stack"],
                    "flash_B": fl, "ram_B": ram})
            print(f"{tag}  {r['status']}  {int(r['lat']) / 1000:9.2f} ms  stack {r['stack']} B  "
                  f"flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB  ({len(lines)} line(s))", flush=True)


def _append(path, cols, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma list of plan numbers")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--max_s", type=int, default=120, help="serial wait per cell")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--candidates", action="store_true",
                    help="measure every candidate setting of CMoS / TimeBase / PhaseFormer (10-07)")
    ap.add_argument("--out_cand", type=Path, default=OUT_CAND)
    ap.add_argument("--project", type=Path, default=bp.DEFAULT_PROJECT)
    ap.add_argument("--power", choices=["active", "idle", "max"], default=None,
                    help="flash ONE cell (--only N) in a power mode and stop: the harness "
                         "prints nothing, read the current on the multimeter in JP6")
    ap.add_argument("--stream", action="store_true",
                    help="10-12: the tightest cell of each channel-streaming port at its real channel "
                         "count (STREAM_CELLS) -> results/board/board_stream_20261012.csv")
    a = ap.parse_args()

    if a.stream:
        run_stream(a)
        return
    if a.candidates:
        run_candidates(a)
        return
    if a.power:
        cells = ([(m, H, D, False) for m, H, D in bp.PLAN] + [(m, H, D, True) for m, H, D in bp.AB_PLAN]
                 + [(m, H, D, False) for m, H, D in bp.EXTRA_PLAN])
        i = int(a.only)
        m, H, D, direct = cells[i - 1]
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "proj"
            shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
            bp.install(proj, m, H, D, a.power, direct)
            elf, fl, ram = build(proj, Path(tmp) / "hal")
            flash_board(elf)
        print(f"flashed {i}. {m} H={H} D={D} power={a.power} - read the multimeter")
        return

    cells = ([(m, H, D, False) for m, H, D in bp.PLAN] + [(m, H, D, True) for m, H, D in bp.AB_PLAN]
             + [(m, H, D, False) for m, H, D in bp.EXTRA_PLAN])
    if a.plan:
        for i, c in enumerate(cells, 1):
            print(i, *c)
        return
    pick = set(int(x) for x in a.only.split(",")) if a.only else None
    done = set()
    if a.out.exists():
        with a.out.open(encoding="utf-8") as f:
            done = {int(r["idx"]) for r in csv.DictReader(f) if r["status"] == "PASS"}
    gcc_ver = run([GCC, "--version"]).splitlines()[0]

    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(a.project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        hal = Path(tmp) / "hal"
        for i, (m, H, D, direct) in enumerate(cells, 1):
            if (pick and i not in pick) or i in done:
                continue
            tag = f"{i:2d}. {m:<11} H={H:<3} D={D:<2}{' direct' if direct else ''}"
            try:
                bp.check_cell(m, H, D)
                bp.install(proj, m, H, D, "none", direct)
                elf, fl, ram = build(proj, hal)
                flash_board(elf)
                name = bp.MODELS[m][1] + (" (direct DFT)" if direct else "")
                lines = read_lines(a.port, name, H, D, 2, a.max_s)
            except Exception as e:                         # keep going
                print(f"{tag}  ERROR  {str(e).splitlines()[0][:120]}", flush=True)
                continue
            if len(lines) < 2 or lines[0]["cycles"] != lines[1]["cycles"]:
                print(f"{tag}  UNSTABLE/NO OUTPUT  {lines}", flush=True)
                continue
            r = lines[0]
            row = dict(idx=i, model=m, H=H, D=D, direct_dft=int(direct), status=r["status"],
                       bad=r["bad"], maxabs_e6=r["maxabs"], cycles=r["cycles"],
                       latency_us=r["lat"], latency_ms=f"{int(r['lat']) / 1000:.2f}",
                       stack_B=r["stack"], flash_B=fl, ram_B=ram, gcc=gcc_ver,
                       timestamp=dt.datetime.now().isoformat(timespec="seconds"))
            new = not a.out.exists()
            with a.out.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"{tag}  {r['status']}  {int(r['lat']) / 1000:9.2f} ms  stack {r['stack']} B  "
                  f"flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB", flush=True)


if __name__ == "__main__":
    main()
