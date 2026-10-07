# -*- coding: utf-8 -*-
"""Automated board measurement (2026-09-29): build -> flash -> read -> CSV.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma list of plan numbers")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--max_s", type=int, default=120, help="serial wait per cell")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--project", type=Path, default=bp.DEFAULT_PROJECT)
    ap.add_argument("--power", choices=["active", "idle", "max"], default=None,
                    help="flash ONE cell (--only N) in a power mode and stop: the harness "
                         "prints nothing, read the current on the multimeter in JP6")
    a = ap.parse_args()

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
            except Exception as e:                         # keep going, record the failure
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
