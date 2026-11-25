# -*- coding: utf-8 -*-
"""Board measurement of the tuned settings: build -> flash -> read -> CSV.

For every pick of the tuning (protocol.tuned(): one per tier, model, dataset, H) at the dataset's real channel count,
on either board, this script
  1. installs the cell into ONE scratch copy of the CubeIDE project
     (board_prep.install: golden export, sources, board_cfg.h, main.c.board);
  2. compiles the whole firmware with the compiler bundled in STM32CubeIDE and
     the flags of the project's own Debug build (so every cell - ours and the
     baselines' - is built by the same toolchain);
  3. flashes it with STM32_Programmer_CLI (write + verify + reset);
  4. reads the harness line from the ST-LINK virtual COM port until the same
     measurement has been printed twice (the harness is deterministic, so two
     identical lines confirm a clean run);
  5. appends one row to the CSV (resumable: finished cells are skipped; a re-tuned pick is measured again).

Only latency, golden PASS/FAIL and peak stack come from here (current: board_prep.py --power firmware, measured by power_run.py).
golden_B = the harness's golden input + output in flash (not part of the deployed model).

    python deployment/board/tools/board_run.py --models FEATHer_core --tiers chip --plan
    python deployment/board/tools/board_run.py --models FEATHer_core --tiers chip                  # F103, COM3
    python deployment/board/tools/board_run.py --models FEATHer_core --tiers chip --board f411 --port COM4
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
import chip_verdict as cv

sys.path.insert(0, str(bp.ROOT / "tools" / "results"))
import protocol as P   # noqa: E402

TOOLS = Path(r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins"
             r"\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin")
CLI = Path(r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins"
           r"\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin"
           r"\STM32_Programmer_CLI.exe")
GCC, SIZE = TOOLS / "arm-none-eabi-gcc.exe", TOOLS / "arm-none-eabi-size.exe"

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
HAL_DIR, STARTUP = "STM32F1xx_HAL_Driver", "startup_stm32f103rbtx.s"

# Second board NUCLEO-F411RE (Cortex-M4F, FPU on), latency only.  Its own CubeMX project (84 MHz, USART2)
# with the harness's USER CODE blocks spliced into its main.c; the same flags with the M4F hard-float ABI.
F411 = dict(project=Path.home() / "STM32Cube_projects" / "f411_test", hal="STM32F4xx_HAL_Driver",
            startup="startup_stm32f411retx.s")
F411_ARCH = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb"]
BLOCK = re.compile(r"/\* USER CODE BEGIN (\w+) \*/.*?/\* USER CODE END \1 \*/", re.S)
BOARD_MEM = {"f103": (bp.FLASH_BYTES, bp.RAM_BYTES), "f411": (512 * 1024, 128 * 1024)}

OUT = {b: bp.ROOT / "results" / "board" / f"board_latency_tuned_{b}.csv" for b in ("f103", "f411")}
COLS = ["board", "tier", "model", "trial", "data", "H", "D", "status", "bad", "maxabs_e6", "cycles",
        "latency_us", "latency_ms", "stack_B", "flash_B", "ram_B", "golden_B", "gcc", "timestamp"]

LINE = re.compile(r"^(?P<name>.+?)\s+H=(?P<H>\d+) D=(?P<D>\d+) (?P<status>PASS|FAIL) "
                  r"bad=(?P<bad>\d+) maxabs\(x1e6\)=(?P<maxabs>\d+) cycles=(?P<cycles>\d+) "
                  r"latency_us=(?P<lat>\d+) stack=(?P<stack>\d+) B")

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


def use_f411() -> None:
    global CFLAGS, ASFLAGS, LDFLAGS, HAL_DIR, STARTUP
    m3 = ("-mcpu=cortex-m3", "-mfloat-abi=soft", "-mthumb")
    arch = lambda flags: F411_ARCH + [f for f in flags if f not in m3]
    CFLAGS = ["-DSTM32F411xE" if f == "-DSTM32F103xB" else f.replace("STM32F1xx", "STM32F4xx") for f in arch(CFLAGS)]
    ASFLAGS = arch(ASFLAGS)
    LDFLAGS = ["STM32F411RETX_FLASH.ld" if f == "STM32F103RBTX_FLASH.ld" else f for f in arch(LDFLAGS)]
    HAL_DIR, STARTUP = F411["hal"], F411["startup"]


def splice_main(proj: Path) -> None:
    """F411: the harness's USER CODE blocks (installed main.c = main.c.board) put into the project's own CubeMX
    main.c (kept by board_prep.install as main.c.bak_pre_board_prep); only the CubeMX init code differs.
    A block the harness lacks (main.c.replay has no 1 / Init / SysInit / WHILE) keeps the project's own."""
    src = proj / "Core" / "Src"
    harness = {}
    for m in BLOCK.finditer((src / "main.c").read_text(encoding="utf-8")):
        harness.setdefault(m.group(1), []).append(m.group(0))
    base = (src / "main.c.bak_pre_board_prep").read_text(encoding="utf-8")
    with (src / "main.c").open("w", encoding="utf-8", newline="\n") as f:
        f.write(BLOCK.sub(lambda m: harness[m.group(1)].pop(0) if harness.get(m.group(1)) else m.group(0), base))


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
        for c in sorted((proj / "Drivers" / HAL_DIR / "Src").glob("*.c")):
            run([GCC, f"Drivers/{HAL_DIR}/Src/{c.name}"] + CFLAGS
                + ["-o", hal_cache / (c.stem + ".o")], cwd=proj)
        run([GCC] + ASFLAGS + [f"Core/Startup/{STARTUP}",
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


def stlink_sn(com: str) -> str:
    """Serial number of the ST-LINK whose virtual COM port is `com` (with two boards attached the programmer
    must be told which one, or it takes the first it finds)."""
    ps = (f"$d = Get-PnpDevice -PresentOnly | Where-Object {{ $_.FriendlyName -match '\\({com}\\)' }}; "
          "(Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_Device_Parent).Data")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    sn = r.stdout.strip().rsplit("\\", 1)[-1]
    if not re.fullmatch(r"[0-9A-F]{24}", sn):
        raise RuntimeError(f"no ST-LINK found behind {com}: {r.stdout.strip()} {r.stderr.strip()}")
    return sn


def flash_board(elf: Path, port=None) -> None:
    conn = [CLI, "-c", "port=SWD"] + ([f"sn={stlink_sn(port)}"] if port else [])
    try:
        out = run(conn + ["-w", elf, "-v", "-rst"])
    except RuntimeError as e:
        if "No STM32 target found" not in str(e):
            raise
        # a core sleeping in WFI (power=idle firmware) cannot be attached:
        # connect while the reset line is held
        out = run(conn + ["mode=UR", "-w", elf, "-v", "-rst"])
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


def cell(tier: str, model: str, data: str, H: int, pick: dict) -> dict:
    """install() arguments of one tuned pick.  The printed name carries the pick (consecutive cells share
    model / H / D, so the reader must not match the previous firmware's lines)."""
    import tune_budget as TB
    bm, config, geo = TB.firmware(model, pick["knobs"])
    if geo is not None:
        geo = {**(cv.geo_of(bm, data, H) or {}), **geo}
        config = f"{tier}:{pick['trial']}"                    # printed name only (no --config for these ports)
    return dict(tier=tier, model=model, trial=pick["trial"], data=data, H=H, bm=bm, config=config, geo=geo)


def tuned_cells(models: list[str], hs: list[int], tiers: list[str]) -> list[dict]:
    out = []
    for key, pick in sorted(P.tuned().items()):
        tier, model, data, H = key.split("|")
        if tier in tiers and model in models and int(H) in hs and data in cv.DATA_D \
                and model in P.tune_models(tier):          # size.yaml tier_excluded
            out.append(cell(tier, model, data, int(H), pick))
    return out


def _append(path, row):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=P.FEATHER_CORE, help="registry models")
    ap.add_argument("--H", default="96,192,336,720")
    ap.add_argument("--tiers", default="1k,chip")
    ap.add_argument("--board", choices=["f103", "f411"], default="f103",
                    help="f411: NUCLEO-F411RE (FPU), its own project and output CSV")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--max_s", type=int, default=120, help="serial wait per cell")
    ap.add_argument("--only", default=None, help="comma list of cell numbers (see --plan)")
    ap.add_argument("--plan", action="store_true", help="print the cells and stop")
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cells = tuned_cells([m.strip() for m in a.models.split(",") if m.strip()], [int(h) for h in a.H.split(",")],
                        [t for t in a.tiers.split(",") if t])
    if a.plan:
        for i, c in enumerate(cells, 1):
            D = cv.DATA_D[c["data"]]
            print(f"{i:3d}. {c['tier']:<4} {c['model']:<14} {c['trial']:<5} {c['data']:<12} D={D:<3} H={c['H']:<3} "
                  f"golden {(96 + c['H']) * D * 4 / 1024:.1f} KB")
        if not cells:
            print(f"no tuned picks for these models / tiers / H in {P.TUNED_FILE}")
        return
    if a.board == "f411":
        use_f411()
    project = a.project or (F411["project"] if a.board == "f411" else bp.DEFAULT_PROJECT)
    out = a.out or OUT[a.board]
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {(r["tier"], r["model"], r["trial"], r["data"], int(r["H"])) for r in csv.DictReader(f)
                    if r["status"] in ("PASS", "NOFIT")}
    flash_max, ram_max = BOARD_MEM[a.board]
    gcc_ver = run([GCC, "--version"]).splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp) / "proj"
        shutil.copytree(project, proj, ignore=shutil.ignore_patterns("Debug", "Release"))
        hal = Path(tmp) / "hal"
        for i, c in enumerate(cells, 1):
            key = (c["tier"], c["model"], c["trial"], c["data"], c["H"])
            if (a.only and str(i) not in a.only.split(",")) or key in done:
                continue
            H, D, bm = c["H"], cv.DATA_D[c["data"]], c["bm"]
            tag = f"{i:3d}. {c['tier']:<4} {c['model']:<14} {c['trial']:<5} {c['data']:<12} D={D:<3} H={H:<3}"
            base = dict(board=a.board, tier=c["tier"], model=c["model"], trial=c["trial"], data=c["data"], H=H, D=D,
                        golden_B=(96 + H) * D * 4, gcc=gcc_ver, timestamp=dt.datetime.now().isoformat(timespec="seconds"))
            try:
                bp.install(proj, bm, H, D, "none", config=c["config"], geo=c["geo"], stream_ch=bm in bp.STREAM_CH,
                           data=c["data"] if bm in bp.CONFIG_MODELS else None)
                if a.board == "f411":
                    splice_main(proj)
                elf, fl, ram = build(proj, hal)
            except Exception as e:
                msg = str(e)
                if "overflow" in msg or "will not fit" in msg:
                    _append(out, {**base, "status": "NOFIT"})
                    print(f"{tag}  does not fit the board (image incl. golden {base['golden_B'] / 1024:.1f} KB)", flush=True)
                else:
                    print(f"{tag}  ERROR  {msg.splitlines()[-1][:140]}", flush=True)
                continue
            if fl > flash_max or ram > ram_max:
                _append(out, {**base, "status": "NOFIT", "flash_B": fl, "ram_B": ram})
                print(f"{tag}  does not fit the board  flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB", flush=True)
                continue
            try:
                flash_board(elf)
                lines = read_lines(a.port, f"{bp.MODELS[bm][1]}[{c['config']}]", H, D, 2, a.max_s)
            except Exception as e:
                print(f"{tag}  ERROR  {str(e).splitlines()[0][:120]}", flush=True)
                continue
            # runs of several seconds can differ by a cycle or two between the two prints (seen: 1 in 334 M); the
            # two lines still count as the same measurement within 1e-6 (the first one is recorded)
            if len(lines) < 2 or abs(int(lines[0]["cycles"]) - int(lines[1]["cycles"])) > max(0, int(lines[0]["cycles"]) // 10**6):
                print(f"{tag}  UNSTABLE/NO OUTPUT  {lines}", flush=True)
                continue
            r = lines[0]
            _append(out, {**base, "status": r["status"], "bad": r["bad"], "maxabs_e6": r["maxabs"],
                          "cycles": r["cycles"], "latency_us": r["lat"], "latency_ms": f"{int(r['lat']) / 1000:.2f}",
                          "stack_B": r["stack"], "flash_B": fl, "ram_B": ram})
            print(f"{tag}  {r['status']}  {int(r['lat']) / 1000:9.2f} ms  stack {r['stack']} B  "
                  f"flash {fl / 1024:.1f} KB  RAM {ram / 1024:.2f} KB", flush=True)


if __name__ == "__main__":
    main()
