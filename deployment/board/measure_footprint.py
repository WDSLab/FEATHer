# -*- coding: utf-8 -*-
"""Measure per-model flash / SRAM footprint by CROSS-COMPILING for Cortex-M3.

Why measured and not estimated: we report (2026-08-24) that params x 4 B
mispredicts flash by up to 84x, so a deployability table built on that
arithmetic would contradict our own finding.  This compiles each port for the
real target and reads the section sizes out of the object file.

    flash = .text + .rodata     (code + weight tables)
    sram  = .data + .bss        (the static arena)

Both EXCLUDE the firmware floor (10.4 KB measured on the F103RB,
BOARD_RESULTS_20260824.md) and the caller's input buffer (L*D*4 B, identical
for every model).  Those are added back when deciding what fits.

SRAM also has to hold the STACK, which .data+.bss does not cover.  Ignoring it
overstates what fits: FEATHer-mix at D=14 lands 504 B under the 20 KB line, so
whether it fits is decided entirely by the stack allowance.  Default 528 B =
the whole-firmware peak stack measured on 2026-08-24 (that figure includes the
HAL and the reporting path, so it is conservative for the model alone; the
frozen port's own call tree is 208 B by -fstack-usage: fea2_forward 16 ->
fea2_begin 104 -> fea_dwconv_same 88, no recursion, no VLAs).  Override with
--stack_bytes and the CSV records what was charged.

This does not need the board — only the ARM toolchain.  Latency and current
still require hardware.

    python deployment/board/measure_footprint.py
    python deployment/board/measure_footprint.py --horizons 96,720
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FLASH_BYTES = 128 * 1024
SRAM_BYTES = 20 * 1024
FIRMWARE_FLOOR = int(10.4 * 1024)     # measured, not assumed
STACK_BYTES = 528                     # measured whole-firmware peak (08-24)
SEQ_LEN, D_MODEL = 96, 7              # MetroPT3 deployment cell


def _geom(pfx):
    """The usual case: one translation unit parameterised by horizon+width."""
    return lambda H, D: [f"-D{pfx}_H={H}", f"-D{pfx}_D={D}"]


# FEATHer-CI is the one two-file port: feather2_infer.c is compiled at WIDTH 1
# and feather2ci_infer.c loops it over the real channel count.  Both objects
# are charged to the model, exactly as the firmware image would carry them.
def _ci_geom(H, D):
    return [f"-DFEA2_H={H}", "-DFEA2_D=1", f"-DFEA2CI_D={D}"]


# name -> (infer source stems, export script stem, -D builder, needs fea_kernels)
MODELS = {
    "FEATHer":       (["feather2_infer"],  "feather2",  _geom("FEA2"), True),
    "FEATHer_CI":    (["feather2_infer",
                       "feather2ci_infer"], "feather2ci", _ci_geom,     True),
    "DLinear":       (["dlinear_infer"],   "dlinear",   _geom("DL"),   False),
    # 2026-09-30, ported for fairness like DLinear.  The cycle table is C x D with
    # C = 298 (MetroPT3's cycle) at every D - the largest mfg cycle, so the flash
    # figure is an upper bound for the other datasets (BlastFurnace C = 24).
    "CycleNet":      (["cyclenet_infer"],  "cyclenet",  _geom("CN"),   False),
    "SparseTSF":     (["stsf_infer"],      "stsf",      _geom("STSF"), True),
    "SparseTSF_MLP": (["stsf_mlp_infer"],  "stsf_mlp",  _geom("SMLP"), True),
    "MixLinear":     (["mixlinear_infer"], "mixlinear", _geom("ML"),   True),
    "DiPE_Linear":   (["dipe_infer"],      "dipe",      _geom("DIPE"), True),
    "FITS":          (["fits_infer"],      "fits",      _geom("FITS"), True),
    # in1 candidate (2026-09-15, not the frozen model): mixer reads the input,
    # sensors finish one at a time.  Two schedules, see src/feather2in_infer.h:
    # _strict keeps x_norm (rule 4), plain recomputes it per sensor.
    "FEATHer_in1r4":        (["feather2in_infer"], "feather2in",
                             lambda H, D: _geom("FEA2I")(H, D)
                             + ["-DFEA2I_RECOMPUTE_NORM=1"], True),
    # 09-04 mix model under the SAME recompute concession as the candidate
    # (2026-09-22 fairness check): no (L, D) arena, bands computed twice per sensor.
    "FEATHer_recompute":    (["feather2r_infer"], "feather2", _geom("FEA2"), True),
    # same, summary R=4 (hidden mixing kept): does the old design reach 43 too?
    "FEATHer_recompute_r4": (["feather2r_infer"], "feather2",
                             lambda H, D: _geom("FEA2")(H, D) + ["-DFEA2_MLP_R=4"], True),
    # residual-MLP width sweep for the ablation table (2026-10-04): same port, R = 2 / 8
    "FEATHer_recompute_r2": (["feather2r_infer"], "feather2",
                             lambda H, D: _geom("FEA2")(H, D) + ["-DFEA2_MLP_R=2"], True),
    "FEATHer_recompute_r8": (["feather2r_infer"], "feather2",
                             lambda H, D: _geom("FEA2")(H, D) + ["-DFEA2_MLP_R=8"], True),
    # CI variant of the re-frozen model (2026-09-23, mlp_rank 4)
    "FEATHer_CI_r4":        (["feather2_infer", "feather2ci_infer"], "feather2ci",
                             lambda H, D: _ci_geom(H, D) + ["-DFEA2_MLP_R=4"], True),
    "FEATHer_in1r4_strict": (["feather2in_infer"], "feather2in",
                             lambda H, D: _geom("FEA2I")(H, D)
                             + ["-DFEA2I_RECOMPUTE_NORM=0"], True),
}

CC = "arm-none-eabi-gcc"
SIZE = "arm-none-eabi-size"
ARM_FLAGS = ["-mcpu=cortex-m3", "-mthumb", "-O2",
             "-ffp-contract=off", "-fno-fast-math",
             "-Isrc", "-Igolden", "-c"]


def wsl(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run inside WSL from the board directory (the ARM toolchain lives there)."""
    if shutil.which(CC):
        return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    inner = "cd \"$(wslpath '%s')\" && %s" % (HERE, " ".join(cmd))
    return subprocess.run(["wsl.exe", "-e", "bash", "-lc", inner],
                          capture_output=True, text=True)


def sections(obj: str) -> dict[str, int]:
    r = wsl([SIZE, "-A", obj])
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(\.\w[\w.]*)\s+(\d+)", line.strip())
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + int(m.group(2))
    return out


# extra export flags for variants that change a weight shape (2026-09-22)
EXPORT_EXTRA = {"FEATHer_recompute_r2": ["--mlp_rank", "2"],
                "FEATHer_recompute_r8": ["--mlp_rank", "8"],
                "FEATHer_recompute_r4": ["--mlp_rank", "4"],
                "FEATHer_CI_r4": ["--mlp_rank", "4"]}


def export(stem: str, H: int, D: int, extra=()) -> tuple[bool, str]:
    py = sys.executable
    r = subprocess.run([py, str(HERE / f"export_golden_{stem}.py"),
                        "--pred_len", str(H),
                        "--d_model", str(D)] + list(extra),
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()
        return False, (tail[-1] if tail else "export failed")
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="96,192,336,720")
    ap.add_argument("--dims", default="7",
                    help="channel counts to sweep, e.g. 1,4,8,16,32")
    ap.add_argument("--out", default="../../results/board_footprint.csv")
    ap.add_argument("--stack_bytes", type=int, default=STACK_BYTES,
                    help="SRAM charged for the stack (see module docstring)")
    ap.add_argument("--models", default=None,
                    help="comma list of MODELS keys (default: all); use with "
                         "--out so a partial run does not replace the main CSV")
    a = ap.parse_args()
    HS = [int(v) for v in a.horizons.split(",")]
    DS = [int(v) for v in a.dims.split(",")]
    if a.models:
        keep = [s.strip() for s in a.models.split(",")]
        unknown = [k for k in keep if k not in MODELS]
        if unknown:
            raise SystemExit("unknown model(s): %s" % ", ".join(unknown))
        run_models = {k: MODELS[k] for k in keep}
    else:
        run_models = MODELS

    if wsl([CC, "--version"]).returncode != 0:
        raise SystemExit(f"{CC} not found (WSL or PATH)")

    # shared kernel object is compiled once; its cost is charged to every model
    # that links it, exactly as the firmware image would carry it
    wsl([CC] + ARM_FLAGS + ["src/fea_kernels.c", "-o", "/tmp/fea_kernels.o"])
    kern = sections("/tmp/fea_kernels.o")
    kern_flash = kern.get(".text", 0) + kern.get(".rodata", 0)

    rows = []
    for name, (srcs, stem, geom, uses_kern) in run_models.items():
      for D in DS:
        for H in HS:
            ok, why = export(stem, H, D, EXPORT_EXTRA.get(name, ()))
            if not ok:
                rows.append(dict(model=name, H=H, D=D, flash=None, sram=None,
                                 fits="NO", note=why[:110]))
                print(f"{name:<14} D={D:<3} H={H:<4} EXCLUDED - {why[:80]}")
                continue
            flash = sram = 0
            failed = None
            for src in srcs:
                obj = f"/tmp/{name}_{src}_{H}_{D}.o"
                r = wsl([CC] + ARM_FLAGS + geom(H, D)
                        + [f"src/{src}.c", "-o", obj])
                if r.returncode != 0:
                    err = [l for l in r.stderr.splitlines() if "error" in l]
                    failed = err[0][:110] if err else "compile error"
                    break
                s = sections(obj)
                flash += s.get(".text", 0) + s.get(".rodata", 0)
                sram += s.get(".data", 0) + s.get(".bss", 0)
            if failed is not None:
                rows.append(dict(model=name, H=H, D=D, flash=None, sram=None,
                                 fits="BUILD FAIL", note=failed))
                print(f"{name:<14} D={D:<3} H={H:<4} BUILD FAIL")
                continue
            if uses_kern:
                flash += kern_flash
            total_flash = flash + FIRMWARE_FLOOR
            # static arena + caller's input buffer + stack
            total_sram = sram + SEQ_LEN * D * 4 + a.stack_bytes
            fits = ("YES" if (total_flash <= FLASH_BYTES
                              and total_sram <= SRAM_BYTES) else "NO")
            rows.append(dict(model=name, H=H, D=D, flash=flash, sram=sram,
                             total_flash=total_flash, total_sram=total_sram,
                             stack=a.stack_bytes,
                             sram_headroom=SRAM_BYTES - total_sram,
                             fits=fits, note=""))
            print(f"{name:<14} D={D:<3} H={H:<4} flash {total_flash/1024:6.1f} KB  "
                  f"sram {total_sram/1024:5.2f} KB   {fits}")

    out = (HERE / a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "D", "H", "flash", "sram", "total_flash", "total_sram",
            "stack", "sram_headroom", "fits", "note"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print(f"budgets: flash {FLASH_BYTES//1024} KB (firmware floor "
          f"{FIRMWARE_FLOOR/1024:.1f} KB) / SRAM {SRAM_BYTES//1024} KB "
          f"(+ input buffer L*D*4 B + {a.stack_bytes} B stack)")


if __name__ == "__main__":
    main()
